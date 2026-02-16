use std::{
    ffi::c_void,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use luminal::{
    egglog_utils::{extract_dtype, extract_expr},
    op::{
        DType, EgglogOp, LLIROp,
        OpParam::{self, *},
    },
    prelude::*,
};

use crate::{
    cudarc::{
        driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr},
        nvrtc::Ptx,
    },
    host::HostOp,
    kernel::cuda_graph::CudaFunctionExt,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashAttentionMode {
    Masked,
    Causal,
}

impl FlashAttentionMode {
    fn kernel_stem(self) -> &'static str {
        match self {
            Self::Masked => "flash_attn_masked",
            Self::Causal => "flash_attn_causal",
        }
    }

    fn term_name(self) -> &'static str {
        match self {
            Self::Masked => "flash_attention_masked",
            Self::Causal => "flash_attention_causal",
        }
    }
}

#[derive(Debug)]
pub struct FlashAttentionOp {
    pub mode: FlashAttentionMode,
    pub batch: Expression,
    pub heads: Expression,
    pub seq_q: Expression,
    pub seq_k: Expression,
    pub head_dim: Expression,
    pub dtype: DType,
    module: OnceLock<Arc<CudaModule>>,
    function: OnceLock<CudaFunction>,
    loaded_config: OnceLock<(i32, usize, FlashAttentionMode)>,
}

impl FlashAttentionOp {
    pub fn new(
        mode: FlashAttentionMode,
        batch: Expression,
        heads: Expression,
        seq_q: Expression,
        seq_k: Expression,
        head_dim: Expression,
        dtype: DType,
    ) -> Self {
        Self {
            mode,
            batch,
            heads,
            seq_q,
            seq_k,
            head_dim,
            dtype,
            module: OnceLock::new(),
            function: OnceLock::new(),
            loaded_config: OnceLock::new(),
        }
    }

    fn kernels_dir() -> PathBuf {
        if let Ok(dir) = std::env::var("LUMINAL_FLASH_ATTN_KERNEL_DIR")
            && !dir.is_empty()
        {
            return PathBuf::from(dir);
        }
        Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels")
    }

    fn kernel_filename(&self, sm: i32, head_dim: usize) -> String {
        format!("{}_h{}_sm{}.cubin", self.mode.kernel_stem(), head_dim, sm)
    }

    fn kernel_entry_name(kernel_path: &Path) -> String {
        let entry_path = PathBuf::from(format!("{}.entry", kernel_path.display()));
        if let Ok(contents) = fs::read_to_string(&entry_path) {
            let entry = contents.trim();
            if !entry.is_empty() {
                return entry.to_string();
            }
        }
        "flash_attn_fwd".to_string()
    }

    fn ensure_kernel_loaded(
        &self,
        stream: &Arc<CudaStream>,
        head_dim: usize,
    ) -> anyhow::Result<&CudaFunction> {
        if head_dim != 64 && head_dim != 128 {
            anyhow::bail!(
                "FlashAttentionOp supports head_dim 64 or 128, got {}",
                head_dim
            );
        }

        let (major, minor) = stream.context().compute_capability()?;
        let sm = major * 10 + minor;

        if let Some((loaded_sm, loaded_head_dim, loaded_mode)) = self.loaded_config.get()
            && (*loaded_sm != sm || *loaded_head_dim != head_dim || *loaded_mode != self.mode)
        {
            anyhow::bail!(
                "FlashAttentionOp was initialized for sm{}_h{}_{:?}, cannot reuse for sm{}_h{}_{:?}",
                loaded_sm,
                loaded_head_dim,
                loaded_mode,
                sm,
                head_dim,
                self.mode
            );
        }

        let kernel_path = Self::kernels_dir().join(self.kernel_filename(sm, head_dim));

        if self.module.get().is_none() {
            let cubin = fs::read(&kernel_path).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to read FlashAttention kernel {}: {e}. Run scripts/compile_flash_attn.py",
                    kernel_path.display()
                )
            })?;
            let module = stream
                .context()
                .load_module(Ptx::from_binary(cubin))
                .map_err(|e| anyhow::anyhow!("Failed to load FlashAttention CUBIN: {e}"))?;
            let _ = self.module.set(module);
            let _ = self.loaded_config.set((sm, head_dim, self.mode));
        }

        if self.function.get().is_none() {
            let module = self
                .module
                .get()
                .expect("FlashAttention module should be initialized");
            let entry_name = Self::kernel_entry_name(&kernel_path);
            let function = module.load_function(&entry_name).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to resolve symbol {} from CUBIN {}: {e}",
                    entry_name,
                    kernel_path.display()
                )
            })?;
            let _ = self.function.set(function);
        }

        Ok(self
            .function
            .get()
            .expect("FlashAttention function should be initialized"))
    }

    fn launch_raw(
        &self,
        stream: &Arc<CudaStream>,
        q_ptr: u64,
        k_ptr: u64,
        v_ptr: u64,
        mask_ptr: u64,
        out_ptr: u64,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        if self.dtype != DType::F32 {
            anyhow::bail!(
                "FlashAttentionOp currently supports only F32 dtype, got {:?}",
                self.dtype
            );
        }

        let batch = self
            .batch
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("Failed to resolve batch expression"))?;
        let heads = self
            .heads
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("Failed to resolve heads expression"))?;
        let seq_q = self
            .seq_q
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("Failed to resolve seq_q expression"))?;
        let seq_k = self
            .seq_k
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("Failed to resolve seq_k expression"))?;
        let head_dim = self
            .head_dim
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("Failed to resolve head_dim expression"))?;

        if batch == 0 || heads == 0 || seq_q == 0 || seq_k == 0 {
            return Ok(());
        }

        if self.mode == FlashAttentionMode::Masked && mask_ptr == 0 {
            anyhow::bail!("FlashAttention masked mode requires a valid mask pointer");
        }

        let function = self.ensure_kernel_loaded(stream, head_dim)?;

        let block_m = 64usize;
        let grid_x = seq_q.div_ceil(block_m) as u32;
        let grid_y = heads as u32;
        let grid_z = batch as u32;

        let block_x = if head_dim == 64 { 128u32 } else { 256u32 };
        let block_y = 1u32;
        let block_z = 1u32;
        let shared_mem = 0u32;

        let mut q_arg = q_ptr;
        let mut k_arg = k_ptr;
        let mut v_arg = v_ptr;
        let mut mask_arg = mask_ptr;
        let mut out_arg = out_ptr;
        let mut batch_arg = batch as i32;
        let mut heads_arg = heads as i32;
        let mut seq_q_arg = seq_q as i32;
        let mut seq_k_arg = seq_k as i32;
        let mut head_dim_arg = head_dim as i32;
        let mut softmax_scale_arg = 1.0f32 / (head_dim as f32).sqrt();

        let mut params = [
            &mut q_arg as *mut u64 as *mut c_void,
            &mut k_arg as *mut u64 as *mut c_void,
            &mut v_arg as *mut u64 as *mut c_void,
            &mut mask_arg as *mut u64 as *mut c_void,
            &mut out_arg as *mut u64 as *mut c_void,
            &mut batch_arg as *mut i32 as *mut c_void,
            &mut heads_arg as *mut i32 as *mut c_void,
            &mut seq_q_arg as *mut i32 as *mut c_void,
            &mut seq_k_arg as *mut i32 as *mut c_void,
            &mut head_dim_arg as *mut i32 as *mut c_void,
            &mut softmax_scale_arg as *mut f32 as *mut c_void,
        ];

        unsafe {
            use crate::cudarc::driver::sys::{CUresult, cuLaunchKernel};
            let result = cuLaunchKernel(
                function.raw_function(),
                grid_x,
                grid_y,
                grid_z,
                block_x,
                block_y,
                block_z,
                shared_mem,
                stream.cu_stream(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if result != CUresult::CUDA_SUCCESS {
                anyhow::bail!(
                    "cuLaunchKernel failed for FlashAttention {:?}: {:?}",
                    self.mode,
                    result
                );
            }
        }

        Ok(())
    }
}

impl Default for FlashAttentionOp {
    fn default() -> Self {
        Self::new(
            FlashAttentionMode::Masked,
            Expression::default(),
            Expression::default(),
            Expression::default(),
            Expression::default(),
            Expression::default(),
            DType::F32,
        )
    }
}

impl EgglogOp for FlashAttentionOp {
    fn term(&self) -> (String, Vec<OpParam>) {
        match self.mode {
            FlashAttentionMode::Masked => (
                self.mode.term_name().to_string(),
                vec![
                    Input, Input, Input, Input, Expr, Expr, Expr, Expr, Expr, Dty,
                ],
            ),
            FlashAttentionMode::Causal => (
                self.mode.term_name().to_string(),
                vec![Input, Input, Input, Expr, Expr, Expr, Expr, Expr, Dty],
            ),
        }
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for FlashAttentionOp {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let q = buffers
            .get(&inputs[0])
            .ok_or_else(|| anyhow::anyhow!("FlashAttention missing Q input"))?;
        let k = buffers
            .get(&inputs[1])
            .ok_or_else(|| anyhow::anyhow!("FlashAttention missing K input"))?;
        let v = buffers
            .get(&inputs[2])
            .ok_or_else(|| anyhow::anyhow!("FlashAttention missing V input"))?;
        let out = buffers
            .get(&self_node)
            .ok_or_else(|| anyhow::anyhow!("FlashAttention missing output buffer"))?;

        let q_ptr = q.device_ptr(stream).0;
        let k_ptr = k.device_ptr(stream).0;
        let v_ptr = v.device_ptr(stream).0;
        let out_ptr = out.device_ptr(stream).0;

        let mask_ptr = if self.mode == FlashAttentionMode::Masked {
            let mask = buffers
                .get(&inputs[3])
                .ok_or_else(|| anyhow::anyhow!("FlashAttention missing mask input"))?;
            mask.device_ptr(stream).0
        } else {
            0
        };

        self.launch_raw(stream, q_ptr, k_ptr, v_ptr, mask_ptr, out_ptr, dyn_map)
    }

    fn warmup_for_capture(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        self.execute(stream, self_node, inputs, buffers, dyn_map)
    }

    fn execute_for_capture_raw(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        raw_buffers: &FxHashMap<NodeIndex, u64>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let q_ptr = *raw_buffers
            .get(&inputs[0])
            .ok_or_else(|| anyhow::anyhow!("FlashAttention missing raw Q pointer"))?;
        let k_ptr = *raw_buffers
            .get(&inputs[1])
            .ok_or_else(|| anyhow::anyhow!("FlashAttention missing raw K pointer"))?;
        let v_ptr = *raw_buffers
            .get(&inputs[2])
            .ok_or_else(|| anyhow::anyhow!("FlashAttention missing raw V pointer"))?;
        let out_ptr = *raw_buffers
            .get(&self_node)
            .ok_or_else(|| anyhow::anyhow!("FlashAttention missing raw output pointer"))?;
        let mask_ptr = if self.mode == FlashAttentionMode::Masked {
            *raw_buffers
                .get(&inputs[3])
                .ok_or_else(|| anyhow::anyhow!("FlashAttention missing raw mask pointer"))?
        } else {
            0
        };

        self.launch_raw(stream, q_ptr, k_ptr, v_ptr, mask_ptr, out_ptr, dyn_map)
    }

    fn output_size(&self) -> Expression {
        self.batch * self.heads * self.seq_q * self.head_dim
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("FlashAttn2")
    }
}

#[derive(Debug, Default)]
pub struct FlashAttentionMasked;

impl EgglogOp for FlashAttentionMasked {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            FlashAttentionMode::Masked.term_name().to_string(),
            vec![
                Input, Input, Input, Input, Expr, Expr, Expr, Expr, Expr, Dty,
            ],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![include_str!("flash_attn_masked_rewrite.egg").to_string()]
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        children: &[&'a ENodeId],
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let op = FlashAttentionOp::new(
            FlashAttentionMode::Masked,
            extract_expr(egraph, children[4], expr_cache).unwrap(),
            extract_expr(egraph, children[5], expr_cache).unwrap(),
            extract_expr(egraph, children[6], expr_cache).unwrap(),
            extract_expr(egraph, children[7], expr_cache).unwrap(),
            extract_expr(egraph, children[8], expr_cache).unwrap(),
            extract_dtype(egraph, children[9]),
        );
        (
            LLIROp::new::<dyn HostOp>(Box::new(op) as Box<dyn HostOp>),
            vec![children[0], children[1], children[2], children[3]],
        )
    }

    fn cleanup(&self) -> bool {
        false
    }
}

#[derive(Debug, Default)]
pub struct FlashAttentionCausal;

impl EgglogOp for FlashAttentionCausal {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            FlashAttentionMode::Causal.term_name().to_string(),
            vec![Input, Input, Input, Expr, Expr, Expr, Expr, Expr, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![include_str!("flash_attn_causal_rewrite.egg").to_string()]
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        children: &[&'a ENodeId],
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let op = FlashAttentionOp::new(
            FlashAttentionMode::Causal,
            extract_expr(egraph, children[3], expr_cache).unwrap(),
            extract_expr(egraph, children[4], expr_cache).unwrap(),
            extract_expr(egraph, children[5], expr_cache).unwrap(),
            extract_expr(egraph, children[6], expr_cache).unwrap(),
            extract_expr(egraph, children[7], expr_cache).unwrap(),
            extract_dtype(egraph, children[8]),
        );
        (
            LLIROp::new::<dyn HostOp>(Box::new(op) as Box<dyn HostOp>),
            vec![children[0], children[1], children[2]],
        )
    }

    fn cleanup(&self) -> bool {
        false
    }
}
