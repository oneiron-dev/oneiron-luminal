use cudarc::cublas::{
    CudaBlas,
    sys::{
        cublasOperation_t, cublasSetStream_v2, cublasSetWorkspace_v2, cublasSgemm_v2,
        cublasStatus_t,
    },
};
use cudarc::driver::{CudaStream, DevicePtr};
use luminal::{
    egglog_utils::extract_expr,
    op::{
        EgglogOp, LLIROp,
        OpParam::{self, *},
    },
    prelude::*,
};
use std::sync::{Arc, OnceLock};
use tracing::{Level, span, trace};

use crate::{cudarc::driver::CudaSlice, host::HostOp};

const CUBLAS_CAPTURE_WORKSPACE_SIZE: usize = 4 * 1024 * 1024;

/// Global shared cuBLAS handle to avoid per-operation workspace allocation
static SHARED_CUBLAS: OnceLock<Arc<CudaBlas>> = OnceLock::new();

/// Parse cuBLAS operation from egglog string (e.g., "\"T\"" -> CUBLAS_OP_T)
pub(crate) fn parse_cublas_op(s: &str) -> cublasOperation_t {
    // Strip quotes if present (egglog strings are stored with quotes)
    let stripped = s.trim_matches('"');
    match stripped {
        "T" => cublasOperation_t::CUBLAS_OP_T,
        "N" => cublasOperation_t::CUBLAS_OP_N,
        "C" => cublasOperation_t::CUBLAS_OP_C,
        other => panic!("Unknown cuBLAS operation: '{other}' (original: '{s}')"),
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct CuBlasSgemmV2 {
    m: Expression,
    n: Expression,
    k: Expression,
    a_layout: cublasOperation_t,
    b_layout: cublasOperation_t,
    lda: Expression,
    ldb: Expression,
    ldc: Expression,
    /// Lazily initialized cuBLAS handle - created on first execute
    cublas: OnceLock<Arc<CudaBlas>>,
    /// Capture-specific cuBLAS handle (must be initialized on the capture stream).
    capture_cublas: OnceLock<Arc<CudaBlas>>,
    /// Capture-only workspace required for cuBLAS stream-capture compatibility.
    capture_workspace: OnceLock<CudaSlice<u8>>,
}

// Useless default for IntoEgglogOp
impl Default for CuBlasSgemmV2 {
    fn default() -> Self {
        Self {
            m: Expression::default(),
            n: Expression::default(),
            k: Expression::default(),
            a_layout: cublasOperation_t::CUBLAS_OP_N, // IGNORE NOT REAL
            b_layout: cublasOperation_t::CUBLAS_OP_T, // IGNORE NOT REAL
            lda: Expression::default(),
            ldb: Expression::default(),
            ldc: Expression::default(),
            cublas: OnceLock::new(),
            capture_cublas: OnceLock::new(),
            capture_workspace: OnceLock::new(),
        }
    }
}

impl EgglogOp for CuBlasSgemmV2 {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "cublasSgemmV2".to_string(),
            //    A      B      m     n      k  , A input Layout, B input Layout,
            vec![Input, Input, Expr, Expr, Expr, Str, Str, Expr, Expr, Expr],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            include_str!["sgemm_v2_RmRm_rewrite.egg"].to_string(), // row row
            include_str!["sgemm_v2_RmCm_rewrite.egg"].to_string(), // row col
            include_str!["sgemm_v2_CmRm_rewrite.egg"].to_string(), // col row
            include_str!["sgemm_v2_CmCm_rewrite.egg"].to_string(), // col col
        ]
    }

    #[allow(unused_variables)]
    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        // Extract dimensions from egglog
        let m = extract_expr(egraph, children[2], expr_cache).unwrap();
        let n = extract_expr(egraph, children[3], expr_cache).unwrap();
        let k = extract_expr(egraph, children[4], expr_cache).unwrap();

        // Extract layout strings from egglog
        let a_layout_str = &egraph.enodes[children[5]].0;
        let b_layout_str = &egraph.enodes[children[6]].0;
        let a_layout = parse_cublas_op(a_layout_str);
        let b_layout = parse_cublas_op(b_layout_str);

        // Extract leading dimensions from egglog
        let lda = extract_expr(egraph, children[7], expr_cache).unwrap();
        let ldb = extract_expr(egraph, children[8], expr_cache).unwrap();
        let ldc = extract_expr(egraph, children[9], expr_cache).unwrap();

        let extracted_state = Self {
            m,
            n,
            k,
            a_layout,
            b_layout,
            lda,
            ldb,
            ldc,
            cublas: OnceLock::new(),
            capture_cublas: OnceLock::new(),
            capture_workspace: OnceLock::new(),
        };
        trace!(?extracted_state);

        let extracted = LLIROp::new::<dyn HostOp>(Box::new(extracted_state) as Box<dyn HostOp>);

        (extracted, vec![children[0], children[1]])
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl CuBlasSgemmV2 {
    fn ensure_capture_workspace<'a>(
        &'a self,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<&'a CudaSlice<u8>> {
        if self.capture_workspace.get().is_none() {
            let workspace = unsafe { stream.alloc::<u8>(CUBLAS_CAPTURE_WORKSPACE_SIZE)? };
            let _ = self.capture_workspace.set(workspace);
        }
        Ok(self
            .capture_workspace
            .get()
            .expect("cuBLAS capture workspace should be initialized"))
    }

    fn run_with_handle(
        &self,
        cublas: &Arc<CudaBlas>,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
        set_stream: bool,
    ) -> anyhow::Result<()> {
        // GEMM parameters
        let m = self.m.exec(dyn_map).unwrap() as i32;
        let n = self.n.exec(dyn_map).unwrap() as i32;
        let k = self.k.exec(dyn_map).unwrap() as i32;
        let a_layout = self.a_layout;
        let b_layout = self.b_layout;
        let lda = self.lda.exec(dyn_map).unwrap() as i32;
        let ldb = self.ldb.exec(dyn_map).unwrap() as i32;
        let ldc = self.ldc.exec(dyn_map).unwrap() as i32;

        let alpha = 1.0f32;
        let beta = 0.0f32;

        // Get buffers: output is self_node, inputs are from graph edges
        let c_buf = buffers[&self_node];
        let a_buf = buffers[&inputs[0]];
        let b_buf = buffers[&inputs[1]];

        // Get device pointers
        let (a_ptr, _a_guard) = a_buf.device_ptr(stream);
        let (b_ptr, _b_guard) = b_buf.device_ptr(stream);
        let (c_ptr, _c_guard) = c_buf.device_ptr(stream);

        // Debug: Check buffer sizes
        trace!(
            "buffer_validation {}=={},{}=={},{}=={}",
            a_buf.len(),
            m * k * 4,
            b_buf.len(),
            k * n * 4,
            c_buf.len(),
            m * n * 4
        );
        let _sgemm_span = span!(
            Level::TRACE,
            "cuBLAS_SGEMM_V2",
            m,
            n,
            k,
            alpha,
            beta,
            lda,
            ldb,
            ldc,
            ?a_layout,
            ?b_layout,
        )
        .entered();

        if set_stream {
            // The CUstream types from cublas::sys and driver::sys are compatible, just cast.
            unsafe {
                cublasSetStream_v2(*cublas.handle(), stream.cu_stream() as _);
            }
        }

        let status = unsafe {
            cublasSgemm_v2(
                *cublas.handle(),
                a_layout,
                b_layout,
                m,
                n,
                k,
                &alpha as *const f32,
                a_ptr as *const f32,
                lda,
                b_ptr as *const f32,
                ldb,
                &beta as *const f32,
                c_ptr as *mut f32,
                ldc,
            )
        };
        if std::env::var("LUMINAL_SYNC_DEBUG").map_or(false, |v| v == "1") {
            stream.synchronize().unwrap();
        }

        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(anyhow::anyhow!(
                "cuBLAS SGEMM TN failed with status: {:?}",
                status
            ));
        }

        Ok(())
    }

    fn run_with_handle_raw(
        &self,
        cublas: &Arc<CudaBlas>,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        raw_buffers: &FxHashMap<NodeIndex, u64>,
        dyn_map: &FxHashMap<char, usize>,
        set_stream: bool,
    ) -> anyhow::Result<()> {
        let m = self.m.exec(dyn_map).unwrap() as i32;
        let n = self.n.exec(dyn_map).unwrap() as i32;
        let k = self.k.exec(dyn_map).unwrap() as i32;
        let a_layout = self.a_layout;
        let b_layout = self.b_layout;
        let lda = self.lda.exec(dyn_map).unwrap() as i32;
        let ldb = self.ldb.exec(dyn_map).unwrap() as i32;
        let ldc = self.ldc.exec(dyn_map).unwrap() as i32;

        let alpha = 1.0f32;
        let beta = 0.0f32;

        let a_ptr = *raw_buffers
            .get(&inputs[0])
            .ok_or_else(|| anyhow::anyhow!("Missing raw pointer for cuBLAS input A"))?;
        let b_ptr = *raw_buffers
            .get(&inputs[1])
            .ok_or_else(|| anyhow::anyhow!("Missing raw pointer for cuBLAS input B"))?;
        let c_ptr = *raw_buffers
            .get(&self_node)
            .ok_or_else(|| anyhow::anyhow!("Missing raw pointer for cuBLAS output C"))?;

        if set_stream {
            unsafe {
                cublasSetStream_v2(*cublas.handle(), stream.cu_stream() as _);
            }
        }

        let status = unsafe {
            cublasSgemm_v2(
                *cublas.handle(),
                a_layout,
                b_layout,
                m,
                n,
                k,
                &alpha as *const f32,
                a_ptr as *const f32,
                lda,
                b_ptr as *const f32,
                ldb,
                &beta as *const f32,
                c_ptr as *mut f32,
                ldc,
            )
        };
        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(anyhow::anyhow!(
                "cuBLAS SGEMM capture failed with status: {:?}",
                status
            ));
        }
        Ok(())
    }
}

impl HostOp for CuBlasSgemmV2 {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        // Use shared cuBLAS handle to avoid per-operation workspace allocation
        let cublas = SHARED_CUBLAS.get_or_init(|| Arc::new(CudaBlas::new(stream.clone()).unwrap()));
        self.run_with_handle(cublas, stream, self_node, inputs, buffers, dyn_map, true)
    }

    fn warmup_for_capture(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let cublas = self
            .capture_cublas
            .get_or_init(|| Arc::new(CudaBlas::new(stream.clone()).unwrap()));
        // Ensure warmup uses the capture stream/handle pairing that will be used for mini-capture.
        self.run_with_handle(cublas, stream, self_node, inputs, buffers, dyn_map, true)
    }

    fn execute_for_capture(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let cublas = self.capture_cublas.get().ok_or_else(|| {
            anyhow::anyhow!("cuBLAS capture handle not initialized; prepare_for_capture missing")
        })?;
        self.run_with_handle(cublas, stream, self_node, inputs, buffers, dyn_map, false)
    }

    fn prepare_for_capture(&self, stream: &Arc<CudaStream>) -> anyhow::Result<()> {
        let cublas = self
            .capture_cublas
            .get_or_init(|| Arc::new(CudaBlas::new(stream.clone()).unwrap()));
        let workspace = self.ensure_capture_workspace(stream)?;
        let (workspace_ptr, _workspace_guard) = workspace.device_ptr(stream);
        let status = unsafe {
            cublasSetWorkspace_v2(
                *cublas.handle(),
                workspace_ptr as *mut std::ffi::c_void,
                CUBLAS_CAPTURE_WORKSPACE_SIZE,
            )
        };
        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(anyhow::anyhow!(
                "cublasSetWorkspace_v2 failed for capture handle: {:?}",
                status
            ));
        }
        Ok(())
    }

    fn execute_for_capture_raw(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        raw_buffers: &FxHashMap<NodeIndex, u64>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let cublas = self.capture_cublas.get().ok_or_else(|| {
            anyhow::anyhow!("cuBLAS capture handle not initialized; prepare_for_capture missing")
        })?;
        self.run_with_handle_raw(
            cublas,
            stream,
            self_node,
            inputs,
            raw_buffers,
            dyn_map,
            false,
        )
    }

    fn output_size(&self) -> Expression {
        self.m * self.n
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("cuBLAS")
    }
}
