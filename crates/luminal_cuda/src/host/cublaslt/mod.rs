use std::sync::{Arc, Mutex, OnceLock};

use luminal::{
    egglog_utils::{extract_dtype, extract_expr},
    op::{
        DType, EgglogOp, LLIROp,
        OpParam::{self, *},
    },
    prelude::{
        tracing::{Level, span, trace},
        *,
    },
};

use crate::{
    cudarc::{
        cublas::sys::cublasOperation_t,
        cublaslt::{
            CudaBlasLT, MatmulShared,
            sys::{
                cublasComputeType_t, cublasLtMatmul, cublasLtMatmulAlgoGetHeuristic,
                cublasLtMatmulDesc_t, cublasLtMatmulDescCreate, cublasLtMatmulDescDestroy,
                cublasLtMatmulDescSetAttribute, cublasLtMatmulHeuristicResult_t,
                cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t,
                cublasLtMatmulPreferenceCreate, cublasLtMatmulPreferenceDestroy,
                cublasLtMatmulPreferenceSetAttribute, cublasLtMatrixLayout_t,
                cublasLtMatrixLayoutCreate, cublasLtMatrixLayoutDestroy, cudaDataType,
            },
        },
        driver::{CudaSlice, CudaStream, DevicePtr},
    },
    host::{HostOp, cublas::parse_cublas_op},
};

const CUBLASLT_WORKSPACE_SIZE: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq)]
struct CapturedMatmulConfig {
    m: u64,
    n: u64,
    k: u64,
    a_layout: cublasOperation_t,
    b_layout: cublasOperation_t,
    lda: i64,
    ldb: i64,
    ldc: i64,
    dtype: DType,
}

#[derive(Debug)]
struct CapturedMatmulState {
    config: CapturedMatmulConfig,
    matmul_desc: cublasLtMatmulDesc_t,
    a_desc: cublasLtMatrixLayout_t,
    b_desc: cublasLtMatrixLayout_t,
    c_desc: cublasLtMatrixLayout_t,
    heuristic: cublasLtMatmulHeuristicResult_t,
}

impl Drop for CapturedMatmulState {
    fn drop(&mut self) {
        unsafe {
            if !self.c_desc.is_null() {
                cublasLtMatrixLayoutDestroy(self.c_desc);
            }
            if !self.b_desc.is_null() {
                cublasLtMatrixLayoutDestroy(self.b_desc);
            }
            if !self.a_desc.is_null() {
                cublasLtMatrixLayoutDestroy(self.a_desc);
            }
            if !self.matmul_desc.is_null() {
                cublasLtMatmulDescDestroy(self.matmul_desc);
            }
        }
    }
}

// SAFETY: descriptors are CUDA-context-scoped opaque handles, and this runtime uses one CUDA
// device/context per process. Access is synchronized via Mutex when mutated.
unsafe impl Send for CapturedMatmulState {}
// SAFETY: same rationale as Send; read-only use after construction is safe under shared access.
unsafe impl Sync for CapturedMatmulState {}

#[derive(Debug)]
#[allow(dead_code)]
pub struct CuBlasLt {
    m: Expression,
    n: Expression,
    k: Expression,
    a_layout: cublasOperation_t,
    b_layout: cublasOperation_t,
    lda: Expression,
    ldb: Expression,
    ldc: Expression,
    dtype: DType,
    cublaslt: OnceLock<Arc<CudaBlasLT>>,
    workspace: OnceLock<CudaSlice<u8>>,
    capture_cublaslt: OnceLock<Arc<CudaBlasLT>>,
    capture_workspace: OnceLock<CudaSlice<u8>>,
    capture_state: Mutex<Option<CapturedMatmulState>>,
}

// Useless default for IntoEgglogOp
impl Default for CuBlasLt {
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
            dtype: DType::F32,
            cublaslt: OnceLock::new(),
            workspace: OnceLock::new(),
            capture_cublaslt: OnceLock::new(),
            capture_workspace: OnceLock::new(),
            capture_state: Mutex::new(None),
        }
    }
}

impl EgglogOp for CuBlasLt {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "cublaslt".to_string(),
            //    A      B      m     n      k  , A input Layout, B input Layout, lda, ldb, ldc, dtype
            vec![
                Input, Input, Expr, Expr, Expr, Str, Str, Expr, Expr, Expr, Dty,
            ],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            include_str!["cublaslt_RmRm_rewrite.egg"].to_string(), // row row
            include_str!["cublaslt_RmCm_rewrite.egg"].to_string(), // row col
            include_str!["cublaslt_CmRm_rewrite.egg"].to_string(), // col row
            include_str!["cublaslt_CmCm_rewrite.egg"].to_string(), // col col
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

        // Extract dtype from egglog
        let dtype = extract_dtype(egraph, children[10]);

        let extracted_state = Self {
            m,
            n,
            k,
            a_layout,
            b_layout,
            lda,
            ldb,
            ldc,
            dtype,
            cublaslt: OnceLock::new(),
            workspace: OnceLock::new(),
            capture_cublaslt: OnceLock::new(),
            capture_workspace: OnceLock::new(),
            capture_state: Mutex::new(None),
        };
        trace!(?extracted_state);

        let extracted = LLIROp::new::<dyn HostOp>(Box::new(extracted_state) as Box<dyn HostOp>);

        (extracted, vec![children[0], children[1]])
    }

    fn cleanup(&self) -> bool {
        false
    }
}

/// Convert DType to CUDA types for cuBLAS LT
/// Returns (matrix_dtype, compute_type, scale_dtype)
fn dtype_to_cuda_types(dtype: DType) -> (cudaDataType, cublasComputeType_t, cudaDataType) {
    match dtype {
        // F32: matrix=f32, compute=f32, scale=f32
        DType::F32 => (
            cudaDataType::CUDA_R_32F,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cudaDataType::CUDA_R_32F,
        ),
        // F16: matrix=f16, compute=f32 (FP32 accumulation for accuracy), scale=f32
        DType::F16 => (
            cudaDataType::CUDA_R_16F,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cudaDataType::CUDA_R_32F,
        ),
        // BF16: matrix=bf16, compute=f32 with tensor cores, scale=f32
        DType::Bf16 => (
            cudaDataType::CUDA_R_16BF,
            cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
            cudaDataType::CUDA_R_32F,
        ),
        DType::Int => panic!("cuBLAS LT does not support integer matmul"),
        DType::Bool => panic!("cuBLAS LT does not support bool matmul"),
        DType::NvFp4 | DType::Mxfp4 => todo!("cuBLAS LT FP4 matmul not yet implemented"),
    }
}

fn destroy_cublaslt_descriptors(
    matmul_desc: &mut cublasLtMatmulDesc_t,
    a_desc: &mut cublasLtMatrixLayout_t,
    b_desc: &mut cublasLtMatrixLayout_t,
    c_desc: &mut cublasLtMatrixLayout_t,
) {
    unsafe {
        if !c_desc.is_null() {
            cublasLtMatrixLayoutDestroy(*c_desc);
            *c_desc = std::ptr::null_mut();
        }
        if !b_desc.is_null() {
            cublasLtMatrixLayoutDestroy(*b_desc);
            *b_desc = std::ptr::null_mut();
        }
        if !a_desc.is_null() {
            cublasLtMatrixLayoutDestroy(*a_desc);
            *a_desc = std::ptr::null_mut();
        }
        if !matmul_desc.is_null() {
            cublasLtMatmulDescDestroy(*matmul_desc);
            *matmul_desc = std::ptr::null_mut();
        }
    }
}

impl CuBlasLt {
    fn ensure_workspace<'a>(
        &'a self,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<&'a CudaSlice<u8>> {
        if self.workspace.get().is_none() {
            let workspace = unsafe { stream.alloc::<u8>(CUBLASLT_WORKSPACE_SIZE)? };
            let _ = self.workspace.set(workspace);
        }
        Ok(self
            .workspace
            .get()
            .expect("cuBLASLt workspace should be initialized"))
    }

    fn ensure_capture_workspace<'a>(
        &'a self,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<&'a CudaSlice<u8>> {
        if self.capture_workspace.get().is_none() {
            let workspace = unsafe { stream.alloc::<u8>(CUBLASLT_WORKSPACE_SIZE)? };
            let _ = self.capture_workspace.set(workspace);
        }
        Ok(self
            .capture_workspace
            .get()
            .expect("cuBLASLt capture workspace should be initialized"))
    }

    fn build_capture_config(&self, dyn_map: &FxHashMap<char, usize>) -> CapturedMatmulConfig {
        CapturedMatmulConfig {
            m: self.m.exec(dyn_map).unwrap() as u64,
            n: self.n.exec(dyn_map).unwrap() as u64,
            k: self.k.exec(dyn_map).unwrap() as u64,
            a_layout: self.a_layout,
            b_layout: self.b_layout,
            lda: self.lda.exec(dyn_map).unwrap() as i64,
            ldb: self.ldb.exec(dyn_map).unwrap() as i64,
            ldc: self.ldc.exec(dyn_map).unwrap() as i64,
            dtype: self.dtype,
        }
    }

    fn ensure_capture_state(
        &self,
        cublaslt: &Arc<CudaBlasLT>,
        config: CapturedMatmulConfig,
    ) -> anyhow::Result<()> {
        let mut state_guard = self
            .capture_state
            .lock()
            .map_err(|_| anyhow::anyhow!("cuBLASLt capture state mutex poisoned"))?;
        if state_guard
            .as_ref()
            .is_some_and(|existing| existing.config == config)
        {
            return Ok(());
        }

        let (cuda_dtype, compute_type, scale_dtype) = dtype_to_cuda_types(config.dtype);
        let mut matmul_desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        let mut a_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut b_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut c_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut preference: cublasLtMatmulPreference_t = std::ptr::null_mut();
        let mut heuristic: cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
        let mut algo_count: i32 = 0;

        let build_res = (|| -> anyhow::Result<()> {
            unsafe {
                cublasLtMatmulDescCreate(&mut matmul_desc, compute_type, scale_dtype).result()?;
                cublasLtMatmulDescSetAttribute(
                    matmul_desc,
                    cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                    &config.a_layout as *const _ as *const std::ffi::c_void,
                    std::mem::size_of::<cublasOperation_t>(),
                )
                .result()?;
                cublasLtMatmulDescSetAttribute(
                    matmul_desc,
                    cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                    &config.b_layout as *const _ as *const std::ffi::c_void,
                    std::mem::size_of::<cublasOperation_t>(),
                )
                .result()?;

                let (a_rows, a_cols) = if config.a_layout == cublasOperation_t::CUBLAS_OP_N {
                    (config.m, config.k)
                } else {
                    (config.k, config.m)
                };
                let (b_rows, b_cols) = if config.b_layout == cublasOperation_t::CUBLAS_OP_N {
                    (config.k, config.n)
                } else {
                    (config.n, config.k)
                };

                cublasLtMatrixLayoutCreate(&mut a_desc, cuda_dtype, a_rows, a_cols, config.lda)
                    .result()?;
                cublasLtMatrixLayoutCreate(&mut b_desc, cuda_dtype, b_rows, b_cols, config.ldb)
                    .result()?;
                cublasLtMatrixLayoutCreate(&mut c_desc, cuda_dtype, config.m, config.n, config.ldc)
                    .result()?;

                cublasLtMatmulPreferenceCreate(&mut preference).result()?;
                cublasLtMatmulPreferenceSetAttribute(
                    preference,
                    cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &CUBLASLT_WORKSPACE_SIZE as *const _ as *const std::ffi::c_void,
                    std::mem::size_of::<usize>(),
                )
                .result()?;

                cublasLtMatmulAlgoGetHeuristic(
                    *cublaslt.handle(),
                    matmul_desc,
                    a_desc,
                    b_desc,
                    c_desc,
                    c_desc,
                    preference,
                    1,
                    &mut heuristic,
                    &mut algo_count,
                )
                .result()?;
            }
            Ok(())
        })();
        unsafe {
            if !preference.is_null() {
                cublasLtMatmulPreferenceDestroy(preference);
            }
        }

        if let Err(err) = build_res {
            destroy_cublaslt_descriptors(&mut matmul_desc, &mut a_desc, &mut b_desc, &mut c_desc);
            return Err(err);
        }
        if algo_count == 0 {
            destroy_cublaslt_descriptors(&mut matmul_desc, &mut a_desc, &mut b_desc, &mut c_desc);
            anyhow::bail!("No suitable cuBLASLT algorithm found");
        }

        *state_guard = Some(CapturedMatmulState {
            config,
            matmul_desc,
            a_desc,
            b_desc,
            c_desc,
            heuristic,
        });
        Ok(())
    }

    fn run_captured_matmul(
        &self,
        cublaslt: &Arc<CudaBlasLT>,
        stream: &Arc<CudaStream>,
        state: &CapturedMatmulState,
        a_ptr: u64,
        b_ptr: u64,
        c_ptr: u64,
        workspace_ptr: u64,
    ) -> anyhow::Result<()> {
        let alpha_f32: f32 = 1.0;
        let beta_f32: f32 = 0.0;
        unsafe {
            cublasLtMatmul(
                *cublaslt.handle(),
                state.matmul_desc,
                &alpha_f32 as *const _ as *const std::ffi::c_void,
                a_ptr as *const std::ffi::c_void,
                state.a_desc,
                b_ptr as *const std::ffi::c_void,
                state.b_desc,
                &beta_f32 as *const _ as *const std::ffi::c_void,
                c_ptr as *const std::ffi::c_void,
                state.c_desc,
                c_ptr as *mut std::ffi::c_void,
                state.c_desc,
                &state.heuristic.algo,
                workspace_ptr as *mut std::ffi::c_void,
                CUBLASLT_WORKSPACE_SIZE,
                stream.cu_stream() as *mut _,
            )
            .result()?;
        }
        Ok(())
    }
}

impl HostOp for CuBlasLt {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        // GEMM parameters
        let m = self.m.exec(dyn_map).unwrap() as u64;
        let n = self.n.exec(dyn_map).unwrap() as u64;
        let k = self.k.exec(dyn_map).unwrap() as u64;
        let a_layout = self.a_layout;
        let b_layout = self.b_layout;
        let lda = self.lda.exec(dyn_map).unwrap() as i64;
        let ldb = self.ldb.exec(dyn_map).unwrap() as i64;
        let ldc = self.ldc.exec(dyn_map).unwrap() as i64;

        // Get CUDA types based on dtype
        let (cuda_dtype, compute_type, scale_dtype) = dtype_to_cuda_types(self.dtype);
        let element_size = match self.dtype {
            DType::F32 => 4u64,
            DType::F16 | DType::Bf16 => 2u64,
            DType::Int | DType::Bool => panic!("cuBLAS LT does not support integer/bool matmul"),
            DType::NvFp4 | DType::Mxfp4 => todo!("cuBLAS LT FP4 matmul not yet implemented"),
        };

        // Alpha/beta scale values (all dtypes use F32 scale type)
        let alpha_f32: f32 = 1.0;
        let beta_f32: f32 = 0.0;

        // Get buffers: output is self_node, inputs are from graph edges
        let c_buf = buffers[&self_node];
        let a_buf = buffers[&inputs[0]];
        let b_buf = buffers[&inputs[1]];

        // Get device pointers
        let (a_ptr, _a_guard) = a_buf.device_ptr(stream);
        let (b_ptr, _b_guard) = b_buf.device_ptr(stream);
        let (c_ptr, _c_guard) = c_buf.device_ptr(stream);

        // Debug tracing
        trace!(
            "buffer_validation {}=={},{}=={},{}=={}",
            a_buf.len(),
            m * k * element_size,
            b_buf.len(),
            k * n * element_size,
            c_buf.len(),
            m * n * element_size
        );
        let _span = span!(
            Level::TRACE,
            "cuBLASLT",
            m, n, k, lda, ldb, ldc, ?a_layout, ?b_layout, ?self.dtype,
        )
        .entered();

        let cublaslt = self
            .cublaslt
            .get_or_init(|| Arc::new(CudaBlasLT::new(stream.clone()).unwrap()));
        let workspace = self.ensure_workspace(stream)?;

        let mut matmul_desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        let mut a_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut b_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut c_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut preference: cublasLtMatmulPreference_t = std::ptr::null_mut();
        let mut heuristic: cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
        let mut algo_count: i32 = 0;

        let (workspace_ptr, _workspace_guard) = workspace.device_ptr(stream);

        unsafe {
            // Create matmul descriptor (compute_type, scale_type for alpha/beta)
            cublasLtMatmulDescCreate(&mut matmul_desc, compute_type, scale_dtype).result()?;

            // Set transpose attributes
            cublasLtMatmulDescSetAttribute(
                matmul_desc,
                cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &a_layout as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<cublasOperation_t>(),
            )
            .result()?;
            cublasLtMatmulDescSetAttribute(
                matmul_desc,
                cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &b_layout as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<cublasOperation_t>(),
            )
            .result()?;

            // Create matrix layout descriptors
            let (a_rows, a_cols) = if a_layout == cublasOperation_t::CUBLAS_OP_N {
                (m, k)
            } else {
                (k, m)
            };
            let (b_rows, b_cols) = if b_layout == cublasOperation_t::CUBLAS_OP_N {
                (k, n)
            } else {
                (n, k)
            };

            cublasLtMatrixLayoutCreate(&mut a_desc, cuda_dtype, a_rows, a_cols, lda).result()?;
            cublasLtMatrixLayoutCreate(&mut b_desc, cuda_dtype, b_rows, b_cols, ldb).result()?;
            cublasLtMatrixLayoutCreate(&mut c_desc, cuda_dtype, m, n, ldc).result()?;

            // Create preference and set workspace size
            cublasLtMatmulPreferenceCreate(&mut preference).result()?;
            cublasLtMatmulPreferenceSetAttribute(
                preference,
                cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &CUBLASLT_WORKSPACE_SIZE as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<usize>(),
            )
            .result()?;

            // Get heuristic (best algorithm)
            cublasLtMatmulAlgoGetHeuristic(
                *cublaslt.handle(),
                matmul_desc,
                a_desc,
                b_desc,
                c_desc,
                c_desc, // D layout same as C
                preference,
                1, // Request 1 result
                &mut heuristic,
                &mut algo_count,
            )
            .result()?;

            if algo_count == 0 {
                // Cleanup before returning error
                cublasLtMatmulPreferenceDestroy(preference);
                cublasLtMatrixLayoutDestroy(c_desc);
                cublasLtMatrixLayoutDestroy(b_desc);
                cublasLtMatrixLayoutDestroy(a_desc);
                cublasLtMatmulDescDestroy(matmul_desc);
                return Err(anyhow::anyhow!("No suitable cuBLASLT algorithm found"));
            }

            // All dtypes use F32 scale type for alpha/beta
            let alpha_ptr = &alpha_f32 as *const _ as *const std::ffi::c_void;
            let beta_ptr = &beta_f32 as *const _ as *const std::ffi::c_void;
            cublasLtMatmul(
                *cublaslt.handle(),
                matmul_desc,
                alpha_ptr,
                a_ptr as *const std::ffi::c_void,
                a_desc,
                b_ptr as *const std::ffi::c_void,
                b_desc,
                beta_ptr,
                c_ptr as *const std::ffi::c_void,
                c_desc,
                c_ptr as *mut std::ffi::c_void,
                c_desc, // D layout same as C
                &heuristic.algo,
                workspace_ptr as *mut std::ffi::c_void,
                CUBLASLT_WORKSPACE_SIZE,
                stream.cu_stream() as *mut _,
            )
            .result()?;

            // Cleanup
            cublasLtMatmulPreferenceDestroy(preference);
            cublasLtMatrixLayoutDestroy(c_desc);
            cublasLtMatrixLayoutDestroy(b_desc);
            cublasLtMatrixLayoutDestroy(a_desc);
            cublasLtMatmulDescDestroy(matmul_desc);
        }

        if std::env::var("LUMINAL_SYNC_DEBUG").map_or(false, |v| v == "1") {
            stream.synchronize()?;
        }
        Ok(())
    }

    fn prepare_for_capture(&self, stream: &Arc<CudaStream>) -> anyhow::Result<()> {
        let _ = self
            .capture_cublaslt
            .get_or_init(|| Arc::new(CudaBlasLT::new(stream.clone()).unwrap()));
        let _ = self.ensure_capture_workspace(stream)?;
        Ok(())
    }

    fn warmup_for_capture(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let config = self.build_capture_config(dyn_map);
        let c_buf = buffers[&self_node];
        let a_buf = buffers[&inputs[0]];
        let b_buf = buffers[&inputs[1]];
        let (a_ptr, _a_guard) = a_buf.device_ptr(stream);
        let (b_ptr, _b_guard) = b_buf.device_ptr(stream);
        let (c_ptr, _c_guard) = c_buf.device_ptr(stream);

        let cublaslt = self
            .capture_cublaslt
            .get_or_init(|| Arc::new(CudaBlasLT::new(stream.clone()).unwrap()));
        self.ensure_capture_state(cublaslt, config)?;
        let workspace = self.ensure_capture_workspace(stream)?;
        let (workspace_ptr, _workspace_guard) = workspace.device_ptr(stream);
        let state_guard = self
            .capture_state
            .lock()
            .map_err(|_| anyhow::anyhow!("cuBLASLt capture state mutex poisoned"))?;
        let state = state_guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("cuBLASLt capture state missing after warmup"))?;
        self.run_captured_matmul(cublaslt, stream, state, a_ptr, b_ptr, c_ptr, workspace_ptr)?;

        if std::env::var("LUMINAL_SYNC_DEBUG").map_or(false, |v| v == "1") {
            stream.synchronize()?;
        }
        Ok(())
    }

    fn execute_for_capture(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let config = self.build_capture_config(dyn_map);
        let cublaslt = self.capture_cublaslt.get().ok_or_else(|| {
            anyhow::anyhow!("cuBLASLt capture handle not initialized; prepare_for_capture missing")
        })?;

        let c_buf = buffers[&self_node];
        let a_buf = buffers[&inputs[0]];
        let b_buf = buffers[&inputs[1]];
        let (a_ptr, _a_guard) = a_buf.device_ptr(stream);
        let (b_ptr, _b_guard) = b_buf.device_ptr(stream);
        let (c_ptr, _c_guard) = c_buf.device_ptr(stream);
        let workspace = self.capture_workspace.get().ok_or_else(|| {
            anyhow::anyhow!(
                "cuBLASLt capture workspace not initialized; prepare_for_capture missing"
            )
        })?;
        let (workspace_ptr, _workspace_guard) = workspace.device_ptr(stream);

        let state_guard = self
            .capture_state
            .lock()
            .map_err(|_| anyhow::anyhow!("cuBLASLt capture state mutex poisoned"))?;
        let state = state_guard.as_ref().ok_or_else(|| {
            anyhow::anyhow!("cuBLASLt capture state missing; warmup_for_capture not run")
        })?;
        if state.config != config {
            return Err(anyhow::anyhow!(
                "cuBLASLt capture state config mismatch; recapture required"
            ));
        }
        self.run_captured_matmul(cublaslt, stream, state, a_ptr, b_ptr, c_ptr, workspace_ptr)?;

        if std::env::var("LUMINAL_SYNC_DEBUG").map_or(false, |v| v == "1") {
            stream.synchronize()?;
        }
        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.m * self.n
    }

    fn output_bytes(&self) -> Expression {
        let elem_size: Expression = match self.dtype {
            DType::F32 | DType::Int => 4,
            DType::F16 | DType::Bf16 => 2,
            DType::Bool => 1,
            DType::NvFp4 | DType::Mxfp4 => todo!("FP4 element size not yet implemented"),
        }
        .into();
        self.output_size() * elem_size
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("cuBLASLt")
    }
}
