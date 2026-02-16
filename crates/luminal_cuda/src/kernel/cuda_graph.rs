#![allow(clippy::missing_safety_doc, clippy::not_unsafe_ptr_arg_deref)]
//! CUDA Graph API wrappers for explicit graph construction and surgical updates.

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DriverError,
    sys::{self, CUevent, CUfunction, CUgraph, CUgraphExec, CUgraphNode},
};

/// A CUDA graph that can be modified and instantiated.
pub struct CudaGraphHandle {
    pub(crate) cu_graph: CUgraph,
    pub(crate) ctx: Arc<CudaContext>,
}

impl CudaGraphHandle {
    /// Creates a new empty CUDA graph.
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self, DriverError> {
        ctx.bind_to_thread()?;
        let mut graph = MaybeUninit::uninit();
        unsafe {
            sys::cuGraphCreate(graph.as_mut_ptr(), 0).result()?;
            Ok(Self {
                cu_graph: graph.assume_init(),
                ctx,
            })
        }
    }

    /// Adds a kernel node to the graph. kernel_params must remain valid for graph lifetime.
    pub unsafe fn add_kernel_node(
        &mut self,
        dependencies: &[CUgraphNode],
        func: CUfunction,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        kernel_params: *mut *mut c_void,
    ) -> Result<CUgraphNode, DriverError> {
        let params = sys::CUDA_KERNEL_NODE_PARAMS {
            func,
            gridDimX: grid_dim.0,
            gridDimY: grid_dim.1,
            gridDimZ: grid_dim.2,
            blockDimX: block_dim.0,
            blockDimY: block_dim.1,
            blockDimZ: block_dim.2,
            sharedMemBytes: shared_mem_bytes,
            kernelParams: kernel_params,
            extra: std::ptr::null_mut(),
            kern: std::ptr::null_mut(), // Not using CUkernel-based launch
            ctx: std::ptr::null_mut(),  // Use default context
        };

        let mut node = MaybeUninit::uninit();
        unsafe {
            sys::cuGraphAddKernelNode_v2(
                node.as_mut_ptr(),
                self.cu_graph,
                dependencies.as_ptr(),
                dependencies.len(),
                &params,
            )
            .result()?;
            Ok(node.assume_init())
        }
    }

    /// Adds an event record node to the graph for timing.
    pub fn add_event_record_node(
        &mut self,
        dependencies: &[CUgraphNode],
        event: CUevent,
    ) -> Result<CUgraphNode, DriverError> {
        let mut node = MaybeUninit::uninit();
        unsafe {
            sys::cuGraphAddEventRecordNode(
                node.as_mut_ptr(),
                self.cu_graph,
                dependencies.as_ptr(),
                dependencies.len(),
                event,
            )
            .result()?;
            Ok(node.assume_init())
        }
    }

    /// Instantiates the graph, creating an executable graph.
    pub fn instantiate(&self) -> Result<CudaGraphExecHandle, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut graph_exec = MaybeUninit::uninit();
        unsafe {
            sys::cuGraphInstantiateWithFlags(graph_exec.as_mut_ptr(), self.cu_graph, 0).result()?;
            Ok(CudaGraphExecHandle {
                cu_graph_exec: graph_exec.assume_init(),
                ctx: self.ctx.clone(),
            })
        }
    }
}

impl Drop for CudaGraphHandle {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        if !self.cu_graph.is_null() {
            unsafe {
                let _ = sys::cuGraphDestroy(self.cu_graph);
            }
        }
    }
}

/// An instantiated CUDA graph that can be launched and updated.
pub struct CudaGraphExecHandle {
    pub(crate) cu_graph_exec: CUgraphExec,
    pub(crate) ctx: Arc<CudaContext>,
}

impl CudaGraphExecHandle {
    /// Launches the graph on the given stream.
    pub fn launch(&self, stream: &CudaStream) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { sys::cuGraphLaunch(self.cu_graph_exec, stream.cu_stream()).result() }
    }

    /// Surgically updates a kernel node's parameters without rebuilding the graph.
    pub unsafe fn update_kernel_node(
        &mut self,
        node: CUgraphNode,
        func: CUfunction,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        kernel_params: *mut *mut c_void,
    ) -> Result<(), DriverError> {
        let params = sys::CUDA_KERNEL_NODE_PARAMS {
            func,
            gridDimX: grid_dim.0,
            gridDimY: grid_dim.1,
            gridDimZ: grid_dim.2,
            blockDimX: block_dim.0,
            blockDimY: block_dim.1,
            blockDimZ: block_dim.2,
            sharedMemBytes: shared_mem_bytes,
            kernelParams: kernel_params,
            extra: std::ptr::null_mut(),
            kern: std::ptr::null_mut(),
            ctx: std::ptr::null_mut(),
        };

        unsafe { sys::cuGraphExecKernelNodeSetParams_v2(self.cu_graph_exec, node, &params) }
            .result()
    }
}

impl Drop for CudaGraphExecHandle {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        if !self.cu_graph_exec.is_null() {
            unsafe {
                let _ = sys::cuGraphExecDestroy(self.cu_graph_exec);
            }
        }
    }
}

/// Extension trait to get the raw CUfunction handle from CudaFunction.
pub trait CudaFunctionExt {
    unsafe fn raw_function(&self) -> CUfunction;
}

impl CudaFunctionExt for CudaFunction {
    unsafe fn raw_function(&self) -> CUfunction {
        // CudaFunction fields are reordered by Rust - cu_function is at offset 8
        debug_assert_eq!(
            std::mem::size_of::<CudaFunction>(),
            std::mem::size_of::<CUfunction>() + std::mem::size_of::<usize>()
        );
        unsafe {
            let ptr = (self as *const CudaFunction as *const u8).add(8) as *const CUfunction;
            std::ptr::read(ptr)
        }
    }
}

/// Stored kernel parameters that persist for the lifetime of a CUDA graph.
#[derive(Debug)]
pub struct KernelParams {
    values: Box<[u64]>,
    ptrs: Box<[*mut c_void]>,
    /// Index of the dyn_dims pointer in values array (if present)
    dyn_dims_idx: Option<usize>,
}

impl KernelParams {
    pub fn new(output_ptr: u64, input_ptrs: &[u64]) -> Self {
        let mut values: Vec<u64> = Vec::with_capacity(1 + input_ptrs.len());
        values.push(output_ptr);
        values.extend_from_slice(input_ptrs);
        let values = values.into_boxed_slice();
        let ptrs: Vec<*mut c_void> = values
            .iter()
            .map(|v| v as *const u64 as *mut c_void)
            .collect();
        Self {
            values,
            ptrs: ptrs.into_boxed_slice(),
            dyn_dims_idx: None,
        }
    }

    /// Create kernel params with a dyn_dims pointer as the last parameter.
    pub fn with_dyn_dims(output_ptr: u64, input_ptrs: &[u64], dyn_dims_ptr: u64) -> Self {
        let mut values: Vec<u64> = Vec::with_capacity(2 + input_ptrs.len());
        values.push(output_ptr);
        values.extend_from_slice(input_ptrs);
        let dyn_dims_idx = values.len();
        values.push(dyn_dims_ptr);
        let values = values.into_boxed_slice();
        let ptrs: Vec<*mut c_void> = values
            .iter()
            .map(|v| v as *const u64 as *mut c_void)
            .collect();
        Self {
            values,
            ptrs: ptrs.into_boxed_slice(),
            dyn_dims_idx: Some(dyn_dims_idx),
        }
    }

    pub fn as_cuda_params(&mut self) -> *mut *mut c_void {
        self.ptrs.as_mut_ptr()
    }

    pub fn update_output(&mut self, ptr: u64) {
        self.values[0] = ptr;
    }

    pub fn update_input(&mut self, index: usize, ptr: u64) {
        self.values[1 + index] = ptr;
    }

    /// Update the dyn_dims pointer if this kernel uses one.
    pub fn update_dyn_dims(&mut self, ptr: u64) {
        if let Some(idx) = self.dyn_dims_idx {
            self.values[idx] = ptr;
        }
    }
}

/// Stored kernel parameters for megakernels that persist for the lifetime of a CUDA graph.
/// Params: tasks, head, ready, queue_lock, timings, start_times, buffers, dyn_dims
#[derive(Debug)]
pub struct MegakernelParams {
    /// Parameter values: [tasks, head, ready, queue_lock, timings, start_times, buffers, dyn_dims]
    values: Box<[u64]>,
    /// Pointer array for CUDA kernel launch
    ptrs: Box<[*mut c_void]>,
}

impl MegakernelParams {
    /// Create megakernel params with all internal buffer pointers and dyn_dims.
    /// Order: tasks, head, ready, queue_lock, timings, start_times, buffers, dyn_dims
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tasks_ptr: u64,
        head_ptr: u64,
        ready_ptr: u64,
        queue_lock_ptr: u64,
        timings_ptr: u64,
        start_times_ptr: u64,
        buffers_ptr: u64,
        dyn_dims_ptr: u64,
    ) -> Self {
        let values: Box<[u64]> = vec![
            tasks_ptr,
            head_ptr,
            ready_ptr,
            queue_lock_ptr,
            timings_ptr,
            start_times_ptr,
            buffers_ptr,
            dyn_dims_ptr,
        ]
        .into_boxed_slice();
        let ptrs: Box<[*mut c_void]> = values
            .iter()
            .map(|v| v as *const u64 as *mut c_void)
            .collect();
        Self { values, ptrs }
    }

    pub fn as_cuda_params(&mut self) -> *mut *mut c_void {
        // Rebuild pointers (in case struct was moved)
        for (i, v) in self.values.iter().enumerate() {
            self.ptrs[i] = v as *const u64 as *mut c_void;
        }
        self.ptrs.as_mut_ptr()
    }

    /// Update the buffers pointer (index 6).
    pub fn update_buffers(&mut self, ptr: u64) {
        self.values[6] = ptr;
    }

    /// Update the dyn_dims pointer (index 7).
    pub fn update_dyn_dims(&mut self, ptr: u64) {
        self.values[7] = ptr;
    }

    /// Get the current buffers pointer value.
    pub fn buffers_ptr(&self) -> u64 {
        self.values[6]
    }
}

/// Timing data for a single kernel in a CUDA graph.
#[derive(Clone, Debug)]
pub struct CudaGraphKernelTiming {
    pub kernel_name: &'static str,
    pub start_ns: u64,
    pub end_ns: u64,
}

/// Timing data for a CUDA graph execution.
#[derive(Clone, Debug)]
pub struct CudaGraphTiming {
    pub kernel_timings: Vec<CudaGraphKernelTiming>,
    /// Time from launch call until first kernel started on GPU
    pub launch_latency_ns: u64,
    /// Elapsed time (in nanoseconds) from span entry to just before graph launch.
    /// This captures the setup overhead (constants, buffers, graph building) that
    /// occurs before the GPU actually starts executing.
    pub setup_duration_ns: u64,
}

pub fn create_cuda_event(ctx: &Arc<CudaContext>) -> Result<CUevent, DriverError> {
    ctx.bind_to_thread()?;
    let mut event = MaybeUninit::uninit();
    unsafe {
        sys::cuEventCreate(
            event.as_mut_ptr(),
            sys::CUevent_flags::CU_EVENT_DEFAULT as u32,
        )
        .result()?;
        Ok(event.assume_init())
    }
}

pub fn destroy_cuda_event(ctx: &Arc<CudaContext>, event: CUevent) {
    if !event.is_null() {
        let _ = ctx.bind_to_thread();
        unsafe {
            let _ = sys::cuEventDestroy_v2(event);
        }
    }
}

pub fn event_elapsed_ms(
    ctx: &Arc<CudaContext>,
    start: CUevent,
    end: CUevent,
) -> Result<f32, DriverError> {
    ctx.bind_to_thread()?;
    let mut ms: f32 = 0.0;
    unsafe {
        // cudarc 0.18.2 generates different bindings depending on the installed CUDA toolkit:
        // - CUDA 12.4 on Linux: cuEventElapsedTime (no _v2 suffix)
        // - macOS (bundled headers): cuEventElapsedTime_v2
        #[cfg(target_os = "macos")]
        sys::cuEventElapsedTime_v2(&mut ms, start, end).result()?;
        #[cfg(not(target_os = "macos"))]
        sys::cuEventElapsedTime(&mut ms, start, end).result()?;
    }
    Ok(ms)
}

pub fn record_event_on_stream(
    ctx: &Arc<CudaContext>,
    event: CUevent,
    stream: &CudaStream,
) -> Result<(), DriverError> {
    ctx.bind_to_thread()?;
    unsafe {
        sys::cuEventRecord(event, stream.cu_stream()).result()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use cudarc::driver::CudaContext;
    use luminal::prelude::*;
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::sync::Arc;

    use crate::cuda_bandwidth_gbps;
    use crate::runtime::CudaRuntime;
    use crate::tests::*;

    #[test]
    fn test_create_empty_graph() {
        let Ok(ctx) = CudaContext::new(0) else { return };
        assert!(CudaGraphHandle::new(ctx).is_ok());
    }

    #[test]
    fn test_kernel_params() {
        let mut params = KernelParams::new(0x1000, &[0x2000, 0x3000]);
        assert!(!params.as_cuda_params().is_null());
        params.update_output(0x4000);
        params.update_input(0, 0x5000);
    }

    #[test]
    fn test_cuda_function_size() {
        assert_eq!(
            std::mem::size_of::<CudaFunction>(),
            std::mem::size_of::<CUfunction>() + std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn test_raw_function_extraction() {
        let Ok(ctx) = CudaContext::new(0) else { return };
        let kernel_src = r#"extern "C" __global__ void test_kernel(float* out) { out[0] = 1.0f; }"#;
        let Ok(ptx) = cudarc::nvrtc::compile_ptx(kernel_src) else {
            return;
        };
        let module = ctx.load_module(ptx).unwrap();
        let func = module.load_function("test_kernel").unwrap();
        let cu_func = unsafe { func.raw_function() };
        assert!(!cu_func.is_null());
        let mut max_threads: i32 = 0;
        let result = unsafe {
            sys::cuFuncGetAttribute(
                &mut max_threads,
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                cu_func,
            )
        };
        assert!(result == sys::cudaError_enum::CUDA_SUCCESS);
    }

    #[test]
    fn test_graph_with_kernel() {
        use cudarc::driver::{CudaSlice, DevicePtr};
        let Ok(ctx) = CudaContext::new(0) else { return };
        let kernel_src = r#"extern "C" __global__ void test_kernel(float* out, float* in1) { if (threadIdx.x == 0) out[0] = in1[0] + 1.0f; }"#;
        let Ok(ptx) = cudarc::nvrtc::compile_ptx(kernel_src) else {
            return;
        };
        let module = ctx.load_module(ptx).unwrap();
        let func = module.load_function("test_kernel").unwrap();
        let stream = ctx.default_stream();
        let output: CudaSlice<f32> = unsafe { stream.alloc(1) }.unwrap();
        let mut input: CudaSlice<f32> = unsafe { stream.alloc(1) }.unwrap();
        stream.memcpy_htod(&[5.0f32], &mut input).unwrap();
        let cu_func = unsafe { func.raw_function() };
        let mut graph = CudaGraphHandle::new(ctx.clone()).unwrap();
        let mut params =
            KernelParams::new(output.device_ptr(&stream).0, &[input.device_ptr(&stream).0]);
        let _node = unsafe {
            graph.add_kernel_node(
                &[],
                cu_func,
                (1, 1, 1),
                (1, 1, 1),
                0,
                params.as_cuda_params(),
            )
        }
        .unwrap();
        let exec = graph.instantiate().unwrap();
        exec.launch(&stream).unwrap();
        stream.synchronize().unwrap();
        let mut result = [0.0f32];
        stream.memcpy_dtoh(&output, &mut result).unwrap();
        assert_eq!(result[0], 6.0f32);
    }

    #[test]
    #[ignore = "Wave B.0 feasibility spike; run manually on a CUDA host"]
    fn wave_b0_stream_capture_with_graph_launch_and_cublaslt() {
        use cudarc::cublas::sys::cublasOperation_t;
        use cudarc::cublaslt::{
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
        };
        use cudarc::driver::{CudaSlice, DevicePtr};

        let Ok(ctx) = CudaContext::new(0) else {
            return;
        };
        if ctx.bind_to_thread().is_err() {
            return;
        }
        // Stream capture requires a non-default stream.
        let stream = ctx.new_stream().unwrap();

        // Child graph that will be launched during stream capture (proxy for CudaGraphOp launch).
        let kernel_src = r#"extern "C" __global__ void test_kernel(float* out, float* in1) { if (threadIdx.x == 0) out[0] = in1[0] + 1.0f; }"#;
        let Ok(ptx) = cudarc::nvrtc::compile_ptx(kernel_src) else {
            return;
        };
        let module = ctx.load_module(ptx).unwrap();
        let func = module.load_function("test_kernel").unwrap();
        let mut output: CudaSlice<f32> = unsafe { stream.alloc(1) }.unwrap();
        let mut input: CudaSlice<f32> = unsafe { stream.alloc(1) }.unwrap();
        stream.memcpy_htod(&[2.0f32], &mut input).unwrap();

        let mut child_graph = CudaGraphHandle::new(ctx.clone()).unwrap();
        let mut params =
            KernelParams::new(output.device_ptr(&stream).0, &[input.device_ptr(&stream).0]);
        unsafe {
            child_graph
                .add_kernel_node(
                    &[],
                    func.raw_function(),
                    (1, 1, 1),
                    (1, 1, 1),
                    0,
                    params.as_cuda_params(),
                )
                .unwrap();
        }
        let child_exec = child_graph.instantiate().unwrap();

        // Buffers for a tiny F32 GEMM on cuBLASLt.
        const M: u64 = 8;
        const N: u64 = 8;
        const K: u64 = 8;
        let mut a: CudaSlice<f32> = unsafe { stream.alloc((M * K) as usize) }.unwrap();
        let mut b: CudaSlice<f32> = unsafe { stream.alloc((K * N) as usize) }.unwrap();
        let c: CudaSlice<f32> = unsafe { stream.alloc((M * N) as usize) }.unwrap();
        stream.memcpy_htod(&vec![1.0f32; (M * K) as usize], &mut a).unwrap();
        stream.memcpy_htod(&vec![2.0f32; (K * N) as usize], &mut b).unwrap();

        let cublaslt = CudaBlasLT::new(stream.clone()).unwrap();
        let (a_ptr, _a_guard) = a.device_ptr(&stream);
        let (b_ptr, _b_guard) = b.device_ptr(&stream);
        let (c_ptr, _c_guard) = c.device_ptr(&stream);

        // Pre-allocate workspace BEFORE capture (cuMemAlloc is not capturable).
        const WORKSPACE_SIZE: usize = 4 * 1024 * 1024;
        let workspace = unsafe { stream.alloc::<u8>(WORKSPACE_SIZE) }.unwrap();
        let (workspace_ptr, _workspace_guard) = workspace.device_ptr(&stream);

        // Pre-create descriptors and find algorithm BEFORE capture (host-side ops).
        let mut matmul_desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        let mut a_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut b_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut c_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut preference: cublasLtMatmulPreference_t = std::ptr::null_mut();
        let mut heuristic: cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
        let mut algo_count: i32 = 0;
        unsafe {
            cublasLtMatmulDescCreate(
                &mut matmul_desc,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cudaDataType::CUDA_R_32F,
            )
            .result()
            .unwrap();
            let layout_n = cublasOperation_t::CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(
                matmul_desc,
                cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &layout_n as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<cublasOperation_t>(),
            )
            .result()
            .unwrap();
            cublasLtMatmulDescSetAttribute(
                matmul_desc,
                cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &layout_n as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<cublasOperation_t>(),
            )
            .result()
            .unwrap();

            cublasLtMatrixLayoutCreate(&mut a_desc, cudaDataType::CUDA_R_32F, M, K, K as i64)
                .result()
                .unwrap();
            cublasLtMatrixLayoutCreate(&mut b_desc, cudaDataType::CUDA_R_32F, K, N, N as i64)
                .result()
                .unwrap();
            cublasLtMatrixLayoutCreate(&mut c_desc, cudaDataType::CUDA_R_32F, M, N, N as i64)
                .result()
                .unwrap();

            cublasLtMatmulPreferenceCreate(&mut preference).result().unwrap();
            cublasLtMatmulPreferenceSetAttribute(
                preference,
                cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &WORKSPACE_SIZE as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<usize>(),
            )
            .result()
            .unwrap();
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
            .result()
            .unwrap();
            assert!(algo_count > 0, "No suitable cuBLASLt algorithm found during capture spike");
        }

        let alpha_f32: f32 = 1.0;
        let beta_f32: f32 = 0.0;

        // Closure that only does the GPU-enqueued matmul call (safe during capture).
        let mut launch_matmul = || -> anyhow::Result<()> {
            unsafe {
                cublasLtMatmul(
                    *cublaslt.handle(),
                    matmul_desc,
                    &alpha_f32 as *const _ as *const std::ffi::c_void,
                    a_ptr as *const std::ffi::c_void,
                    a_desc,
                    b_ptr as *const std::ffi::c_void,
                    b_desc,
                    &beta_f32 as *const _ as *const std::ffi::c_void,
                    c_ptr as *const std::ffi::c_void,
                    c_desc,
                    c_ptr as *mut std::ffi::c_void,
                    c_desc,
                    &heuristic.algo,
                    workspace_ptr as *mut std::ffi::c_void,
                    WORKSPACE_SIZE,
                    stream.cu_stream() as *mut _,
                )
                .result()?;
            }
            Ok(())
        };

        // Warmup: run one matmul + child graph launch before capture to trigger any
        // one-time internal allocations inside cuBLASLt (handle-level state, algo caching).
        // This mirrors the runtime's warmup-then-capture pattern.
        child_exec.launch(&stream).unwrap();
        launch_matmul().unwrap();
        stream.synchronize().unwrap();

        // Capture child graph launch + cuBLASLt call into a parent graph.
        // NOTE: During stream capture, cuCtxSetCurrent (called by bind_to_thread) may
        // be illegal. Use raw cuGraphLaunch instead of the wrapper's launch() method.
        unsafe {
            sys::cuStreamBeginCapture_v2(
                stream.cu_stream(),
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            )
            .result()
            .unwrap();

            sys::cuGraphLaunch(child_exec.cu_graph_exec, stream.cu_stream())
                .result()
                .unwrap();
        }
        launch_matmul().unwrap();

        let mut captured_graph = std::ptr::null_mut();
        unsafe {
            sys::cuStreamEndCapture(stream.cu_stream(), &mut captured_graph)
                .result()
                .unwrap();
        }
        assert!(!captured_graph.is_null(), "capture should return a graph");

        let mut captured_exec = std::ptr::null_mut();
        unsafe {
            sys::cuGraphInstantiateWithFlags(&mut captured_exec, captured_graph, 0)
                .result()
                .unwrap();
            sys::cuGraphLaunch(captured_exec, stream.cu_stream())
                .result()
                .unwrap();
        }
        stream.synchronize().unwrap();

        let mut child_out = [0.0f32; 1];
        stream.memcpy_dtoh(&output, &mut child_out).unwrap();
        assert!(
            (child_out[0] - 3.0).abs() < 1e-6,
            "captured child graph kernel result mismatch"
        );

        let mut c_out = vec![0.0f32; (M * N) as usize];
        stream.memcpy_dtoh(&c, &mut c_out).unwrap();
        // A is all ones and B is all twos -> each C element is 2*K.
        let expected = (2 * K) as f32;
        assert!(
            (c_out[0] - expected).abs() < 1e-3,
            "captured cuBLASLt result mismatch: got {}, expected {}",
            c_out[0],
            expected
        );

        unsafe {
            sys::cuGraphExecDestroy(captured_exec).result().unwrap();
            sys::cuGraphDestroy(captured_graph).result().unwrap();
            cublasLtMatmulPreferenceDestroy(preference);
            cublasLtMatrixLayoutDestroy(c_desc);
            cublasLtMatrixLayoutDestroy(b_desc);
            cublasLtMatrixLayoutDestroy(a_desc);
            cublasLtMatmulDescDestroy(matmul_desc);
        }
    }

    // CUDA Graph Tests

    #[test]
    fn test_cuda_graph_basic_execution() {
        let Some(stream) = get_cuda_stream() else {
            return;
        };
        let size = 1024;
        let mut cx = Graph::default();
        let a = cx.tensor(size);
        let b = cx.tensor(size);
        let c = ((a + b) * a + b).output();
        cx.build_search_space_exclude_ops::<CudaRuntime, crate::block::Ops>();
        let mut rt = CudaRuntime::initialize(stream);
        let data_a = random_vec(size);
        let data_b = random_vec(size);
        rt.set_data(a, data_a.clone());
        rt.set_data(b, data_b.clone());
        rt = cx.search(rt, 5);
        rt.execute(&cx.dyn_map);
        let result1 = rt.get_f32(c);
        rt.execute(&cx.dyn_map);
        assert_close(&result1, &rt.get_f32(c));
        let expected: Vec<f32> = data_a
            .iter()
            .zip(&data_b)
            .map(|(a, b)| (a + b) * a + b)
            .collect();
        assert_close(&result1, &expected);
    }

    #[test]
    fn test_cuda_graph_multiple_executions() {
        let Some(stream) = get_cuda_stream() else {
            return;
        };
        let size = 2048;
        let mut cx = Graph::default();
        let a = cx.tensor(size);
        let b = cx.tensor(size);
        let c = (a + b + a + b).output();
        cx.build_search_space_exclude_ops::<CudaRuntime, crate::block::Ops>();
        let mut rt = CudaRuntime::initialize(stream);
        let data_a = random_vec(size);
        let data_b = random_vec(size);
        rt.set_data(a, data_a.clone());
        rt.set_data(b, data_b.clone());
        rt = cx.search(rt, 5);
        let mut results = Vec::new();
        for _ in 0..5 {
            rt.execute(&cx.dyn_map);
            results.push(rt.get_f32(c));
        }
        for result in &results {
            assert_close(result, &results[0]);
        }
        let expected: Vec<f32> = data_a
            .iter()
            .zip(&data_b)
            .map(|(a, b)| a + b + a + b)
            .collect();
        assert_close(&results[0], &expected);
    }

    #[test]
    fn test_cuda_graph_dyn_dims_surgical_update() {
        let Some(stream) = get_cuda_stream() else {
            return;
        };
        let size = 512;
        let mut cx = Graph::default();
        let a = cx.tensor('s');
        let b = cx.tensor('s');
        let c = (a + b).output();
        let d = (c * a).output();
        cx.build_search_space_exclude_ops::<CudaRuntime, crate::block::Ops>();
        let mut rt = CudaRuntime::initialize(stream);
        let data_a = random_vec(size);
        let data_b = random_vec(size);
        rt.set_data(a, data_a.clone());
        rt.set_data(b, data_b.clone());
        cx.set_dim('s', size);
        rt = cx.search(rt, 5);
        rt.execute(&cx.dyn_map);
        let expected: Vec<f32> = data_a
            .iter()
            .zip(&data_b)
            .map(|(a, b)| (a + b) * a)
            .collect();
        assert_close(&rt.get_f32(d), &expected);
        let size = 1024;
        let data_a2 = random_vec(size);
        let data_b2 = random_vec(size);
        rt.set_data(a, data_a2.clone());
        rt.set_data(b, data_b2.clone());
        cx.set_dim('s', size);
        rt.execute(&cx.dyn_map);
        let expected2: Vec<f32> = data_a2
            .iter()
            .zip(&data_b2)
            .map(|(a, b)| (a + b) * a)
            .collect();
        assert_close(&rt.get_f32(d), &expected2);
    }

    #[test]
    fn test_single_kernel_in_graph() {
        let Some(stream) = get_cuda_stream() else {
            return;
        };
        let size = 1024;
        let mut cx = Graph::default();
        let a = cx.tensor(size);
        let b = cx.tensor(size);
        let c = (a + b).output();
        cx.build_search_space_exclude_ops::<CudaRuntime, crate::block::Ops>();
        let mut rt = CudaRuntime::initialize(stream);
        let data_a = random_vec(size);
        let data_b = random_vec(size);
        rt.set_data(a, data_a.clone());
        rt.set_data(b, data_b.clone());
        rt = cx.search(rt, 5);
        rt.execute(&cx.dyn_map);
        let expected: Vec<f32> = data_a.iter().zip(&data_b).map(|(a, b)| a + b).collect();
        assert_close(&rt.get_f32(c), &expected);
        assert!(rt.last_kernel_stats.iter().any(|s| s.name == "CudaGraph"));
    }

    #[test]
    fn test_cuda_graph_chain_performance() {
        let Some(stream) = get_cuda_stream() else {
            return;
        };
        let size = 4096;
        let mut cx = Graph::default();
        let a = cx.tensor(size);
        let b = cx.tensor(size);
        let mut result = a + b;
        for _ in 0..5 {
            result += a;
            result *= b;
        }
        let output = result.output();
        cx.build_search_space_exclude_ops::<CudaRuntime, crate::block::Ops>();
        let mut rt = CudaRuntime::initialize(stream);
        let data_a = random_vec(size);
        let data_b = random_vec(size);
        rt.set_data(a, data_a.clone());
        rt.set_data(b, data_b.clone());
        rt = cx.search(rt, 5);
        for _ in 0..10 {
            rt.execute(&cx.dyn_map);
        }
        let mut expected: Vec<f32> = data_a.iter().zip(&data_b).map(|(a, b)| a + b).collect();
        for _ in 0..5 {
            expected = expected.iter().zip(&data_a).map(|(r, a)| r + a).collect();
            expected = expected.iter().zip(&data_b).map(|(r, b)| r * b).collect();
        }
        assert_close_precision(&rt.get_f32(output), &expected, 1e-2);
    }
}
