use crate::{
    block::{BlockOp, N_TIMING_SLOTS, SMEvent, record_block_op_timings},
    host::HostOp,
    kernel::{
        CudaGraphExecHandle, CudaGraphHandle, CudaGraphOp, CudaGraphTiming, KernelOp,
        record_cuda_graph_timings,
    },
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice};

use fixedbitset::FixedBitSet;
use itertools::Itertools;
use luminal::hlir::*;
use luminal::prelude::{
    petgraph::{
        Directed, Direction,
        algo::{Cycle, toposort},
        prelude::StableGraph,
        visit::{EdgeRef, NodeIndexable},
    },
    *,
};

use luminal_tracing::PerfettoGuard;
use memmap2::MmapOptions;
use prost::Message;
use safetensors::SafeTensors;
use std::{
    collections::{VecDeque, hash_map::Entry},
    fmt::Debug,
    fs::File,
    sync::Arc,
    time::Duration,
};
use tracing::{Level, enabled, span, trace};
use uuid::Uuid;

pub enum CudaInput {
    Buffer(CudaSlice<u8>),
    Ptr(u64),
}

/// Executable operation in the runtime graph.
/// All operations (including CUDA graphs) are now HostOps.
struct ExecutableHostOp {
    stream: Arc<CudaStream>,
    inputs: Vec<NodeIndex>,
    output: NodeIndex,
    internal: Arc<Box<dyn HostOp>>,
}

/// Statistics for a single kernel execution
#[derive(Debug, Clone)]
pub struct KernelStats {
    pub name: &'static str,
    pub execution_time_us: f64,
    pub bytes_loaded: usize,
    pub bytes_stored: usize,
    pub flops: usize,
    pub bandwidth_gbps: f64,
    pub tflops: f64,
}

/// Statistics for a single block op execution (aggregated across all SMs)
#[derive(Debug, Clone)]
pub struct BlockOpStats {
    pub name: &'static str,
    pub execution_time_us: f64,
    pub bytes_loaded: usize,
    pub bytes_stored: usize,
    pub flops: usize,
    pub bandwidth_gbps: f64,
    pub tflops: f64,
    pub count: usize,
}
impl Debug for ExecutableHostOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HostOp: ({:?})", self.internal)
    }
}

/// Pending timing data using pinned memory for async copies
struct PendingTimingData {
    timing_buffer_idx: usize,
    start_time_idx: usize,
    span_id: Uuid,
}

#[derive(Clone, PartialEq, Eq)]
struct RuntimeReplaySignature {
    dyn_map: Vec<(char, usize)>,
    cached_ptrs: Vec<(usize, u64)>,
    force_zero_all: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RuntimeReplayState {
    Disabled,
    WarmupPending,
    CapturePending,
    Ready,
}

struct RuntimeReplayCache {
    state: RuntimeReplayState,
    signature: Option<RuntimeReplaySignature>,
    graph_exec: Option<CudaGraphExecHandle>,
    prepared_capture_stream: Option<u64>,
    last_error: Option<String>,
}

impl Default for RuntimeReplayCache {
    fn default() -> Self {
        Self {
            state: RuntimeReplayState::Disabled,
            signature: None,
            graph_exec: None,
            prepared_capture_stream: None,
            last_error: None,
        }
    }
}

pub struct CudaRuntime {
    pub hlir_buffers: FxHashMap<NodeIndex, CudaInput>,
    pub buffers: FxHashMap<NodeIndex, CudaSlice<u8>>,
    pub llir_graph: luminal::graph::LLIRGraph,
    cuda_stream: Arc<CudaStream>,
    replay_capture_stream: Arc<CudaStream>,
    timing_stream: Arc<CudaStream>,
    exec_graph: StableGraph<ExecutableHostOp, (), Directed>,
    node_to_exec: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) timings: Vec<Vec<(Vec<SMEvent>, u64, Uuid)>>,
    pub(crate) cuda_graph_timings: Vec<(CudaGraphTiming, Uuid)>,
    /// Pending timing data collected asynchronously using pinned memory
    pending_timing_data: Vec<PendingTimingData>,
    /// Pool of pre-allocated pinned timing buffers (one per slot)
    pinned_timing_pool: Vec<PinnedHostSlice<SMEvent>>,
    /// Pool of pre-allocated pinned start time buffers (one per slot)
    pinned_start_pool: Vec<PinnedHostSlice<u64>>,
    /// Index of next available slot in the pinned buffer pools
    next_pinned_slot: usize,
    last_dyn_map: FxHashMap<char, usize>,
    intermediate_buffer_dims: FxHashSet<char>,
    llir_to_hlir: FxHashMap<NodeIndex, NodeIndex>,
    hlir_to_llir: FxHashMap<NodeIndex, NodeIndex>,
    changed_hlir: FxHashSet<NodeIndex>,
    /// Intermediate buffers that must be zeroed before execution.
    buffers_requiring_zero: FxHashSet<NodeIndex>,
    /// Cached buffer pointers to avoid repeated device_ptr() calls (keyed by llir_node)
    cached_buffer_ptrs: FxHashMap<NodeIndex, u64>,
    pub last_kernel_stats: Vec<KernelStats>,
    pub last_total_time_us: f64,
    kernel_cache: FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    num_sms: usize,
    runtime_replay: RuntimeReplayCache,
}

impl CudaRuntime {
    /// Creates a new CudaRuntime with default configuration:
    /// - Device 0
    /// - Blocking sync scheduling
    /// - Default stream
    pub fn new() -> Result<Self, cudarc::driver::DriverError> {
        let ctx = cudarc::driver::CudaContext::new(0)?;
        ctx.bind_to_thread()?;
        ctx.set_flags(cudarc::driver::sys::CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC)?;
        // Use a non-default stream so that CUDA stream capture works.
        // cuStreamBeginCapture requires a non-default stream.
        let stream = ctx.new_stream()?;

        Ok(Self::initialize(stream))
    }

    #[tracing::instrument(skip_all)]
    pub fn load_safetensors(&mut self, cx: &Graph, file_path: &str) {
        let f = File::open(file_path).unwrap();
        let mmap = unsafe { MmapOptions::new().map(&f).unwrap() };
        let st = SafeTensors::deserialize(&mmap).unwrap();
        for node in cx.graph.node_indices() {
            if let Some(Input { label, .. }) = (*cx.graph[node]).as_any().downcast_ref::<Input>()
                && let Ok(tensor) = st.tensor(label)
            {
                self.changed_hlir.insert(node);
                match tensor.dtype() {
                    safetensors::Dtype::F32 => {
                        let bytes = tensor.data();
                        let f32s: &[f32] = bytemuck::cast_slice(bytes);
                        let dev = f32s.to_cuda_input(&self.cuda_stream);
                        self.hlir_buffers.insert(node, dev);
                    }
                    safetensors::Dtype::F16 => {
                        let f32s = tensor
                            .data()
                            .chunks_exact(2)
                            .map(|chunk| {
                                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                                half::f16::from_bits(bits).to_f32()
                            })
                            .collect_vec();
                        let dev = f32s.to_cuda_input(&self.cuda_stream);
                        self.hlir_buffers.insert(node, dev);
                    }
                    safetensors::Dtype::U8 => {
                        let bytes = tensor.data();
                        let dev = bytes.to_cuda_input(&self.cuda_stream);
                        self.hlir_buffers.insert(node, dev);
                    }
                    dtype => unimplemented!("{dtype} loading not supported yet"),
                }
            }
        }
    }

    pub fn set_data(&mut self, id: impl ToId, data: impl ToCudaInput) {
        let id = id.to_id();
        let cuda_input = data.to_cuda_input(&self.cuda_stream);
        self.hlir_buffers.insert(id, cuda_input);
        self.changed_hlir.insert(id);
    }

    /// Upload f32 input data, reusing an existing device buffer when size matches.
    /// This keeps input pointers stable across frames for runtime-level CUDA graph replay.
    pub fn set_data_f32(&mut self, id: impl ToId, data: &[f32]) {
        let id = id.to_id();
        let byte_data: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        self.set_data_bytes_reuse(id, byte_data);
    }

    /// Upload i32 input data, reusing an existing device buffer when size matches.
    pub fn set_data_i32(&mut self, id: impl ToId, data: &[i32]) {
        let id = id.to_id();
        let byte_data: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        self.set_data_bytes_reuse(id, byte_data);
    }

    fn set_data_bytes_reuse(&mut self, id: NodeIndex, byte_data: &[u8]) {
        if let Some(CudaInput::Buffer(buf)) = self.hlir_buffers.get_mut(&id)
            && buf.len() == byte_data.len()
        {
            self.cuda_stream.memcpy_htod(byte_data, buf).unwrap();
            return;
        }

        self.hlir_buffers.insert(
            id,
            CudaInput::Buffer(self.cuda_stream.clone_htod(byte_data).unwrap()),
        );
        self.changed_hlir.insert(id);
    }

    /// Write partial data into an existing HLIR buffer at a byte offset.
    /// The buffer must have been previously created via `set_data`.
    /// Does NOT mark the buffer as changed (pointer is unchanged).
    pub fn update_buffer_slice(&mut self, id: impl ToId, byte_offset: usize, data: &[f32]) {
        let id = id.to_id();
        let byte_data: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        let byte_end = byte_offset + byte_data.len();
        let buf = match self.hlir_buffers.get_mut(&id) {
            Some(CudaInput::Buffer(buf)) => buf,
            Some(CudaInput::Ptr(_)) => {
                panic!("update_buffer_slice: {:?} is a Ptr, not a Buffer", id)
            }
            None => panic!("update_buffer_slice: no buffer for {:?}", id),
        };
        let mut view = buf.slice_mut(byte_offset..byte_end);
        self.cuda_stream.memcpy_htod(byte_data, &mut view).unwrap();
    }

    #[tracing::instrument(skip_all)]
    fn get_output_data(&self, id: impl ToId) -> Vec<u8> {
        let id = id.to_id();
        let output_id = self
            .llir_graph
            .node_indices()
            .find(|n| {
                if let Some(Output { node }) = self.llir_graph[*n].to_op::<Output>() {
                    *node == id.index()
                } else {
                    false
                }
            })
            .expect("Cannot find output tensor!");
        let data_id = self
            .llir_graph
            .neighbors_directed(output_id, Direction::Incoming)
            .next()
            .unwrap();

        let _span = span!(Level::TRACE, "dtoh").entered();
        self.cuda_stream
            .clone_dtoh(
                self.buffers
                    .get(&data_id)
                    .expect("Cannot find tensor in runtime!"),
            )
            .unwrap()
    }

    pub fn get_f32(&self, id: impl ToId) -> Vec<f32> {
        let bytes = self.get_output_data(id);
        let bytes = bytes.leak();
        let n_bytes = bytes.len();
        let bytes_ptr = bytes.as_mut_ptr();
        let float_ptr = bytes_ptr as *mut f32;
        unsafe { Vec::from_raw_parts(float_ptr, n_bytes / 4, n_bytes / 4) }
    }

    pub fn get_i32(&self, id: impl ToId) -> Vec<i32> {
        self.get_output_data(id)
            .chunks_exact(4)
            .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect_vec()
    }

    #[tracing::instrument(skip_all)]
    fn allocate_intermediate_buffers(&mut self, dyn_dims: &FxHashMap<char, usize>) {
        // Log GPU memory before allocation
        if let Ok((free, total)) = cudarc::driver::result::mem_get_info() {
            let used = total - free;
            eprintln!(
                "    [gpu_mem] before alloc: {:.1} GB used / {:.1} GB total ({:.1} GB free)",
                used as f64 / 1e9,
                total as f64 / 1e9,
                free as f64 / 1e9,
            );
        }
        self.intermediate_buffer_dims.clear();
        let mut total_alloc_bytes: usize = 0;
        for node in self.llir_graph.node_indices().collect_vec() {
            if self.llir_graph[node].to_op::<Input>().is_some() {
                continue;
            }
            if let Some(op) = self.llir_graph[node].to_dialect::<dyn BlockOp>() {
                let out_bytes = op.output_bytes();
                let exec_bytes = out_bytes.exec(dyn_dims).unwrap();
                // Skip allocation for ops with zero output size
                if exec_bytes == 0 {
                    continue;
                }
                self.intermediate_buffer_dims.extend(out_bytes.dyn_vars());
                let alloc_bytes = exec_bytes;
                total_alloc_bytes += alloc_bytes;
                self.buffers.insert(
                    node,
                    self.cuda_stream
                        .alloc_zeros(alloc_bytes)
                        .unwrap_or_else(|e| {
                            let mem_info = cudarc::driver::result::mem_get_info()
                                .map(|(f, t)| format!("{:.1} GB free / {:.1} GB total", f as f64 / 1e9, t as f64 / 1e9))
                                .unwrap_or_else(|_| "unknown".to_string());
                            panic!(
                                "Failed to alloc {} bytes ({:.1} MB) for BlockOp node {:?} (total so far: {:.1} MB, gpu: {}): {:?}",
                                alloc_bytes, alloc_bytes as f64 / 1e6, node, total_alloc_bytes as f64 / 1e6, mem_info, e
                            )
                        }),
                );
                let ptr = self.buffers[&node].device_ptr(&self.cuda_stream).0;
                self.cached_buffer_ptrs.insert(node, ptr);
            } else if let Some(op) = self.llir_graph[node].to_dialect::<dyn KernelOp>() {
                // Kernel nodes remain in the graph for buffer allocation.
                // Their execution is handled by CudaGraphOp.
                let out_bytes = op.output_bytes();
                let exec_size = out_bytes.exec(dyn_dims).unwrap();
                // Skip allocation for kernels with zero output size (e.g., megakernels)
                if exec_size == 0 {
                    continue;
                }
                self.intermediate_buffer_dims.extend(out_bytes.dyn_vars());
                let alloc_bytes = exec_size;
                total_alloc_bytes += alloc_bytes;
                self.buffers.insert(
                    node,
                    self.cuda_stream
                        .alloc_zeros(alloc_bytes)
                        .unwrap_or_else(|e| {
                            let mem_info = cudarc::driver::result::mem_get_info()
                                .map(|(f, t)| format!("{:.1} GB free / {:.1} GB total", f as f64 / 1e9, t as f64 / 1e9))
                                .unwrap_or_else(|_| "unknown".to_string());
                            panic!(
                                "Failed to alloc {} bytes ({:.1} MB) for KernelOp node {:?} (total so far: {:.1} MB, gpu: {}): {:?}",
                                alloc_bytes, alloc_bytes as f64 / 1e6, node, total_alloc_bytes as f64 / 1e6, mem_info, e
                            )
                        }),
                );
                let ptr = self.buffers[&node].device_ptr(&self.cuda_stream).0;
                self.cached_buffer_ptrs.insert(node, ptr);
            } else if let Some(op) = self.llir_graph[node].to_dialect::<dyn HostOp>() {
                // Allocate output buffer for this HostOp if it has one
                let out_bytes = op.output_bytes().exec(dyn_dims).unwrap();
                if out_bytes > 0 {
                    let alloc_bytes = out_bytes;
                    total_alloc_bytes += alloc_bytes;
                    self.buffers.insert(
                        node,
                        self.cuda_stream
                            .alloc_zeros(alloc_bytes)
                            .unwrap_or_else(|e| {
                                let mem_info = cudarc::driver::result::mem_get_info()
                                    .map(|(f, t)| format!("{:.1} GB free / {:.1} GB total", f as f64 / 1e9, t as f64 / 1e9))
                                    .unwrap_or_else(|_| "unknown".to_string());
                                panic!(
                                    "Failed to alloc {} bytes ({:.1} MB) for HostOp node {:?} (total so far: {:.1} MB, gpu: {}): {:?}",
                                    alloc_bytes, alloc_bytes as f64 / 1e6, node, total_alloc_bytes as f64 / 1e6, mem_info, e
                                )
                            }),
                    );
                    let ptr = self.buffers[&node].device_ptr(&self.cuda_stream).0;
                    self.cached_buffer_ptrs.insert(node, ptr);
                }
            }
        }
        eprintln!(
            "    [gpu_mem] allocated {:.1} MB for {} intermediate buffers",
            total_alloc_bytes as f64 / 1e6,
            self.buffers.len(),
        );
    }

    /// Pre-allocate buffers with the given dynamic dimension values.
    /// CUDA graph building is handled internally by CudaGraphOp on first execution.
    #[tracing::instrument(skip_all)]
    pub fn prebuild_graphs(&mut self, dyn_map: &FxHashMap<char, usize>) {
        // 1. Allocate intermediate buffers (needed for buffer pointers)
        if self.buffers.is_empty() {
            self.last_dyn_map = dyn_map.clone();
            self.allocate_intermediate_buffers(dyn_map);
        }

        // 2. Process changed HLIR inputs to get their buffer pointers
        if !self.changed_hlir.is_empty() {
            let to_process: Vec<(NodeIndex, NodeIndex, u64)> = self
                .changed_hlir
                .iter()
                .filter_map(|hlir_node| {
                    self.hlir_buffers.get(hlir_node).map(|input| {
                        let llir_node = self.hlir_to_llir[hlir_node];
                        let ptr = match input {
                            CudaInput::Buffer(buf) => buf.device_ptr(&self.cuda_stream).0,
                            CudaInput::Ptr(p) => *p,
                        };
                        (*hlir_node, llir_node, ptr)
                    })
                })
                .collect();

            for (hlir_node, llir_node, ptr) in to_process {
                self.cached_buffer_ptrs.insert(llir_node, ptr);
                self.changed_hlir.remove(&hlir_node);
            }
        }

        // CUDA graph building is now handled internally by CudaGraphOp on first execution
    }
}

pub trait ToCudaInput {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput;
}

impl ToCudaInput for &[f32] {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(
            stream
                .clone_htod(unsafe {
                    std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4)
                })
                .unwrap(),
        )
    }
}

impl ToCudaInput for Vec<i32> {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(
            stream
                .clone_htod(unsafe {
                    std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4)
                })
                .unwrap(),
        )
    }
}

impl ToCudaInput for Vec<f32> {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(
            stream
                .clone_htod(unsafe {
                    std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4)
                })
                .unwrap(),
        )
    }
}

impl ToCudaInput for &[u8] {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(stream.clone_htod(self).unwrap())
    }
}

impl ToCudaInput for Vec<u8> {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(stream.clone_htod(&self).unwrap())
    }
}

impl Runtime for CudaRuntime {
    type Ops = (
        crate::logical::Ops,
        crate::kernel::Ops,
        crate::block::Ops,
        crate::host::Ops,
    );
    type CompileArg = Arc<CudaStream>;
    type ExecReturn = ();
    type ProfileMetric = Duration;

    fn initialize(stream: Self::CompileArg) -> Self {
        let ctx = stream.context();
        let num_sms = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .unwrap() as usize;
        let timing_stream = ctx.new_stream().expect("Failed to create timing stream");
        let replay_capture_stream = ctx
            .new_stream()
            .expect("Failed to create replay capture stream");

        // Pre-allocate pinned buffer pools for async timing collection
        // We allocate enough slots to handle many megakernels between flushes
        const PINNED_POOL_SIZE: usize = 64;
        let timing_len = num_sms * N_TIMING_SLOTS;
        let pinned_timing_pool;
        let pinned_start_pool;
        if enabled!(Level::TRACE) {
            pinned_timing_pool = (0..PINNED_POOL_SIZE)
                .map(|_| unsafe {
                    ctx.alloc_pinned::<SMEvent>(timing_len)
                        .expect("Failed to allocate pinned timing buffer")
                })
                .collect::<Vec<_>>();
            pinned_start_pool = (0..PINNED_POOL_SIZE)
                .map(|_| unsafe {
                    ctx.alloc_pinned::<u64>(num_sms)
                        .expect("Failed to allocate pinned start time buffer")
                })
                .collect::<Vec<_>>();
        } else {
            pinned_timing_pool = vec![];
            pinned_start_pool = vec![]
        }

        Self {
            num_sms,
            hlir_buffers: FxHashMap::default(),
            buffers: FxHashMap::default(),
            cuda_stream: stream,
            replay_capture_stream,
            timing_stream,
            llir_graph: StableGraph::default(),
            exec_graph: StableGraph::default(),
            node_to_exec: FxHashMap::default(),
            hlir_to_llir: FxHashMap::default(),
            llir_to_hlir: FxHashMap::default(),
            changed_hlir: FxHashSet::default(),
            buffers_requiring_zero: FxHashSet::default(),
            cached_buffer_ptrs: FxHashMap::default(),
            timings: vec![],
            cuda_graph_timings: vec![],
            pending_timing_data: vec![],
            pinned_timing_pool,
            pinned_start_pool,
            next_pinned_slot: 0,
            last_dyn_map: FxHashMap::default(),
            intermediate_buffer_dims: FxHashSet::default(),
            last_kernel_stats: vec![],
            last_total_time_us: 0.0,
            kernel_cache: FxHashMap::default(),
            runtime_replay: RuntimeReplayCache::default(),
        }
    }

    #[tracing::instrument(skip_all)]
    fn load_llir(&mut self, llir_graph: &LLIRGraph) {
        // Sync before clearing old data to ensure all operations complete
        self.cuda_stream
            .synchronize()
            .expect("Failed to sync at start of load_llir");

        // exec_graph entries are ExecutableHostOp which are dropped automatically

        // Clear intermediate buffers when loading new graph - they need to be
        // reallocated and re-registered with the new work_queue
        self.buffers.clear();
        self.cached_buffer_ptrs.clear();
        self.runtime_replay = RuntimeReplayCache::default();
        // Mark all HLIR inputs as changed so their pointers get re-cached in execute
        self.changed_hlir.extend(self.hlir_buffers.keys().copied());
        self.exec_graph.clear();
        self.buffers_requiring_zero.clear();

        // Sync after clearing all buffers to ensure CUDA resources are freed
        self.cuda_stream
            .synchronize()
            .expect("Failed to sync after clearing buffers");

        // Rebind CUDA context to thread after cleanup to ensure valid state
        self.cuda_stream
            .context()
            .bind_to_thread()
            .expect("Failed to bind CUDA context after cleanup");

        let mut exec_graph = StableGraph::default();
        let mut node_to_exec = FxHashMap::default();

        // Clone llir_graph so we can modify it to add megakernel nodes
        let mut llir_graph = llir_graph.clone();

        // Compile BlockOp subgraphs into MegakernelOps and add them to the llir_graph
        let megakernel_to_blocks = crate::block::block_to_kernel(
            &mut llir_graph,
            &self.cuda_stream,
            &mut self.kernel_cache,
        );

        // Compile kernel subgraphs into CudaGraphOps (which implement HostOp)
        // This adds CudaGraphOp nodes to llir_graph and removes the original kernel nodes.
        // After this, only HostOps remain in the llir_graph.
        crate::kernel::kernel_to_host(
            &mut llir_graph,
            &self.cuda_stream,
            &mut self.kernel_cache,
            &megakernel_to_blocks,
        );

        // Add host ops
        {
            let _span = span!(Level::TRACE, "compile_host_ops").entered();
            for host_op_node_index in llir_graph.node_indices() {
                if let Some(host_op) = llir_graph[host_op_node_index].to_dialect::<dyn HostOp>() {
                    let inputs = llir_graph
                        .edges_directed(host_op_node_index, Direction::Incoming)
                        .sorted_by_key(|e| e.id())
                        .map(|e| e.source())
                        .collect_vec();
                    node_to_exec.insert(
                        host_op_node_index,
                        exec_graph.add_node(ExecutableHostOp {
                            stream: Arc::clone(&self.cuda_stream),
                            inputs,
                            output: host_op_node_index,
                            internal: Arc::clone(host_op),
                        }),
                    );
                }
            }
        }

        // Add edges
        for edge in llir_graph.edge_indices() {
            let (start, end) = llir_graph.edge_endpoints(edge).unwrap();
            if !node_to_exec.contains_key(&start) || !node_to_exec.contains_key(&end) {
                continue;
            }
            let (exec_start, exec_end) = (node_to_exec[&start], node_to_exec[&end]);
            if exec_start != exec_end
                && exec_graph
                    .edges_connecting(exec_start, exec_end)
                    .next()
                    .is_none()
            {
                exec_graph.add_edge(exec_start, exec_end, ());
            }
        }

        for host_op_node_index in llir_graph.node_indices() {
            if let Some(host_op) = llir_graph[host_op_node_index].to_dialect::<dyn HostOp>() {
                self.buffers_requiring_zero
                    .extend(host_op.zero_output_nodes());
            }
        }

        self.exec_graph = exec_graph;
        self.llir_graph = llir_graph.clone();
        self.node_to_exec = node_to_exec;
        self.hlir_to_llir.clear();
        self.llir_to_hlir.clear();
        self.changed_hlir.clear();
        for (hlir_node, llir_node) in self
            .llir_graph
            .node_indices()
            .filter_map(|n| self.llir_graph[n].to_op::<Input>().map(|op| (op.node, n)))
            .collect_vec()
        {
            self.llir_to_hlir
                .insert(llir_node, NodeIndex::new(hlir_node));
            self.hlir_to_llir
                .insert(NodeIndex::new(hlir_node), llir_node);
            self.changed_hlir.insert(NodeIndex::new(hlir_node));
        }

        // Prebuild CUDA graphs if we have a previous dyn_map (e.g., from search/profile)
        // This avoids rebuild overhead on first execute after load_llir
        if !self.last_dyn_map.is_empty() {
            let dyn_map = self.last_dyn_map.clone();
            self.prebuild_graphs(&dyn_map);
        }
    }

    fn allocate_dummy_input(&mut self, node_index: usize, num_elements: usize) {
        let buf = self
            .cuda_stream
            .alloc_zeros(num_elements * std::mem::size_of::<f32>())
            .unwrap();
        let id = NodeIndex::new(node_index);
        self.hlir_buffers.insert(id, CudaInput::Buffer(buf));
        self.changed_hlir.insert(id);
    }

    fn clear_intermediate_buffers(&mut self) {
        self.cuda_stream.synchronize().unwrap();
        self.buffers.clear();
        self.cached_buffer_ptrs.clear();
        self.runtime_replay = RuntimeReplayCache::default();
    }

    #[tracing::instrument(skip_all)]
    fn profile(
        &mut self,
        llir_graph: &LLIRGraph,
        dyn_map: &FxHashMap<char, usize>,
        _trials: usize,
    ) -> (Self::ProfileMetric, String) {
        self.buffers.clear();
        self.load_llir(llir_graph);
        let start = std::time::Instant::now();
        self.execute(dyn_map);
        let duration = start.elapsed();

        // Flush pending timing data so it's available for stats
        self.flush_pending_timings();

        // Compute aggregates for profile display
        let block_op_stats = Self::compute_block_op_stats(
            &self.llir_graph,
            self.timings.last().unwrap_or(&vec![]),
            &self.last_dyn_map,
            self.num_sms,
        );

        let total_bytes: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.bytes_loaded + s.bytes_stored)
            .sum::<usize>()
            + block_op_stats
                .iter()
                .map(|s| s.bytes_loaded + s.bytes_stored)
                .sum::<usize>();
        let total_flops: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.flops)
            .sum::<usize>()
            + block_op_stats.iter().map(|s| s.flops).sum::<usize>();
        let aggregate_bw = if self.last_total_time_us > 0.0 {
            (total_bytes as f64) / (self.last_total_time_us * 1e-6) / 1e9
        } else {
            0.0
        };
        let aggregate_tf = if self.last_total_time_us > 0.0 {
            (total_flops as f64) / (self.last_total_time_us * 1e-6) / 1e12
        } else {
            0.0
        };

        let peak_bw = crate::cuda_bandwidth_gbps(self.cuda_stream.context());
        let peak_tf = crate::cuda_compute_f32_tflops(self.cuda_stream.context());
        let mbu = peak_bw.map(|p| aggregate_bw / p as f64);
        let mfu = peak_tf.map(|p| aggregate_tf / p as f64);

        let duration_str = pretty_duration::pretty_duration(&duration, None);
        let mbu_str = mbu.map_or("-".to_string(), |v| format!("{:.1}%", v * 100.0));
        let mfu_str = mfu.map_or("-".to_string(), |v| format!("{:.1}%", v * 100.0));
        let display = format!(
            "{duration_str} | MBU: {mbu_str} | MFU: {mfu_str} [BLK: {} KRN: {} HOST: {}]",
            llir_graph
                .node_weights()
                .filter(|n| n.to_dialect::<dyn BlockOp>().is_some())
                .count(),
            llir_graph
                .node_weights()
                .filter(|n| n.to_dialect::<dyn KernelOp>().is_some())
                .count(),
            llir_graph
                .node_weights()
                .filter(|n| n.to_dialect::<dyn HostOp>().is_some())
                .count()
        );

        (duration, display)
    }

    #[tracing::instrument(skip_all)]
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) -> Self::ExecReturn {
        let buffers_empty = self.buffers.is_empty();
        let dyn_map_len_changed = dyn_map.len() != self.last_dyn_map.len();
        let dyn_dims_changed = dyn_map
            .iter()
            .filter(|(d, _)| self.intermediate_buffer_dims.contains(*d))
            .any(|(d, v)| self.last_dyn_map.get(d).map(|n| *n != *v).unwrap_or(true));
        let needs_realloc = buffers_empty || dyn_map_len_changed || dyn_dims_changed;
        if needs_realloc {
            self.last_dyn_map = dyn_map.clone();
            self.allocate_intermediate_buffers(dyn_map);
            self.runtime_replay = RuntimeReplayCache::default();
        }

        let rt_profiling = std::env::var("LUMINAL_PROFILE").map_or(false, |v| v == "1");
        let sync_debug = std::env::var("LUMINAL_SYNC_DEBUG").map_or(false, |v| v == "1");
        let force_zero_all = std::env::var("LUMINAL_FORCE_ZERO_ALL").map_or(false, |v| v == "1");
        let replay_enabled = !sync_debug
            && std::env::var("LUMINAL_RUNTIME_GRAPH_REPLAY").map_or(false, |v| v == "1");
        let replay_debug = std::env::var("LUMINAL_RUNTIME_GRAPH_DEBUG").map_or(false, |v| v == "1");
        let force_recapture =
            std::env::var("LUMINAL_RUNTIME_GRAPH_FORCE_RECAPTURE").map_or(false, |v| v == "1");

        let single_stream = self
            .exec_graph
            .node_indices()
            .all(|n| Arc::ptr_eq(&self.exec_graph[n].stream, &self.cuda_stream));
        assert!(
            single_stream || sync_debug,
            "Detected multi-stream host op execution. Set LUMINAL_SYNC_DEBUG=1 or add stream event dependencies before disabling per-op sync.",
        );

        // Cache HLIR input pointers
        if !self.changed_hlir.is_empty() {
            for hlir_node in self.changed_hlir.clone() {
                // Skip HLIR nodes not present in the current LLIR graph (e.g., from other chunks)
                let Some(&llir_node) = self.hlir_to_llir.get(&hlir_node) else {
                    continue;
                };
                let ptr = match &self.hlir_buffers[&hlir_node] {
                    CudaInput::Buffer(buf) => buf.device_ptr(&self.cuda_stream).0,
                    CudaInput::Ptr(p) => *p,
                };
                self.cached_buffer_ptrs.insert(llir_node, ptr);
            }
            self.changed_hlir.clear();
        }

        // Ensure all CUDA graphs are built (handles first execute and any missing graphs)
        let prebuild_t = std::time::Instant::now();
        self.prebuild_graphs(dyn_map);
        let prebuild_us = prebuild_t.elapsed().as_micros() as f64;

        if !replay_enabled {
            self.runtime_replay = RuntimeReplayCache::default();
        }
        if replay_enabled && self.runtime_replay.state == RuntimeReplayState::Disabled {
            self.runtime_replay.state = RuntimeReplayState::WarmupPending;
        }

        let signature = self.build_runtime_replay_signature(dyn_map, force_zero_all);
        if replay_enabled
            && self
                .runtime_replay
                .signature
                .as_ref()
                .is_some_and(|prev| prev != &signature)
        {
            if replay_debug {
                eprintln!("    [runtime_replay] invalidating cached parent graph");
            }
            self.runtime_replay = RuntimeReplayCache::default();
            self.runtime_replay.state = RuntimeReplayState::WarmupPending;
        }
        if replay_enabled
            && force_recapture
            && self.runtime_replay.state == RuntimeReplayState::Ready
        {
            self.runtime_replay.graph_exec = None;
            self.runtime_replay.state = RuntimeReplayState::CapturePending;
        }
        if replay_enabled {
            self.runtime_replay.signature = Some(signature.clone());
        }

        let mut n_zeroed_bufs = 0usize;
        let mut zero_us = 0.0;
        let mut replay_mode = "dispatch";
        let (n_ops, exec_ops_us) = match self.runtime_replay.state {
            RuntimeReplayState::Ready if replay_enabled => {
                if replay_debug
                    && self
                        .runtime_replay
                        .signature
                        .as_ref()
                        .is_some_and(|sig| sig != &signature)
                {
                    panic!("runtime_replay signature changed without invalidation");
                }
                if let Some(exec) = self.runtime_replay.graph_exec.as_ref() {
                    // Parent replay launches frozen child graphs only; refresh CudaGraphOp
                    // pre_execute state (e.g. Megakernel queues/counters) each frame.
                    if let Err(err) = self.refresh_cuda_graph_ops_for_replay(dyn_map) {
                        if replay_debug {
                            eprintln!(
                                "    [runtime_replay] pre-launch refresh failed, falling back: {err}"
                            );
                        }
                        self.runtime_replay.last_error = Some(err.to_string());
                        self.runtime_replay.state = RuntimeReplayState::WarmupPending;
                        let (n_zeroed, zero_time_us) =
                            self.zero_buffers_for_execute(force_zero_all, true);
                        n_zeroed_bufs = n_zeroed;
                        zero_us = zero_time_us;
                        replay_mode = "fallback";
                        self.execute_host_ops_loop(dyn_map, sync_debug)
                    } else {
                        let launch_t = std::time::Instant::now();
                        // Replay launches the frozen parent graph after per-frame CudaGraphOp pre_execute refresh.
                        if let Err(err) = exec.launch(&self.cuda_stream) {
                            if replay_debug {
                                eprintln!(
                                    "    [runtime_replay] launch failed, falling back: {err}"
                                );
                            }
                            self.runtime_replay.last_error = Some(err.to_string());
                            self.runtime_replay.state = RuntimeReplayState::WarmupPending;
                            let (n_zeroed, zero_time_us) =
                                self.zero_buffers_for_execute(force_zero_all, true);
                            n_zeroed_bufs = n_zeroed;
                            zero_us = zero_time_us;
                            replay_mode = "fallback";
                            self.execute_host_ops_loop(dyn_map, sync_debug)
                        } else {
                            replay_mode = "replay";
                            (1, launch_t.elapsed().as_micros() as f64)
                        }
                    }
                } else {
                    self.runtime_replay.state = RuntimeReplayState::WarmupPending;
                    let (n_zeroed, zero_time_us) =
                        self.zero_buffers_for_execute(force_zero_all, true);
                    n_zeroed_bufs = n_zeroed;
                    zero_us = zero_time_us;
                    replay_mode = "fallback";
                    self.execute_host_ops_loop(dyn_map, sync_debug)
                }
            }
            RuntimeReplayState::CapturePending if replay_enabled => {
                let capture_stream = Arc::clone(&self.replay_capture_stream);
                self.prepare_host_ops_for_capture(&capture_stream)
                    .unwrap_or_else(|e| {
                        panic!("HostOp capture preparation failed: {e:#}");
                    });
                let prepared_stream = self.runtime_replay.prepared_capture_stream;
                let capture_stream_ptr = capture_stream.cu_stream() as u64;
                if prepared_stream != Some(capture_stream_ptr) {
                    if replay_debug {
                        eprintln!(
                            "    [runtime_replay] prep/capture stream mismatch, falling back: prepared={prepared_stream:?} capture=0x{capture_stream_ptr:016x}"
                        );
                    }
                    self.runtime_replay.last_error = Some(format!(
                        "prepare/capture stream mismatch: prepared={prepared_stream:?} capture=0x{capture_stream_ptr:016x}"
                    ));
                    self.runtime_replay.graph_exec = None;
                    self.runtime_replay.state = RuntimeReplayState::WarmupPending;
                    let (n_zeroed, zero_time_us) =
                        self.zero_buffers_for_execute(force_zero_all, true);
                    n_zeroed_bufs = n_zeroed;
                    zero_us = zero_time_us;
                    replay_mode = "fallback";
                    self.execute_host_ops_loop(dyn_map, sync_debug)
                } else {
                    match self.build_runtime_graph(dyn_map, force_zero_all, &capture_stream) {
                        Ok((n_zeroed, ops, build_us, replay_launch_us, mini_captures)) => {
                            n_zeroed_bufs = n_zeroed;
                            zero_us = 0.0;
                            // launch time is the meaningful host-side cost after capture is built.
                            self.runtime_replay.state = RuntimeReplayState::Ready;
                            replay_mode = "capture";
                            if replay_debug {
                                eprintln!(
                                    "    [runtime_replay] built parent graph ({mini_captures} mini-captures, build: {:.2}ms)",
                                    build_us / 1000.0
                                );
                            }
                            (ops, replay_launch_us)
                        }
                        Err(err) => {
                            if replay_debug {
                                eprintln!(
                                    "    [runtime_replay] capture failed, falling back: {err:#}"
                                );
                            }
                            self.runtime_replay.last_error = Some(err.to_string());
                            self.runtime_replay.graph_exec = None;
                            self.runtime_replay.state = RuntimeReplayState::WarmupPending;
                            let (n_zeroed, zero_time_us) =
                                self.zero_buffers_for_execute(force_zero_all, true);
                            n_zeroed_bufs = n_zeroed;
                            zero_us = zero_time_us;
                            replay_mode = "fallback";
                            self.execute_host_ops_loop(dyn_map, sync_debug)
                        }
                    }
                }
            }
            RuntimeReplayState::WarmupPending if replay_enabled => {
                let capture_stream = Arc::clone(&self.replay_capture_stream);
                self.prepare_host_ops_for_capture(&capture_stream)
                    .unwrap_or_else(|e| {
                        panic!("HostOp capture preparation failed: {e:#}");
                    });
                let (n_zeroed, zero_time_us) = self.zero_buffers_for_execute(force_zero_all, true);
                n_zeroed_bufs = n_zeroed;
                zero_us = zero_time_us;
                self.runtime_replay.state = RuntimeReplayState::CapturePending;
                replay_mode = "warmup";
                self.execute_host_ops_loop(dyn_map, sync_debug)
            }
            _ => {
                let (n_zeroed, zero_time_us) = self.zero_buffers_for_execute(force_zero_all, true);
                n_zeroed_bufs = n_zeroed;
                zero_us = zero_time_us;
                self.execute_host_ops_loop(dyn_map, sync_debug)
            }
        };

        // Final sync to ensure all operations completed successfully
        let sync_t = std::time::Instant::now();
        self.cuda_stream
            .synchronize()
            .expect("Final sync failed in execute");
        let final_sync_us = sync_t.elapsed().as_micros() as f64;

        self.last_total_time_us = exec_ops_us;

        if rt_profiling {
            let total_us = zero_us + prebuild_us + exec_ops_us + final_sync_us;
            eprintln!(
                "    [runtime] mode: {} | zero_buffers: {:.2}ms ({} bufs) | prebuild: {:.2}ms | exec_ops: {:.2}ms ({} ops) | final_sync: {:.2}ms | total: {:.2}ms",
                replay_mode,
                zero_us / 1000.0,
                n_zeroed_bufs,
                prebuild_us / 1000.0,
                exec_ops_us / 1000.0,
                n_ops,
                final_sync_us / 1000.0,
                total_us / 1000.0,
            );
        }

        // Populate last_kernel_stats from HostOps that report stats
        self.last_kernel_stats.clear();
        for exec_node in self.exec_graph.node_indices() {
            let exec_op = &self.exec_graph[exec_node];
            if let Some(name) = exec_op.internal.stats_name() {
                self.last_kernel_stats.push(KernelStats {
                    name,
                    execution_time_us: 0.0,
                    bytes_loaded: 0,
                    bytes_stored: 0,
                    flops: 0,
                    bandwidth_gbps: 0.0,
                    tflops: 0.0,
                });
            }
        }
    }
}

impl CudaRuntime {
    fn build_runtime_replay_signature(
        &self,
        dyn_map: &FxHashMap<char, usize>,
        force_zero_all: bool,
    ) -> RuntimeReplaySignature {
        let mut dyn_entries = dyn_map.iter().map(|(k, v)| (*k, *v)).collect_vec();
        dyn_entries.sort_by_key(|(k, _)| *k);

        let mut ptr_entries = self
            .cached_buffer_ptrs
            .iter()
            .map(|(node, ptr)| (node.index(), *ptr))
            .collect_vec();
        ptr_entries.sort_by_key(|(node, _)| *node);

        RuntimeReplaySignature {
            dyn_map: dyn_entries,
            cached_ptrs: ptr_entries,
            force_zero_all,
        }
    }

    fn prepare_host_ops_for_capture(
        &mut self,
        prep_stream: &Arc<CudaStream>,
    ) -> anyhow::Result<()> {
        for exec_node in self.exec_graph.node_indices() {
            let exec_op = &self.exec_graph[exec_node];
            exec_op.internal.prepare_for_capture(prep_stream)?;
        }
        self.runtime_replay.prepared_capture_stream = Some(prep_stream.cu_stream() as u64);
        Ok(())
    }

    fn zero_buffers_for_execute(&mut self, force_zero_all: bool, sync_after: bool) -> (usize, f64) {
        let zero_t = std::time::Instant::now();
        let mut n_zeroed_bufs = 0usize;
        if force_zero_all {
            for buffer in self.buffers.values_mut() {
                self.cuda_stream.memset_zeros(buffer).unwrap();
                n_zeroed_bufs += 1;
            }
        } else {
            let zero_nodes = self.buffers_requiring_zero.iter().copied().collect_vec();
            for node in zero_nodes {
                if let Some(buffer) = self.buffers.get_mut(&node) {
                    self.cuda_stream.memset_zeros(buffer).unwrap();
                    n_zeroed_bufs += 1;
                }
            }
        }
        if sync_after && n_zeroed_bufs > 0 {
            self.cuda_stream.synchronize().unwrap();
        }
        (n_zeroed_bufs, zero_t.elapsed().as_micros() as f64)
    }

    fn refresh_cuda_graph_ops_for_replay(
        &self,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let exec_order = toposort(&self.exec_graph, None).unwrap();
        for exec_node in exec_order {
            let exec_op = &self.exec_graph[exec_node];
            let Some(cuda_graph_op) = exec_op.internal.as_any().downcast_ref::<CudaGraphOp>()
            else {
                continue;
            };
            let buffer_map = self.build_host_op_buffer_map(exec_op);
            cuda_graph_op.ensure_built_and_updated_for_runtime_graph(
                &exec_op.stream,
                &buffer_map,
                dyn_map,
            )?;
        }
        Ok(())
    }

    fn build_host_op_buffer_map<'a>(
        &'a self,
        exec_op: &'a ExecutableHostOp,
    ) -> FxHashMap<NodeIndex, &'a CudaSlice<u8>> {
        // Build buffer map for the HostOp interface
        let mut buffer_map: FxHashMap<NodeIndex, &CudaSlice<u8>> = FxHashMap::default();
        // Add output buffer
        if let Some(buf) = self.buffers.get(&exec_op.output) {
            buffer_map.insert(exec_op.output, buf);
        }
        // Add input buffers
        for inp in exec_op.inputs.iter() {
            if let Some(buf) = self.buffers.get(inp) {
                buffer_map.insert(*inp, buf);
            } else if let Some(hlir_node) = self.llir_to_hlir.get(inp)
                && let Some(CudaInput::Buffer(buf)) = self.hlir_buffers.get(hlir_node)
            {
                buffer_map.insert(*inp, buf);
            }
        }
        // Add extra buffer nodes (for CudaGraphOp)
        let extra_nodes = exec_op.internal.extra_buffer_nodes();
        for extra_node in extra_nodes {
            if let Entry::Vacant(e) = buffer_map.entry(extra_node) {
                if let Some(buf) = self.buffers.get(&extra_node) {
                    e.insert(buf);
                } else if let Some(hlir_node) = self.llir_to_hlir.get(&extra_node)
                    && let Some(CudaInput::Buffer(buf)) = self.hlir_buffers.get(hlir_node)
                {
                    e.insert(buf);
                } else {
                    panic!(
                        "Missing buffer for extra_buffer_node {:?}. has_llir_to_hlir={}, node_in_llir={}",
                        extra_node,
                        self.llir_to_hlir.contains_key(&extra_node),
                        self.llir_graph.node_weight(extra_node).is_some()
                    );
                }
            }
        }
        buffer_map
    }

    fn execute_host_ops_loop(
        &mut self,
        dyn_map: &FxHashMap<char, usize>,
        sync_debug: bool,
    ) -> (usize, f64) {
        let total_start = std::time::Instant::now();
        let exec_order = toposort(&self.exec_graph, None).unwrap();
        let n_ops = exec_order.len();
        for exec_node in exec_order {
            let exec_op = &self.exec_graph[exec_node];
            trace!("Executing: {:?}", exec_op);

            let buffer_map = self.build_host_op_buffer_map(exec_op);
            let _span = span!(
                Level::TRACE,
                "host_op_execute",
                n_inputs = exec_op.inputs.len()
            )
            .entered();
            exec_op
                .internal
                .execute(
                    &exec_op.stream,
                    exec_op.output,
                    &exec_op.inputs,
                    &buffer_map,
                    dyn_map,
                )
                .unwrap_or_else(|e| {
                    let op_name = exec_op.internal.stats_name().unwrap_or("unknown");
                    panic!(
                        "HostOp failed: {e:#}\n  Op: {op_name}\n  Output: {:?}\n  Inputs: {:?}\n  Buffers: {}",
                        exec_op.output, exec_op.inputs, buffer_map.len()
                    );
                });
            if sync_debug {
                self.cuda_stream.synchronize().unwrap();
            }
        }
        (n_ops, total_start.elapsed().as_micros() as f64)
    }

    fn capture_single_host_op_graph(
        &self,
        exec_op: &ExecutableHostOp,
        capture_stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<cudarc::driver::sys::CUgraph> {
        capture_stream.context().bind_to_thread()?;
        // Warm up the op on the capture stream outside stream-capture.
        // cuBLAS/cuBLASLt may JIT kernels and allocate internal resources on first execute.
        exec_op
            .internal
            .warmup_for_capture(
                capture_stream,
                exec_op.output,
                &exec_op.inputs,
                buffers,
                dyn_map,
            )
            .map_err(|err| {
                anyhow::anyhow!("Mini-capture warmup host op execute failed: {err:#}")
            })?;
        capture_stream
            .synchronize()
            .map_err(|err| anyhow::anyhow!("Mini-capture warmup sync failed: {err:#}"))?;

        let prev_capture_flag = crate::CUDA_STREAM_CAPTURING.get();
        crate::CUDA_STREAM_CAPTURING.set(true);
        let capture_res = (|| -> anyhow::Result<cudarc::driver::sys::CUgraph> {
            unsafe {
                cudarc::driver::sys::cuStreamBeginCapture_v2(
                    capture_stream.cu_stream(),
                    cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_GLOBAL,
                )
                .result()?;
            }
            let execute_res = exec_op.internal.execute_for_capture(
                capture_stream,
                exec_op.output,
                &exec_op.inputs,
                buffers,
                dyn_map,
            );
            if let Err(err) = execute_res {
                let mut discarded_graph = std::ptr::null_mut();
                unsafe {
                    let _ = cudarc::driver::sys::cuStreamEndCapture(
                        capture_stream.cu_stream(),
                        &mut discarded_graph,
                    );
                    if !discarded_graph.is_null() {
                        let _ = cudarc::driver::sys::cuGraphDestroy(discarded_graph);
                    }
                }
                return Err(anyhow::anyhow!(
                    "Mini-capture host op execute failed: {err:#}"
                ));
            }

            let mut captured_graph = std::ptr::null_mut();
            unsafe {
                cudarc::driver::sys::cuStreamEndCapture(
                    capture_stream.cu_stream(),
                    &mut captured_graph,
                )
                .result()?;
            }
            if captured_graph.is_null() {
                anyhow::bail!("Mini-capture returned null graph");
            }
            Ok(captured_graph)
        })();
        crate::CUDA_STREAM_CAPTURING.set(prev_capture_flag);
        capture_res
    }

    fn build_runtime_graph(
        &mut self,
        dyn_map: &FxHashMap<char, usize>,
        force_zero_all: bool,
        capture_stream: &Arc<CudaStream>,
    ) -> anyhow::Result<(usize, usize, f64, f64, usize)> {
        let build_t = std::time::Instant::now();
        let mut parent_graph = CudaGraphHandle::new(self.cuda_stream.context().clone())?;
        let exec_order = toposort(&self.exec_graph, None).unwrap();
        let n_ops = exec_order.len();

        let mut n_zeroed_bufs = 0usize;
        let mut mini_capture_count = 0usize;
        let mut op_nodes: FxHashMap<NodeIndex, cudarc::driver::sys::CUgraphNode> =
            FxHashMap::default();
        let mut zero_nodes: FxHashMap<NodeIndex, cudarc::driver::sys::CUgraphNode> =
            FxHashMap::default();
        let mut touched_buffers: FxHashMap<NodeIndex, FxHashSet<NodeIndex>> = FxHashMap::default();

        // Add memset nodes for buffers requiring zeroing.
        let zero_targets = if force_zero_all {
            self.buffers.keys().copied().collect_vec()
        } else {
            self.buffers_requiring_zero.iter().copied().collect_vec()
        };
        for node in zero_targets.into_iter().sorted_by_key(|n| n.index()) {
            if let Some(buffer) = self.buffers.get(&node) {
                let ptr = buffer.device_ptr(&self.cuda_stream).0;
                let memset_node = parent_graph.add_memset_zero_node_u8(&[], ptr, buffer.len())?;
                zero_nodes.insert(node, memset_node);
                n_zeroed_bufs += 1;
            }
        }

        // Build one parent node per HostOp.
        for exec_node in exec_order {
            let exec_op = &self.exec_graph[exec_node];
            let buffer_map = self.build_host_op_buffer_map(exec_op);
            touched_buffers.insert(exec_node, buffer_map.keys().copied().collect());

            let graph_node = if let Some(cuda_graph_op) =
                exec_op.internal.as_any().downcast_ref::<CudaGraphOp>()
            {
                // Build/update child graph without launching; parent graph will launch it later.
                cuda_graph_op.ensure_built_and_updated_for_runtime_graph(
                    &exec_op.stream,
                    &buffer_map,
                    dyn_map,
                )?;
                let child_graph = cuda_graph_op.cu_graph().ok_or_else(|| {
                    anyhow::anyhow!("CudaGraphOp child graph missing after ensure_built")
                })?;
                parent_graph.add_child_graph_node(&[], child_graph)?
            } else {
                let op_name = exec_op.internal.stats_name().unwrap_or("unknown");
                if op_name != "cuBLAS" && op_name != "cuBLASLt" {
                    anyhow::bail!("Unsupported HostOp for runtime graph build: {op_name}");
                }

                // Mini-capture uses current buffer_map pointers; parent node clones this graph.
                // Capture prep and mini-capture must use the same CUDA stream handle.
                // If pointers change, signature invalidation forces a full parent graph rebuild.
                let captured = self.capture_single_host_op_graph(
                    exec_op,
                    capture_stream,
                    &buffer_map,
                    dyn_map,
                )?;
                mini_capture_count += 1;
                let node = parent_graph.add_child_graph_node(&[], captured)?;
                unsafe {
                    cudarc::driver::sys::cuGraphDestroy(captured).result()?;
                }
                node
            };

            op_nodes.insert(exec_node, graph_node);
        }

        // Add data dependencies from exec_graph edges.
        let mut added_deps: FxHashSet<(usize, usize)> = FxHashSet::default();
        for edge in self.exec_graph.edge_indices() {
            let (src, dst) = self.exec_graph.edge_endpoints(edge).unwrap();
            let Some(&from_node) = op_nodes.get(&src) else {
                continue;
            };
            let Some(&to_node) = op_nodes.get(&dst) else {
                continue;
            };
            let key = (from_node as usize, to_node as usize);
            if added_deps.insert(key) {
                parent_graph.add_dependency(from_node, to_node)?;
            }
        }

        // Zeroing dependencies: memset(X) must happen before every op that touches X.
        for (exec_node, buffers) in touched_buffers {
            let Some(&op_node) = op_nodes.get(&exec_node) else {
                continue;
            };
            for buffer_node in buffers {
                let Some(&zero_node) = zero_nodes.get(&buffer_node) else {
                    continue;
                };
                let key = (zero_node as usize, op_node as usize);
                if added_deps.insert(key) {
                    parent_graph.add_dependency(zero_node, op_node)?;
                }
            }
        }

        let graph_exec = parent_graph.instantiate()?;
        let build_us = build_t.elapsed().as_micros() as f64;
        self.runtime_replay.graph_exec = Some(graph_exec);

        let launch_t = std::time::Instant::now();
        self.runtime_replay
            .graph_exec
            .as_ref()
            .expect("runtime graph exec should be initialized")
            .launch(&self.cuda_stream)?;
        let launch_us = launch_t.elapsed().as_micros() as f64;

        Ok((
            n_zeroed_bufs,
            n_ops,
            build_us,
            launch_us,
            mini_capture_count,
        ))
    }

    fn compute_block_op_stats(
        llir_graph: &LLIRGraph,
        timings: &[(Vec<SMEvent>, u64, Uuid)],
        dyn_map: &FxHashMap<char, usize>,
        sm_count: usize,
    ) -> Vec<BlockOpStats> {
        // Get unique op names (same order as in interpreter)
        let op_names: Vec<&'static str> = llir_graph
            .node_indices()
            .filter_map(|n| llir_graph[n].to_dialect::<dyn BlockOp>())
            .map(|bo| (bo.op_name(), bo.clone()))
            .collect::<FxHashMap<_, _>>()
            .into_iter()
            .sorted_by_key(|(n, _)| *n)
            .map(|(n, _)| n)
            .collect();

        if op_names.is_empty() {
            return vec![];
        }

        // Build map from op_name to index for event decoding
        let op_name_to_idx: FxHashMap<&'static str, usize> =
            op_names.iter().enumerate().map(|(i, n)| (*n, i)).collect();

        // Sum up bytes_loaded, bytes_stored, flops across ALL instances of each op type
        let mut op_bytes_loaded: Vec<usize> = vec![0; op_names.len()];
        let mut op_bytes_stored: Vec<usize> = vec![0; op_names.len()];
        let mut op_flops: Vec<usize> = vec![0; op_names.len()];

        // Sum up prologue metrics across ALL instances of each op type
        let mut prologue_a_bytes_loaded: Vec<usize> = vec![0; op_names.len()];
        let mut prologue_a_flops: Vec<usize> = vec![0; op_names.len()];
        let mut prologue_b_bytes_loaded: Vec<usize> = vec![0; op_names.len()];
        let mut prologue_b_flops: Vec<usize> = vec![0; op_names.len()];
        let mut prologue_c_bytes_loaded: Vec<usize> = vec![0; op_names.len()];
        let mut prologue_c_flops: Vec<usize> = vec![0; op_names.len()];

        for node in llir_graph.node_indices() {
            if let Some(op) = llir_graph[node].to_dialect::<dyn BlockOp>()
                && let Some(&idx) = op_name_to_idx.get(op.op_name())
            {
                let flops_expr = op.flops();
                let flops_val = flops_expr.exec(dyn_map).unwrap();
                op_bytes_loaded[idx] += op.bytes_loaded().exec(dyn_map).unwrap();
                op_bytes_stored[idx] += op.bytes_stored().exec(dyn_map).unwrap();
                op_flops[idx] += flops_val;

                // Aggregate prologue metrics
                prologue_a_bytes_loaded[idx] +=
                    op.prologue_a_bytes_loaded().exec(dyn_map).unwrap_or(0);
                prologue_a_flops[idx] += op.prologue_a_flops().exec(dyn_map).unwrap_or(0);
                prologue_b_bytes_loaded[idx] +=
                    op.prologue_b_bytes_loaded().exec(dyn_map).unwrap_or(0);
                prologue_b_flops[idx] += op.prologue_b_flops().exec(dyn_map).unwrap_or(0);
                prologue_c_bytes_loaded[idx] +=
                    op.prologue_c_bytes_loaded().exec(dyn_map).unwrap_or(0);
                prologue_c_flops[idx] += op.prologue_c_flops().exec(dyn_map).unwrap_or(0);
            }
        }

        // Aggregate timing per op type across all SMs
        // Event encoding:
        // 0: Issue
        // 1: Wait
        // 2 to 2 + n_ops - 1: Main ops
        // 2 + n_ops + op_idx * 3 + 0: Prologue A
        // 2 + n_ops + op_idx * 3 + 1: Prologue B
        // 2 + n_ops + op_idx * 3 + 2: Prologue C
        let n_ops = op_names.len();
        let mut op_times_ns: Vec<u64> = vec![0; n_ops];
        let mut op_counts: Vec<usize> = vec![0; n_ops];
        let mut prologue_a_times_ns: Vec<u64> = vec![0; n_ops];
        let mut prologue_a_counts: Vec<usize> = vec![0; n_ops];
        let mut prologue_b_times_ns: Vec<u64> = vec![0; n_ops];
        let mut prologue_b_counts: Vec<usize> = vec![0; n_ops];
        let mut prologue_c_times_ns: Vec<u64> = vec![0; n_ops];
        let mut prologue_c_counts: Vec<usize> = vec![0; n_ops];
        let mut issue_time_ns: u64 = 0;
        let mut issue_count: usize = 0;
        let mut wait_time_ns: u64 = 0;
        let mut wait_count: usize = 0;

        for (sm_timings, _start_time, _) in timings {
            for sm_chunk in sm_timings.chunks(N_TIMING_SLOTS) {
                for event in sm_chunk.iter() {
                    if event.start == 0 {
                        break; // No more events recorded for this SM
                    }
                    let stop = if event.stop == 0 {
                        event.start
                    } else {
                        event.stop
                    };
                    let duration = stop.saturating_sub(event.start);
                    let event_code = event.event as usize;
                    if event_code == 0 {
                        issue_time_ns += duration;
                        issue_count += 1;
                    } else if event_code == 1 {
                        wait_time_ns += duration;
                        wait_count += 1;
                    } else if event_code >= 2 && event_code < 2 + n_ops {
                        // Main op
                        let op_idx = event_code - 2;
                        op_times_ns[op_idx] += duration;
                        op_counts[op_idx] += 1;
                    } else if event_code >= 2 + n_ops {
                        // Prologue event
                        let prologue_event = event_code - 2 - n_ops;
                        let op_idx = prologue_event / 3;
                        let prologue_type = prologue_event % 3;
                        if op_idx < n_ops {
                            match prologue_type {
                                0 => {
                                    prologue_a_times_ns[op_idx] += duration;
                                    prologue_a_counts[op_idx] += 1;
                                }
                                1 => {
                                    prologue_b_times_ns[op_idx] += duration;
                                    prologue_b_counts[op_idx] += 1;
                                }
                                2 => {
                                    prologue_c_times_ns[op_idx] += duration;
                                    prologue_c_counts[op_idx] += 1;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        // Compute bandwidth and TFLOPS from aggregated metrics
        // Divide total SM-time by sm_count to get wall-clock time
        let mut stats: Vec<BlockOpStats> = op_names
            .iter()
            .enumerate()
            .filter(|(i, _)| op_times_ns[*i] > 0)
            .map(|(i, &name)| {
                // Total SM-time divided by number of SMs gives wall-clock time
                let time_us = (op_times_ns[i] as f64 / sm_count as f64) / 1000.0;
                let bytes_loaded = op_bytes_loaded[i];
                let bytes_stored = op_bytes_stored[i];
                let flop_count = op_flops[i];

                let total_bytes = bytes_loaded + bytes_stored;
                let bandwidth_gbps = if time_us > 0.0 {
                    (total_bytes as f64) / (time_us * 1e-6) / 1e9
                } else {
                    0.0
                };
                let tflops = if time_us > 0.0 {
                    (flop_count as f64) / (time_us * 1e-6) / 1e12
                } else {
                    0.0
                };

                BlockOpStats {
                    name,
                    execution_time_us: time_us,
                    bytes_loaded,
                    bytes_stored,
                    flops: flop_count,
                    bandwidth_gbps,
                    tflops,
                    count: op_counts[i],
                }
            })
            .collect();

        // Add prologue timing stats (only for prologues that were recorded)
        for (i, &name) in op_names.iter().enumerate() {
            if prologue_a_times_ns[i] > 0 {
                let time_us = (prologue_a_times_ns[i] as f64 / sm_count as f64) / 1000.0;
                let bytes_loaded = prologue_a_bytes_loaded[i];
                let flop_count = prologue_a_flops[i];
                let bandwidth_gbps = if time_us > 0.0 && bytes_loaded > 0 {
                    (bytes_loaded as f64) / (time_us * 1e-6) / 1e9
                } else {
                    0.0
                };
                let tflops = if time_us > 0.0 && flop_count > 0 {
                    (flop_count as f64) / (time_us * 1e-6) / 1e12
                } else {
                    0.0
                };
                stats.push(BlockOpStats {
                    name: Box::leak(format!("{} (prologue A)", name).into_boxed_str()),
                    execution_time_us: time_us,
                    bytes_loaded,
                    bytes_stored: 0,
                    flops: flop_count,
                    bandwidth_gbps,
                    tflops,
                    count: prologue_a_counts[i],
                });
            }
            if prologue_b_times_ns[i] > 0 {
                let time_us = (prologue_b_times_ns[i] as f64 / sm_count as f64) / 1000.0;
                let bytes_loaded = prologue_b_bytes_loaded[i];
                let flop_count = prologue_b_flops[i];
                let bandwidth_gbps = if time_us > 0.0 && bytes_loaded > 0 {
                    (bytes_loaded as f64) / (time_us * 1e-6) / 1e9
                } else {
                    0.0
                };
                let tflops = if time_us > 0.0 && flop_count > 0 {
                    (flop_count as f64) / (time_us * 1e-6) / 1e12
                } else {
                    0.0
                };
                stats.push(BlockOpStats {
                    name: Box::leak(format!("{} (prologue B)", name).into_boxed_str()),
                    execution_time_us: time_us,
                    bytes_loaded,
                    bytes_stored: 0,
                    flops: flop_count,
                    bandwidth_gbps,
                    tflops,
                    count: prologue_b_counts[i],
                });
            }
            if prologue_c_times_ns[i] > 0 {
                let time_us = (prologue_c_times_ns[i] as f64 / sm_count as f64) / 1000.0;
                let bytes_loaded = prologue_c_bytes_loaded[i];
                let flop_count = prologue_c_flops[i];
                let bandwidth_gbps = if time_us > 0.0 && bytes_loaded > 0 {
                    (bytes_loaded as f64) / (time_us * 1e-6) / 1e9
                } else {
                    0.0
                };
                let tflops = if time_us > 0.0 && flop_count > 0 {
                    (flop_count as f64) / (time_us * 1e-6) / 1e12
                } else {
                    0.0
                };
                stats.push(BlockOpStats {
                    name: Box::leak(format!("{} (prologue C)", name).into_boxed_str()),
                    execution_time_us: time_us,
                    bytes_loaded,
                    bytes_stored: 0,
                    flops: flop_count,
                    bandwidth_gbps,
                    tflops,
                    count: prologue_c_counts[i],
                });
            }
        }

        // Add Issue and Wait timing stats
        if issue_time_ns > 0 {
            let time_us = (issue_time_ns as f64 / sm_count as f64) / 1000.0;
            stats.push(BlockOpStats {
                name: "Issue",
                execution_time_us: time_us,
                bytes_loaded: 0,
                bytes_stored: 0,
                flops: 0,
                bandwidth_gbps: 0.0,
                tflops: 0.0,
                count: issue_count,
            });
        }
        if wait_time_ns > 0 {
            let time_us = (wait_time_ns as f64 / sm_count as f64) / 1000.0;
            stats.push(BlockOpStats {
                name: "Wait",
                execution_time_us: time_us,
                bytes_loaded: 0,
                bytes_stored: 0,
                flops: 0,
                bandwidth_gbps: 0.0,
                tflops: 0.0,
                count: wait_count,
            });
        }

        stats
    }

    /// Print execution statistics for the last execution (computes block op stats lazily).
    pub fn print_execution_stats(&self) {
        if self.last_kernel_stats.is_empty() && self.timings.is_empty() {
            println!("No execution stats available.");
            return;
        }

        // Compute block op stats lazily
        let sm_count = self
            .cuda_stream
            .context()
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .unwrap() as usize;
        let block_op_stats = Self::compute_block_op_stats(
            &self.llir_graph,
            self.timings.last().map(|v| v.as_slice()).unwrap_or(&[]),
            &self.last_dyn_map,
            sm_count,
        );

        // Compute aggregates
        let total_bytes_loaded: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.bytes_loaded)
            .sum::<usize>()
            + block_op_stats.iter().map(|s| s.bytes_loaded).sum::<usize>();
        let total_bytes_stored: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.bytes_stored)
            .sum::<usize>()
            + block_op_stats.iter().map(|s| s.bytes_stored).sum::<usize>();
        let total_flops: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.flops)
            .sum::<usize>()
            + block_op_stats.iter().map(|s| s.flops).sum::<usize>();
        let total_bytes = total_bytes_loaded + total_bytes_stored;
        let aggregate_bw = if self.last_total_time_us > 0.0 {
            (total_bytes as f64) / (self.last_total_time_us * 1e-6) / 1e9
        } else {
            0.0
        };
        let aggregate_tf = if self.last_total_time_us > 0.0 {
            (total_flops as f64) / (self.last_total_time_us * 1e-6) / 1e12
        } else {
            0.0
        };

        let peak_bw = crate::cuda_bandwidth_gbps(self.cuda_stream.context());
        let peak_tf = crate::cuda_compute_f32_tflops(self.cuda_stream.context());

        // Print kernel stats
        if !self.last_kernel_stats.is_empty() {
            println!("\n=== Kernel Execution Statistics ===\n");
            println!(
                "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
                "Kernel",
                "Time (us)",
                "Loaded",
                "Stored",
                "Agg FLOPS",
                "BW (GB/s)",
                "TFLOPS",
                "MBU",
                "MFU"
            );
            println!("{}", "-".repeat(116));
            for s in &self.last_kernel_stats {
                self.print_stat_row(
                    s.name,
                    s.execution_time_us,
                    None,
                    s.bytes_loaded,
                    s.bytes_stored,
                    s.flops,
                    s.bandwidth_gbps,
                    s.tflops,
                    peak_bw,
                    peak_tf,
                );
            }
            println!("{}", "-".repeat(116));
        }

        // Print block op stats
        if !block_op_stats.is_empty() {
            println!("\n=== Block Op Execution Statistics ===\n");
            println!(
                "{:<20} {:>12} {:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
                "BlockOp",
                "Time (us)",
                "Count",
                "Loaded",
                "Stored",
                "Agg FLOPS",
                "BW (GB/s)",
                "TFLOPS",
                "MBU",
                "MFU"
            );
            println!("{}", "-".repeat(124));
            for s in &block_op_stats {
                self.print_stat_row(
                    s.name,
                    s.execution_time_us,
                    Some(s.count),
                    s.bytes_loaded,
                    s.bytes_stored,
                    s.flops,
                    s.bandwidth_gbps,
                    s.tflops,
                    peak_bw,
                    peak_tf,
                );
            }
            println!("{}", "-".repeat(124));
        }

        // Print aggregate stats
        println!("\n=== Aggregate Statistics ===\n");
        println!(
            "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
            "", "Time (us)", "Loaded", "Stored", "Agg FLOPS", "BW (GB/s)", "TFLOPS", "MBU", "MFU"
        );
        println!("{}", "-".repeat(116));
        let (mbu, mfu) = match (peak_bw, peak_tf) {
            (Some(pb), Some(pt)) => (
                format!("{:.1}%", aggregate_bw / pb as f64 * 100.0),
                format!("{:.1}%", aggregate_tf / pt as f64 * 100.0),
            ),
            _ => ("-".into(), "-".into()),
        };
        println!(
            "{:<20} {:>12.2} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
            "Total",
            self.last_total_time_us,
            format_size(total_bytes_loaded),
            format_size(total_bytes_stored),
            format_flops(total_flops),
            format!("{:.2}", aggregate_bw),
            format!("{:.4}", aggregate_tf),
            mbu,
            mfu
        );

        if let (Some(pb), Some(pt)) = (peak_bw, peak_tf) {
            println!("\nDevice peak: {} GB/s bandwidth, {} TFLOPS (F32)", pb, pt);
        }
        println!();
    }

    #[allow(clippy::too_many_arguments)]
    fn print_stat_row(
        &self,
        name: &str,
        time_us: f64,
        count: Option<usize>,
        loaded: usize,
        stored: usize,
        flops: usize,
        bw: f64,
        tf: f64,
        peak_bw: Option<usize>,
        peak_tf: Option<usize>,
    ) {
        let total = loaded + stored;
        let ld = if loaded > 0 {
            format_size(loaded)
        } else {
            "-".into()
        };
        let st = if stored > 0 {
            format_size(stored)
        } else {
            "-".into()
        };
        let fl = if flops > 0 {
            format_flops(flops)
        } else {
            "-".into()
        };
        let bw_s = if total > 0 {
            format!("{bw:.2}")
        } else {
            "-".into()
        };
        let tf_s = if flops > 0 {
            format!("{tf:.4}")
        } else {
            "-".into()
        };
        let mbu = peak_bw
            .filter(|_| total > 0)
            .map(|p| format!("{:.1}%", bw / p as f64 * 100.0))
            .unwrap_or("-".into());
        let mfu = peak_tf
            .filter(|_| flops > 0)
            .map(|p| format!("{:.1}%", tf / p as f64 * 100.0))
            .unwrap_or("-".into());

        match count {
            Some(c) => println!(
                "{name:<20} {time_us:>12.2} {c:>8} {ld:>12} {st:>12} {fl:>12} {bw_s:>12} {tf_s:>12} {mbu:>8} {mfu:>8}"
            ),
            None => println!(
                "{name:<20} {time_us:>12.2} {ld:>12} {st:>12} {fl:>12} {bw_s:>12} {tf_s:>12} {mbu:>8} {mfu:>8}"
            ),
        }
    }

    /// Flush pending timing data to the timings collection.
    fn flush_pending_timings(&mut self) {
        if self.pending_timing_data.is_empty() {
            return;
        }

        // Sync timing stream first to ensure all async copies are complete
        self.timing_stream
            .synchronize()
            .expect("Failed to sync timing stream");

        // Extract data from pinned memory pool
        let mut timing_entries = Vec::with_capacity(self.pending_timing_data.len());
        for pending in self.pending_timing_data.drain(..) {
            let timing_data = self.pinned_timing_pool[pending.timing_buffer_idx]
                .as_slice()
                .expect("Failed to read timing buffer")
                .to_vec();
            let min_start = self.pinned_start_pool[pending.start_time_idx]
                .as_slice()
                .expect("Failed to read start time buffer")
                .iter()
                .copied()
                .min()
                .unwrap_or(0);
            timing_entries.push((timing_data, min_start, pending.span_id));
        }

        // Reset pool index so buffers can be reused
        self.next_pinned_slot = 0;

        if !timing_entries.is_empty() {
            self.timings.push(timing_entries);
        }
    }

    /// Record GPU timings to an existing perfetto trace file.
    pub fn record_cuda_perfetto_trace(&mut self, perfetto_guard: PerfettoGuard) {
        // Flush any pending timing copies first
        self.flush_pending_timings();

        perfetto_guard.stop();
        let ops: Vec<Arc<Box<dyn BlockOp>>> = self
            .llir_graph
            .node_indices()
            .filter_map(|n| self.llir_graph[n].to_dialect::<dyn BlockOp>())
            .map(|bo| (bo.op_name(), bo.clone()))
            .collect::<std::collections::HashMap<_, _>>()
            .into_iter()
            .sorted_by_key(|(n, _)| *n)
            .map(|(_, o)| o)
            .collect();
        let data = std::fs::read(&perfetto_guard.path).unwrap();
        let mut trace = tracing_perfetto_sdk_schema::Trace::decode(data.as_slice()).unwrap();
        let mut extra_packets = record_block_op_timings(&trace, &ops, &self.timings);
        extra_packets.extend(record_cuda_graph_timings(&trace, &self.cuda_graph_timings));
        trace.packet.extend(extra_packets);
        // Sort ALL packets by timestamp for proper Perfetto visualization
        trace.packet.sort_by_key(|p| p.timestamp.unwrap_or(0));
        let mut buf = Vec::with_capacity(trace.encoded_len());
        trace.encode(&mut buf).unwrap();
        std::fs::write(perfetto_guard.path, buf).unwrap();
    }
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.2} KB", bytes as f64 / 1e3)
    } else {
        format!("{} B", bytes)
    }
}

fn format_flops(flops: usize) -> String {
    if flops >= 1_000_000_000_000 {
        format!("{:.2} T", flops as f64 / 1e12)
    } else if flops >= 1_000_000_000 {
        format!("{:.2} G", flops as f64 / 1e9)
    } else if flops >= 1_000_000 {
        format!("{:.2} M", flops as f64 / 1e6)
    } else if flops >= 1_000 {
        format!("{:.2} K", flops as f64 / 1e3)
    } else {
        format!("{}", flops)
    }
}

pub(crate) fn partition_marked_convex<T, E>(
    g: &StableGraph<T, E, Directed>,
    marked: &FxHashSet<NodeIndex>,
) -> Result<Vec<FxHashSet<NodeIndex>>, Cycle<NodeIndex>> {
    if marked.is_empty() {
        return Ok(vec![]);
    }

    // --- Global topo order (also validates DAG) ---
    let topo = toposort(g, None)?;
    let topo_len = topo.len();

    // Map NodeIndex <-> topo position
    let mut idx_to_pos: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    let mut pos_to_idx: Vec<NodeIndex> = Vec::with_capacity(topo_len);
    for (pos, &ni) in topo.iter().enumerate() {
        idx_to_pos.insert(ni, pos);
        pos_to_idx.push(ni);
    }

    // --- Full-graph reachability: reach[upos] contains all vpos reachable from u ---
    // (Bitset DP over topo order)
    let mut reach: Vec<FixedBitSet> = (0..topo_len)
        .map(|_| {
            let mut b = FixedBitSet::with_capacity(topo_len);
            b.grow(topo_len);
            b
        })
        .collect();

    for &u in topo.iter().rev() {
        let upos = idx_to_pos[&u];
        for v in g.neighbors_directed(u, Direction::Outgoing) {
            if let Some(&vpos) = idx_to_pos.get(&v) {
                reach[upos].insert(vpos);
                let rv = reach[vpos].clone();
                reach[upos].union_with(&rv);
            }
        }
    }

    // --- 1) Weakly-connected components in the marked-induced subgraph ---
    let components = marked_weak_components(g, marked);

    let mut results: Vec<FxHashSet<NodeIndex>> = Vec::new();

    for comp in components {
        // Component nodes in topo positions (sorted)
        let mut comp_pos: Vec<usize> = comp
            .iter()
            .filter_map(|ni| idx_to_pos.get(ni).copied())
            .collect();
        comp_pos.sort_unstable();

        // Membership: in_comp_pos bitset over topo positions
        let mut in_comp_pos = FixedBitSet::with_capacity(topo_len);
        in_comp_pos.grow(topo_len);
        for &p in &comp_pos {
            in_comp_pos.insert(p);
        }

        // Membership: in_comp_idx vec over NodeIndex::index() for component-relative DP
        let mut in_comp_idx = vec![false; g.node_bound()];
        for &n in &comp {
            in_comp_idx[n.index()] = true;
        }

        // --- Component-relative "between" witnesses (path-wise, correct) ---
        // has_comp_anc[x] == true if x has a component node as an ancestor (or is in comp)
        let mut has_comp_anc = vec![false; g.node_bound()];
        for &u in &topo {
            let mut v = in_comp_idx[u.index()];
            for p in g.neighbors_directed(u, Direction::Incoming) {
                v |= has_comp_anc[p.index()];
                if v {
                    break;
                }
            }
            has_comp_anc[u.index()] = v;
        }

        // has_comp_des[x] == true if x has a component node as a descendant (or is in comp)
        let mut has_comp_des = vec![false; g.node_bound()];
        for &u in topo.iter().rev() {
            let mut v = in_comp_idx[u.index()];
            for s in g.neighbors_directed(u, Direction::Outgoing) {
                v |= has_comp_des[s.index()];
                if v {
                    break;
                }
            }
            has_comp_des[u.index()] = v;
        }

        // --- Build witness constraints Px/Sx only for true witnesses of THIS component ---
        // Witness x is UNMARKED and lies on some path comp_node ->* x ->* comp_node.
        // For each witness x:
        //   Px(x) = {u in comp | u ->* x}
        //   Sx(x) = {v in comp | x ->* v}
        // A valid block cannot contain nodes from both Px(x) and Sx(x).
        let mut px_map: FxHashMap<NodeIndex, FixedBitSet> = FxHashMap::default();
        let mut sx_map: FxHashMap<NodeIndex, FixedBitSet> = FxHashMap::default();
        let mut px_witnesses: FxHashMap<usize, Vec<NodeIndex>> = FxHashMap::default(); // upos -> witnesses where upos ∈ Px
        let mut sx_witnesses: FxHashMap<usize, Vec<NodeIndex>> = FxHashMap::default(); // vpos -> witnesses where vpos ∈ Sx

        for x in g.node_indices() {
            if marked.contains(&x) {
                continue; // must be outside the block (unmarked) to be a witness
            }
            if !(has_comp_anc[x.index()] && has_comp_des[x.index()]) {
                continue; // not between this component's marked nodes
            }

            let Some(&xpos) = idx_to_pos.get(&x) else {
                continue;
            };
            // Sx = reachable-from-x ∩ component
            let mut sx = reach[xpos].clone();
            sx.intersect_with(&in_comp_pos);
            if sx.is_empty() {
                continue;
            }

            // Px = {u in comp | u can reach x}
            let mut px = FixedBitSet::with_capacity(topo_len);
            px.grow(topo_len);
            for &upos in &comp_pos {
                if reach[upos].contains(xpos) {
                    px.insert(upos);
                }
            }
            if px.is_empty() {
                continue;
            }

            px_map.insert(x, px.clone());
            sx_map.insert(x, sx.clone());

            for upos in px.ones() {
                px_witnesses.entry(upos).or_default().push(x);
            }
            for vpos in sx.ones() {
                sx_witnesses.entry(vpos).or_default().push(x);
            }
        }

        // --- 3) Deterministic topo sweep partition within this component ---
        let mut current: FxHashSet<NodeIndex> = FxHashSet::default();
        let mut block_bits = FixedBitSet::with_capacity(topo_len);
        block_bits.grow(topo_len);

        for &p in &comp_pos {
            let violates = would_violate(
                p,
                &block_bits,
                &px_witnesses,
                &sx_witnesses,
                &px_map,
                &sx_map,
            );

            if violates && !current.is_empty() {
                results.push(std::mem::take(&mut current));
                block_bits.clear(); // keeps length
            }

            let ni = pos_to_idx[p];
            current.insert(ni);
            block_bits.insert(p);
        }

        if !current.is_empty() {
            results.push(current);
        }
    }

    Ok(results)
}

/// Deterministic “contiguous marked” components: weakly-connected in the marked-induced subgraph.
fn marked_weak_components<T, E>(
    g: &StableGraph<T, E, Directed>,
    marked: &FxHashSet<NodeIndex>,
) -> Vec<Vec<NodeIndex>> {
    let mut seen: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut comps: Vec<Vec<NodeIndex>> = Vec::new();

    for start in g.node_indices() {
        if !marked.contains(&start) || seen.contains(&start) {
            continue;
        }

        let mut q = VecDeque::new();
        q.push_back(start);
        seen.insert(start);

        let mut comp = Vec::new();
        while let Some(u) = q.pop_front() {
            comp.push(u);
            for v in g.neighbors_undirected(u) {
                if marked.contains(&v) && seen.insert(v) {
                    q.push_back(v);
                }
            }
        }
        comps.push(comp);
    }

    comps
}

fn would_violate(
    p: usize,
    block_bits: &FixedBitSet,
    px_witnesses: &FxHashMap<usize, Vec<NodeIndex>>,
    sx_witnesses: &FxHashMap<usize, Vec<NodeIndex>>,
    px_map: &FxHashMap<NodeIndex, FixedBitSet>,
    sx_map: &FxHashMap<NodeIndex, FixedBitSet>,
) -> bool {
    // If p ∈ Px(x), block cannot contain any node in Sx(x)
    if let Some(ws) = px_witnesses.get(&p) {
        for &x in ws {
            if let Some(sx) = sx_map.get(&x)
                && intersects(block_bits, sx)
            {
                return true;
            }
        }
    }

    // If p ∈ Sx(x), block cannot contain any node in Px(x)
    if let Some(ws) = sx_witnesses.get(&p) {
        for &x in ws {
            if let Some(px) = px_map.get(&x)
                && intersects(block_bits, px)
            {
                return true;
            }
        }
    }

    false
}

fn intersects(a: &FixedBitSet, b: &FixedBitSet) -> bool {
    let mut tmp = a.clone();
    tmp.intersect_with(b);
    // Note: is_empty() checks if length is 0, not if there are no bits set
    // Use count_ones() to check if there are any set bits after intersection
    tmp.count_ones(..) > 0
}
