//! Compiles KernelOp subgraphs into HostOp (CudaGraphOp).
//!
//! CudaGraphOp wraps a subgraph of KernelOps into a single executable unit
//! that can be executed like any other HostOp.

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr,
    sys::{CUgraph, CUgraphNode},
};
use itertools::Itertools;
use luminal::{
    graph::LLIRGraph,
    op::{EgglogOp, LLIROp},
    prelude::{
        petgraph::{Direction, algo::toposort, visit::EdgeRef},
        *,
    },
};
use tracing::{Level, enabled, span};

use crate::{
    host::HostOp,
    kernel::{
        CudaFunctionExt, CudaGraphExecHandle, CudaGraphHandle, KernelOp, create_cuda_event,
        destroy_cuda_event,
    },
    runtime::partition_marked_convex,
};

/// A compiled kernel within a CudaGraphOp.
#[derive(Debug)]
struct CompiledKernel {
    /// The node index in the original llir_graph
    node: NodeIndex,
    /// The compiled CUDA function
    function: CudaFunction,
    /// Launch grid dimensions (blocks)
    grid: (Expression, Expression, Expression),
    /// Launch block dimensions (threads)
    block: (Expression, Expression, Expression),
    /// Shared memory size
    shared_mem: Expression,
    /// Input node indices (for buffer lookup)
    inputs: Vec<NodeIndex>,
    /// Reference to the KernelOp for trait methods
    kernel_op: Arc<Box<dyn KernelOp>>,
    /// Internal buffers allocated for this kernel
    internal_bufs: Vec<CudaSlice<u8>>,
    /// Device constants from compile()
    constants: FxHashMap<char, CudaSlice<u8>>,
    /// Graph node handle (set after graph is built)
    graph_node: Option<CUgraphNode>,
    /// Kernel name for profiling
    kernel_name: &'static str,
}

impl CompiledKernel {
    #[allow(clippy::too_many_arguments)]
    fn new(
        node: NodeIndex,
        function: CudaFunction,
        grid: (Expression, Expression, Expression),
        block: (Expression, Expression, Expression),
        shared_mem: Expression,
        inputs: Vec<NodeIndex>,
        kernel_op: Arc<Box<dyn KernelOp>>,
        constants: FxHashMap<char, CudaSlice<u8>>,
        kernel_name: &'static str,
    ) -> Self {
        Self {
            node,
            function,
            grid,
            block,
            shared_mem,
            inputs,
            kernel_op,
            internal_bufs: Vec::new(),
            constants,
            graph_node: None,
            kernel_name,
        }
    }
}

/// Unified kernel params that can hold any number of u64 values.
struct UnifiedKernelParams {
    values: Vec<u64>,
    ptrs: Vec<*mut std::ffi::c_void>,
}

impl UnifiedKernelParams {
    fn new(values: Vec<u64>) -> Self {
        let ptrs = values
            .iter()
            .map(|v| v as *const u64 as *mut std::ffi::c_void)
            .collect();
        Self { values, ptrs }
    }

    fn as_cuda_params(&mut self) -> *mut *mut std::ffi::c_void {
        // Rebuild pointers (in case struct was moved)
        for (i, v) in self.values.iter().enumerate() {
            self.ptrs[i] = v as *const u64 as *mut std::ffi::c_void;
        }
        self.ptrs.as_mut_ptr()
    }
}

/// Mutable state for CudaGraphOp that needs interior mutability.
struct CudaGraphOpState {
    /// Compiled kernels in topological order
    kernels: Vec<CompiledKernel>,
    /// Shared device buffer for dynamic dimensions
    dyn_dims_buffer: Option<CudaSlice<i32>>,
    /// CUDA graph handle
    cuda_graph: Option<CudaGraphHandle>,
    /// CUDA graph exec handle
    cuda_graph_exec: Option<CudaGraphExecHandle>,
    /// Mapping from kernel node to graph node
    node_to_graph_node: FxHashMap<NodeIndex, CUgraphNode>,
    /// Kernel params for each kernel
    kernel_params: Vec<UnifiedKernelParams>,
    /// Last dynamic dimension values (for change detection)
    last_dyn_values: FxHashMap<char, usize>,
    /// Last buffer pointers (for change detection)
    last_buffer_ptrs: FxHashMap<NodeIndex, u64>,
    /// Fingerprint of internal pointers used when runtime replay graph was captured.
    /// Includes all kernel internal buffers and the dyn-dims pointer (if any).
    runtime_capture_internal_ptrs: Option<Vec<u64>>,
    /// Timing events for profiling
    timing_events: Vec<cudarc::driver::sys::CUevent>,
}

impl CudaGraphOpState {
    fn new(kernels: Vec<CompiledKernel>) -> Self {
        Self {
            kernels,
            dyn_dims_buffer: None,
            cuda_graph: None,
            cuda_graph_exec: None,
            node_to_graph_node: FxHashMap::default(),
            kernel_params: Vec::new(),
            last_dyn_values: FxHashMap::default(),
            last_buffer_ptrs: FxHashMap::default(),
            runtime_capture_internal_ptrs: None,
            timing_events: Vec::new(),
        }
    }
}

/// A CUDA graph operation that implements HostOp.
///
/// This wraps a subgraph of KernelOps into a single executable CUDA graph.
/// It manages graph building, execution, and dynamic updates.
pub struct CudaGraphOp {
    /// All nodes that this graph needs buffers for (kernels + their inputs)
    buffer_nodes: Vec<NodeIndex>,
    /// Buffer size requirements for extra nodes (node -> size in bytes)
    buffer_sizes: FxHashMap<NodeIndex, Expression>,
    /// Nodes whose outputs must be zeroed before execution due to accumulation semantics.
    zero_output_nodes: Vec<NodeIndex>,
    /// Dynamic dimensions used by this graph (sorted alphabetically)
    dyn_dims_order: Vec<char>,
    /// The CUDA stream (needed for operations)
    stream: Arc<CudaStream>,
    /// Mutable state wrapped in RefCell for interior mutability
    state: RefCell<CudaGraphOpState>,
}

impl CudaGraphOp {
    fn new(
        buffer_nodes: Vec<NodeIndex>,
        buffer_sizes: FxHashMap<NodeIndex, Expression>,
        zero_output_nodes: Vec<NodeIndex>,
        dyn_dims_order: Vec<char>,
        stream: Arc<CudaStream>,
        state: CudaGraphOpState,
    ) -> Self {
        Self {
            buffer_nodes,
            buffer_sizes,
            zero_output_nodes,
            dyn_dims_order,
            stream,
            state: RefCell::new(state),
        }
    }
}

impl std::fmt::Debug for CudaGraphOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.borrow();
        f.debug_struct("CudaGraphOp")
            .field("n_kernels", &state.kernels.len())
            .field("n_buffer_nodes", &self.buffer_nodes.len())
            .finish()
    }
}

impl EgglogOp for CudaGraphOp {
    fn term(&self) -> (String, Vec<luminal::op::OpParam>) {
        ("CudaGraphOp".to_string(), vec![])
    }

    fn rewrites(&self) -> Vec<String> {
        vec![]
    }

    fn extract<'a>(
        &'a self,
        _egraph: &'a luminal::egglog_utils::SerializedEGraph,
        _children: &[&'a luminal::prelude::ENodeId],
        _list_cache: &mut FxHashMap<&'a luminal::prelude::ENodeId, Vec<Expression>>,
        _expr_cache: &mut FxHashMap<&'a luminal::prelude::ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a luminal::prelude::ENodeId>) {
        panic!("CudaGraphOp should not be extracted from egglog")
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for CudaGraphOp {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        _self_node: NodeIndex,
        _inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        self.execute_internal(stream, buffers, dyn_map, true)
    }

    fn output_size(&self) -> Expression {
        // CudaGraphOp doesn't have a single output - individual kernels have outputs
        0.into()
    }

    fn extra_buffer_nodes(&self) -> Vec<NodeIndex> {
        // Only return nodes that actually have buffers
        // Filter out nodes in buffer_sizes with size 0 (like MegakernelOps)
        // Keep nodes not in buffer_sizes (external inputs that have their own buffers)
        self.buffer_nodes
            .iter()
            .filter(|n| {
                match self.buffer_sizes.get(n) {
                    Some(size) => size.exec(&FxHashMap::default()).unwrap_or(1) != 0,
                    None => true, // Not a kernel output, might be an external input
                }
            })
            .copied()
            .collect()
    }

    fn extra_buffer_sizes(&self) -> FxHashMap<NodeIndex, Expression> {
        self.buffer_sizes.clone()
    }

    fn zero_output_nodes(&self) -> Vec<NodeIndex> {
        self.zero_output_nodes.clone()
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("CudaGraph")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayRefreshOutcome {
    Ready,
    NeedsRecapture(String),
}

impl CudaGraphOp {
    /// Ensures the child CUDA graph is built and updated for the given buffers and dynamic dims.
    /// This path intentionally does NOT launch the child graph.
    pub fn ensure_built_and_updated_for_runtime_graph(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        self.execute_internal(stream, buffers, dyn_map, false)
    }

    /// Returns the raw child CUgraph handle if this op has built its graph.
    pub fn cu_graph(&self) -> Option<CUgraph> {
        self.state
            .borrow()
            .cuda_graph
            .as_ref()
            .map(CudaGraphHandle::raw_graph)
    }

    fn collect_buffer_ptrs_from_raw(
        &self,
        raw_ptrs: &FxHashMap<NodeIndex, u64>,
    ) -> FxHashMap<NodeIndex, u64> {
        let mut ptrs = FxHashMap::default();
        for &node in &self.buffer_nodes {
            if let Some(ptr) = raw_ptrs.get(&node) {
                ptrs.insert(node, *ptr);
            }
        }
        ptrs
    }

    fn validate_required_buffer_ptrs(
        &self,
        buffer_ptrs: &FxHashMap<NodeIndex, u64>,
        context: &str,
    ) -> anyhow::Result<()> {
        let missing: Vec<_> = self
            .buffer_nodes
            .iter()
            .filter(|n| !buffer_ptrs.contains_key(n))
            .filter(|n| match self.buffer_sizes.get(n) {
                Some(size) => size.exec(&FxHashMap::default()).unwrap_or(1) != 0,
                None => true,
            })
            .copied()
            .collect();
        if !missing.is_empty() {
            anyhow::bail!(
                "{context}: {} required buffer pointers missing: {:?}",
                missing.len(),
                missing
            );
        }
        Ok(())
    }

    fn internal_pointer_fingerprint(
        state: &CudaGraphOpState,
        stream: &Arc<CudaStream>,
    ) -> Vec<u64> {
        let mut fingerprint = Vec::new();
        for kernel in &state.kernels {
            for buf in &kernel.internal_bufs {
                fingerprint.push(buf.device_ptr(stream).0);
            }
        }
        let dyn_dims_ptr = state
            .dyn_dims_buffer
            .as_ref()
            .map(|buf| buf.device_ptr(stream).0)
            .unwrap_or(0);
        fingerprint.push(dyn_dims_ptr);
        fingerprint
    }

    fn build_kernel_params_from_ptrs(
        state: &mut CudaGraphOpState,
        stream: &Arc<CudaStream>,
        dyn_map: &FxHashMap<char, usize>,
        current_buffer_ptrs: &FxHashMap<NodeIndex, u64>,
    ) -> anyhow::Result<()> {
        let dyn_dims_ptr = state
            .dyn_dims_buffer
            .as_ref()
            .map(|buf| buf.device_ptr(stream).0)
            .unwrap_or(0);

        let num_kernels = state.kernels.len();
        state.kernel_params.clear();
        state.kernel_params.reserve(num_kernels);

        for idx in 0..num_kernels {
            let kernel = &state.kernels[idx];
            let has_output = kernel
                .kernel_op
                .output_size()
                .exec(&FxHashMap::default())
                .unwrap_or(1)
                != 0;
            let output_ptr = current_buffer_ptrs.get(&kernel.node).copied().unwrap_or(0);
            if has_output && output_ptr == 0 {
                anyhow::bail!(
                    "Missing output pointer for kernel '{}' node {:?}",
                    kernel.kernel_name,
                    kernel.node
                );
            }
            let mut input_ptrs = Vec::with_capacity(kernel.inputs.len());
            for input in &kernel.inputs {
                let ptr = current_buffer_ptrs.get(input).copied().ok_or_else(|| {
                    anyhow::anyhow!(
                        "Missing input pointer for kernel '{}' input node {:?}",
                        kernel.kernel_name,
                        input
                    )
                })?;
                input_ptrs.push(ptr);
            }
            let param_values = kernel.kernel_op.build_params(
                stream,
                output_ptr,
                &input_ptrs,
                &kernel.internal_bufs,
                dyn_dims_ptr,
            );
            state
                .kernel_params
                .push(UnifiedKernelParams::new(param_values));
        }
        Ok(())
    }

    pub fn prepare_for_runtime_capture(
        &self,
        stream: &Arc<CudaStream>,
        raw_ptrs: &FxHashMap<NodeIndex, u64>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let mut state = self.state.borrow_mut();

        let dyn_map_changed = dyn_map.len() != state.last_dyn_values.len()
            || dyn_map
                .iter()
                .any(|(k, v)| state.last_dyn_values.get(k) != Some(v));

        let mut needs_internal_realloc = false;
        for kernel in state.kernels.iter() {
            let internal_dims = kernel.kernel_op.internal_buffer_dyn_dims();
            if internal_dims
                .iter()
                .any(|d| dyn_map.get(d) != state.last_dyn_values.get(d))
            {
                needs_internal_realloc = true;
                break;
            }
        }

        if needs_internal_realloc {
            for kernel in state.kernels.iter_mut() {
                kernel.internal_bufs = kernel.kernel_op.allocate_internal_buffers(stream, dyn_map);
            }
            state.cuda_graph = None;
            state.cuda_graph_exec = None;
            state.node_to_graph_node.clear();
            state.kernel_params.clear();
        }

        // Ensure internal buffers are initialized on first runtime-capture preparation.
        let mut allocated_missing_internal_bufs = false;
        for kernel in state.kernels.iter_mut() {
            if kernel.internal_bufs.is_empty() {
                kernel.internal_bufs = kernel.kernel_op.allocate_internal_buffers(stream, dyn_map);
                allocated_missing_internal_bufs = true;
            }
        }
        if allocated_missing_internal_bufs {
            state.kernel_params.clear();
        }

        if !self.dyn_dims_order.is_empty() && state.dyn_dims_buffer.is_none() {
            state.dyn_dims_buffer = Some(
                stream
                    .alloc_zeros::<i32>(self.dyn_dims_order.len())
                    .expect("Failed to allocate dyn_dims buffer"),
            );
        }

        if dyn_map_changed && !self.dyn_dims_order.is_empty() {
            let values: Vec<i32> = self
                .dyn_dims_order
                .iter()
                .map(|d| dyn_map.get(d).copied().unwrap_or(0) as i32)
                .collect();
            if let Some(buf) = state.dyn_dims_buffer.as_mut() {
                stream.memcpy_htod(&values, buf)?;
            }
        }

        let current_buffer_ptrs = self.collect_buffer_ptrs_from_raw(raw_ptrs);
        self.validate_required_buffer_ptrs(&current_buffer_ptrs, "prepare_for_runtime_capture")?;

        for idx in 0..state.kernels.len() {
            let kernel = &mut state.kernels[idx];
            kernel.kernel_op.pre_execute(
                stream,
                &mut kernel.internal_bufs,
                &mut kernel.constants,
                &current_buffer_ptrs,
                dyn_map,
            );
        }

        let buffer_ptrs_changed = current_buffer_ptrs != state.last_buffer_ptrs;
        let need_params_rebuild = needs_internal_realloc
            || allocated_missing_internal_bufs
            || dyn_map_changed
            || buffer_ptrs_changed
            || state.kernel_params.len() != state.kernels.len();
        if need_params_rebuild {
            Self::build_kernel_params_from_ptrs(&mut state, stream, dyn_map, &current_buffer_ptrs)?;
        }

        state.last_dyn_values = dyn_map.clone();
        state.last_buffer_ptrs = current_buffer_ptrs;
        state.runtime_capture_internal_ptrs =
            Some(Self::internal_pointer_fingerprint(&state, stream));
        Ok(())
    }

    pub fn refresh_for_runtime_replay(
        &self,
        stream: &Arc<CudaStream>,
        raw_ptrs: &FxHashMap<NodeIndex, u64>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<ReplayRefreshOutcome> {
        let mut state = self.state.borrow_mut();
        let Some(captured_internal_ptrs) = state.runtime_capture_internal_ptrs.clone() else {
            return Ok(ReplayRefreshOutcome::NeedsRecapture(
                "missing capture internal pointer fingerprint".to_string(),
            ));
        };

        if dyn_map.len() != state.last_dyn_values.len()
            || dyn_map
                .iter()
                .any(|(k, v)| state.last_dyn_values.get(k) != Some(v))
        {
            return Ok(ReplayRefreshOutcome::NeedsRecapture(
                "dynamic dimensions changed".to_string(),
            ));
        }

        let current_buffer_ptrs = self.collect_buffer_ptrs_from_raw(raw_ptrs);
        self.validate_required_buffer_ptrs(&current_buffer_ptrs, "refresh_for_runtime_replay")?;
        if current_buffer_ptrs != state.last_buffer_ptrs {
            return Ok(ReplayRefreshOutcome::NeedsRecapture(
                "buffer pointers changed".to_string(),
            ));
        }
        if state.kernel_params.len() != state.kernels.len() {
            return Ok(ReplayRefreshOutcome::NeedsRecapture(
                "kernel params missing or stale".to_string(),
            ));
        }
        let pre_refresh_fingerprint = Self::internal_pointer_fingerprint(&state, stream);
        if pre_refresh_fingerprint != captured_internal_ptrs {
            return Ok(ReplayRefreshOutcome::NeedsRecapture(
                "internal pointer drift detected before replay refresh".to_string(),
            ));
        }

        for idx in 0..state.kernels.len() {
            let kernel = &mut state.kernels[idx];
            kernel.kernel_op.pre_execute(
                stream,
                &mut kernel.internal_bufs,
                &mut kernel.constants,
                &current_buffer_ptrs,
                dyn_map,
            );
        }

        let post_refresh_fingerprint = Self::internal_pointer_fingerprint(&state, stream);
        if post_refresh_fingerprint != captured_internal_ptrs {
            return Ok(ReplayRefreshOutcome::NeedsRecapture(
                "internal pointer drift detected after replay refresh".to_string(),
            ));
        }

        Ok(ReplayRefreshOutcome::Ready)
    }

    pub fn replay_kernels_onto_stream(
        &self,
        stream: &Arc<CudaStream>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let mut state = self.state.borrow_mut();
        if state.kernel_params.len() != state.kernels.len() {
            anyhow::bail!(
                "replay_kernels_onto_stream: kernel params length {} does not match kernel count {}",
                state.kernel_params.len(),
                state.kernels.len()
            );
        }

        for idx in 0..state.kernels.len() {
            let (cu_func, kernel_name, kernel_node, grid_dim, block_dim, shared_mem) = {
                let kernel = &state.kernels[idx];
                (
                    unsafe { kernel.function.raw_function() },
                    kernel.kernel_name,
                    kernel.node,
                    (
                        kernel.grid.0.exec(dyn_map).unwrap() as u32,
                        kernel.grid.1.exec(dyn_map).unwrap() as u32,
                        kernel.grid.2.exec(dyn_map).unwrap() as u32,
                    ),
                    (
                        kernel.block.0.exec(dyn_map).unwrap() as u32,
                        kernel.block.1.exec(dyn_map).unwrap() as u32,
                        kernel.block.2.exec(dyn_map).unwrap() as u32,
                    ),
                    kernel.shared_mem.exec(dyn_map).unwrap() as u32,
                )
            };

            let params_ptr = state.kernel_params[idx].as_cuda_params();
            unsafe {
                use cudarc::driver::sys::{CUresult, cuLaunchKernel};
                let result = cuLaunchKernel(
                    cu_func,
                    grid_dim.0,
                    grid_dim.1,
                    grid_dim.2,
                    block_dim.0,
                    block_dim.1,
                    block_dim.2,
                    shared_mem,
                    stream.cu_stream(),
                    params_ptr,
                    std::ptr::null_mut(),
                );
                if result != CUresult::CUDA_SUCCESS {
                    anyhow::bail!(
                        "cuLaunchKernel failed for kernel '{}' node {:?} idx {}: {:?}",
                        kernel_name,
                        kernel_node,
                        idx,
                        result
                    );
                }
            }
        }

        Ok(())
    }

    /// Execute the CUDA graph with the given buffers and dynamic dimensions.
    fn execute_internal(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
        launch_graph: bool,
    ) -> anyhow::Result<()> {
        let mut state = self.state.borrow_mut();
        let _span = span!(Level::TRACE, "cuda_graph", kernels = state.kernels.len()).entered();
        let sync_debug = std::env::var("LUMINAL_SYNC_DEBUG").map_or(false, |v| v == "1");

        // Check if dyn_map changed
        let dyn_map_changed = dyn_map.len() != state.last_dyn_values.len()
            || dyn_map
                .iter()
                .any(|(k, v)| state.last_dyn_values.get(k) != Some(v));

        // Check if any kernel's internal buffer dimensions changed
        let mut needs_internal_realloc = false;
        for kernel in state.kernels.iter() {
            let internal_dims = kernel.kernel_op.internal_buffer_dyn_dims();
            if internal_dims
                .iter()
                .any(|d| dyn_map.get(d) != state.last_dyn_values.get(d))
            {
                needs_internal_realloc = true;
                break;
            }
        }

        // Reallocate internal buffers if needed
        if needs_internal_realloc {
            for kernel in state.kernels.iter_mut() {
                kernel.internal_bufs = kernel.kernel_op.allocate_internal_buffers(stream, dyn_map);
            }
            // Internal buffer pointers changed, need to rebuild CUDA graph
            state.cuda_graph = None;
            state.cuda_graph_exec = None;
            state.node_to_graph_node.clear();
            state.kernel_params.clear();
        }

        // Allocate dyn_dims_buffer if needed
        if !self.dyn_dims_order.is_empty() && state.dyn_dims_buffer.is_none() {
            state.dyn_dims_buffer = Some(
                stream
                    .alloc_zeros::<i32>(self.dyn_dims_order.len())
                    .expect("Failed to allocate dyn_dims buffer"),
            );
        }

        // Update shared dyn_dims buffer if dyn_map changed
        if dyn_map_changed && !self.dyn_dims_order.is_empty() {
            let values: Vec<i32> = self
                .dyn_dims_order
                .iter()
                .map(|d| {
                    dyn_map.get(d).copied().unwrap_or_else(|| {
                        eprintln!(
                            "WARNING: dyn dim '{}' missing from dyn_map, defaulting to 0",
                            d
                        );
                        0
                    }) as i32
                })
                .collect();
            if let Some(buf) = state.dyn_dims_buffer.as_mut() {
                stream.memcpy_htod(&values, buf)?;
            }
        }

        // Build CUDA graph if needed
        let mut graph_rebuilt = false;
        if state.cuda_graph.is_none() {
            self.build_graph(&mut state, stream, buffers, dyn_map)?;
            graph_rebuilt = true;
        }

        // Collect current buffer pointers
        let mut current_buffer_ptrs: FxHashMap<NodeIndex, u64> = FxHashMap::default();
        for &node in &self.buffer_nodes {
            if let Some(buf) = buffers.get(&node) {
                current_buffer_ptrs.insert(node, buf.device_ptr(stream).0);
            }
        }

        // Verify all buffer_nodes that should have buffers actually do.
        // Nodes with zero output size (e.g., Megakernels) won't have allocated buffers.
        let missing: Vec<_> = self
            .buffer_nodes
            .iter()
            .filter(|n| !current_buffer_ptrs.contains_key(n))
            .filter(|n| {
                // Only flag as missing if this node is expected to have a buffer
                match self.buffer_sizes.get(n) {
                    Some(size) => size.exec(&FxHashMap::default()).unwrap_or(1) != 0,
                    None => true, // External input — should always have a buffer
                }
            })
            .collect();
        if !missing.is_empty() {
            panic!(
                "CudaGraphOp: {} buffer_nodes missing from buffers map: {:?}",
                missing.len(),
                missing
            );
        }

        // Check if we need to update the graph
        let buffer_ptrs_changed = current_buffer_ptrs != state.last_buffer_ptrs;
        let needs_update = dyn_map_changed || buffer_ptrs_changed;

        // Always call pre_execute for each kernel, even when needs_update is false.
        // Megakernels modify internal state (head counter, barriers, task queue remaining)
        // during execution, and these must be reset before every re-launch.
        for idx in 0..state.kernels.len() {
            let kernel = &mut state.kernels[idx];
            kernel.kernel_op.pre_execute(
                stream,
                &mut kernel.internal_bufs,
                &mut kernel.constants,
                &current_buffer_ptrs,
                dyn_map,
            );
        }

        if needs_update {
            // Update kernel params
            let dyn_dims_ptr = state
                .dyn_dims_buffer
                .as_ref()
                .map(|buf| buf.device_ptr(stream).0)
                .unwrap_or(0);

            // Build params for each kernel first
            let num_kernels = state.kernels.len();
            for idx in 0..num_kernels {
                let kernel = &state.kernels[idx];
                let has_output = kernel
                    .kernel_op
                    .output_size()
                    .exec(&FxHashMap::default())
                    .unwrap_or(1)
                    != 0;
                let output_ptr = current_buffer_ptrs.get(&kernel.node).copied().unwrap_or_else(|| {
                    if has_output {
                        panic!(
                            "CudaGraphOp::execute_internal: missing output buffer for node {:?} (kernel '{}', idx {}). \
                             buffer_nodes: {}, provided buffers: {}",
                            kernel.node, kernel.kernel_name, idx,
                            self.buffer_nodes.len(), buffers.len()
                        )
                    }
                    0
                });
                let input_ptrs: Vec<u64> =
                    kernel
                        .inputs
                        .iter()
                        .enumerate()
                        .map(|(inp_idx, inp)| {
                            current_buffer_ptrs.get(inp).copied().unwrap_or_else(|| {
                            panic!(
                                "CudaGraphOp::execute_internal: missing input buffer for node {:?} \
                                 (kernel '{}', idx {}, input_idx {}). \
                                 buffer_nodes: {}, provided buffers: {}",
                                inp, kernel.kernel_name, idx, inp_idx,
                                self.buffer_nodes.len(), buffers.len()
                            )
                        })
                        })
                        .collect();

                let param_values = kernel.kernel_op.build_params(
                    stream,
                    output_ptr,
                    &input_ptrs,
                    &kernel.internal_bufs,
                    dyn_dims_ptr,
                );
                state.kernel_params[idx] = UnifiedKernelParams::new(param_values);
            }

            // Now update CUDA graph nodes.
            // Skip bind_to_thread during parent-level stream capture.
            if !crate::CUDA_STREAM_CAPTURING.get() {
                state
                    .cuda_graph_exec
                    .as_ref()
                    .unwrap()
                    .ctx
                    .bind_to_thread()?;
            }

            for idx in 0..num_kernels {
                let kernel = &state.kernels[idx];
                let graph_node = state.node_to_graph_node[&kernel.node];

                let grid_dim = (
                    kernel.grid.0.exec(dyn_map).unwrap() as u32,
                    kernel.grid.1.exec(dyn_map).unwrap() as u32,
                    kernel.grid.2.exec(dyn_map).unwrap() as u32,
                );
                let block_dim = (
                    kernel.block.0.exec(dyn_map).unwrap() as u32,
                    kernel.block.1.exec(dyn_map).unwrap() as u32,
                    kernel.block.2.exec(dyn_map).unwrap() as u32,
                );
                let shared_mem = kernel.shared_mem.exec(dyn_map).unwrap() as u32;
                let cu_func = unsafe { kernel.function.raw_function() };

                // Get params pointer first to avoid borrowing state twice
                let params_ptr = state.kernel_params[idx].as_cuda_params();
                let exec = state.cuda_graph_exec.as_mut().unwrap();
                unsafe {
                    exec.update_kernel_node(
                        graph_node, cu_func, grid_dim, block_dim, shared_mem, params_ptr,
                    )?;
                }
            }

            state.last_dyn_values = dyn_map.clone();
            state.last_buffer_ptrs = current_buffer_ptrs.clone();
        }

        // Keep pre-launch sync only when debugging or when graph params changed this call.
        // Skip during CUDA stream capture (synchronize is illegal on a captured stream).
        if !crate::CUDA_STREAM_CAPTURING.get() && (sync_debug || graph_rebuilt || needs_update) {
            stream.synchronize()?;
        }

        if !launch_graph {
            return Ok(());
        }

        // Check if we should bypass the CUDA graph and launch kernels individually
        // for diagnostics (set LUMINAL_NO_CUDA_GRAPH=1 to enable)
        let no_cuda_graph = std::env::var("LUMINAL_NO_CUDA_GRAPH").map_or(false, |v| v == "1");

        if no_cuda_graph {
            // Launch each kernel individually for fault isolation
            eprintln!(
                "[DIAG] Launching {} kernels individually (LUMINAL_NO_CUDA_GRAPH=1)",
                state.kernels.len()
            );
            let dyn_dims_ptr = state
                .dyn_dims_buffer
                .as_ref()
                .map(|buf| buf.device_ptr(stream).0)
                .unwrap_or(0);

            for idx in 0..state.kernels.len() {
                let kernel = &state.kernels[idx];
                let has_output = kernel
                    .kernel_op
                    .output_size()
                    .exec(&FxHashMap::default())
                    .unwrap_or(1)
                    != 0;
                let output_ptr = current_buffer_ptrs.get(&kernel.node).copied().unwrap_or(0);
                let input_ptrs: Vec<u64> = kernel
                    .inputs
                    .iter()
                    .map(|inp| current_buffer_ptrs.get(inp).copied().unwrap_or(0))
                    .collect();

                let param_values = kernel.kernel_op.build_params(
                    stream,
                    output_ptr,
                    &input_ptrs,
                    &kernel.internal_bufs,
                    dyn_dims_ptr,
                );

                let grid_dim = (
                    kernel.grid.0.exec(dyn_map).unwrap() as u32,
                    kernel.grid.1.exec(dyn_map).unwrap() as u32,
                    kernel.grid.2.exec(dyn_map).unwrap() as u32,
                );
                let block_dim = (
                    kernel.block.0.exec(dyn_map).unwrap() as u32,
                    kernel.block.1.exec(dyn_map).unwrap() as u32,
                    kernel.block.2.exec(dyn_map).unwrap() as u32,
                );
                let shared_mem = kernel.shared_mem.exec(dyn_map).unwrap() as u32;

                eprintln!(
                    "[DIAG] Kernel {}/{}: '{}' node={:?} grid=({},{},{}) block=({},{},{}) smem={} output=0x{:x} inputs={} has_output={}",
                    idx + 1,
                    state.kernels.len(),
                    kernel.kernel_name,
                    kernel.node,
                    grid_dim.0,
                    grid_dim.1,
                    grid_dim.2,
                    block_dim.0,
                    block_dim.1,
                    block_dim.2,
                    shared_mem,
                    output_ptr,
                    input_ptrs.len(),
                    has_output
                );

                // For Megakernels, print param_values (internal_buf ptrs) and buffer array contents
                if kernel.kernel_name == "Megakernel" {
                    eprintln!(
                        "[DIAG]   params ({} values): {:?}",
                        param_values.len(),
                        param_values
                            .iter()
                            .enumerate()
                            .map(|(i, v)| format!("[{}]=0x{:x}", i, v))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                    // Read buffer_array back from GPU (internal_bufs[6])
                    if kernel.internal_bufs.len() > 6 {
                        let buf_array_size =
                            kernel.internal_bufs[6].len() / std::mem::size_of::<u64>();
                        if buf_array_size > 0 {
                            let mut host_buf: Vec<u64> = vec![0u64; buf_array_size];
                            let view = kernel.internal_bufs[6].as_view();
                            let typed = unsafe { view.transmute::<u64>(buf_array_size).unwrap() };
                            stream.memcpy_dtoh(&typed, &mut host_buf).unwrap();
                            stream.synchronize().unwrap();
                            eprintln!("[DIAG]   buffer_array ({} entries):", buf_array_size);
                            for (i, ptr) in host_buf.iter().enumerate() {
                                let status = if *ptr == 0 { " *** NULL ***" } else { "" };
                                eprintln!("[DIAG]     [{}] = 0x{:x}{}", i, ptr, status);
                            }

                            // Probe each buffer pointer for accessibility
                            eprintln!("[DIAG]   Buffer pointer accessibility probe:");
                            for (i, &ptr) in host_buf.iter().enumerate() {
                                if ptr == 0 {
                                    eprintln!("[DIAG]     [{}] = NULL - skipping", i);
                                    continue;
                                }
                                unsafe {
                                    let mut probe_data = [0u8; 8];
                                    let res = cudarc::driver::sys::cuMemcpyDtoH_v2(
                                        probe_data.as_mut_ptr() as *mut std::ffi::c_void,
                                        ptr,
                                        8,
                                    );
                                    if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                                        eprintln!(
                                            "[DIAG]     [{}] = 0x{:x} *** INACCESSIBLE: {:?} ***",
                                            i, ptr, res
                                        );
                                    } else {
                                        let first_f32 =
                                            f32::from_ne_bytes(probe_data[..4].try_into().unwrap());
                                        eprintln!(
                                            "[DIAG]     [{}] = 0x{:x} OK (first f32: {})",
                                            i, ptr, first_f32
                                        );
                                    }
                                }
                            }
                        }
                    }
                    // Print work queue info
                    if !kernel.internal_bufs.is_empty() {
                        let task_buf_bytes = kernel.internal_bufs[0].len();
                        eprintln!(
                            "[DIAG]   task_buffer_size={} bytes, internal_bufs_count={}",
                            task_buf_bytes,
                            kernel.internal_bufs.len()
                        );
                    }

                    // Print buffer sizes for all external buffers used by this CudaGraphOp
                    eprintln!("[DIAG]   External buffer sizes (buffer_nodes):");
                    for &node in &self.buffer_nodes {
                        if let Some(buf) = buffers.get(&node) {
                            let ptr = buf.device_ptr(stream).0;
                            eprintln!(
                                "[DIAG]     {:?}: {} bytes ({} f32 elements) ptr=0x{:x}",
                                node,
                                buf.len(),
                                buf.len() / 4,
                                ptr
                            );
                        } else {
                            eprintln!("[DIAG]     {:?}: NOT IN BUFFERS MAP", node);
                        }
                    }
                }

                // Launch kernel directly via CUfunction
                unsafe {
                    let mut params_storage: Vec<*mut std::ffi::c_void> = param_values
                        .iter()
                        .map(|v| v as *const u64 as *mut std::ffi::c_void)
                        .collect();

                    use cudarc::driver::sys::*;
                    let result = cuLaunchKernel(
                        kernel.function.raw_function(),
                        grid_dim.0,
                        grid_dim.1,
                        grid_dim.2,
                        block_dim.0,
                        block_dim.1,
                        block_dim.2,
                        shared_mem,
                        stream.cu_stream(),
                        params_storage.as_mut_ptr(),
                        std::ptr::null_mut(),
                    );
                    if result != CUresult::CUDA_SUCCESS {
                        panic!(
                            "[DIAG] cuLaunchKernel FAILED for kernel {}/{} '{}' node={:?}: {:?}",
                            idx + 1,
                            state.kernels.len(),
                            kernel.kernel_name,
                            kernel.node,
                            result
                        );
                    }
                }

                // Sync after each kernel to catch errors immediately
                if let Err(e) = stream.synchronize() {
                    panic!(
                        "[DIAG] CUDA ERROR after kernel {}/{} '{}' node={:?}: {:#}\n  \
                         grid=({},{},{}) block=({},{},{}) smem={}\n  \
                         output_ptr=0x{:x} input_ptrs={:?} has_output={}",
                        idx + 1,
                        state.kernels.len(),
                        kernel.kernel_name,
                        kernel.node,
                        e,
                        grid_dim.0,
                        grid_dim.1,
                        grid_dim.2,
                        block_dim.0,
                        block_dim.1,
                        block_dim.2,
                        shared_mem,
                        output_ptr,
                        input_ptrs,
                        has_output
                    );
                }
            }
            eprintln!(
                "[DIAG] All {} kernels completed successfully!",
                state.kernels.len()
            );
        } else {
            // Launch the CUDA graph. During parent-level stream capture,
            // bind_to_thread() (cuCtxSetCurrent) is illegal, so use raw cuGraphLaunch.
            let exec = state.cuda_graph_exec.as_ref().unwrap();
            if crate::CUDA_STREAM_CAPTURING.get() {
                unsafe {
                    cudarc::driver::sys::cuGraphLaunch(exec.cu_graph_exec, stream.cu_stream())
                        .result()?;
                }
            } else {
                exec.launch(stream)?;
            }
        }

        // Keep post-launch sync only in explicit debug mode.
        if sync_debug {
            stream.synchronize()?;
        }

        Ok(())
    }

    /// Build the CUDA graph from compiled kernels.
    fn build_graph(
        &self,
        state: &mut std::cell::RefMut<'_, CudaGraphOpState>,
        stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let ctx = stream.context().clone();
        let mut graph = CudaGraphHandle::new(ctx.clone())?;

        let num_kernels = state.kernels.len();
        state.kernel_params.clear();
        state.kernel_params.reserve(num_kernels);

        let tracing_enabled = enabled!(Level::TRACE);
        if tracing_enabled {
            let needed_events = num_kernels + 1;
            while state.timing_events.len() < needed_events {
                state.timing_events.push(create_cuda_event(&ctx)?);
            }
        }

        // Collect buffer pointers
        let mut buffer_ptrs: FxHashMap<NodeIndex, u64> = FxHashMap::default();
        for &node in &self.buffer_nodes {
            if let Some(buf) = buffers.get(&node) {
                buffer_ptrs.insert(node, buf.device_ptr(stream).0);
            }
        }

        let dyn_dims_ptr = state
            .dyn_dims_buffer
            .as_ref()
            .map(|buf| buf.device_ptr(stream).0)
            .unwrap_or(0);

        graph.ctx.bind_to_thread()?;

        let mut prev_graph_node: Option<CUgraphNode> = None;

        for idx in 0..num_kernels {
            // Allocate internal buffers if not already done
            {
                let kernel = &mut state.kernels[idx];
                if kernel.internal_bufs.is_empty() {
                    kernel.internal_bufs =
                        kernel.kernel_op.allocate_internal_buffers(stream, dyn_map);
                }
            }

            // Call pre_execute to initialize internal state (e.g., populate buffer array for MegakernelOps)
            {
                let kernel = &mut state.kernels[idx];
                kernel.kernel_op.pre_execute(
                    stream,
                    &mut kernel.internal_bufs,
                    &mut kernel.constants,
                    &buffer_ptrs,
                    dyn_map,
                );
            }

            let kernel = &state.kernels[idx];
            let grid_dim = (
                kernel.grid.0.exec(dyn_map).unwrap() as u32,
                kernel.grid.1.exec(dyn_map).unwrap() as u32,
                kernel.grid.2.exec(dyn_map).unwrap() as u32,
            );
            let block_dim = (
                kernel.block.0.exec(dyn_map).unwrap() as u32,
                kernel.block.1.exec(dyn_map).unwrap() as u32,
                kernel.block.2.exec(dyn_map).unwrap() as u32,
            );
            let shared_mem = kernel.shared_mem.exec(dyn_map).unwrap() as u32;

            // Kernels with zero output size (e.g., Megakernels) don't have output buffers —
            // they write to internal buffer arrays instead. Allow null (0) for those.
            let has_output = kernel
                .kernel_op
                .output_size()
                .exec(&FxHashMap::default())
                .unwrap_or(1)
                != 0;
            let output_ptr = buffer_ptrs.get(&kernel.node).copied().unwrap_or_else(|| {
                if has_output {
                    panic!(
                        "CudaGraphOp::build_graph: missing output buffer for node {:?} (kernel '{}', idx {}). \
                         buffer_nodes: {}, provided buffers: {}",
                        kernel.node, kernel.kernel_name, idx,
                        self.buffer_nodes.len(), buffers.len()
                    )
                }
                0
            });
            let input_ptrs: Vec<u64> = kernel
                .inputs
                .iter()
                .enumerate()
                .map(|(inp_idx, inp)| {
                    buffer_ptrs.get(inp).copied().unwrap_or_else(|| {
                        panic!(
                            "CudaGraphOp::build_graph: missing input buffer for node {:?} \
                             (kernel '{}', idx {}, input_idx {}). \
                             buffer_nodes: {}, provided buffers: {}",
                            inp,
                            kernel.kernel_name,
                            idx,
                            inp_idx,
                            self.buffer_nodes.len(),
                            buffers.len()
                        )
                    })
                })
                .collect();

            let param_values = kernel.kernel_op.build_params(
                stream,
                output_ptr,
                &input_ptrs,
                &kernel.internal_bufs,
                dyn_dims_ptr,
            );
            let mut params = UnifiedKernelParams::new(param_values);

            let cu_func = unsafe { kernel.function.raw_function() };
            let kernel_node = kernel.node;

            // Get timing event for this index (separate access from kernels)
            let timing_event = if tracing_enabled {
                Some(state.timing_events[idx])
            } else {
                None
            };

            let deps: &[CUgraphNode] = match (&prev_graph_node, timing_event) {
                (Some(prev), Some(event)) => {
                    let event_node = graph.add_event_record_node(&[*prev], event)?;
                    prev_graph_node = Some(event_node);
                    std::slice::from_ref(prev_graph_node.as_ref().unwrap())
                }
                (None, Some(event)) => {
                    let event_node = graph.add_event_record_node(&[], event)?;
                    prev_graph_node = Some(event_node);
                    std::slice::from_ref(prev_graph_node.as_ref().unwrap())
                }
                (Some(prev), None) => std::slice::from_ref(prev),
                (None, None) => &[],
            };

            let graph_node = unsafe {
                graph.add_kernel_node(
                    deps,
                    cu_func,
                    grid_dim,
                    block_dim,
                    shared_mem,
                    params.as_cuda_params(),
                )?
            };

            state.node_to_graph_node.insert(kernel_node, graph_node);
            state.kernels[idx].graph_node = Some(graph_node);
            state.kernel_params.push(params);
            prev_graph_node = Some(graph_node);
        }

        if tracing_enabled && let Some(prev) = prev_graph_node {
            graph.add_event_record_node(&[prev], state.timing_events[num_kernels])?;
        }

        let exec = graph.instantiate()?;

        state.cuda_graph = Some(graph);
        state.cuda_graph_exec = Some(exec);
        state.last_dyn_values = dyn_map.clone();
        state.last_buffer_ptrs = buffer_ptrs;

        Ok(())
    }
}

impl Drop for CudaGraphOp {
    fn drop(&mut self) {
        let mut state = self.state.borrow_mut();

        // Destroy timing events - extract ctx first to avoid borrow issues
        let ctx = state.cuda_graph_exec.as_ref().map(|exec| exec.ctx.clone());
        if let Some(ctx) = ctx {
            for event in state.timing_events.drain(..) {
                destroy_cuda_event(&ctx, event);
            }
        }

        // Forget dyn_dims buffer (managed by runtime)
        if let Some(buf) = state.dyn_dims_buffer.take() {
            std::mem::forget(buf);
        }

        // Handle kernel resources
        for kernel in state.kernels.iter_mut() {
            // Forget constants (they point to __constant__ memory)
            let constants = std::mem::take(&mut kernel.constants);
            for (_k, v) in constants {
                std::mem::forget(v);
            }
            // Forget internal buffers (managed by runtime)
            for buf in kernel.internal_bufs.drain(..) {
                std::mem::forget(buf);
            }
        }
    }
}

/// Compile KernelOp subgraphs in the LLIR graph into CudaGraphOps.
///
/// This function:
/// 1. Finds all KernelOp nodes in the graph
/// 2. Partitions them into convex subgraphs
/// 3. For each subgraph, creates a CudaGraphOp (which implements HostOp)
/// 4. Adds the CudaGraphOp node to the llir_graph with appropriate edges
///
/// Note: KernelOp nodes remain in the graph for buffer allocation and edge tracking.
/// Their execution is handled by the CudaGraphOp via the CUDA graph API.
#[allow(clippy::type_complexity)]
pub fn kernel_to_host(
    llir_graph: &mut LLIRGraph,
    cuda_stream: &Arc<CudaStream>,
    kernel_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    megakernel_to_blocks: &FxHashMap<NodeIndex, Vec<NodeIndex>>,
) {
    let _span = span!(Level::TRACE, "kernel_to_host").entered();

    let kernel_ops_in_graph = llir_graph
        .node_indices()
        .filter(|n| llir_graph[*n].to_dialect::<dyn KernelOp>().is_some())
        .collect::<FxHashSet<_>>();

    if kernel_ops_in_graph.is_empty() {
        return;
    }

    let kernel_subgraphs = partition_marked_convex(llir_graph, &kernel_ops_in_graph).unwrap();

    // Track which kernel node belongs to which CudaGraphOp (for later edge creation)
    let mut kernel_to_cuda_graph: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    // Track all CudaGraphOp nodes and their subgraphs for edge creation
    let mut cuda_graph_subgraphs: Vec<(NodeIndex, FxHashSet<NodeIndex>)> = Vec::new();

    for subgraph in kernel_subgraphs {
        // Compile kernels in topological order
        let topo_order: Vec<_> = toposort(&*llir_graph, None)
            .unwrap()
            .into_iter()
            .filter(|n| subgraph.contains(n))
            .collect();

        let mut kernels = Vec::with_capacity(topo_order.len());
        let mut all_dyn_dims = FxHashSet::default();
        let mut all_buffer_nodes = FxHashSet::default();
        let mut all_buffer_sizes: FxHashMap<NodeIndex, Expression> = FxHashMap::default();
        let mut all_zero_output_nodes: FxHashSet<NodeIndex> = FxHashSet::default();

        for kernel_node_idx in &topo_order {
            let kernel_op_ref = llir_graph[*kernel_node_idx]
                .to_dialect::<dyn KernelOp>()
                .unwrap();

            let (kernel_function, _, kernel_str, grid, block, shared_mem, constants) =
                kernel_op_ref.compile(cuda_stream, kernel_cache);

            // Collect inputs from graph edges
            let mut inputs: Vec<NodeIndex> = llir_graph
                .edges_directed(*kernel_node_idx, Direction::Incoming)
                .sorted_by_key(|e| e.id())
                .map(|e| e.source())
                .collect_vec();

            // If this is a megakernel, include all its block op nodes for buffer access
            if let Some(block_nodes) = megakernel_to_blocks.get(kernel_node_idx) {
                inputs.extend(block_nodes.iter().copied());
            }

            // Collect dyn dims used by this kernel
            all_dyn_dims.extend(grid.0.dyn_vars());
            all_dyn_dims.extend(grid.1.dyn_vars());
            all_dyn_dims.extend(grid.2.dyn_vars());
            all_dyn_dims.extend(block.0.dyn_vars());
            all_dyn_dims.extend(block.1.dyn_vars());
            all_dyn_dims.extend(block.2.dyn_vars());
            all_dyn_dims.extend(shared_mem.dyn_vars());
            all_dyn_dims.extend(kernel_op_ref.output_bytes().dyn_vars());

            // Collect buffer nodes and sizes
            // Only add kernel nodes with non-zero output size (MegakernelOps have size 0)
            let output_bytes = kernel_op_ref.output_bytes();
            if output_bytes.exec(&FxHashMap::default()).unwrap_or(1) != 0 {
                all_buffer_nodes.insert(*kernel_node_idx);
                all_buffer_sizes.insert(*kernel_node_idx, output_bytes);

                // Kernels that accumulate into outputs need zeroed output buffers.
                // Megakernels are treated conservatively because they execute mixed task types.
                let requires_zero = if kernel_str.trim().is_empty() {
                    // Conservative fallback when we cannot inspect generated source.
                    true
                } else {
                    kernel_str.contains("atomicAdd") || kernel_op_ref.kernel_name() == "Megakernel"
                };
                if requires_zero {
                    all_zero_output_nodes.insert(*kernel_node_idx);
                }
            }
            all_buffer_nodes.extend(inputs.iter().copied());

            let kernel_op: Arc<Box<dyn KernelOp>> = Arc::clone(kernel_op_ref);

            kernels.push(CompiledKernel::new(
                *kernel_node_idx,
                kernel_function,
                grid,
                block,
                shared_mem,
                inputs,
                kernel_op.clone(),
                constants,
                kernel_op.kernel_name(),
            ));
        }

        // Sort dyn dims alphabetically for consistent buffer layout
        let mut dyn_dims_order: Vec<char> = all_dyn_dims.into_iter().collect();
        dyn_dims_order.sort();

        let buffer_nodes: Vec<NodeIndex> = all_buffer_nodes.into_iter().collect();
        let zero_output_nodes: Vec<NodeIndex> = all_zero_output_nodes.into_iter().collect();

        // Create CudaGraphOp with RefCell for interior mutability
        let state = CudaGraphOpState::new(kernels);

        let cuda_graph_op = CudaGraphOp::new(
            buffer_nodes,
            all_buffer_sizes,
            zero_output_nodes,
            dyn_dims_order,
            cuda_stream.clone(),
            state,
        );

        // Add CudaGraphOp to llir_graph as a HostOp
        let cuda_graph_node =
            llir_graph.add_node(LLIROp::new(Box::new(cuda_graph_op) as Box<dyn HostOp>));

        // Track which kernel nodes belong to this CudaGraphOp
        for kernel_node in &subgraph {
            kernel_to_cuda_graph.insert(*kernel_node, cuda_graph_node);
        }
        // Also track block op nodes inside megakernels
        for kernel_node in &subgraph {
            if let Some(block_nodes) = megakernel_to_blocks.get(kernel_node) {
                for block_node in block_nodes {
                    kernel_to_cuda_graph.insert(*block_node, cuda_graph_node);
                }
            }
        }
        cuda_graph_subgraphs.push((cuda_graph_node, subgraph.clone()));

        // Find external inputs: nodes outside subgraph that have edges into subgraph
        let external_inputs: FxHashSet<NodeIndex> = subgraph
            .iter()
            .flat_map(|&node| {
                llir_graph
                    .edges_directed(node, Direction::Incoming)
                    .map(|e| e.source())
                    .filter(|src| !subgraph.contains(src))
            })
            .collect();

        // Add edges from external inputs to CudaGraphOp
        for input in &external_inputs {
            llir_graph.add_edge(*input, cuda_graph_node, ());
        }

        // Note: We intentionally keep the kernel nodes in the graph.
        // They are needed for:
        // 1. Buffer allocation (their output_size determines buffer sizes)
        // 2. Edge tracking (other ops like cuBLAS reference specific kernel outputs)
        // The CudaGraphOp handles their execution via the CUDA graph API.
    }

    // Second pass: Add edges between CudaGraphOps based on kernel dependencies.
    // This ensures proper execution ordering when a kernel in one CudaGraphOp
    // produces output consumed by a kernel (or BlockOp inside a megakernel) in another CudaGraphOp.
    let mut edges_to_add: Vec<(NodeIndex, NodeIndex)> = Vec::new();

    for (cuda_graph_node, subgraph) in &cuda_graph_subgraphs {
        // Find all nodes that this subgraph produces output for (including BlockOp nodes in megakernels)
        let mut all_producer_nodes: FxHashSet<NodeIndex> = subgraph.clone();
        for kernel_node in subgraph {
            if let Some(block_nodes) = megakernel_to_blocks.get(kernel_node) {
                all_producer_nodes.extend(block_nodes.iter().copied());
            }
        }

        // Find external consumers that are kernels belonging to other CudaGraphOps
        for producer_node in &all_producer_nodes {
            for edge in llir_graph.edges_directed(*producer_node, Direction::Outgoing) {
                let consumer = edge.target();
                if all_producer_nodes.contains(&consumer) {
                    continue; // Same subgraph
                }
                // Check if consumer is a kernel in another CudaGraphOp
                if let Some(&consumer_cuda_graph) = kernel_to_cuda_graph.get(&consumer)
                    && consumer_cuda_graph != *cuda_graph_node
                {
                    edges_to_add.push((*cuda_graph_node, consumer_cuda_graph));
                }
                // Also add edges to HostOps (like cuBLAS ops) that consume our outputs
                if llir_graph[consumer]
                    .to_dialect::<dyn super::super::host::HostOp>()
                    .is_some()
                {
                    edges_to_add.push((*cuda_graph_node, consumer));
                }
            }
        }
    }

    // Add collected edges (deduplicate), skipping back-edges to preserve DAG property
    let edges_to_add: FxHashSet<(NodeIndex, NodeIndex)> = edges_to_add.into_iter().collect();
    let topo = toposort(&*llir_graph, None).unwrap();
    let mut topo_pos: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    for (i, n) in topo.iter().enumerate() {
        topo_pos.insert(*n, i);
    }
    for (src, dst) in edges_to_add {
        // Only add forward edges (src before dst in topo order) to avoid creating cycles
        let src_pos = topo_pos.get(&src).copied().unwrap_or(usize::MAX);
        let dst_pos = topo_pos.get(&dst).copied().unwrap_or(usize::MAX);
        if src_pos >= dst_pos {
            continue; // Skip back-edges
        }
        if !llir_graph.edges_connecting(src, dst).any(|_| true) {
            llir_graph.add_edge(src, dst, ());
        }
    }
}
