use std::{fmt::Debug, sync::Arc};

use crate::cudarc::driver::{CudaSlice, CudaStream};
use luminal::{op::EgglogOp, prelude::*};
mod cublas;
mod cublaslt;

pub type Ops = (cublaslt::CuBlasLt, cublas::CuBlasSgemmV2);

/// Host operations that execute on the CPU but orchestrate GPU work.
///
/// This includes operations like cuBLAS calls and CUDA graph executions.
pub trait HostOp: Debug + as_any::AsAny + EgglogOp {
    /// Execute the operation with access to buffers via a map.
    ///
    /// # Arguments
    /// * `stream` - The CUDA stream to execute on
    /// * `self_node` - The NodeIndex of this op in the llir_graph (used as output buffer)
    /// * `inputs` - NodeIndices of input nodes (in edge order from the graph)
    /// * `buffers` - Map from NodeIndex to device buffer for all allocated nodes
    /// * `dyn_map` - Dynamic dimension values
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()>;

    /// Initialize resources that should not be created inside CUDA stream capture.
    ///
    /// Default: no-op.
    fn prepare_for_capture(&self, _stream: &Arc<CudaStream>) -> anyhow::Result<()> {
        Ok(())
    }

    /// Warm up capture-specific state before stream capture begins.
    ///
    /// Default: use normal execute path.
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

    /// Execute using capture-safe operations inside a stream capture section.
    ///
    /// Default: use normal execute path.
    fn execute_for_capture(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        self.execute(stream, self_node, inputs, buffers, dyn_map)
    }

    /// Execute inside capture using pre-resolved raw device pointers.
    ///
    /// This avoids `CudaSlice::device_ptr()` during active capture, which injects
    /// stream event management not suitable for mini-capture.
    fn execute_for_capture_raw(
        &self,
        _stream: &Arc<CudaStream>,
        _self_node: NodeIndex,
        _inputs: &[NodeIndex],
        _raw_buffers: &FxHashMap<NodeIndex, u64>,
        _dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("execute_for_capture_raw not implemented for this HostOp");
    }

    /// Returns the output buffer size in elements.
    /// Return 0 if this op doesn't have a single output buffer (e.g., CudaGraphOp).
    fn output_size(&self) -> Expression;

    /// Returns the output buffer size in bytes (accounts for dtype when relevant).
    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    /// Returns output buffer nodes that must be zeroed before execution.
    /// Used for kernels with accumulation semantics (e.g., atomicAdd).
    fn zero_output_nodes(&self) -> Vec<NodeIndex> {
        vec![]
    }

    /// Returns additional nodes (beyond graph edges) that this op needs buffers for.
    ///
    /// For most ops, this returns empty (buffers determined by graph edges).
    /// For CudaGraphOp, this returns all internal kernel nodes.
    fn extra_buffer_nodes(&self) -> Vec<NodeIndex> {
        vec![]
    }

    /// Returns buffer size requirements for extra nodes (node -> size in bytes).
    ///
    /// Called during buffer allocation to ensure all required buffers exist.
    /// For CudaGraphOp, this returns sizes for all internal kernel output buffers.
    fn extra_buffer_sizes(&self) -> FxHashMap<NodeIndex, Expression> {
        FxHashMap::default()
    }

    /// Returns the name of this host op for stats reporting, or None if not reportable.
    fn stats_name(&self) -> Option<&'static str> {
        None
    }
}
