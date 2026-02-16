pub mod backend;
pub mod code_predictor;
#[cfg(feature = "cuda")]
pub mod flash_attn;
pub mod model;
pub mod pipeline;
pub mod speech_decoder;
pub mod weight_loader;

use luminal::prelude::GraphTensor;

/// Graph break that only activates on CUDA builds. NativeRuntime does not
/// support multi-chunk stitching in `set_data`, so we skip breaks when
/// testing without the `cuda` feature.
#[inline(always)]
pub fn maybe_graph_break(t: GraphTensor) -> GraphTensor {
    #[cfg(feature = "cuda")]
    {
        t.graph_break()
    }
    #[cfg(not(feature = "cuda"))]
    {
        t
    }
}

#[cfg(test)]
mod tests;
