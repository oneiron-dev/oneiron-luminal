use crate::weight_loader::load_weights_from_map;
use luminal::{graph::Graph, prelude::ToId};
use std::collections::HashMap;

#[cfg(any(feature = "metal", feature = "cuda"))]
use crate::weight_loader::expected_element_count;

#[cfg(any(feature = "metal", feature = "cuda"))]
use luminal::hlir::Input;

#[cfg(feature = "metal")]
use luminal_metal::MetalRuntime;

#[cfg(feature = "cuda")]
use luminal_cuda::runtime::CudaRuntime;

/// TileMatmul BlockOps to exclude when compiled megakernel is active.
/// Excluding these forces cuBLASLt HostOp for all matmuls, which:
/// 1. Removes matmuls from BlockOp convex subgraphs
/// 2. Lets remaining element-wise ops form compiled-eligible subgraphs
/// 3. Uses cuBLASLt's optimized GEMV for decode (faster than TileMatmul)
#[cfg(feature = "cuda")]
type TileMatmulOps = (
    luminal_cuda::block::TileMatmulFullSplit,
    luminal_cuda::block::TileMatmulNvFp4,
    luminal_cuda::block::TileMatmulMxfp4,
);

// Type alias: cuda > metal > native
#[cfg(feature = "cuda")]
pub type Rt = CudaRuntime;

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub type Rt = MetalRuntime;

#[cfg(not(any(feature = "metal", feature = "cuda")))]
pub type Rt = luminal::prelude::NativeRuntime;

pub fn build_search_space(cx: &mut Graph) {
    #[cfg(feature = "cuda")]
    {
        let use_compiled =
            std::env::var("LUMINAL_COMPILED_MEGAKERNEL").map_or(false, |v| v == "1");
        if use_compiled {
            cx.build_search_space_exclude_ops::<CudaRuntime, TileMatmulOps>();
        } else {
            cx.build_search_space::<CudaRuntime>();
        }
    }

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    cx.build_search_space::<MetalRuntime>();

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    cx.build_search_space::<luminal::prelude::NativeRuntime>();
}

pub fn compile(cx: &mut Graph, weights: &HashMap<String, Vec<f32>>) -> Rt {
    // Enable e-graph caching to skip egglog on subsequent runs.
    // Use a separate cache dir when compiled megakernel is active because the
    // e-graph is different (no TileMatmul ops) and the cache key doesn't include
    // which ops were excluded.
    let mut cache_dir =
        std::env::var("LUMINAL_CACHE_DIR").unwrap_or_else(|_| ".luminal_cache".to_string());
    if std::env::var("LUMINAL_COMPILED_MEGAKERNEL").map_or(false, |v| v == "1") {
        cache_dir.push_str("_compiled");
    }
    cx.cache_dir = Some(std::path::PathBuf::from(cache_dir));

    build_search_space(cx);

    #[cfg(feature = "cuda")]
    {
        let rt = CudaRuntime::new().expect("Failed to initialize CUDA runtime");
        compile_accelerated(cx, weights, rt)
    }

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let rt = MetalRuntime::initialize(());
        let mut rt = compile_accelerated(cx, weights, rt);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        let mut rt = cx.search(luminal::prelude::NativeRuntime::default(), 1);
        load_weights_from_map(&mut rt, cx, weights);
        rt
    }
}

/// Shared compile flow for accelerated backends (Metal, CUDA).
/// Loads weights, zero-fills unresolved inputs, then runs search.
#[cfg(any(feature = "metal", feature = "cuda"))]
fn compile_accelerated(cx: &mut Graph, weights: &HashMap<String, Vec<f32>>, mut rt: Rt) -> Rt {
    load_weights_from_map(&mut rt, cx, weights);

    for node in cx.graph.node_indices() {
        let Some(input) = cx.graph[node].as_any().downcast_ref::<Input>() else {
            continue;
        };
        if rt.hlir_buffers.contains_key(&node) {
            continue;
        }

        let n_elements = expected_element_count(cx, node).unwrap_or_else(|| {
            panic!(
                "Could not resolve element count for unresolved input '{}' ({node:?})",
                input.label
            )
        });
        set_data(&mut rt, node, vec![0.0; n_elements]);
    }

    cx.search(rt, 1)
}

pub fn set_data(rt: &mut Rt, id: impl ToId, data: Vec<f32>) {
    #[cfg(feature = "cuda")]
    rt.set_data_f32(id, &data);

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    rt.set_data(id, &data);

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    rt.set_data(id, data);
}

pub fn set_data_i32(rt: &mut Rt, id: impl ToId, data: Vec<i32>) {
    #[cfg(feature = "cuda")]
    rt.set_data_i32(id, &data);

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let data: Vec<f32> = data.into_iter().map(|v| v as f32).collect();
        rt.set_data(id, &data);
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    rt.set_data(id, data);
}

pub fn update_data_slice(rt: &mut Rt, id: impl ToId, byte_offset: usize, data: &[f32]) {
    #[cfg(feature = "cuda")]
    rt.update_buffer_slice(id, byte_offset, data);

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let _ = (rt, id, byte_offset, data);
        unimplemented!("Metal partial update not yet implemented");
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        let _ = (rt, id, byte_offset, data);
        unimplemented!("Native partial update not yet implemented");
    }
}

pub fn get_f32(rt: &Rt, id: impl ToId) -> Vec<f32> {
    #[cfg(any(feature = "metal", feature = "cuda"))]
    {
        rt.get_f32(id)
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        rt.get_f32(id).clone()
    }
}
