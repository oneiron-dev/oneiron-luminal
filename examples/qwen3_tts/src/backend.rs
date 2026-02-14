use crate::weight_loader::load_weights_from_map;
use luminal::{graph::Graph, prelude::ToId};
use std::collections::HashMap;

#[cfg(any(feature = "metal", feature = "cuda"))]
use crate::weight_loader::expected_element_count;

#[cfg(any(feature = "metal", feature = "cuda"))]
use luminal::{hlir::Input, op::Runtime};

#[cfg(feature = "metal")]
use luminal_metal::MetalRuntime;

#[cfg(feature = "cuda")]
use luminal_cuda::runtime::CudaRuntime;

// Type alias: cuda > metal > native
#[cfg(feature = "cuda")]
pub type Rt = CudaRuntime;

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub type Rt = MetalRuntime;

#[cfg(not(any(feature = "metal", feature = "cuda")))]
pub type Rt = luminal::prelude::NativeRuntime;

pub fn build_search_space(cx: &mut Graph) {
    #[cfg(feature = "cuda")]
    cx.build_search_space::<CudaRuntime>();

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    cx.build_search_space::<MetalRuntime>();

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    cx.build_search_space::<luminal::prelude::NativeRuntime>();
}

pub fn compile(cx: &mut Graph, weights: &HashMap<String, Vec<f32>>) -> Rt {
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
    rt.set_data(id, data);

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    rt.set_data(id, &data);

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    rt.set_data(id, data);
}

pub fn set_data_i32(rt: &mut Rt, id: impl ToId, data: Vec<i32>) {
    #[cfg(feature = "cuda")]
    rt.set_data(id, data);

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let data: Vec<f32> = data.into_iter().map(|v| v as f32).collect();
        rt.set_data(id, &data);
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    rt.set_data(id, data);
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
