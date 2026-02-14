use crate::weight_loader::load_weights_from_map;
use luminal::{graph::Graph, prelude::ToId};
use std::collections::HashMap;

#[cfg(feature = "metal")]
use crate::weight_loader::expected_element_count;

#[cfg(feature = "metal")]
use luminal::{hlir::Input, op::Runtime};

#[cfg(feature = "metal")]
use luminal_metal::MetalRuntime;

#[cfg(feature = "metal")]
pub type Rt = MetalRuntime;

#[cfg(not(feature = "metal"))]
pub type Rt = luminal::prelude::NativeRuntime;

pub fn build_search_space(cx: &mut Graph) {
    #[cfg(feature = "metal")]
    cx.build_search_space::<MetalRuntime>();

    #[cfg(not(feature = "metal"))]
    cx.build_search_space::<luminal::prelude::NativeRuntime>();
}

pub fn compile(cx: &mut Graph, weights: &HashMap<String, Vec<f32>>) -> Rt {
    build_search_space(cx);

    #[cfg(feature = "metal")]
    {
        let mut rt = MetalRuntime::initialize(());
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

        let mut rt = cx.search(rt, 1);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt
    }

    #[cfg(not(feature = "metal"))]
    {
        let mut rt = cx.search(luminal::prelude::NativeRuntime::default(), 1);
        load_weights_from_map(&mut rt, cx, weights);
        rt
    }
}

pub fn set_data(rt: &mut Rt, id: impl ToId, data: Vec<f32>) {
    #[cfg(feature = "metal")]
    rt.set_data(id, &data);

    #[cfg(not(feature = "metal"))]
    rt.set_data(id, data);
}

pub fn set_data_i32(rt: &mut Rt, id: impl ToId, data: Vec<i32>) {
    #[cfg(feature = "metal")]
    {
        let data: Vec<f32> = data.into_iter().map(|v| v as f32).collect();
        rt.set_data(id, &data);
    }

    #[cfg(not(feature = "metal"))]
    rt.set_data(id, data);
}

pub fn get_f32(rt: &Rt, id: impl ToId) -> Vec<f32> {
    #[cfg(feature = "metal")]
    {
        rt.get_f32(id)
    }

    #[cfg(not(feature = "metal"))]
    {
        rt.get_f32(id).clone()
    }
}
