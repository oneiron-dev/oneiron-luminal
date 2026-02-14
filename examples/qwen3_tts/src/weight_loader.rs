use crate::backend;
use luminal::hlir::Input;
use luminal::prelude::petgraph::Direction;
use luminal::prelude::{Graph, NativeRuntime};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;

fn to_f32_vec(dtype: Dtype, data: &[u8], name: &str) -> Result<Vec<f32>, String> {
    match dtype {
        Dtype::F32 => {
            if data.len() % 4 != 0 {
                return Err(format!(
                    "Tensor {name} has invalid F32 byte length {}",
                    data.len()
                ));
            }
            Ok(data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        Dtype::F16 => {
            if data.len() % 2 != 0 {
                return Err(format!(
                    "Tensor {name} has invalid F16 byte length {}",
                    data.len()
                ));
            }
            Ok(data
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect())
        }
        Dtype::BF16 => {
            if data.len() % 2 != 0 {
                return Err(format!(
                    "Tensor {name} has invalid BF16 byte length {}",
                    data.len()
                ));
            }
            Ok(data
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect())
        }
        other => Err(format!(
            "Unsupported dtype {other:?} for tensor {name}. Only F32/F16/BF16 are supported."
        )),
    }
}

pub fn expected_element_count(cx: &Graph, node: luminal::prelude::NodeIndex) -> Option<usize> {
    let edge = cx.graph.edges_directed(node, Direction::Outgoing).next()?;
    edge.weight().n_elements().exec(&cx.dyn_map)
}

pub fn load_weights_from_map(
    rt: &mut backend::Rt,
    cx: &Graph,
    weights: &HashMap<String, Vec<f32>>,
) -> usize {
    let mut loaded = 0usize;
    for node in cx.graph.node_indices() {
        let Some(input) = cx.graph[node].as_any().downcast_ref::<Input>() else {
            continue;
        };
        let Some(data) = weights.get(&input.label) else {
            continue;
        };
        backend::set_data(rt, node, data.clone());
        loaded += 1;
    }
    loaded
}

pub fn load_safetensors_to_native(
    rt: &mut NativeRuntime,
    cx: &Graph,
    path: &Path,
) -> Result<usize, String> {
    let data =
        std::fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    let tensors =
        SafeTensors::deserialize(&data).map_err(|e| format!("Failed to parse safetensors: {e}"))?;

    let mut loaded = 0usize;
    for node in cx.graph.node_indices() {
        let Some(input) = cx.graph[node].as_any().downcast_ref::<Input>() else {
            continue;
        };

        let Ok(view) = tensors.tensor(&input.label) else {
            continue;
        };

        let values = to_f32_vec(view.dtype(), view.data(), &input.label)?;
        if let Some(expected) = expected_element_count(cx, node) {
            if expected != values.len() {
                return Err(format!(
                    "Tensor {} has {} elements in safetensors, expected {} from graph",
                    input.label,
                    values.len(),
                    expected
                ));
            }
        }

        rt.set_data(node, values);
        loaded += 1;
    }

    Ok(loaded)
}

pub fn load_safetensors_to_map(path: &Path) -> Result<HashMap<String, Vec<f32>>, String> {
    let data =
        std::fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    let tensors =
        SafeTensors::deserialize(&data).map_err(|e| format!("Failed to parse safetensors: {e}"))?;

    let mut map = HashMap::new();
    for (name, _) in tensors.iter() {
        let view = tensors
            .tensor(name)
            .map_err(|e| format!("Failed to access tensor {name}: {e}"))?;
        let values = to_f32_vec(view.dtype(), view.data(), name)?;
        map.insert(name.to_string(), values);
    }
    Ok(map)
}
