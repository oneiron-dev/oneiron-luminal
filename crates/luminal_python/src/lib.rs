use luminal::prelude::*;
use luminal::visualization::ToDot;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

#[derive(Debug, Clone)]
struct PyTorchGraphNode {
    name: String,
    op: String, // "placeholder", "call_function", "output", "get_attr"
    target: String,
    args: Vec<String>,
    kwargs: HashMap<String, String>,
    shape: Vec<usize>,
    dtype: String,
}

#[pyfunction]
#[pyo3(signature = (nodes, inputs, verbose=false))]
fn compile(
    nodes: Vec<(
        String,
        String,
        String,
        Vec<String>,
        HashMap<String, String>,
        Vec<usize>,
        String,
    )>,
    inputs: HashMap<String, Vec<f32>>,
    verbose: bool,
) -> PyResult<HashMap<String, Vec<f32>>> {
    if verbose {
        eprintln!("Loading PyTorch graph with {} nodes", nodes.len());
    }

    let mut graph_nodes: Vec<PyTorchGraphNode> = Vec::new();

    for (name, op, target, args, kwargs, shape, dtype) in nodes {
        if verbose {
            eprintln!("  Node: {} | op={} | target={}", name, op, target);
        }
        graph_nodes.push(PyTorchGraphNode {
            name,
            op,
            target,
            args,
            kwargs,
            shape,
            dtype,
        });
    }

    let mut cx = Graph::new();
    let mut tensor_map: HashMap<String, GraphTensor> = HashMap::new();
    let mut output_names: Vec<String> = Vec::new();

    for node in &graph_nodes {
        match node.op.as_str() {
            "placeholder" => {
                if verbose {
                    eprintln!(
                        "  Creating input: {} with shape {:?} dtype={}",
                        node.name, node.shape, node.dtype
                    );
                }

                let mut tensor = match node.shape.len() {
                    0 => cx.tensor((1,)),
                    1 => cx.tensor((node.shape[0],)),
                    2 => cx.tensor((node.shape[0], node.shape[1])),
                    3 => cx.tensor((node.shape[0], node.shape[1], node.shape[2])),
                    4 => cx.tensor((node.shape[0], node.shape[1], node.shape[2], node.shape[3])),
                    _ => {
                        eprintln!(
                            "    WARNING: Unsupported shape length: {}",
                            node.shape.len()
                        );
                        cx.tensor((1,))
                    }
                };

                if node.dtype.contains("int") || node.dtype.contains("long") {
                    if verbose {
                        eprintln!("    -> Marking tensor as Int dtype");
                    }
                    tensor = tensor.as_dtype(DType::Int);
                }

                if verbose {
                    eprintln!("    ✓ Created input tensor (shape: {:?})", tensor.shape);
                }
                tensor_map.insert(node.name.clone(), tensor);
            }

            "call_function" => {
                if verbose {
                    eprintln!("  Creating operation: {} ({})", node.name, node.target);
                }

                match node.target.as_str() {
                    "aten.add.Tensor" | "aten.add" => {
                        if node.args.len() < 2 {
                            panic!(
                                "aten.add requires at least 2 arguments, got {}",
                                node.args.len()
                            );
                        }

                        let left_hand_name = &node.args[0];
                        let right_hand_name = &node.args[1];

                        if verbose {
                            eprintln!("    -> Computing: {} + {}", left_hand_name, right_hand_name);
                        }

                        let left_hand = tensor_map.get(left_hand_name);
                        let right_hand = tensor_map.get(right_hand_name);

                        let output = match (left_hand, right_hand) {
                            (Some(left), Some(right)) => {
                                if verbose {
                                    eprintln!("    -> tensor + tensor");
                                }
                                *left + *right
                            }
                            (Some(tensor), None) => {
                                let scalar: f32 = right_hand_name.parse().unwrap_or_else(|_| {
                                    panic!(
                                        "Could not parse '{}' as scalar for add operation",
                                        right_hand_name
                                    )
                                });
                                if verbose {
                                    eprintln!("    -> tensor + scalar ({})", scalar);
                                }
                                *tensor + scalar
                            }
                            (None, Some(tensor)) => {
                                let scalar: f32 = left_hand_name.parse().unwrap_or_else(|_| {
                                    panic!(
                                        "Could not parse '{}' as scalar for add operation",
                                        left_hand_name
                                    )
                                });
                                if verbose {
                                    eprintln!("    -> scalar ({}) + tensor", scalar);
                                }
                                scalar + *tensor
                            }
                            (None, None) => {
                                panic!(
                                    "Both operands of add are missing from tensor_map: {} and {}",
                                    left_hand_name, right_hand_name
                                );
                            }
                        };

                        if verbose {
                            eprintln!("    ✓ Created addition");
                            eprintln!("    Output shape: {:?}", output.shape);
                        }
                        tensor_map.insert(node.name.clone(), output);
                    }

                    "aten.mul.Tensor" | "aten.mul" => {
                        if node.args.len() < 2 {
                            panic!(
                                "aten.mul requires at least 2 arguments, got {}",
                                node.args.len()
                            );
                        }

                        let left_hand_name = &node.args[0];
                        let right_hand_name = &node.args[1];

                        if verbose {
                            eprintln!("    -> Computing: {} * {}", left_hand_name, right_hand_name);
                        }

                        let left_hand = tensor_map.get(left_hand_name);
                        let right_hand = tensor_map.get(right_hand_name);

                        let output = match (left_hand, right_hand) {
                            (Some(left), Some(right)) => {
                                if verbose {
                                    eprintln!("    -> tensor * tensor");
                                }
                                *left * *right
                            }
                            (Some(tensor), None) => {
                                let scalar: f32 = right_hand_name.parse().unwrap_or_else(|_| {
                                    panic!(
                                        "Could not parse '{}' as scalar for mul operation",
                                        right_hand_name
                                    )
                                });
                                if verbose {
                                    eprintln!("    -> tensor * scalar ({})", scalar);
                                }
                                *tensor * scalar
                            }
                            (None, Some(tensor)) => {
                                let scalar: f32 = left_hand_name.parse().unwrap_or_else(|_| {
                                    panic!(
                                        "Could not parse '{}' as scalar for mul operation",
                                        left_hand_name
                                    )
                                });
                                if verbose {
                                    eprintln!("    -> scalar ({}) * tensor", scalar);
                                }
                                scalar * *tensor
                            }
                            (None, None) => {
                                panic!(
                                    "Both operands of mul are missing from tensor_map: {} and {}",
                                    left_hand_name, right_hand_name
                                );
                            }
                        };

                        if verbose {
                            eprintln!("    ✓ Created multiplication");
                            eprintln!("    Output shape: {:?}", output.shape);
                        }
                        tensor_map.insert(node.name.clone(), output);
                    }

                    "aten.linear.default" => {
                        if node.args.len() < 2 {
                            panic!(
                                "aten.linear requires at least 2 arguments, got {}",
                                node.args.len()
                            );
                        }

                        let input_name = &node.args[0];
                        let weight_name = &node.args[1];

                        if verbose {
                            eprintln!("    -> Computing: {} @ {}.T", input_name, weight_name);
                        }

                        let input = tensor_map.get(input_name).unwrap_or_else(|| {
                            panic!(
                                "Could not find input '{}' in tensor_map for linear operation",
                                input_name
                            )
                        });
                        let weight = tensor_map.get(weight_name).unwrap_or_else(|| {
                            panic!(
                                "Could not find weight '{}' in tensor_map for linear operation",
                                weight_name
                            )
                        });

                        let weight_t = weight.permute((1, 0));
                        let mut output = input.matmul(weight_t);

                        if node.args.len() >= 3 {
                            let bias_name = &node.args[2];
                            if let Some(bias) = tensor_map.get(bias_name) {
                                if verbose {
                                    eprintln!("    -> Adding bias: {}", bias_name);
                                }
                                let batch_size = output.dims()[0];
                                let expanded_bias = bias.expand_lhs([batch_size]);
                                output = output + expanded_bias;
                            }
                        }

                        if verbose {
                            eprintln!("    ✓ Created linear layer (HLIR: Mul + SumReduce + Add)");
                            eprintln!("    Output shape: {:?}", output.shape);
                        }
                        tensor_map.insert(node.name.clone(), output);
                    }

                    "aten.view.default" => {
                        if node.args.len() < 2 {
                            panic!(
                                "aten.view requires at least 2 arguments (tensor, *shape), got {}",
                                node.args.len()
                            );
                        }

                        let input_name = &node.args[0];

                        if verbose {
                            eprintln!(
                                "    -> Computing: {}.view({:?})",
                                input_name,
                                &node.args[1..]
                            );
                        }

                        let input = tensor_map.get(input_name).unwrap_or_else(|| {
                            panic!(
                                "Could not find input '{}' in tensor_map for view operation",
                                input_name
                            )
                        });

                        let mut target_shape: Vec<i32> = Vec::new();
                        for arg in &node.args[1..] {
                            let dim: i32 = arg.parse().unwrap_or_else(|_| {
                                panic!(
                                    "Could not parse dimension '{}' as integer for view operation",
                                    arg
                                )
                            });
                            target_shape.push(dim);
                        }

                        if verbose {
                            eprintln!("    -> Target shape: {:?}", target_shape);
                            eprintln!("    -> Input shape: {:?}", input.dims());
                        }

                        let input_dims = input.dims();
                        let mut total_elements = Expression::from(1);
                        for dim in &input_dims {
                            total_elements *= dim;
                        }

                        let mut inferred_idx: Option<usize> = None;
                        let mut known_product = Expression::from(1);
                        for (i, &dim) in target_shape.iter().enumerate() {
                            if dim == -1 {
                                if inferred_idx.is_some() {
                                    panic!(
                                        "Only one dimension can be inferred (set to -1) in view operation"
                                    );
                                }
                                inferred_idx = Some(i);
                            } else if dim > 0 {
                                known_product *= dim as usize;
                            } else {
                                panic!(
                                    "Invalid dimension {} in view operation (must be positive or -1)",
                                    dim
                                );
                            }
                        }

                        let mut final_shape: Vec<Expression> = Vec::new();
                        for (i, &dim) in target_shape.iter().enumerate() {
                            if Some(i) == inferred_idx {
                                let inferred_dim = total_elements / known_product;
                                final_shape.push(inferred_dim);
                            } else {
                                final_shape.push(Expression::from(dim as usize));
                            }
                        }

                        if verbose {
                            eprintln!("    -> Final shape: {:?}", final_shape);
                        }

                        let mut output = *input;
                        output.shape = ShapeTracker::new(final_shape);

                        if verbose {
                            eprintln!("    ✓ Created view operation");
                            eprintln!("    Output shape: {:?}", output.shape);
                        }
                        tensor_map.insert(node.name.clone(), output);
                    }

                    "aten.transpose.int" => {
                        if node.args.len() < 3 {
                            panic!(
                                "aten.transpose requires 3 arguments (tensor, dim0, dim1), got {}",
                                node.args.len()
                            );
                        }

                        let input_name = &node.args[0];
                        let dim0_str = &node.args[1];
                        let dim1_str = &node.args[2];

                        if verbose {
                            eprintln!(
                                "    -> Computing: {}.transpose({}, {})",
                                input_name, dim0_str, dim1_str
                            );
                        }

                        let input = tensor_map.get(input_name).unwrap_or_else(|| {
                            panic!(
                                "Could not find input '{}' in tensor_map for transpose operation",
                                input_name
                            )
                        });

                        // Parse dimension indices
                        let mut dim0: i32 = dim0_str.parse().unwrap_or_else(|_| {
                            panic!(
                                "Could not parse dim0 '{}' as integer for transpose operation",
                                dim0_str
                            )
                        });
                        let mut dim1: i32 = dim1_str.parse().unwrap_or_else(|_| {
                            panic!(
                                "Could not parse dim1 '{}' as integer for transpose operation",
                                dim1_str
                            )
                        });

                        // Get number of dimensions
                        let num_dims = input.dims().len() as i32;

                        // Handle negative indices (Python-style indexing)
                        if dim0 < 0 {
                            dim0 += num_dims;
                        }
                        if dim1 < 0 {
                            dim1 += num_dims;
                        }

                        // Validate dimensions
                        if dim0 < 0 || dim0 >= num_dims {
                            panic!(
                                "dim0 {} is out of bounds for tensor with {} dimensions",
                                dim0, num_dims
                            );
                        }
                        if dim1 < 0 || dim1 >= num_dims {
                            panic!(
                                "dim1 {} is out of bounds for tensor with {} dimensions",
                                dim1, num_dims
                            );
                        }

                        if verbose {
                            eprintln!("    -> Input shape: {:?}", input.dims());
                            eprintln!("    -> Transposing dimensions {} and {}", dim0, dim1);
                        }

                        let output = input.transpose(dim0 as usize, dim1 as usize) * 1.0;

                        if verbose {
                            eprintln!("    ✓ Created transpose operation");
                            eprintln!("    Output shape: {:?}", output.shape);
                        }
                        tensor_map.insert(node.name.clone(), output);
                    }

                    "aten.dropout.default" => {
                        if node.args.len() >= 1 {
                            let input_name = &node.args[0];

                            if let Some(input) = tensor_map.get(input_name) {
                                if verbose {
                                    eprintln!("    -> Dropout (no-op)");
                                }
                                let output = *input;
                                tensor_map.insert(node.name.clone(), output);
                            } else if verbose {
                                eprintln!("    ERROR: Could not find tensor {}", input_name);
                            }
                        }
                    }

                    "aten.lift_fresh_copy.default" => {
                        // This operation creates a fresh copy of a tensor
                        // Used by PyTorch export to track constants/parameters
                        if node.args.is_empty() {
                            panic!(
                                "aten.lift_fresh_copy requires at least 1 argument (tensor), got 0"
                            );
                        }

                        let input_name = &node.args[0];

                        if verbose {
                            eprintln!("    -> Creating fresh copy of: {}", input_name);
                        }

                        let input = *tensor_map.get(input_name).unwrap_or_else(|| {
                            panic!(
                                "Could not find input '{}' in tensor_map for lift_fresh_copy operation",
                                input_name
                            )
                        });

                        // Just pass through - luminal handles the graph structure
                        let output = input;

                        if verbose {
                            eprintln!("    ✓ Created fresh copy (pass-through)");
                            eprintln!("    Output shape: {:?}", output.shape);
                        }
                        tensor_map.insert(node.name.clone(), output);
                    }

                    "aten.layer_norm.default" => {
                        if node.args.len() >= 1 {
                            let input_name = &node.args[0];

                            if let Some(input) = tensor_map.get(input_name) {
                                if verbose {
                                    eprintln!("    -> Computing: layer_norm({})", input_name);
                                    eprintln!("    Args ({:?})", node.args);
                                }

                                // Get epsilon from kwargs, default to 1e-5
                                let eps: f32 = node
                                    .kwargs
                                    .get("eps")
                                    .and_then(|s| s.parse().ok())
                                    .unwrap_or(1e-5);

                                let dims = input.dims();
                                let norm_axis = dims.len() - 1;
                                let mut output = input.layer_norm(norm_axis, eps);

                                if node.args.len() >= 3 && node.args[2] != "None" {
                                    let weight_name = &node.args[2];
                                    if let Some(weight) = tensor_map.get(weight_name) {
                                        if verbose {
                                            eprintln!(
                                                "    -> Applying weight (gamma): {}",
                                                weight_name
                                            );
                                        }
                                        let batch_size = output.dims()[0];
                                        let expanded_weight = weight.expand_lhs([batch_size]);
                                        output = output * expanded_weight;
                                    }
                                }

                                if node.args.len() >= 4 && node.args[3] != "None" {
                                    let bias_name = &node.args[3];
                                    if let Some(bias) = tensor_map.get(bias_name) {
                                        if verbose {
                                            eprintln!("    -> Applying bias (beta): {}", bias_name);
                                        }
                                        let batch_size = output.dims()[0];
                                        let expanded_bias = bias.expand_lhs([batch_size]);
                                        output = output + expanded_bias;
                                    }
                                }

                                if verbose {
                                    eprintln!("    ✓ Created layer normalization");
                                    eprintln!("    Output shape: {:?}", output.shape);
                                }
                                tensor_map.insert(node.name.clone(), output);
                            } else if verbose {
                                eprintln!("    ERROR: Could not find tensor {}", input_name);
                            }
                        }
                    }

                    "aten.split.Tensor" => {
                        // Split doesn't actually transform the data - it just conceptually divides it
                        // The actual slicing happens in getitem
                        if node.args.len() < 2 {
                            panic!(
                                "aten.split requires at least 2 arguments (tensor, split_size), got {}",
                                node.args.len()
                            );
                        }

                        let input_name = &node.args[0];
                        let split_size_str = &node.args[1];
                        let dim_str = if node.args.len() >= 3 {
                            &node.args[2]
                        } else {
                            "0"
                        };

                        if verbose {
                            eprintln!(
                                "    -> Split operation: {}.split({}, dim={})",
                                input_name, split_size_str, dim_str
                            );
                        }

                        let input = *tensor_map.get(input_name).unwrap_or_else(|| {
                            panic!(
                                "Could not find input '{}' in tensor_map for split operation",
                                input_name
                            )
                        });

                        tensor_map.insert(node.name.clone(), input);

                        if verbose {
                            eprintln!("    ✓ Split metadata recorded");
                        }
                    }

                    "<built-in function getitem>" | "operator.getitem" | "getitem" => {
                        if node.args.len() < 2 {
                            panic!(
                                "getitem requires 2 arguments (container, index), got {}",
                                node.args.len()
                            );
                        }

                        let container_name = &node.args[0];
                        let index_str = &node.args[1];

                        if verbose {
                            eprintln!("    -> Computing: {}[{}]", container_name, index_str);
                        }

                        let index: usize = index_str.parse().unwrap_or_else(|_| {
                            panic!(
                                "Could not parse index '{}' as integer for getitem operation",
                                index_str
                            )
                        });

                        let container = *tensor_map.get(container_name).unwrap_or_else(|| {
                            panic!(
                                "Could not find container '{}' in tensor_map for getitem operation",
                                container_name
                            )
                        });

                        let split_node = graph_nodes.iter().find(|n| n.name == *container_name);

                        if let Some(split_node) = split_node {
                            if split_node.target == "aten.split.Tensor"
                                && split_node.args.len() >= 2
                            {
                                let split_size: usize = split_node.args[1].parse().unwrap();
                                let dim_str = if split_node.args.len() >= 3 {
                                    &split_node.args[2]
                                } else {
                                    "0"
                                };
                                let mut dim: i32 = dim_str.parse().unwrap();

                                let num_dims = container.dims().len() as i32;
                                if dim < 0 {
                                    dim += num_dims;
                                }
                                let dim_usize = dim as usize;

                                let start = index * split_size;
                                let dim_size = container.dims()[dim_usize].to_usize().unwrap();
                                let end = std::cmp::min(start + split_size, dim_size);

                                if verbose {
                                    eprintln!("    -> Extracting chunk {} from split", index);
                                    eprintln!(
                                        "    -> Slicing dimension {} from {} to {}",
                                        dim, start, end
                                    );
                                }

                                let chunk = container.slice_along(start..end, dim_usize) * 1.0;

                                if verbose {
                                    eprintln!("    ✓ Retrieved chunk");
                                    eprintln!("    Chunk shape: {:?}", chunk.shape);
                                }

                                tensor_map.insert(node.name.clone(), chunk);
                            } else {
                                panic!(
                                    "getitem container '{}' is not a split operation",
                                    container_name
                                );
                            }
                        } else {
                            panic!(
                                "Could not find split node '{}' for getitem operation",
                                container_name
                            );
                        }
                    }

                    "aten.unsqueeze.default" => {
                        if node.args.len() < 2 {
                            panic!(
                                "aten.unsqueeze requires 2 arguments (tensor, dim), got {}",
                                node.args.len()
                            );
                        }

                        let input_name = &node.args[0];
                        let dim_str = &node.args[1];

                        if verbose {
                            eprintln!("    -> Computing: {}.unsqueeze({})", input_name, dim_str);
                        }

                        let input = *tensor_map.get(input_name).unwrap_or_else(|| {
                            panic!(
                                "Could not find input '{}' in tensor_map for unsqueeze operation",
                                input_name
                            )
                        });

                        // Parse dimension
                        let mut dim: i32 = dim_str.parse().unwrap_or_else(|_| {
                            panic!(
                                "Could not parse dim '{}' as integer for unsqueeze operation",
                                dim_str
                            )
                        });

                        // Get number of dimensions (after unsqueeze will be +1)
                        let num_dims = input.dims().len() as i32;
                        let new_num_dims = num_dims + 1;

                        // Handle negative indices
                        if dim < 0 {
                            dim += new_num_dims;
                        }

                        // Validate dimension
                        if dim < 0 || dim >= new_num_dims {
                            panic!(
                                "dim {} is out of bounds for unsqueeze operation (valid range: 0 to {})",
                                dim,
                                new_num_dims - 1
                            );
                        }

                        if verbose {
                            eprintln!("    -> Input shape: {:?}", input.dims());
                            eprintln!("    -> Adding dimension of size 1 at position {}", dim);
                        }

                        // Apply unsqueeze
                        let output = input.unsqueeze(dim as usize);

                        if verbose {
                            eprintln!("    ✓ Created unsqueeze operation");
                            eprintln!("    Output shape: {:?}", output.shape);
                        }
                        tensor_map.insert(node.name.clone(), output);
                    }

                    "aten.squeeze.dim" => {
                        if node.args.len() < 2 {
                            panic!(
                                "aten.squeeze requires 2 arguments (tensor, dim), got {}",
                                node.args.len()
                            );
                        }

                        let input_name = &node.args[0];
                        let dim_str = &node.args[1];

                        if verbose {
                            eprintln!("    -> Computing: {}.squeeze({})", input_name, dim_str);
                        }

                        let input = *tensor_map.get(input_name).unwrap_or_else(|| {
                            panic!(
                                "Could not find input '{}' in tensor_map for squeeze operation",
                                input_name
                            )
                        });

                        // Parse dimension
                        let mut dim: i32 = dim_str.parse().unwrap_or_else(|_| {
                            panic!(
                                "Could not parse dim '{}' as integer for squeeze operation",
                                dim_str
                            )
                        });

                        // Get number of dimensions
                        let num_dims = input.dims().len() as i32;

                        // Handle negative indices (relative to current shape)
                        if dim < 0 {
                            dim += num_dims;
                        }

                        // Validate dimension
                        if dim < 0 || dim >= num_dims {
                            panic!(
                                "dim {} is out of bounds for squeeze operation (valid range: 0 to {})",
                                dim,
                                num_dims - 1
                            );
                        }

                        // Validate that the dimension has size 1
                        let dim_size = input.dims()[dim as usize].to_usize().unwrap();
                        if dim_size != 1 {
                            panic!(
                                "Cannot squeeze dimension {} with size {} (only dimensions of size 1 can be squeezed)",
                                dim, dim_size
                            );
                        }

                        if verbose {
                            eprintln!("    -> Input shape: {:?}", input.dims());
                            eprintln!("    -> Removing dimension {} (size 1)", dim);
                        }

                        // Apply squeeze
                        let output = input.squeeze(dim as usize);

                        if verbose {
                            eprintln!("    ✓ Created squeeze operation");
                            eprintln!("    Output shape: {:?}", output.shape);
                        }
                        tensor_map.insert(node.name.clone(), output);
                    }

                    "aten.arange.default" => {
                        // torch.arange(end) - creates [0, 1, 2, ..., end-1]
                        if node.args.is_empty() {
                            panic!("aten.arange.default requires at least 1 argument (end), got 0");
                        }

                        let end_str = &node.args[0];

                        if verbose {
                            eprintln!("    -> Computing: arange({})", end_str);
                            eprintln!("    DEBUG: arange node name: {}", node.name);
                            eprintln!("    DEBUG: arange args: {:?}", node.args);
                            eprintln!("    DEBUG: arange shape: {:?}", node.shape);
                        }

                        // Parse end value
                        let end: usize = end_str.parse().unwrap_or_else(|_| {
                            panic!(
                                "Could not parse end '{}' as integer for arange operation",
                                end_str
                            )
                        });

                        if verbose {
                            eprintln!("    -> Creating range [0, 1, ..., {}]", end - 1);
                            eprintln!("    DEBUG: Parsed end value: {}", end);
                        }

                        // Create arange [0, 1, ..., end-1]
                        let output = cx.arange(end);

                        if verbose {
                            eprintln!("    ✓ Created arange operation");
                            eprintln!("    Output shape: {:?}", output.shape);
                            eprintln!("    DEBUG: arange tensor ID: {:?}", output.id);
                            eprintln!("    DEBUG: arange dtype: {:?}", output.dtype);
                        }
                        tensor_map.insert(node.name.clone(), output);
                    }

                    "aten.arange.start" => {
                        // torch.arange(start, end) - creates [start, start+1, ..., end-1]
                        if node.args.len() < 2 {
                            panic!(
                                "aten.arange.start requires 2 arguments (start, end), got {}",
                                node.args.len()
                            );
                        }

                        let start_str = &node.args[0];
                        let end_str = &node.args[1];

                        if verbose {
                            eprintln!("    -> Computing: arange({}, {})", start_str, end_str);
                        }

                        // Parse start and end values
                        let start: f32 = start_str.parse().unwrap_or_else(|_| {
                            panic!(
                                "Could not parse start '{}' as number for arange operation",
                                start_str
                            )
                        });
                        let end: f32 = end_str.parse().unwrap_or_else(|_| {
                            panic!(
                                "Could not parse end '{}' as number for arange operation",
                                end_str
                            )
                        });

                        // Calculate length
                        let length = (end - start).ceil() as usize;

                        if verbose {
                            eprintln!(
                                "    -> Creating range [{}, {}, ..., {}]",
                                start,
                                start + 1.0,
                                end - 1.0
                            );
                            eprintln!("    -> Length: {}", length);
                        }

                        // Create base arange [0, 1, ..., length-1] and shift by start
                        let output = cx.arange(length) + start;

                        if verbose {
                            eprintln!("    ✓ Created arange operation");
                            eprintln!("    Output shape: {:?}", output.shape);
                        }
                        tensor_map.insert(node.name.clone(), output);
                    }

                    "aten.arange.start_step" => {
                        // torch.arange(start, end, step) - creates [start, start+step, start+2*step, ...]
                        if node.args.len() < 3 {
                            panic!(
                                "aten.arange.start_step requires 3 arguments (start, end, step), got {}",
                                node.args.len()
                            );
                        }

                        let start_str = &node.args[0];
                        let end_str = &node.args[1];
                        let step_str = &node.args[2];

                        if verbose {
                            eprintln!(
                                "    -> Computing: arange({}, {}, {})",
                                start_str, end_str, step_str
                            );
                        }

                        // Parse start, end, and step values
                        let start: f32 = start_str.parse().unwrap_or_else(|_| {
                            panic!(
                                "Could not parse start '{}' as number for arange operation",
                                start_str
                            )
                        });
                        let end: f32 = end_str.parse().unwrap_or_else(|_| {
                            panic!(
                                "Could not parse end '{}' as number for arange operation",
                                end_str
                            )
                        });
                        let step: f32 = step_str.parse().unwrap_or_else(|_| {
                            panic!(
                                "Could not parse step '{}' as number for arange operation",
                                step_str
                            )
                        });

                        // Calculate length: ceil((end - start) / step)
                        let length = ((end - start) / step).ceil() as usize;

                        if verbose {
                            eprintln!(
                                "    -> Creating range [{}, {}, ..., <{}]",
                                start,
                                start + step,
                                end
                            );
                            eprintln!("    -> Length: {}, Step: {}", length, step);
                        }

                        // Create base arange [0, 1, ..., length-1], scale by step, and shift by start
                        let output = cx.arange(length) * step + start;

                        if verbose {
                            eprintln!("    ✓ Created arange operation");
                            eprintln!("    Output shape: {:?}", output.shape);
                        }
                        tensor_map.insert(node.name.clone(), output);
                    }

                    "aten.scaled_dot_product_attention.default" => {
                        if node.args.len() < 3 {
                            panic!(
                                "aten.scaled_dot_product_attention requires at least 3 arguments (query, key, value), got {}",
                                node.args.len()
                            );
                        }

                        let query_name = &node.args[0];
                        let key_name = &node.args[1];
                        let value_name = &node.args[2];

                        if verbose {
                            eprintln!(
                                "    -> Computing: scaled_dot_product_attention({}, {}, {})",
                                query_name, key_name, value_name
                            );
                        }

                        let query = *tensor_map.get(query_name).unwrap_or_else(|| {
                            panic!(
                                "Could not find query '{}' in tensor_map for SDPA operation",
                                query_name
                            )
                        });
                        let key = *tensor_map.get(key_name).unwrap_or_else(|| {
                            panic!(
                                "Could not find key '{}' in tensor_map for SDPA operation",
                                key_name
                            )
                        });
                        let value = *tensor_map.get(value_name).unwrap_or_else(|| {
                            panic!(
                                "Could not find value '{}' in tensor_map for SDPA operation",
                                value_name
                            )
                        });

                        if verbose {
                            eprintln!("    -> Query shape: {:?}", query.dims());
                            eprintln!("    -> Key shape: {:?}", key.dims());
                            eprintln!("    -> Value shape: {:?}", value.dims());
                        }

                        // Get head dimension (last dimension)
                        let query_dims = query.dims();
                        let head_dim = query_dims[query_dims.len() - 1].to_usize().unwrap();
                        let scale = 1.0 / (head_dim as f32).sqrt();

                        if verbose {
                            eprintln!("    -> Head dimension: {}", head_dim);
                            eprintln!("    -> Scale factor: {}", scale);
                        }

                        // Transpose key on last two dimensions
                        let num_dims = key.dims().len();
                        let key_t = key.transpose(num_dims - 2, num_dims - 1);

                        if verbose {
                            eprintln!("    -> Key transposed shape: {:?}", key_t.dims());
                        }

                        // Compute Q @ K^T
                        let scores = query.matmul(key_t);

                        if verbose {
                            eprintln!("    -> Scores shape (Q @ K^T): {:?}", scores.dims());
                        }

                        // Scale by 1/sqrt(d_k)
                        let scaled_scores = scores * scale;

                        // Apply softmax on last dimension
                        let attention_weights = scaled_scores.softmax(num_dims - 1);

                        if verbose {
                            eprintln!(
                                "    -> Attention weights shape: {:?}",
                                attention_weights.dims()
                            );
                        }

                        // Compute attention @ V
                        let output = attention_weights.matmul(value);

                        if verbose {
                            eprintln!("    ✓ Created scaled dot product attention");
                            eprintln!("    Output shape: {:?}", output.shape);
                        }
                        tensor_map.insert(node.name.clone(), output);
                    }

                    "aten.gelu.default" => {
                        // GELU activation - TANH APPROXIMATION ONLY
                        let is_tanh_approx = node
                            .kwargs
                            .get("approximate")
                            .map(|s| s == "tanh")
                            .unwrap_or(false);

                        if !is_tanh_approx {
                            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                                "Exact GELU (erf-based) is not supported.\n\
                                 Luminal only supports the tanh approximation of GELU.\n\
                                 Please use: torch.nn.functional.gelu(x, approximate='tanh')\n\
                                 \n\
                                 Note: The tanh approximation differs from exact GELU by ~0.0004 max error.\n\
                                 Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))",
                            ));
                        }

                        if node.args.len() >= 1 {
                            let input_name = &node.args[0];

                            if let Some(input) = tensor_map.get(input_name) {
                                if verbose {
                                    eprintln!(
                                        "    -> Computing: gelu({}) [tanh approximation]",
                                        input_name
                                    );
                                }

                                let output = input.gelu();

                                if verbose {
                                    eprintln!("    ✓ Created GELU activation (tanh approximation)");
                                    eprintln!("    Output shape: {:?}", output.shape);
                                }
                                tensor_map.insert(node.name.clone(), output);
                            } else if verbose {
                                eprintln!("    ERROR: Could not find tensor {}", input_name);
                            }
                        }
                    }

                    "aten.embedding.default" => {
                        if node.args.len() < 2 {
                            panic!(
                                "aten.embedding requires at least 2 arguments, got {}",
                                node.args.len()
                            );
                        }

                        let weight_name = &node.args[0];
                        let indices_name = &node.args[1];

                        if verbose {
                            eprintln!(
                                "    -> Computing: embedding({}, {})",
                                weight_name, indices_name
                            );
                        }

                        let weight = tensor_map.get(weight_name).unwrap_or_else(|| {
                            panic!(
                                "Could not find weight '{}' in tensor_map for embedding operation",
                                weight_name
                            )
                        });
                        let indices = tensor_map.get(indices_name).unwrap_or_else(|| {
                            panic!(
                                "Could not find indices '{}' in tensor_map for embedding operation",
                                indices_name
                            )
                        });

                        if verbose {
                            eprintln!("    Weight shape: {:?}", weight.shape);
                            eprintln!("    Indices shape: {:?}", indices.shape);
                            eprintln!("    Indices dtype: {:?}", indices.dtype);
                        }

                        let weight_dims = weight.dims();
                        let embedding_dim = weight_dims[1]; // num_embeddings x embedding_dim
                        let indices_dims = indices.dims();

                        let scaled_indices = *indices * embedding_dim;

                        let expanded_scaled =
                            scaled_indices.expand_dim(indices_dims.len(), embedding_dim);

                        let mut offset = cx.arange(embedding_dim);
                        for (i, &dim) in indices_dims.iter().enumerate() {
                            offset = offset.expand_dim(i, dim);
                        }

                        let gather_indices = expanded_scaled + offset;

                        let output = weight.gather(gather_indices);

                        if verbose {
                            eprintln!("    ✓ Created embedding lookup (HLIR: Gather)");
                            eprintln!("    Output shape: {:?}", output.shape);
                        }
                        tensor_map.insert(node.name.clone(), output);
                    }

                    "aten.index.Tensor" => {
                        // Expect: args[0] = input tensor name
                        let input_name = node.args.get(0).unwrap().clone();
                        let input = *tensor_map.get(&input_name).unwrap_or_else(|| {
                            panic!("aten.index.Tensor: missing input tensor '{}'", input_name)
                        });

                        // Metadata injected from Python
                        let index_dim: usize = node.kwargs.get("index_dim")
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or_else(|| {
                                panic!(
                                    "aten.index.Tensor: missing/invalid kwargs['index_dim'] (spec={:?})",
                                    node.kwargs.get("index_spec")
                                )
                            });

                        let index_tensor_name =
                            node.kwargs.get("index_tensor").cloned().unwrap_or_else(|| {
                                panic!(
                                    "aten.index.Tensor: missing kwargs['index_tensor'] (spec={:?})",
                                    node.kwargs.get("index_spec")
                                )
                            });

                        let indices = *tensor_map.get(&index_tensor_name).unwrap_or_else(|| {
                            panic!(
                                "aten.index.Tensor: missing index tensor '{}'",
                                index_tensor_name
                            )
                        });

                        let in_dims = input.dims();
                        let rank = in_dims.len();
                        if index_dim >= rank {
                            panic!(
                                "aten.index.Tensor: index_dim={} out of range for rank={}",
                                index_dim, rank
                            );
                        }

                        if verbose {
                            eprintln!(
                                "  -> aten.index.Tensor input={} shape={:?} index_dim={} index_tensor={} idx_shape={:?} idx_dtype={:?} spec={}",
                                input_name,
                                in_dims,
                                index_dim,
                                index_tensor_name,
                                indices.dims(),
                                indices.dtype, // if you store dtype
                                node.kwargs.get("index_spec").cloned().unwrap_or_default()
                            );
                        }

                        // --- Simple supported subset ---
                        // We implement: out = input with exactly ONE dimension 'index_dim' gathered by an int tensor.
                        // All other dimensions behave like ':' (full slice).
                        //
                        // Strategy:
                        // 1) Flatten input to 2D: (prefix, indexed_dim, suffix_flat)
                        //    -> reshape to (prefix*indexed_dim, suffix_flat) or (prefix, indexed_dim*suffix_flat)
                        //    Choose a layout that lets us gather rows with one index per prefix entry.
                        // 2) Convert indices into "row indices" into the flattened matrix
                        // 3) Use gather to fetch full suffix_flat block
                        // 4) Reshape back to the correct output shape, preserving indexed dim size from indices

                        // Compute sizes (require static dims for now)
                        let mut prefix = 1usize;
                        for d in 0..index_dim {
                            prefix *= in_dims[d]
                                .to_usize()
                                .expect("aten.index.Tensor: dynamic prefix dim not supported yet");
                        }
                        let indexed = in_dims[index_dim]
                            .to_usize()
                            .expect("aten.index.Tensor: dynamic indexed dim not supported yet");

                        let mut suffix = 1usize;
                        for d in (index_dim + 1)..rank {
                            suffix *= in_dims[d]
                                .to_usize()
                                .expect("aten.index.Tensor: dynamic suffix dim not supported yet");
                        }

                        // Reshape input to (prefix, indexed, suffix)
                        // then flatten to (prefix*indexed, suffix) so we can gather "rows" (a whole suffix block)
                        let mut x3 = input;
                        x3.shape = ShapeTracker::new(vec![
                            Expression::from(prefix),
                            Expression::from(indexed),
                            Expression::from(suffix),
                        ]);

                        let mut x2 = x3;
                        x2.shape = ShapeTracker::new(vec![
                            Expression::from(prefix * indexed),
                            Expression::from(suffix),
                        ]);

                        // Normalize indices shape:
                        // We support either:
                        //  - scalar/len1 -> broadcast to (prefix,)
                        //  - (prefix,)   -> use directly
                        //
                        // (This is still generic: prefix is derived from shape, not GPT-specific.)
                        let mut idx = indices;
                        let idx_dims = idx.dims();
                        if idx_dims.len() == 0 {
                            // scalar -> treat as (1,) then broadcast
                            idx = idx.unsqueeze(0);
                        }
                        let idx_len = idx
                            .dims()
                            .iter()
                            .fold(1usize, |acc, e| acc * e.to_usize().unwrap());
                        if idx_len == 1 && prefix != 1 {
                            idx = idx.expand_lhs([Expression::from(prefix)]);
                        } else if idx_len != prefix {
                            panic!(
                                "aten.index.Tensor: unsupported indices size. expected 1 or prefix={}, got {} (idx_shape={:?})",
                                prefix,
                                idx_len,
                                idx.dims()
                            );
                        }

                        // Now idx is effectively (prefix,)
                        // Compute flat row indices: row = arange(prefix) * indexed + idx
                        // NOTE: this assumes idx contains integer values in [0, indexed) or negative indexing already normalized upstream.
                        // If you need negative handling, do it either in Python (preferred) or add a normalize op later.
                        let base = cx.arange(prefix) * (indexed as f32); // ideally int arange; use your existing for now
                        let row = base + idx; // (prefix,)

                        // Gather full suffix blocks from x2 (shape (prefix*indexed, suffix))
                        let suffix_expr = Expression::from(suffix);
                        let row_dims = row.dims(); // should be (prefix,)

                        let scaled = row * suffix_expr;
                        let expanded_scaled = scaled.expand_dim(row_dims.len(), suffix_expr);

                        let mut offset = cx.arange(suffix);
                        for (i, &d) in row_dims.iter().enumerate() {
                            offset = offset.expand_dim(i, d);
                        }
                        let gather_idx = expanded_scaled + offset;
                        let gathered = x2.gather(gather_idx); // (prefix, suffix)

                        // Reshape back to original rank with indexed dim replaced by 1 (since we're selecting one index per prefix entry)
                        // Output shape becomes: original dims, but indexed dim becomes 1, and suffix dims restored.
                        //
                        // We currently produce (prefix, suffix) and reshape to:
                        //   dims[0:index_dim] + [1] + dims[index_dim+1:]
                        let mut out_dims: Vec<Expression> = Vec::new();
                        for d in 0..index_dim {
                            out_dims.push(in_dims[d]);
                        }
                        out_dims.push(Expression::from(1));
                        for d in (index_dim + 1)..rank {
                            out_dims.push(in_dims[d]);
                        }

                        let mut out = gathered;
                        out.shape = ShapeTracker::new(out_dims);

                        tensor_map.insert(node.name.clone(), out);
                    }

                    _ => {
                        panic!(
                            "Unsupported operation: {} (node: {}). This operation needs to be implemented.",
                            node.target, node.name
                        );
                    }
                }
            }

            "output" => {
                if verbose {
                    eprintln!("  Marking outputs");
                    eprintln!("    Output args requested: {:?}", node.args);
                    eprintln!(
                        "    Available tensors: {:?}",
                        tensor_map.keys().collect::<Vec<_>>()
                    );
                }

                for arg in &node.args {
                    // ✅ Skip torch.export tuple entries like (logits, None)
                    if arg == "None" {
                        if verbose {
                            eprintln!("    -> Skipping None output");
                        }
                        continue;
                    }

                    if let Some(tensor) = tensor_map.get(arg).copied() {
                        let output_tensor = tensor.output();
                        tensor_map.insert(arg.clone(), output_tensor);
                        output_names.push(arg.clone());
                        if verbose {
                            eprintln!("    ✓ Marked {} as output", arg);
                        }
                    } else if verbose {
                        eprintln!(
                            "    ⚠ WARNING: Output tensor {} not found in tensor_map",
                            arg
                        );
                    }
                }
            }

            "get_attr" => {
                if verbose {
                    eprintln!(
                        "  Creating parameter: {} with shape {:?}",
                        node.name, node.shape
                    );
                }

                let tensor = match node.shape.len() {
                    0 => cx.tensor((1,)),
                    1 => cx.tensor((node.shape[0],)),
                    2 => cx.tensor((node.shape[0], node.shape[1])),
                    3 => cx.tensor((node.shape[0], node.shape[1], node.shape[2])),
                    4 => cx.tensor((node.shape[0], node.shape[1], node.shape[2], node.shape[3])),
                    _ => {
                        if verbose {
                            eprintln!(
                                "    WARNING: Unsupported shape length: {}",
                                node.shape.len()
                            );
                        }
                        cx.tensor((1,))
                    }
                };

                if verbose {
                    eprintln!("    ✓ Created parameter tensor (shape: {:?})", tensor.shape);
                }
                tensor_map.insert(node.name.clone(), tensor);
            }

            _ => {
                if verbose {
                    eprintln!("  Unknown op type: {}", node.op);
                }
            }
        }
    }

    if verbose {
        eprintln!("\n✓ Successfully built HLIR graph!");
        eprintln!("  Tensors: {}", tensor_map.len());
        eprintln!("  HLIR nodes: {}", cx.graph.node_count());
        eprintln!("  HLIR edges: {}", cx.graph.edge_count());

        eprintln!("\nExporting HLIR graph to dot file...");
        match cx.graph.to_dot() {
            Ok(dot_string) => {
                let filename = "hlir_graph.dot";
                match File::create(filename) {
                    Ok(mut file) => {
                        if let Err(e) = file.write_all(dot_string.as_bytes()) {
                            eprintln!("  ⚠ Failed to write to {}: {}", filename, e);
                        } else {
                            eprintln!("  ✓ HLIR graph exported to {}", filename);
                            eprintln!("    View it at: https://dreampuf.github.io/GraphvizOnline/");
                        }
                    }
                    Err(e) => {
                        eprintln!("  ⚠ Failed to create {}: {}", filename, e);
                    }
                }
            }
            Err(e) => {
                eprintln!("  ⚠ Failed to generate dot representation: {}", e);
            }
        }

        eprintln!("\nBuilding search space (populates op metadata)...");

        // Add detailed node information before building search space
        eprintln!("\n==== PRE-SEARCH SPACE DEBUG ====");
        eprintln!("  Total HLIR nodes: {}", cx.graph.node_count());
        eprintln!("  Total HLIR edges: {}", cx.graph.edge_count());

        // Log all tensors in the map with their properties
        eprintln!("\n  Tensor map contents:");
        for (name, tensor) in tensor_map.iter() {
            eprintln!(
                "    - {}: shape={:?}, ID={:?}",
                name, tensor.shape, tensor.id
            );
        }

        eprintln!("\n  Output tensors to retrieve:");
        for output_name in &output_names {
            if let Some(tensor) = tensor_map.get(output_name) {
                eprintln!(
                    "    - {}: shape={:?}, ID={:?}",
                    output_name, tensor.shape, tensor.id
                );
            } else {
                eprintln!("    - {} (NOT FOUND IN TENSOR MAP!)", output_name);
            }
        }
        eprintln!("================================");
    }

    cx.build_search_space::<NativeRuntime>();

    if verbose {
        eprintln!("Egglog running...");
    }

    if verbose {
        eprintln!("  ✓ Search space built");

        eprintln!("\nCompiling graph...");
        eprintln!("\n==== PRE-COMPILE DEBUG ====");
        eprintln!("  About to call cx.search() with NativeRuntime");
        eprintln!(
            "  Graph state: {} nodes, {} edges",
            cx.graph.node_count(),
            cx.graph.edge_count()
        );
        eprintln!("============================");
    }

    let mut runtime = cx.search(NativeRuntime::default(), 1);
    if verbose {
        eprintln!("  ✓ Compilation complete!");

        eprintln!("\nExporting optimized LLIR graph to dot file...");
        match runtime.graph.to_dot() {
            Ok(dot_string) => {
                let filename = "llir_graph_optimized.dot";
                match File::create(filename) {
                    Ok(mut file) => {
                        if let Err(e) = file.write_all(dot_string.as_bytes()) {
                            eprintln!("  ⚠ Failed to write to {}: {}", filename, e);
                        } else {
                            eprintln!("  ✓ Optimized LLIR graph exported to {}", filename);
                            eprintln!("    LLIR nodes: {}", runtime.graph.node_count());
                            eprintln!("    LLIR edges: {}", runtime.graph.edge_count());
                            eprintln!("    View it at: https://dreampuf.github.io/GraphvizOnline/");
                        }
                    }
                    Err(e) => {
                        eprintln!("  ⚠ Failed to create {}: {}", filename, e);
                    }
                }
            }
            Err(e) => {
                eprintln!("  ⚠ Failed to generate dot representation: {}", e);
            }
        }

        eprintln!("\nSetting input data...");
    }
    for (name, data) in &inputs {
        if let Some(tensor) = tensor_map.get(name) {
            if verbose {
                eprintln!(
                    "  Setting data for input: {} (dtype: {:?})",
                    name, tensor.dtype
                );
            }
            match tensor.dtype {
                DType::Int => {
                    let int_data: Vec<i32> = data.iter().map(|&f| f as i32).collect();
                    runtime.set_data(tensor.id, int_data);
                }
                _ => {
                    runtime.set_data(tensor.id, data.clone());
                }
            }
        } else if verbose {
            eprintln!("  WARNING: Input {} not found in tensor_map", name);
        }
    }

    if verbose {
        eprintln!("\nExecuting graph...");
    }
    runtime.execute(&cx.dyn_map);
    if verbose {
        eprintln!("  ✓ Execution complete!");

        eprintln!("\nCollecting outputs...");
    }
    let mut outputs = HashMap::new();
    if verbose {
        eprintln!("  Number of output names: {}", output_names.len());
        if !output_names.is_empty() {
            eprintln!("  Output names: {:?}", output_names);
            eprintln!(
                "  First few tensor map keys: {:?}",
                tensor_map.keys().take(10).collect::<Vec<_>>()
            );
        }
    }
    for name in &output_names {
        if let Some(tensor) = tensor_map.get(name) {
            if verbose {
                eprintln!("  Found tensor {} with id: {:?}", name, tensor.id);
            }
            let output_data = runtime.get_f32(tensor.id).to_vec();
            if verbose {
                eprintln!("  Output {} has {} elements", name, output_data.len());
            }
            outputs.insert(name.clone(), output_data);
        } else if verbose {
            eprintln!("  WARNING: Output {} not found in tensor_map", name);
            eprintln!(
                "    Available keys containing '{}': {:?}",
                name,
                tensor_map
                    .keys()
                    .filter(|k| k.contains(name))
                    .collect::<Vec<_>>()
            );
        }
    }

    if verbose {
        eprintln!("\n✓ Successfully executed HLIR graph!");
        eprintln!(
            "  Built {} HLIR nodes from {} PyTorch nodes",
            cx.graph.node_count(),
            graph_nodes.len()
        );
        eprintln!("  Returned {} outputs", outputs.len());
    }

    Ok(outputs)
}

#[pymodule]
fn luminal_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    Ok(())
}
