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

                    // TODO: Implement transpose
                    "aten.transpose.int" => {
                        if verbose {
                            eprintln!("    TODO: transpose operation not yet implemented");
                        }
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

                    // TODO: Implement tensor split
                    "aten.split.Tensor" => {
                        if verbose {
                            eprintln!("    TODO: split operation not yet implemented");
                        }
                    }

                    // TODO: Implement scaled dot product attention
                    "aten.scaled_dot_product_attention.default" => {
                        if verbose {
                            eprintln!(
                                "    TODO: scaled_dot_product_attention operation not yet implemented"
                            );
                        }
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
    }
    cx.build_search_space::<NativeRuntime>();
    if verbose {
        eprintln!("  ✓ Search space built");

        eprintln!("\nCompiling graph...");
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
    for name in &output_names {
        if let Some(tensor) = tensor_map.get(name) {
            let output_data = runtime.get_f32(tensor.id).to_vec();
            if verbose {
                eprintln!("  Output {} has {} elements", name, output_data.len());
            }
            outputs.insert(name.clone(), output_data);
        } else if verbose {
            eprintln!("  WARNING: Output {} not found in tensor_map", name);
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
