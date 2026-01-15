use luminal::prelude::*;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Represents a node in the exported PyTorch graph
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

/// Main function: Load a PyTorch exported graph, build HLIR, and execute
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

    // Build HLIR graph
    let mut cx = Graph::new();
    let mut tensor_map: HashMap<String, GraphTensor> = HashMap::new();
    let mut output_names: Vec<String> = Vec::new();

    for node in &graph_nodes {
        match node.op.as_str() {
            "placeholder" => {
                if verbose {
                    eprintln!(
                        "  Creating input: {} with shape {:?}",
                        node.name, node.shape
                    );
                }

                // Create tensor with actual shape from PyTorch
                let tensor = match node.shape.len() {
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

                if verbose {
                    eprintln!("    ✓ Created input tensor (shape: {:?})", tensor.shape);
                }
                tensor_map.insert(node.name.clone(), tensor);
            }

            "call_function" => {
                if verbose {
                    eprintln!("  Creating operation: {} ({})", node.name, node.target);
                }

                // Map aten operations to HLIR
                match node.target.as_str() {
                    "aten.add.Tensor" | "aten.add" => {
                        if node.args.len() >= 2 {
                            let left_hand_name = &node.args[0];
                            let right_hand_name = &node.args[1];

                            if let (Some(left_hand), Some(right_hand)) = (
                                tensor_map.get(left_hand_name),
                                tensor_map.get(right_hand_name),
                            ) {
                                if verbose {
                                    eprintln!(
                                        "    -> Computing: {} + {}",
                                        left_hand_name, right_hand_name
                                    );
                                }

                                // Add two tensors (dereference since get() returns &GraphTensor)
                                let output = *left_hand + *right_hand;

                                if verbose {
                                    eprintln!("    ✓ Created tensor addition (HLIR: Add)");
                                    eprintln!("    Output shape: {:?}", output.shape);
                                }
                                tensor_map.insert(node.name.clone(), output);
                            } else if verbose {
                                eprintln!(
                                    "    ERROR: Could not find tensors {} or {}",
                                    left_hand_name, right_hand_name
                                );
                            }
                        }
                    }

                    "aten.mul.Tensor" | "aten.mul" => {
                        if node.args.len() >= 2 {
                            let left_hand_name = &node.args[0];
                            let right_hand_name = &node.args[1];

                            if let (Some(left_hand), Some(right_hand)) = (
                                tensor_map.get(left_hand_name),
                                tensor_map.get(right_hand_name),
                            ) {
                                if verbose {
                                    eprintln!(
                                        "    -> Computing: {} * {}",
                                        left_hand_name, right_hand_name
                                    );
                                }

                                // Multiply two tensors (dereference since get() returns &GraphTensor)
                                let output = *left_hand * *right_hand;

                                if verbose {
                                    eprintln!("    ✓ Created tensor multiplication (HLIR: Mul)");
                                    eprintln!("    Output shape: {:?}", output.shape);
                                }
                                tensor_map.insert(node.name.clone(), output);
                            } else if verbose {
                                eprintln!(
                                    "    ERROR: Could not find tensors {} or {}",
                                    left_hand_name, right_hand_name
                                );
                            }
                        }
                    }

                    "aten.linear.default" => {
                        // linear(input, weight, bias=None)
                        // output = input @ weight.T + bias
                        if node.args.len() >= 2 {
                            let input_name = &node.args[0];
                            let weight_name = &node.args[1];

                            if let (Some(input), Some(weight)) =
                                (tensor_map.get(input_name), tensor_map.get(weight_name))
                            {
                                if verbose {
                                    eprintln!(
                                        "    -> Computing: {} @ {}.T",
                                        input_name, weight_name
                                    );
                                }

                                // Perform matmul with transposed weight: input @ weight.T
                                // PyTorch linear: weight is (out_features, in_features), needs transpose
                                let weight_t = weight.permute((1, 0));
                                let mut output = input.matmul(weight_t);

                                // Add bias if present (args[2])
                                if node.args.len() >= 3 {
                                    let bias_name = &node.args[2];
                                    if let Some(bias) = tensor_map.get(bias_name) {
                                        if verbose {
                                            eprintln!("    -> Adding bias: {}", bias_name);
                                        }
                                        // Bias has shape (out_features,), need to expand to (batch, out_features)
                                        // Get batch size from output (first dimension)
                                        let batch_size = output.dims()[0];
                                        let expanded_bias = bias.expand_lhs([batch_size]);
                                        output = output + expanded_bias;
                                    }
                                }

                                if verbose {
                                    eprintln!(
                                        "    ✓ Created linear layer (HLIR: Mul + SumReduce + Add)"
                                    );
                                    eprintln!("    Output shape: {:?}", output.shape);
                                }
                                tensor_map.insert(node.name.clone(), output);
                            } else if verbose {
                                eprintln!(
                                    "    ERROR: Could not find tensors {} or {}",
                                    input_name, weight_name
                                );
                            }
                        }
                    }

                    // TODO: Implement view/reshape operations
                    "aten.view.default" => {
                        if verbose {
                            eprintln!("    TODO: view operation not yet implemented");
                        }
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

                    // TODO: Implement layer normalization
                    "aten.layer_norm.default" => {
                        if verbose {
                            eprintln!("    TODO: layer_norm operation not yet implemented");
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
                        //
                        // Check kwargs to ensure tanh approximation was requested
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

                    // TODO: Implement embedding lookup
                    "aten.embedding.default" => {
                        if verbose {
                            eprintln!("    TODO: embedding operation not yet implemented");
                        }
                    }

                    _ => {
                        if verbose {
                            eprintln!("    WARNING: Unsupported operation: {}", node.target);
                        }
                    }
                }
            }

            "output" => {
                if verbose {
                    eprintln!("  Marking outputs");
                }
                // Mark all output tensors
                for arg in &node.args {
                    if let Some(tensor) = tensor_map.get(arg).copied() {
                        let output_tensor = tensor.output();
                        tensor_map.insert(arg.clone(), output_tensor);
                        output_names.push(arg.clone());
                        if verbose {
                            eprintln!("    ✓ Marked {} as output", arg);
                        }
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

                // Create tensor for parameters (weights, biases)
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

    // Display graph summary
    if verbose {
        eprintln!("\n✓ Successfully built HLIR graph!");
        eprintln!("  Tensors: {}", tensor_map.len());
        eprintln!("  HLIR nodes: {}", cx.graph.node_count());
        eprintln!("  HLIR edges: {}", cx.graph.edge_count());

        // Build search space to populate operation metadata
        eprintln!("\nBuilding search space (populates op metadata)...");
    }
    cx.build_search_space::<NativeRuntime>();
    if verbose {
        eprintln!("  ✓ Search space built");

        // Compile and get runtime
        eprintln!("\nCompiling graph...");
    }
    let mut runtime = cx.search(NativeRuntime::default(), 1);
    if verbose {
        eprintln!("  ✓ Compilation complete!");

        // Set input data
        eprintln!("\nSetting input data...");
    }
    for (name, data) in &inputs {
        if let Some(tensor) = tensor_map.get(name) {
            if verbose {
                eprintln!("  Setting data for input: {}", name);
            }
            runtime.set_data(tensor.id, data.clone());
        } else if verbose {
            eprintln!("  WARNING: Input {} not found in tensor_map", name);
        }
    }

    // Execute
    if verbose {
        eprintln!("\nExecuting graph...");
    }
    runtime.execute(&cx.dyn_map);
    if verbose {
        eprintln!("  ✓ Execution complete!");

        // Get outputs
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

/// The Python module
#[pymodule]
fn luminal_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    Ok(())
}
