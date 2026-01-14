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
    shape: Vec<usize>,
    dtype: String,
}

/// Main function: Load a PyTorch exported graph, build HLIR, and execute
#[pyfunction]
#[pyo3(signature = (nodes, inputs, verbose=false))]
fn compile(
    nodes: Vec<(String, String, String, Vec<String>, Vec<usize>, String)>,
    inputs: HashMap<String, Vec<f32>>,
    verbose: bool,
) -> PyResult<HashMap<String, Vec<f32>>> {
    if verbose {
        eprintln!("Loading PyTorch graph with {} nodes", nodes.len());
    }

    let mut graph_nodes: Vec<PyTorchGraphNode> = Vec::new();

    for (name, op, target, args, shape, dtype) in nodes {
        if verbose {
            eprintln!("  Node: {} | op={} | target={}", name, op, target);
        }
        graph_nodes.push(PyTorchGraphNode {
            name,
            op,
            target,
            args,
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
                        if verbose && node.args.len() >= 2 {
                            eprintln!("add.Tensor!");
                        }
                    }

                    "aten.mul.Tensor" | "aten.mul" => {
                        if verbose && node.args.len() >= 2 {
                            eprintln!("mul.Tensor!");
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
