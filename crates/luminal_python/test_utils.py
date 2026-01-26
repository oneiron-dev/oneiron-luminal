"""Shared test utilities for luminal_python tests"""

import pytest
import torch

def extract_graph_and_inputs(model, x, verbose=False):
    # Handle both single tensor and tuple of tensors
    if isinstance(x, tuple):
        input_tensors = x
        exported = torch.export.export(model, input_tensors)
    else:
        input_tensors = (x,)
        exported = torch.export.export(model, (x,))

    nodes = []

    def _flatten_arg(arg, out):
        """Append arg(s) to out as strings, matching existing Rust expectations."""
        if hasattr(arg, "name"):
            out.append(arg.name)
        elif arg is None:
            out.append("None")
        elif isinstance(arg, (int, float, bool, str)):
            out.append(str(arg))
        elif isinstance(arg, (list, tuple)):
            for sub in arg:
                _flatten_arg(sub, out)
        else:
            # fallback: keep string form
            out.append(str(arg))

    for node in exported.graph.nodes:
        # ---- args: keep current flattened positional behavior ----
        args = []
        for arg in node.args:
            _flatten_arg(arg, args)

        # ---- kwargs: keep existing behavior, but allow adding extra metadata ----
        kwargs = {}
        if hasattr(node, "kwargs") and node.kwargs:
            for key, value in node.kwargs.items():
                kwargs[str(key)] = str(value)

        # ---- NEW: extra metadata for aten.index.Tensor (without changing args) ----
        if str(node.target) == "aten.index.Tensor":
            # Export usually has (input, indices) where indices is a tuple/list like (None, idx_node, None)
            # BUT since args are flattened we can’t recover dim position from args alone.
            # So inspect the structured node.args directly here and stash the results in kwargs.
            if len(node.args) >= 2:
                indices_obj = node.args[1]

                index_dim = None
                index_tensor = None
                spec_parts = []

                if isinstance(indices_obj, (list, tuple)):
                    for i, item in enumerate(indices_obj):
                        if item is None:
                            spec_parts.append("None")
                            continue
                        if hasattr(item, "name"):
                            spec_parts.append(item.name)
                            # pick the first non-None tensor index
                            if index_dim is None:
                                index_dim = i
                                index_tensor = item.name
                        else:
                            spec_parts.append(str(item))
                else:
                    # Sometimes it’s not a tuple/list; just record it for debug
                    spec_parts.append(str(indices_obj))

                if index_dim is not None and index_tensor is not None:
                    kwargs["index_dim"] = str(index_dim)
                    kwargs["index_tensor"] = str(index_tensor)
                kwargs["index_spec"] = ",".join(spec_parts)

        # ---- meta: shape + dtype ----
        shape = []
        dtype = "f32"
        if hasattr(node, "meta") and "val" in node.meta:
            val = node.meta["val"]
            if hasattr(val, "shape"):
                shape = [int(dim) for dim in val.shape]
            if hasattr(val, "dtype"):
                dtype = str(val.dtype)

        nodes.append((node.name, node.op, str(node.target), args, kwargs, shape, dtype))

    # ---- Collect input placeholders from the graph (unchanged) ----
    inputs = {}
    placeholder_nodes = [n for n in exported.graph.nodes if n.op == "placeholder"]

    if verbose:
        print(f"DEBUG: Processing {len(placeholder_nodes)} placeholder nodes")
        print(f"DEBUG: Available input tensors: {len(input_tensors)}")
        print(f"DEBUG: Placeholder node names in order:")
        for i, node in enumerate(placeholder_nodes):
            print(f"  {i}: {node.name}")

    input_placeholder_count = 0
    params_dict = dict(model.named_parameters())

    for node in placeholder_nodes:
        is_parameter = False

        # First check if it's a parameter placeholder (starts with 'p_')
        if node.name.startswith('p_'):
            # Try to find matching parameter
            for param_name, param in params_dict.items():
                normalized_name = 'p_' + param_name.replace(".", "_")
                if node.name == normalized_name:
                    inputs[node.name] = param.detach().flatten().tolist()
                    is_parameter = True
                    if verbose:
                        print(f"  {node.name}: Parameter (shape={list(param.shape)})")
                    break

            # If not found in named_parameters but starts with 'p_', it might be weight-tied
            # Use the embedding weight for lm_head (weight tying in GPT)
            if not is_parameter:
                if 'lm_head' in node.name and 'transformer.wte.weight' in params_dict:
                    # Use transformer.wte.weight for lm_head due to weight tying
                    param = params_dict['transformer.wte.weight']
                    inputs[node.name] = param.detach().flatten().tolist()
                    is_parameter = True
                    if verbose:
                        print(f"  {node.name}: Weight-tied parameter (shape={list(param.shape)})")
                elif hasattr(node, 'meta') and 'val' in node.meta:
                    # If we have metadata, use it
                    val = node.meta['val']
                    if hasattr(val, 'detach'):
                        inputs[node.name] = val.detach().flatten().tolist()
                        is_parameter = True
                        if verbose:
                            print(f"  {node.name}: Parameter from metadata")
                else:
                    # This shouldn't happen but let's handle it gracefully
                    print(f"WARNING: Parameter placeholder '{node.name}' not found in model")
                    is_parameter = True  # Mark as parameter to avoid consuming input tensor

        if not is_parameter:
            # Handle lifted constant tensors (e.g., c_lifted_tensor_0)
            if node.name.startswith("c_lifted_tensor"):
                # These are constants that PyTorch lifted to placeholders
                # For arange operations, this is typically 0
                if hasattr(node, "meta") and "val" in node.meta:
                    val = node.meta["val"]
                    if hasattr(val, "item"):
                        # Single value tensor
                        inputs[node.name] = [float(val.item())]
                        if verbose:
                            print(f"  {node.name}: Lifted constant = {val.item()}")
                    elif hasattr(val, "numpy"):
                        # Convert to list
                        inputs[node.name] = val.detach().numpy().flatten().tolist()
                        if verbose:
                            print(f"  {node.name}: Lifted constant array")
                    else:
                        # Default to 0 for arange start value
                        inputs[node.name] = [0.0]
                        if verbose:
                            print(f"  {node.name}: Lifted constant (default=0)")
                else:
                    # Default to 0 for constants without metadata
                    inputs[node.name] = [0.0]
                    if verbose:
                        print(f"  {node.name}: Lifted constant (no metadata, default=0)")
            elif input_placeholder_count < len(input_tensors):
                # This should handle regular input tensors like 'idx'
                inputs[node.name] = input_tensors[input_placeholder_count].flatten().tolist()
                if verbose:
                    print(f"  {node.name}: Input tensor {input_placeholder_count} (shape={list(input_tensors[input_placeholder_count].shape)})")
                input_placeholder_count += 1
            else:
                # If we've run out of input tensors but still have placeholders, print a warning
                print(f"WARNING: Placeholder '{node.name}' has no corresponding input tensor")
                print(f"  Available input tensors: {len(input_tensors)}")
                print(f"  Current placeholder count: {input_placeholder_count}")

    return nodes, inputs



def run_luminal_and_compare(model, x, atol, verbose=False):
    """Run model through both PyTorch and Luminal and compare results

    Args:
        model: PyTorch model to test
        x: Input tensor
        atol: Absolute tolerance for comparison
    """
    with torch.no_grad():
        pytorch_output = model(x)

    nodes, inputs = extract_graph_and_inputs(model, x, verbose=verbose)

    luminal_native = pytest.importorskip("luminal_native")
    outputs = luminal_native.compile(nodes, inputs, verbose=verbose)

    output_key = next(iter(outputs.keys()))
    print(f"keys: {output_key}")
    luminal_output = torch.tensor(outputs[output_key]).reshape(pytorch_output.shape)

    torch.testing.assert_close(luminal_output, pytorch_output, atol=atol, rtol=0.0)
