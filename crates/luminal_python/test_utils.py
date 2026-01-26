"""Shared test utilities for luminal_python tests"""

import pytest
import torch

def extract_graph_and_inputs(model, x):
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

    input_placeholder_count = 0
    for node in placeholder_nodes:
        is_parameter = False
        params_dict = dict(model.named_parameters())
        for param_name, param in params_dict.items():
            normalized_name = param_name.replace(".", "_")
            if normalized_name in node.name:
                inputs[node.name] = param.detach().flatten().tolist()
                is_parameter = True
                break

        if not is_parameter:
            if input_placeholder_count < len(input_tensors):
                inputs[node.name] = input_tensors[input_placeholder_count].flatten().tolist()
                input_placeholder_count += 1

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

    nodes, inputs = extract_graph_and_inputs(model, x)

    luminal_native = pytest.importorskip("luminal_native")
    outputs = luminal_native.compile(nodes, inputs, verbose=verbose)

    output_key = next(iter(outputs.keys()))
    print(f"keys: {output_key}")
    luminal_output = torch.tensor(outputs[output_key]).reshape(pytorch_output.shape)

    torch.testing.assert_close(luminal_output, pytorch_output, atol=atol, rtol=0.0)
