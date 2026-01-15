"""Shared test utilities for luminal_python tests"""

import pytest
import torch


def extract_graph_and_inputs(model, x):
    """Extract graph nodes and inputs from a PyTorch model

    Args:
        model: PyTorch model to export
        x: Input tensor for the model

    Returns:
        tuple: (nodes, inputs) where nodes is a list of graph node tuples
               and inputs is a dict of input tensors
    """
    exported = torch.export.export(model, (x,))

    nodes = []
    for node in exported.graph.nodes:
        args = []
        for arg in node.args:
            if hasattr(arg, "name"):
                args.append(arg.name)
            elif isinstance(arg, (list, tuple)):
                for sub_arg in arg:
                    if hasattr(sub_arg, "name"):
                        args.append(sub_arg.name)

        # Extract kwargs - convert values to strings for Rust HashMap<String, String>
        kwargs = {}
        if hasattr(node, "kwargs") and node.kwargs:
            for key, value in node.kwargs.items():
                kwargs[str(key)] = str(value)

        shape = []
        dtype = "f32"
        if hasattr(node, "meta") and "val" in node.meta:
            val = node.meta["val"]
            if hasattr(val, "shape"):
                shape = [int(dim) for dim in val.shape]
            if hasattr(val, "dtype"):
                dtype = str(val.dtype)

        nodes.append((node.name, node.op, str(node.target), args, kwargs, shape, dtype))

    inputs = {"x": x.flatten().tolist()}

    placeholder_nodes = [
        n for n in exported.graph.nodes if n.op == "placeholder" and n.name != "x"
    ]
    params_dict = dict(model.named_parameters())

    for node in placeholder_nodes:
        for param_name, param in params_dict.items():
            normalized_name = param_name.replace(".", "_")
            if normalized_name in node.name:
                inputs[node.name] = param.detach().flatten().tolist()
                break

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
    luminal_output = torch.tensor(outputs[output_key]).reshape(pytorch_output.shape)

    torch.testing.assert_close(luminal_output, pytorch_output, atol=atol, rtol=0.0)
