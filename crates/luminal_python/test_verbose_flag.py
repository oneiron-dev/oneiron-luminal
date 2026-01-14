# test_verbose_flag.py
import pytest
import torch
import torch.nn as nn


def extract_graph_and_inputs(model, x):
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

        shape = []
        dtype = "f32"
        if hasattr(node, "meta") and "val" in node.meta:
            val = node.meta["val"]
            if hasattr(val, "shape"):
                shape = [int(dim) for dim in val.shape]
            if hasattr(val, "dtype"):
                dtype = str(val.dtype)

        nodes.append((node.name, node.op, str(node.target), args, shape, dtype))

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


def test_compile_verbose_produces_more_output(capfd):
    luminal_native = pytest.importorskip("luminal_native")

    class SimpleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)

        def forward(self, x):
            return self.linear(x)

    model = SimpleLinear()
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    nodes, inputs = extract_graph_and_inputs(model, x)

    luminal_native.compile(nodes, inputs, verbose=False)
    out1, err1 = capfd.readouterr()
    text1 = out1 + err1

    luminal_native.compile(nodes, inputs, verbose=True)
    out2, err2 = capfd.readouterr()
    text2 = out2 + err2

    assert len(text2) > len(text1), (
        "Expected verbose=True to produce more output than verbose=False.\n"
        f"len(non_verbose)={len(text1)} len(verbose)={len(text2)}\n"
        "Non-verbose output:\n" + text1 + "\n"
        "Verbose output:\n" + text2
    )
