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


def run_luminal_and_compare(model, x, atol):
    with torch.no_grad():
        pytorch_output = model(x)

    nodes, inputs = extract_graph_and_inputs(model, x)

    luminal_native = pytest.importorskip("luminal_native")
    outputs = luminal_native.compile(nodes, inputs, verbose=False)

    output_key = next(iter(outputs.keys()))
    luminal_output = torch.tensor(outputs[output_key]).reshape(pytorch_output.shape)

    torch.testing.assert_close(luminal_output, pytorch_output, atol=atol, rtol=0.0)


def test_mul_simple():
    """Test simple element-wise multiplication of two linear layer outputs"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 4, bias=False)
            self.linear2 = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear1.weight.fill_(2.0)
                self.linear2.weight.fill_(3.0)

        def forward(self, x):
            a = self.linear1(x)
            b = self.linear2(x)
            return a * b  # Element-wise multiplication

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_mul_multiple():
    """Test multiple multiplications in sequence"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 4, bias=False)
            self.linear2 = nn.Linear(3, 4, bias=False)
            self.linear3 = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear1.weight.fill_(2.0)
                self.linear2.weight.fill_(1.5)
                self.linear3.weight.fill_(0.5)

        def forward(self, x):
            a = self.linear1(x)
            b = self.linear2(x)
            c = self.linear3(x)
            return a * b * c  # Chain multiple multiplications

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_mul_with_bias():
    """Test multiplication with biased linear layers"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 4, bias=True)
            self.linear2 = nn.Linear(3, 4, bias=True)
            with torch.no_grad():
                self.linear1.weight.fill_(1.5)
                self.linear1.bias.fill_(0.1)
                self.linear2.weight.fill_(2.0)
                self.linear2.bias.fill_(0.2)

        def forward(self, x):
            a = self.linear1(x)
            b = self.linear2(x)
            return a * b

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_mul_gating():
    """Test gating mechanism (common in GLU, gated linear units)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Linear(3, 8, bias=False)
            self.gate = nn.Linear(3, 8, bias=False)
            with torch.no_grad():
                self.value.weight.fill_(1.0)
                self.gate.weight.fill_(0.5)

        def forward(self, x):
            value = self.value(x)
            gate = self.gate(x)
            return value * gate  # Gating pattern

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_mul_add_combined():
    """Test combining multiplication and addition (common pattern)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 4, bias=False)
            self.linear2 = nn.Linear(3, 4, bias=False)
            self.linear3 = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear1.weight.fill_(1.0)
                self.linear2.weight.fill_(0.5)
                self.linear3.weight.fill_(2.0)

        def forward(self, x):
            a = self.linear1(x)
            b = self.linear2(x)
            c = self.linear3(x)
            # Combine multiplication and addition: (a * b) + c
            return (a * b) + c

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_mul_parallel_paths():
    """Test multiplication in parallel network paths"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            # First path
            self.path1_linear1 = nn.Linear(3, 8, bias=True)
            self.path1_linear2 = nn.Linear(8, 4, bias=True)
            # Second path
            self.path2_linear1 = nn.Linear(3, 8, bias=True)
            self.path2_linear2 = nn.Linear(8, 4, bias=True)

            with torch.no_grad():
                self.path1_linear1.weight.fill_(0.8)
                self.path1_linear1.bias.fill_(0.05)
                self.path1_linear2.weight.fill_(0.6)
                self.path1_linear2.bias.fill_(0.1)

                self.path2_linear1.weight.fill_(0.7)
                self.path2_linear1.bias.fill_(0.08)
                self.path2_linear2.weight.fill_(0.5)
                self.path2_linear2.bias.fill_(0.12)

        def forward(self, x):
            # Two parallel paths
            path1 = self.path1_linear2(self.path1_linear1(x))
            path2 = self.path2_linear2(self.path2_linear1(x))

            # Multiply paths together
            return path1 * path2

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-4)


def test_mul_swish_approximation():
    """Test multiplication pattern similar to SwiGLU/Swish gating"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(3, 8, bias=False)
            self.up_proj = nn.Linear(3, 8, bias=False)
            self.down_proj = nn.Linear(8, 4, bias=False)
            with torch.no_grad():
                self.gate_proj.weight.fill_(0.8)
                self.up_proj.weight.fill_(0.6)
                self.down_proj.weight.fill_(0.5)

        def forward(self, x):
            # Pattern: (gate * up) @ down (simplified SwiGLU without activation)
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            gated = gate * up
            return self.down_proj(gated)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)
