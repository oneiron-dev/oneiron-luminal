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


def test_add_simple():
    """Test simple addition of two linear layer outputs"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 4, bias=False)
            self.linear2 = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear1.weight.fill_(1.0)
                self.linear2.weight.fill_(2.0)

        def forward(self, x):
            a = self.linear1(x)
            b = self.linear2(x)
            return a + b  # Add two tensors

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_add_multiple():
    """Test multiple additions in sequence"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 4, bias=False)
            self.linear2 = nn.Linear(3, 4, bias=False)
            self.linear3 = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear1.weight.fill_(1.0)
                self.linear2.weight.fill_(0.5)
                self.linear3.weight.fill_(0.25)

        def forward(self, x):
            a = self.linear1(x)
            b = self.linear2(x)
            c = self.linear3(x)
            return a + b + c  # Chain multiple additions

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_add_with_bias():
    """Test addition with biased linear layers"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 4, bias=True)
            self.linear2 = nn.Linear(3, 4, bias=True)
            with torch.no_grad():
                self.linear1.weight.fill_(1.0)
                self.linear1.bias.fill_(0.1)
                self.linear2.weight.fill_(2.0)
                self.linear2.bias.fill_(0.2)

        def forward(self, x):
            a = self.linear1(x)
            b = self.linear2(x)
            return a + b

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_add_residual():
    """Test residual connection pattern (x + linear(x))"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 4, bias=False)
            self.linear2 = nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear2.weight.fill_(0.5)

        def forward(self, x):
            # Transform to 4 features first
            out = self.linear1(x)
            # Residual connection: add input to output
            out2 = self.linear2(out)
            return out + out2

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_add_complex_network():
    """Test complex network with multiple additions"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 8, bias=True)
            self.linear2 = nn.Linear(3, 8, bias=True)
            self.linear3 = nn.Linear(8, 4, bias=True)
            self.linear4 = nn.Linear(8, 4, bias=True)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear1.bias.fill_(0.1)
                self.linear2.weight.fill_(0.3)
                self.linear2.bias.fill_(0.05)
                self.linear3.weight.fill_(0.4)
                self.linear3.bias.fill_(0.15)
                self.linear4.weight.fill_(0.2)
                self.linear4.bias.fill_(0.08)

        def forward(self, x):
            # Two parallel paths that get added
            path1 = self.linear1(x)
            path2 = self.linear2(x)
            merged = path1 + path2

            # Split again
            out1 = self.linear3(merged)
            out2 = self.linear4(merged)

            return out1 + out2

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-4)


def test_add_many():
    """Test adding many tensors together"""
    class Model(nn.Module):
        def __init__(self, num_branches=8):
            super().__init__()
            self.branches = nn.ModuleList([
                nn.Linear(3, 4, bias=False) for _ in range(num_branches)
            ])
            with torch.no_grad():
                for i, branch in enumerate(self.branches):
                    # Different weight for each branch
                    branch.weight.fill_(0.1 * (i + 1))

        def forward(self, x):
            # Add all branches together
            result = self.branches[0](x)
            for branch in self.branches[1:]:
                result = result + branch(x)
            return result

    model = Model(num_branches=8)
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-4)
