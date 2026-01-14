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


def test_linear_1_layer():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=True)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)
                self.linear.bias.fill_(0.5)

        def forward(self, x):
            return self.linear(x)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_linear_2_layers():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 8, bias=True)
            self.linear2 = nn.Linear(8, 4, bias=True)
            with torch.no_grad():
                self.linear1.weight.fill_(1.0)
                self.linear1.bias.fill_(0.5)
                self.linear2.weight.fill_(1.0)
                self.linear2.bias.fill_(0.5)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_linear_with_bias():
    """Test linear layer specifically for bias functionality"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=True)
            with torch.no_grad():
                # Use simple values to verify bias is added correctly
                self.linear.weight.fill_(1.0)
                self.linear.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

        def forward(self, x):
            return self.linear(x)

    model = Model()
    x = torch.tensor([[1.0, 1.0, 1.0]])  # Simple input: sum should be 3.0

    # Expected: [3.0 + 1.0, 3.0 + 2.0, 3.0 + 3.0, 3.0 + 4.0] = [4.0, 5.0, 6.0, 7.0]
    run_luminal_and_compare(model, x, atol=1e-5)


def test_linear_mixed_bias():
    """Test mixing layers with and without bias"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 8, bias=True)   # With bias
            self.linear2 = nn.Linear(8, 6, bias=False)  # Without bias
            self.linear3 = nn.Linear(6, 4, bias=True)   # With bias
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear1.bias.fill_(0.1)
                self.linear2.weight.fill_(0.5)
                # linear2 has no bias
                self.linear3.weight.fill_(0.5)
                self.linear3.bias.fill_(0.2)

        def forward(self, x):
            x = self.linear1(x)  # Has bias
            x = self.linear2(x)  # No bias
            x = self.linear3(x)  # Has bias
            return x

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_linear_32_layers():
    class Model(nn.Module):
        def __init__(self, num_layers=32):
            super().__init__()
            layers = []

            layers.append(nn.Linear(3, 8, bias=True))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(8, 8, bias=True))
            layers.append(nn.Linear(8, 4, bias=True))

            self.layers = nn.ModuleList(layers)

            with torch.no_grad():
                for layer in self.layers:
                    layer.weight.fill_(0.1)
                    layer.bias.fill_(0.01)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = Model(num_layers=32)
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-4)
