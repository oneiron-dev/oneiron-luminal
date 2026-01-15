import pytest
import torch
import torch.nn as nn

from test_utils import extract_graph_and_inputs, run_luminal_and_compare

def test_gelu_simple():
    """Test GELU activation after linear layer"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)

        def forward(self, x):
            x = self.linear(x)
            return torch.nn.functional.gelu(x, approximate="tanh")  # Default exact GELU

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_gelu_with_bias():
    """Test GELU with biased linear layer"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=True)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)
                self.linear.bias.fill_(0.5)

        def forward(self, x):
            x = self.linear(x)
            return torch.nn.functional.gelu(x, approximate="tanh")

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_gelu_mlp():
    """Test GELU in MLP pattern (linear -> gelu -> linear)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 8, bias=True)
            self.linear2 = nn.Linear(8, 4, bias=True)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear1.bias.fill_(0.1)
                self.linear2.weight.fill_(0.5)
                self.linear2.bias.fill_(0.1)

        def forward(self, x):
            x = self.linear1(x)
            x = torch.nn.functional.gelu(x, approximate="tanh")
            x = self.linear2(x)
            return x

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_gelu_residual():
    """Test GELU with residual connection"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 4, bias=True)
            self.linear2 = nn.Linear(4, 4, bias=True)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear1.bias.fill_(0.05)
                self.linear2.weight.fill_(0.5)
                self.linear2.bias.fill_(0.05)

        def forward(self, x):
            identity = x
            x = self.linear1(x)
            x = torch.nn.functional.gelu(x, approximate="tanh")
            x = self.linear2(x)
            return x + identity  # Residual connection

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_gelu_negative_inputs():
    """Test GELU with negative input values"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=True)
            with torch.no_grad():
                # Use weights that will produce negative values
                self.linear.weight.fill_(-0.5)
                self.linear.bias.fill_(-1.0)

        def forward(self, x):
            x = self.linear(x)
            return torch.nn.functional.gelu(x, approximate="tanh")

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_gelu_deep_network():
    """Test GELU in deep network with multiple activations"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 8, bias=True)
            self.linear2 = nn.Linear(8, 8, bias=True)
            self.linear3 = nn.Linear(8, 8, bias=True)
            self.linear4 = nn.Linear(8, 4, bias=True)

            with torch.no_grad():
                for layer in [self.linear1, self.linear2, self.linear3, self.linear4]:
                    layer.weight.fill_(0.3)
                    layer.bias.fill_(0.05)

        def forward(self, x):
            x = torch.nn.functional.gelu(self.linear1(x), approximate="tanh")
            x = torch.nn.functional.gelu(self.linear2(x), approximate="tanh")
            x = torch.nn.functional.gelu(self.linear3(x), approximate="tanh")
            x = self.linear4(x)
            return x

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_gelu_range():
    """Test GELU across a range of input values"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 5, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)

        def forward(self, x):
            x = self.linear(x)
            return torch.nn.functional.gelu(x, approximate="tanh")

    model = Model()
    # Test range from negative to positive
    x = torch.tensor([[-5.0, -2.5, 0.0, 2.5, 5.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_gelu_exact_not_supported():
    """Test that exact GELU (without approximate='tanh') raises an error"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)

        def forward(self, x):
            x = self.linear(x)
            return torch.nn.functional.gelu(x)  # Default exact GELU - should fail

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Extract graph and inputs
    nodes, inputs = extract_graph_and_inputs(model, x)

    # This should raise NotImplementedError
    luminal_native = pytest.importorskip("luminal_native")
    with pytest.raises(NotImplementedError) as exc_info:
        luminal_native.compile(nodes, inputs, verbose=False)

    # Verify the error message contains helpful guidance
    error_msg = str(exc_info.value)
    assert "approximate='tanh'" in error_msg
    assert "not supported" in error_msg.lower()
