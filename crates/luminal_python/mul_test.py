import torch
import torch.nn as nn

from test_utils import extract_graph_and_inputs, run_luminal_and_compare


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
