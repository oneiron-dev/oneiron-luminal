import torch
import torch.nn as nn

from test_utils import extract_graph_and_inputs, run_luminal_and_compare


def test_layernorm_simple():
    """Test simple LayerNorm with default parameters"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layernorm = nn.LayerNorm(4)
            with torch.no_grad():
                self.layernorm.weight.fill_(1.0)
                self.layernorm.bias.fill_(0.0)

        def forward(self, x):
            return self.layernorm(x)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_layernorm_with_affine():
    """Test LayerNorm with learnable affine parameters"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layernorm = nn.LayerNorm(4, elementwise_affine=True)
            with torch.no_grad():
                # Set weight (gamma) and bias (beta) to specific values
                self.layernorm.weight.copy_(torch.tensor([1.0, 2.0, 0.5, 1.5]))
                self.layernorm.bias.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))

        def forward(self, x):
            return self.layernorm(x)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_layernorm_no_affine():
    """Test LayerNorm without learnable parameters"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layernorm = nn.LayerNorm(4, elementwise_affine=False)

        def forward(self, x):
            return self.layernorm(x)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_layernorm_after_linear():
    """Test LayerNorm after linear layer (common pattern)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=True)
            self.layernorm = nn.LayerNorm(4)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)
                self.linear.bias.fill_(0.1)
                self.layernorm.weight.fill_(1.0)
                self.layernorm.bias.fill_(0.0)

        def forward(self, x):
            x = self.linear(x)
            x = self.layernorm(x)
            return x

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_layernorm_in_mlp():
    """Test LayerNorm in MLP pattern (linear -> layernorm -> linear)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 8, bias=True)
            self.layernorm = nn.LayerNorm(8)
            self.linear2 = nn.Linear(8, 4, bias=True)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear1.bias.fill_(0.1)
                self.layernorm.weight.fill_(1.0)
                self.layernorm.bias.fill_(0.0)
                self.linear2.weight.fill_(0.5)
                self.linear2.bias.fill_(0.1)

        def forward(self, x):
            x = self.linear1(x)
            x = self.layernorm(x)
            x = self.linear2(x)
            return x

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_layernorm_with_gelu():
    """Test LayerNorm with GELU activation"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 8, bias=True)
            self.layernorm = nn.LayerNorm(8)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)
                self.linear.bias.fill_(0.1)
                self.layernorm.weight.fill_(1.0)
                self.layernorm.bias.fill_(0.0)

        def forward(self, x):
            x = self.linear(x)
            x = self.layernorm(x)
            x = torch.nn.functional.gelu(x, approximate="tanh")
            return x

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_layernorm_residual():
    """Test LayerNorm with residual connection"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 4, bias=True)
            self.layernorm = nn.LayerNorm(4)
            self.linear2 = nn.Linear(4, 4, bias=True)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear1.bias.fill_(0.05)
                self.layernorm.weight.fill_(1.0)
                self.layernorm.bias.fill_(0.0)
                self.linear2.weight.fill_(0.5)
                self.linear2.bias.fill_(0.05)

        def forward(self, x):
            identity = x
            x = self.linear1(x)
            x = torch.nn.functional.gelu(x, approximate="tanh")
            x = self.linear2(x)
            x = self.layernorm(x + identity)  # Post-LayerNorm residual
            return x

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_layernorm_multiple():
    """Test multiple LayerNorms in sequence"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 8, bias=True)
            self.layernorm1 = nn.LayerNorm(8)
            self.linear2 = nn.Linear(8, 8, bias=True)
            self.layernorm2 = nn.LayerNorm(8)
            self.linear3 = nn.Linear(8, 4, bias=True)

            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear1.bias.fill_(0.1)
                self.layernorm1.weight.fill_(1.0)
                self.layernorm1.bias.fill_(0.0)
                self.linear2.weight.fill_(0.5)
                self.linear2.bias.fill_(0.1)
                self.layernorm2.weight.fill_(1.0)
                self.layernorm2.bias.fill_(0.0)
                self.linear3.weight.fill_(0.5)
                self.linear3.bias.fill_(0.1)

        def forward(self, x):
            x = self.linear1(x)
            x = self.layernorm1(x)
            x = torch.nn.functional.gelu(x, approximate="tanh")
            x = self.linear2(x)
            x = self.layernorm2(x)
            x = torch.nn.functional.gelu(x, approximate="tanh")
            x = self.linear3(x)
            return x

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_layernorm_different_shapes():
    """Test LayerNorm with different normalized shapes"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            # Normalize over last dimension only
            self.layernorm = nn.LayerNorm(6)
            with torch.no_grad():
                self.layernorm.weight.fill_(1.0)
                self.layernorm.bias.fill_(0.0)

        def forward(self, x):
            return self.layernorm(x)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_layernorm_negative_values():
    """Test LayerNorm with negative input values"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=True)
            self.layernorm = nn.LayerNorm(4)
            with torch.no_grad():
                # Use weights that will produce negative values
                self.linear.weight.fill_(-0.5)
                self.linear.bias.fill_(-1.0)
                self.layernorm.weight.fill_(1.0)
                self.layernorm.bias.fill_(0.0)

        def forward(self, x):
            x = self.linear(x)
            x = self.layernorm(x)
            return x

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_layernorm_varied_affine():
    """Test LayerNorm with varied affine parameters"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layernorm = nn.LayerNorm(4)
            with torch.no_grad():
                # Use non-uniform weights and biases
                self.layernorm.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
                self.layernorm.bias.copy_(torch.tensor([-0.5, 0.0, 0.5, 1.0]))

        def forward(self, x):
            return self.layernorm(x)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_layernorm_transformer_block():
    """Test LayerNorm in transformer-like block"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            # Simplified transformer block: Linear -> LayerNorm -> MLP -> LayerNorm
            self.linear1 = nn.Linear(4, 4, bias=True)
            self.layernorm1 = nn.LayerNorm(4)
            self.mlp1 = nn.Linear(4, 8, bias=True)
            self.mlp2 = nn.Linear(8, 4, bias=True)
            self.layernorm2 = nn.LayerNorm(4)

            with torch.no_grad():
                self.linear1.weight.fill_(0.3)
                self.linear1.bias.fill_(0.05)
                self.layernorm1.weight.fill_(1.0)
                self.layernorm1.bias.fill_(0.0)
                self.mlp1.weight.fill_(0.3)
                self.mlp1.bias.fill_(0.05)
                self.mlp2.weight.fill_(0.3)
                self.mlp2.bias.fill_(0.05)
                self.layernorm2.weight.fill_(1.0)
                self.layernorm2.bias.fill_(0.0)

        def forward(self, x):
            # First sublayer with residual
            identity = x
            x = self.linear1(x)
            x = self.layernorm1(x + identity)

            # MLP sublayer with residual
            identity = x
            x = self.mlp1(x)
            x = torch.nn.functional.gelu(x, approximate="tanh")
            x = self.mlp2(x)
            x = self.layernorm2(x + identity)

            return x

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    run_luminal_and_compare(model, x, atol=1e-5)
