import torch
import torch.nn as nn

from test_utils import extract_graph_and_inputs, run_luminal_and_compare


def test_view_flatten():
    """Test simple flatten using view"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 12)
            return out.view(2, 12)  # Explicit reshape (no-op in this case)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_view_reshape_2d():
    """Test reshaping output into different 2D shape"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 12)
            return out.view(4, 6)  # Reshape to (4, 6)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_view_inferred_dim():
    """Test view with -1 for inferred dimension"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 12)
            return out.view(-1, 6)  # Reshape to (?, 6) - infers first dim as 4

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_view_flatten_all():
    """Test flattening to 1D using view(-1)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 8, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 8)
            return out.view(-1)  # Flatten to (16,)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_view_between_layers():
    """Test view operation between two linear layers"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 12, bias=False)
            self.linear2 = nn.Linear(6, 4, bias=False)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear2.weight.fill_(0.8)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear1(x)  # out shape: (2, 12)
            out = out.view(4, 6)  # Reshape to (4, 6)
            return self.linear2(out)  # out shape: (4, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_view_3d_reshape():
    """Test reshaping to 3D tensor"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 24, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 24)
            return out.view(2, 4, 6)  # Reshape to (2, 4, 6)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_view_multiple():
    """Test multiple view operations in sequence"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 24, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 24)
            out = out.view(6, 8)  # Reshape to (6, 8)
            out = out.view(12, 4)  # Reshape to (12, 4)
            return out.view(2, 24)  # Back to (2, 24)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_view_with_batch():
    """Test view preserving batch dimension"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 16, bias=True)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)
                self.linear.bias.fill_(0.1)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 16)
            # Reshape keeping batch dim: (batch, 4, 4)
            return out.view(2, 4, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_view_attention_pattern():
    """Test view pattern common in attention mechanisms"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            # Multi-head attention-like pattern
            self.qkv = nn.Linear(3, 24, bias=False)  # 3 heads * 8 dim
            with torch.no_grad():
                self.qkv.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            qkv = self.qkv(x)  # shape: (2, 24)
            # Reshape for multi-head: (batch, seq_len=2, num_heads=3, head_dim=4)
            # Here we treat the batch*seq as flattened, so reshape to (2, 3, 4)
            return qkv.view(2, 3, 8)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_view_inferred_last_dim():
    """Test view with -1 for inferred last dimension"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 12)
            return out.view(3, -1)  # Reshape to (3, ?) - infers last dim as 8

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_view_inferred_middle_dim():
    """Test view with -1 for inferred middle dimension"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 24, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 24)
            return out.view(2, -1, 4)  # Reshape to (2, ?, 4) - infers middle dim as 6

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_view_with_arithmetic():
    """Test view combined with arithmetic operations"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 12, bias=False)
            self.linear2 = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear2.weight.fill_(0.3)

        def forward(self, x):
            # x shape: (2, 3)
            out1 = self.linear1(x)  # out shape: (2, 12)
            out2 = self.linear2(x)  # out shape: (2, 12)

            # Add them
            combined = out1 + out2  # (2, 12)

            # Reshape
            reshaped = combined.view(4, 6)  # (4, 6)

            return reshaped

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_view_with_multiplication():
    """Test view with element-wise multiplication"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 12, bias=False)
            self.linear2 = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear2.weight.fill_(0.8)

        def forward(self, x):
            # x shape: (2, 3)
            out1 = self.linear1(x)  # out shape: (2, 12)
            out2 = self.linear2(x)  # out shape: (2, 12)

            # Multiply them
            combined = out1 * out2  # (2, 12)

            # Reshape to 3D
            reshaped = combined.view(2, 3, 4)  # (2, 3, 4)

            return reshaped

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_view_4d():
    """Test view with 4D tensor (common in convolutions)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 48, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 48)
            # Reshape to 4D like (batch, channels, height, width)
            return out.view(2, 3, 4, 4)  # (2, 3, 4, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_view_4d_to_2d():
    """Test view from 4D back to 2D (flatten spatial dimensions)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 48, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 48)
            # First reshape to 4D
            out = out.view(2, 3, 4, 4)  # (2, 3, 4, 4)
            # Then flatten back to 2D
            return out.view(2, -1)  # (2, 48)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_view_powers_of_two():
    """Test view with power-of-2 dimensions (common in neural networks)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 64, bias=True)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)
                self.linear.bias.fill_(0.1)

        def forward(self, x):
            # x shape: (2, 4)
            out = self.linear(x)  # out shape: (2, 64)
            # Reshape using power-of-2 dimensions
            return out.view(2, 8, 8)  # (2, 8, 8)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_view_chain_with_layers():
    """Test multiple views interleaved with linear layers"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 12, bias=False)
            self.linear2 = nn.Linear(4, 8, bias=False)
            self.linear3 = nn.Linear(8, 6, bias=False)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear2.weight.fill_(0.7)
                self.linear3.weight.fill_(0.6)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear1(x)  # (2, 12)
            out = out.view(6, 4)   # (6, 4)
            out = self.linear2(out) # (6, 8)
            out = out.view(12, 4)  # (12, 4)
            out = out.view(6, 8)   # (6, 8)
            out = self.linear3(out) # (6, 6)
            return out.view(3, 12) # (3, 12)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_view_single_element_dim():
    """Test view with dimension of size 1"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 12)
            # Reshape with a dimension of size 1
            return out.view(2, 1, 12)  # (2, 1, 12)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)
