import torch
import torch.nn as nn

from test_utils import extract_graph_and_inputs, run_luminal_and_compare


def test_transpose_simple():
    """Test simple 2D transpose"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 4)
            return out.transpose(0, 1)  # (4, 2)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_transpose_identity():
    """Test transpose with same dimensions (no-op)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 4)
            return out.transpose(0, 0)  # (2, 4) - no change

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_transpose_double():
    """Test double transpose returns to original"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 4)
            out = out.transpose(0, 1)  # (4, 2)
            return out.transpose(0, 1)  # (2, 4) - back to original

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_transpose_3d_dims01():
    """Test transpose on 3D tensor (dims 0 and 1)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 12)
            out = out.view(2, 3, 4)  # (2, 3, 4)
            return out.transpose(0, 1)  # (3, 2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_transpose_3d_dims02():
    """Test transpose on 3D tensor (dims 0 and 2)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 12)
            out = out.view(2, 3, 4)  # (2, 3, 4)
            return out.transpose(0, 2)  # (4, 3, 2)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_transpose_3d_dims12():
    """Test transpose on 3D tensor (dims 1 and 2)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 12)
            out = out.view(2, 3, 4)  # (2, 3, 4)
            return out.transpose(1, 2)  # (2, 4, 3)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_transpose_with_matmul():
    """Test transpose in matrix multiplication pattern"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 4, bias=False)
            self.linear2 = nn.Linear(4, 2, bias=False)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear2.weight.fill_(0.3)

        def forward(self, x):
            # x shape: (2, 3)
            out1 = self.linear1(x)  # (2, 4)
            out2 = self.linear2(out1)  # (2, 2)
            return out2.transpose(0, 1)  # (2, 2) transposed

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_transpose_with_add():
    """Test transpose with addition"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 4, bias=False)
            self.linear2 = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear2.weight.fill_(0.3)

        def forward(self, x):
            # x shape: (2, 3)
            out1 = self.linear1(x)  # (2, 4)
            out2 = self.linear2(x)  # (2, 4)
            combined = out1 + out2  # (2, 4)
            return combined.transpose(0, 1)  # (4, 2)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_transpose_with_mul():
    """Test transpose with element-wise multiplication"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 4, bias=False)
            self.linear2 = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear2.weight.fill_(0.7)

        def forward(self, x):
            # x shape: (2, 3)
            out1 = self.linear1(x)  # (2, 4)
            out2 = self.linear2(x)  # (2, 4)
            combined = out1 * out2  # (2, 4)
            return combined.transpose(0, 1)  # (4, 2)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_transpose_4d():
    """Test transpose on 4D tensor"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 24, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 24)
            out = out.view(2, 3, 2, 4)  # (2, 3, 2, 4)
            return out.transpose(1, 3)  # (2, 4, 2, 3)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_transpose_negative_dims():
    """Test transpose with negative dimension indices"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 12)
            out = out.view(2, 3, 4)  # (2, 3, 4)
            return out.transpose(-2, -1)  # Same as transpose(1, 2) -> (2, 4, 3)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_transpose_attention_pattern():
    """Test transpose in attention mechanism pattern"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 24, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 24)
            # Reshape for multi-head attention: (batch=2, seq=3, heads=2, head_dim=4)
            out = out.view(2, 3, 2, 4)
            # Transpose to (batch, heads, seq, head_dim) for attention
            return out.transpose(1, 2)  # (2, 2, 3, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_transpose_multiple():
    """Test multiple transpose operations"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 24, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 24)
            out = out.view(2, 3, 4, 2)  # (2, 3, 4, 2)
            out = out.transpose(0, 1)  # (3, 2, 4, 2)
            out = out.transpose(2, 3)  # (3, 2, 2, 4)
            return out.transpose(0, 2)  # (2, 2, 3, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_transpose_with_bias():
    """Test transpose with biased linear layer"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 6, bias=True)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)
                self.linear.bias.fill_(0.1)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 6)
            return out.transpose(0, 1)  # (6, 2)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_transpose_permute_equivalent():
    """Test that transpose is equivalent to specific permute"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 12)
            out = out.view(2, 3, 4)  # (2, 3, 4)
            # transpose(1, 2) should be same as permute(0, 2, 1)
            return out.transpose(1, 2)  # (2, 4, 3)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_transpose_batch_dims():
    """Test transpose maintaining batch dimension"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 20, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 20)
            out = out.view(2, 4, 5)  # (2, 4, 5) - batch_size=2, then 4x5
            # Transpose the non-batch dimensions
            return out.transpose(1, 2)  # (2, 5, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)
