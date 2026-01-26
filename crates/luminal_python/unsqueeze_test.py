import torch
import torch.nn as nn

from test_utils import extract_graph_and_inputs, run_luminal_and_compare


def test_unsqueeze_simple():
    """Test simple unsqueeze adding dimension at position 0"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 4)
            return out.unsqueeze(0)  # (1, 2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_unsqueeze_dim1():
    """Test unsqueeze adding dimension at position 1"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 4)
            return out.unsqueeze(1)  # (2, 1, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_unsqueeze_last_dim():
    """Test unsqueeze adding dimension at last position"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 4)
            return out.unsqueeze(2)  # (2, 4, 1)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_unsqueeze_negative_dim():
    """Test unsqueeze with negative dimension"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 4)
            return out.unsqueeze(-1)  # (2, 4, 1) - last dimension

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_unsqueeze_multiple():
    """Test multiple unsqueeze operations"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 4)
            out = out.unsqueeze(0)  # (1, 2, 4)
            out = out.unsqueeze(3)  # (1, 2, 4, 1)
            return out

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_unsqueeze_between_layers():
    """Test unsqueeze between linear layers"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 4, bias=False)
            self.linear2 = nn.Linear(4, 6, bias=False)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear2.weight.fill_(0.7)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear1(x)  # (2, 4)
            out = out.unsqueeze(1)  # (2, 1, 4)
            # Process each element in middle dim
            batch, _, features = out.shape
            out = out.view(batch, features)  # (2, 4)
            return self.linear2(out)  # (2, 6)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_unsqueeze_with_add():
    """Test unsqueeze with addition"""
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
            return combined.unsqueeze(1)  # (2, 1, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_unsqueeze_with_mul():
    """Test unsqueeze with multiplication"""
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
            return combined.unsqueeze(0)  # (1, 2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_unsqueeze_before_transpose():
    """Test unsqueeze followed by transpose"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 4)
            out = out.unsqueeze(0)  # (1, 2, 4)
            return out.transpose(0, 1)  # (2, 1, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_unsqueeze_attention_pattern():
    """Test unsqueeze in attention-like pattern (adding head dimension)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 8, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 8)
            # Add head dimension for attention: (batch, heads, seq, features)
            # First view as (2, 2, 4) then add head dim
            out = out.view(2, 2, 4)  # (2, 2, 4)
            out = out.unsqueeze(1)  # (2, 1, 2, 4) - single head
            return out

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_unsqueeze_with_view():
    """Test unsqueeze followed by view"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 12)
            out = out.unsqueeze(1)  # (2, 1, 12)
            return out.view(2, 3, 4)  # (2, 3, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_unsqueeze_with_bias():
    """Test unsqueeze with biased linear layer"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=True)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)
                self.linear.bias.fill_(0.1)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 4)
            return out.unsqueeze(1)  # (2, 1, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_unsqueeze_1d_to_2d():
    """Test unsqueeze converting 1D to 2D"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 8, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 8)
            out = out.view(-1)  # (16,) - flatten
            return out.unsqueeze(0)  # (1, 16)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_unsqueeze_middle_of_3d():
    """Test unsqueeze in middle of 3D tensor"""
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
            return out.unsqueeze(2)  # (2, 3, 1, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_unsqueeze_chain():
    """Test chain of unsqueeze and view operations"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 8, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 8)
            out = out.unsqueeze(1)  # (2, 1, 8)
            out = out.view(2, 8)  # (2, 8)
            out = out.unsqueeze(0)  # (1, 2, 8)
            return out.view(2, 8)  # (2, 8)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


# ============================================================================
# SQUEEZE TESTS
# ============================================================================


def test_squeeze_simple():
    """Test simple squeeze removing dimension at position 0"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 4)
            out = out.unsqueeze(0)  # (1, 2, 4)
            return out.squeeze(0)  # (2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_squeeze_dim1():
    """Test squeeze removing dimension at position 1"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 4)
            out = out.unsqueeze(1)  # (2, 1, 4)
            return out.squeeze(1)  # (2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_squeeze_last_dim():
    """Test squeeze removing dimension at last position"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 4)
            out = out.unsqueeze(2)  # (2, 4, 1)
            return out.squeeze(2)  # (2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_squeeze_negative_dim():
    """Test squeeze with negative dimension"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 4)
            out = out.unsqueeze(-1)  # (2, 4, 1)
            return out.squeeze(-1)  # (2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_squeeze_unsqueeze_roundtrip():
    """Test that squeeze and unsqueeze are inverses"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 4)
            out = out.unsqueeze(0)  # (1, 2, 4)
            out = out.unsqueeze(3)  # (1, 2, 4, 1)
            out = out.squeeze(3)  # (1, 2, 4)
            out = out.squeeze(0)  # (2, 4)
            return out

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_squeeze_between_layers():
    """Test squeeze between linear layers"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 4, bias=False)
            self.linear2 = nn.Linear(4, 6, bias=False)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear2.weight.fill_(0.7)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear1(x)  # (2, 4)
            out = out.unsqueeze(1)  # (2, 1, 4)
            out = out.squeeze(1)  # (2, 4)
            return self.linear2(out)  # (2, 6)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_squeeze_with_add():
    """Test squeeze with addition"""
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
            combined = combined.unsqueeze(1)  # (2, 1, 4)
            return combined.squeeze(1)  # (2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_squeeze_with_mul():
    """Test squeeze with multiplication"""
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
            combined = combined.unsqueeze(0)  # (1, 2, 4)
            return combined.squeeze(0)  # (2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_squeeze_after_transpose():
    """Test squeeze after transpose"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 4)
            out = out.unsqueeze(0)  # (1, 2, 4)
            out = out.transpose(0, 1)  # (2, 1, 4)
            return out.squeeze(1)  # (2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_squeeze_attention_pattern():
    """Test squeeze in attention-like pattern (removing head dimension)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 8, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 8)
            out = out.view(2, 2, 4)  # (2, 2, 4)
            out = out.unsqueeze(1)  # (2, 1, 2, 4) - single head
            # After attention, remove head dimension
            return out.squeeze(1)  # (2, 2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_squeeze_with_view():
    """Test squeeze followed by view"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 12)
            out = out.unsqueeze(1)  # (2, 1, 12)
            out = out.squeeze(1)  # (2, 12)
            return out.view(2, 3, 4)  # (2, 3, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_squeeze_multiple_dims():
    """Test squeezing multiple dimensions sequentially"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 4)
            out = out.unsqueeze(0)  # (1, 2, 4)
            out = out.unsqueeze(2)  # (1, 2, 1, 4)
            out = out.squeeze(2)  # (1, 2, 4)
            out = out.squeeze(0)  # (2, 4)
            return out

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_squeeze_with_bias():
    """Test squeeze with biased linear layer"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=True)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)
                self.linear.bias.fill_(0.1)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # (2, 4)
            out = out.unsqueeze(1)  # (2, 1, 4)
            return out.squeeze(1)  # (2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_squeeze_middle_dimension():
    """Test squeeze removing middle dimension"""
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
            out = out.unsqueeze(2)  # (2, 3, 1, 4)
            return out.squeeze(2)  # (2, 3, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)
