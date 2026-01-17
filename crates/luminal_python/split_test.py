import torch
import torch.nn as nn

from test_utils import extract_graph_and_inputs, run_luminal_and_compare


def test_split_simple():
    """Test simple split into equal chunks"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 12)
            # Split into 3 chunks of size 4 along dim 1
            chunks = out.split(4, dim=1)
            # Return first chunk
            return chunks[0]  # (2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_split_last_chunk():
    """Test accessing last chunk from split"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 12)
            # Split into 3 chunks of size 4 along dim 1
            chunks = out.split(4, dim=1)
            # Return last chunk
            return chunks[2]  # (2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_split_middle_chunk():
    """Test accessing middle chunk from split"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 12)
            # Split into 3 chunks of size 4 along dim 1
            chunks = out.split(4, dim=1)
            # Return middle chunk
            return chunks[1]  # (2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_split_uneven():
    """Test split with uneven chunk size (last chunk is smaller)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 10, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 10)
            # Split into chunks of size 4 (last will be size 2)
            chunks = out.split(4, dim=1)
            # Return last chunk which is smaller
            return chunks[2]  # (2, 2)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_split_dim0():
    """Test split along dimension 0"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 8, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 8)
            # Split along dim 0 (batch dimension)
            chunks = out.split(1, dim=0)
            # Return first chunk
            return chunks[0]  # (1, 8)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_split_3d_tensor():
    """Test split on 3D tensor"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 24, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 24)
            out = out.view(2, 6, 4)  # (2, 6, 4)
            # Split along middle dimension
            chunks = out.split(2, dim=1)
            # Return first chunk
            return chunks[0]  # (2, 2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_split_with_add():
    """Test split followed by addition of chunks"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 8, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 8)
            # Split into 2 chunks of size 4
            chunks = out.split(4, dim=1)
            # Add the two chunks together
            return chunks[0] + chunks[1]  # (2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_split_with_mul():
    """Test split followed by multiplication of chunks"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 8, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 8)
            # Split into 2 chunks of size 4
            chunks = out.split(4, dim=1)
            # Multiply the two chunks together (gating pattern)
            return chunks[0] * chunks[1]  # (2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_split_glu_pattern():
    """Test split in GLU (Gated Linear Unit) pattern"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            # GLU typically projects to 2x the hidden size then splits
            self.linear = nn.Linear(3, 16, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 16)
            # Split into value and gate
            value, gate = out.split(8, dim=1)
            # GLU: value * sigmoid(gate), but we'll just use value * gate for simplicity
            return value * gate  # (2, 8)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_split_between_layers():
    """Test split between linear layers"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 12, bias=False)
            self.linear2 = nn.Linear(4, 6, bias=False)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear2.weight.fill_(0.7)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear1(x)  # out shape: (2, 12)
            # Split into chunks
            chunks = out.split(4, dim=1)
            # Process first chunk through another layer
            return self.linear2(chunks[0])  # (2, 6)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_split_all_chunks_used():
    """Test using all chunks from split"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 12)
            # Split into 4 chunks of size 3
            c0, c1, c2, c3 = out.split(3, dim=1)
            # Combine all chunks with operations
            return (c0 + c1) * (c2 + c3)  # (2, 3)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_split_single_chunk():
    """Test split that produces a single chunk"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 8, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 8)
            # Split with chunk size equal to dimension size
            chunks = out.split(8, dim=1)
            # Should produce single chunk
            return chunks[0]  # (2, 8)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_split_with_view():
    """Test split combined with view operations"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 24, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 24)
            # Split into chunks
            chunks = out.split(8, dim=1)
            # Reshape first chunk
            return chunks[0].view(2, 4, 2)  # (2, 4, 2)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_split_negative_dim():
    """Test split with negative dimension index"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 12)
            # Split along last dimension using negative index
            chunks = out.split(4, dim=-1)
            # Return first chunk
            return chunks[0]  # (2, 4)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_split_with_bias():
    """Test split with biased linear layer"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12, bias=True)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)
                self.linear.bias.fill_(0.1)

        def forward(self, x):
            # x shape: (2, 3)
            out = self.linear(x)  # out shape: (2, 12)
            # Split into chunks
            chunks = out.split(6, dim=1)
            # Return first chunk
            return chunks[0]  # (2, 6)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)
