import pytest
import torch
import torch.nn as nn

from test_utils import extract_graph_and_inputs, run_luminal_and_compare


def test_embedding_basic():
    """Test basic embedding lookup"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10, 4)  # 10 vocab size, 4 embedding dim
            with torch.no_grad():
                # Initialize with simple values for easy verification
                self.embedding.weight.fill_(1.0)

        def forward(self, x):
            return self.embedding(x)

    model = Model()
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_embedding_single_token():
    """Test embedding with single token input"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(5, 3)
            with torch.no_grad():
                self.embedding.weight.copy_(torch.arange(15.0).reshape(5, 3))

        def forward(self, x):
            return self.embedding(x)

    model = Model()
    x = torch.tensor([[2]])  # Single token
    run_luminal_and_compare(model, x, atol=1e-5)


def test_embedding_varied_vocab_sizes():
    """Test embeddings with different vocabulary sizes"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(50, 8)  # Larger vocab
            with torch.no_grad():
                self.embedding.weight.fill_(0.5)

        def forward(self, x):
            return self.embedding(x)

    model = Model()
    x = torch.tensor([[0, 10, 20, 30], [5, 15, 25, 35]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_embedding_sequence_length():
    """Test embedding with varying sequence lengths"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(20, 6)
            with torch.no_grad():
                self.embedding.weight.fill_(0.25)

        def forward(self, x):
            return self.embedding(x)

    model = Model()
    # Longer sequence
    x = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_embedding_batch_size():
    """Test embedding with larger batch sizes"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(15, 5)
            with torch.no_grad():
                self.embedding.weight.fill_(0.75)

        def forward(self, x):
            return self.embedding(x)

    model = Model()
    # Larger batch
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_embedding_sequential_lookups():
    """Test multiple embedding layers in sequence"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding1 = nn.Embedding(10, 4)
            self.embedding2 = nn.Embedding(10, 4)
            with torch.no_grad():
                self.embedding1.weight.fill_(1.0)
                self.embedding2.weight.fill_(0.5)

        def forward(self, x):
            # Use same input for both embeddings (not typical but tests functionality)
            e1 = self.embedding1(x)
            e2 = self.embedding2(x)
            # Sum the embeddings
            return e1 + e2

    model = Model()
    x = torch.tensor([[1, 2, 3]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_embedding_with_operations():
    """Test embedding with subsequent operations"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 4)
            with torch.no_grad():
                self.embedding.weight.fill_(2.0)

        def forward(self, x):
            x = self.embedding(x)
            # Apply operations on embedding output
            x = x * 0.5  # Scale
            x = x + 1.0  # Shift
            return x

    model = Model()
    x = torch.tensor([[0, 1, 2], [3, 4, 5]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_embedding_edge_indices():
    """Test embedding with edge case indices (0 and max)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10, 3)
            with torch.no_grad():
                # Use distinct values per embedding vector
                for i in range(10):
                    self.embedding.weight[i] = float(i)

        def forward(self, x):
            return self.embedding(x)

    model = Model()
    x = torch.tensor([[0, 9, 0, 9]])  # Test first and last indices
    run_luminal_and_compare(model, x, atol=1e-5)
