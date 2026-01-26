import torch
import torch.nn as nn

from test_utils import extract_graph_and_inputs, run_luminal_and_compare


def test_arange_basic():
    """Test most basic arange - just create and return it"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))

        def forward(self, x):
            # x shape: (4,)
            # Create range [0, 1, 2, 3]
            idx = torch.arange(4, dtype=torch.float32)
            # Multiply with weight (element-wise)
            return x + self.weight * idx

    model = Model()
    x = torch.ones(4)
    run_luminal_and_compare(model, x, atol=1e-5)


def test_arange_start():
    """Test arange with start parameter - needed for nanogpt"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

        def forward(self, x):
            # x shape: (3,)
            # Create range [2, 3, 4]
            idx = torch.arange(2, 5, dtype=torch.float32)
            return x + self.weight * idx

    model = Model()
    x = torch.ones(3)
    run_luminal_and_compare(model, x, atol=1e-5)


def test_arange_with_step():
    """Test arange with step parameter"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

        def forward(self, x):
            # x shape: (3,)
            # Create range [0, 2, 4]
            idx = torch.arange(0, 6, 2, dtype=torch.float32)
            return x + self.weight * idx

    model = Model()
    x = torch.ones(3)
    run_luminal_and_compare(model, x, atol=1e-5)
