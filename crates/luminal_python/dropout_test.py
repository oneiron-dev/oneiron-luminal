import torch
import torch.nn as nn

from test_utils import extract_graph_and_inputs, run_luminal_and_compare


def test_dropout_simple():
    """Test dropout after linear layer (should be no-op in inference)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            self.dropout = nn.Dropout(p=0.5)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)

        def forward(self, x):
            x = self.linear(x)
            x = self.dropout(x)
            return x

    model = Model()
    model.eval()  # Inference mode - dropout is disabled
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_dropout_with_bias():
    """Test dropout with biased linear layer"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=True)
            self.dropout = nn.Dropout(p=0.3)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)
                self.linear.bias.fill_(0.5)

        def forward(self, x):
            x = self.linear(x)
            x = self.dropout(x)
            return x

    model = Model()
    model.eval()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_dropout_mlp():
    """Test dropout in MLP pattern (linear -> dropout -> linear)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 8, bias=True)
            self.dropout1 = nn.Dropout(p=0.2)
            self.linear2 = nn.Linear(8, 4, bias=True)
            self.dropout2 = nn.Dropout(p=0.5)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear1.bias.fill_(0.1)
                self.linear2.weight.fill_(0.5)
                self.linear2.bias.fill_(0.1)

        def forward(self, x):
            x = self.linear1(x)
            x = self.dropout1(x)
            x = self.linear2(x)
            x = self.dropout2(x)
            return x

    model = Model()
    model.eval()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_dropout_with_gelu():
    """Test dropout combined with GELU activation"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 8, bias=True)
            self.dropout = nn.Dropout(p=0.4)
            self.linear2 = nn.Linear(8, 4, bias=True)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear1.bias.fill_(0.1)
                self.linear2.weight.fill_(0.5)
                self.linear2.bias.fill_(0.1)

        def forward(self, x):
            x = self.linear1(x)
            x = torch.nn.functional.gelu(x, approximate='tanh')
            x = self.dropout(x)
            x = self.linear2(x)
            return x

    model = Model()
    model.eval()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_dropout_residual():
    """Test dropout with residual connection"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 4, bias=True)
            self.dropout = nn.Dropout(p=0.3)
            self.linear2 = nn.Linear(4, 4, bias=True)
            with torch.no_grad():
                self.linear1.weight.fill_(0.5)
                self.linear1.bias.fill_(0.05)
                self.linear2.weight.fill_(0.5)
                self.linear2.bias.fill_(0.05)

        def forward(self, x):
            identity = x
            x = self.linear1(x)
            x = self.dropout(x)
            x = self.linear2(x)
            return x + identity  # Residual connection

    model = Model()
    model.eval()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_dropout_multiple():
    """Test multiple dropout layers with different probabilities"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 8, bias=True)
            self.dropout1 = nn.Dropout(p=0.1)
            self.linear2 = nn.Linear(8, 8, bias=True)
            self.dropout2 = nn.Dropout(p=0.3)
            self.linear3 = nn.Linear(8, 4, bias=True)
            self.dropout3 = nn.Dropout(p=0.5)
            with torch.no_grad():
                self.linear1.weight.fill_(0.4)
                self.linear1.bias.fill_(0.05)
                self.linear2.weight.fill_(0.4)
                self.linear2.bias.fill_(0.05)
                self.linear3.weight.fill_(0.4)
                self.linear3.bias.fill_(0.05)

        def forward(self, x):
            x = self.linear1(x)
            x = self.dropout1(x)
            x = self.linear2(x)
            x = self.dropout2(x)
            x = self.linear3(x)
            x = self.dropout3(x)
            return x

    model = Model()
    model.eval()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    run_luminal_and_compare(model, x, atol=1e-5)


def test_dropout_high_probability():
    """Test dropout with very high probability (still no-op in inference)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            self.dropout = nn.Dropout(p=0.99)  # Very high dropout
            with torch.no_grad():
                self.linear.weight.fill_(1.0)

        def forward(self, x):
            x = self.linear(x)
            x = self.dropout(x)
            return x

    model = Model()
    model.eval()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # Should still match exactly since dropout is disabled in eval mode
    run_luminal_and_compare(model, x, atol=1e-5)
