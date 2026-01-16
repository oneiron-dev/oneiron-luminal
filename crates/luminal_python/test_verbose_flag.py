# test_verbose_flag.py
import pytest
import torch
import torch.nn as nn
import os

from test_utils import extract_graph_and_inputs, run_luminal_and_compare


def test_compile_verbose_produces_more_output(capfd):
    luminal_native = pytest.importorskip("luminal_native")

    class SimpleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)

        def forward(self, x):
            return self.linear(x)

    model = SimpleLinear()
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    nodes, inputs = extract_graph_and_inputs(model, x)

    luminal_native.compile(nodes, inputs, verbose=False)
    out1, err1 = capfd.readouterr()
    text1 = out1 + err1

    luminal_native.compile(nodes, inputs, verbose=True)
    out2, err2 = capfd.readouterr()
    text2 = out2 + err2

    assert len(text2) > len(text1), (
        "Expected verbose=True to produce more output than verbose=False.\n"
        f"len(non_verbose)={len(text1)} len(verbose)={len(text2)}\n"
        "Non-verbose output:\n" + text1 + "\n"
        "Verbose output:\n" + text2
    )


def test_verbose_creates_dot_files():
    """Test that verbose=True creates hlir_graph.dot and llir_graph_optimized.dot files"""
    luminal_native = pytest.importorskip("luminal_native")

    # Clean up any existing .dot files first
    for filename in ["hlir_graph.dot", "llir_graph_optimized.dot"]:
        if os.path.exists(filename):
            os.remove(filename)

    # Create a simple model
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            return self.linear(x)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Extract graph and inputs
    nodes, inputs = extract_graph_and_inputs(model, x)

    # Run with verbose=True
    outputs = luminal_native.compile(nodes, inputs, verbose=True)

    # Verify both .dot files were created
    assert os.path.exists("hlir_graph.dot"), "hlir_graph.dot was not created"
    assert os.path.exists("llir_graph_optimized.dot"), "llir_graph_optimized.dot was not created"

    # Verify files are not empty
    assert os.path.getsize("hlir_graph.dot") > 0, "hlir_graph.dot is empty"
    assert os.path.getsize("llir_graph_optimized.dot") > 0, "llir_graph_optimized.dot is empty"

    # Verify files contain dot graph content
    with open("hlir_graph.dot", "r") as f:
        hlir_content = f.read()
        assert "digraph" in hlir_content, "hlir_graph.dot doesn't contain valid dot graph"

    with open("llir_graph_optimized.dot", "r") as f:
        llir_content = f.read()
        assert "digraph" in llir_content, "llir_graph_optimized.dot doesn't contain valid dot graph"

    # Clean up
    os.remove("hlir_graph.dot")
    os.remove("llir_graph_optimized.dot")


def test_no_dot_files_without_verbose():
    """Test that verbose=False does not create .dot files"""
    luminal_native = pytest.importorskip("luminal_native")

    # Clean up any existing .dot files first
    for filename in ["hlir_graph.dot", "llir_graph_optimized.dot"]:
        if os.path.exists(filename):
            os.remove(filename)

    # Create a simple model
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)

        def forward(self, x):
            return self.linear(x)

    model = Model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Extract graph and inputs
    nodes, inputs = extract_graph_and_inputs(model, x)

    # Run with verbose=False (default)
    outputs = luminal_native.compile(nodes, inputs, verbose=False)

    # Verify .dot files were NOT created
    assert not os.path.exists("hlir_graph.dot"), "hlir_graph.dot should not be created with verbose=False"
    assert not os.path.exists("llir_graph_optimized.dot"), "llir_graph_optimized.dot should not be created with verbose=False"
