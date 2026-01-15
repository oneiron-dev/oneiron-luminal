# test_verbose_flag.py
import pytest
import torch
import torch.nn as nn

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
