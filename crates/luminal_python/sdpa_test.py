import torch
import torch.nn as nn
import torch.nn.functional as F

from test_utils import extract_graph_and_inputs, run_luminal_and_compare


def test_sdpa_simple():
    """Test simple scaled dot product attention"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(8, 24, bias=False)
            with torch.no_grad():
                self.qkv.weight.fill_(0.1)

        def forward(self, x):
            # x shape: (2, 4, 8) - (batch, seq, features)
            qkv = self.qkv(x)  # (2, 4, 24)
            # Split into q, k, v
            q, k, v = qkv.split(8, dim=-1)  # Each (2, 4, 8)
            # Apply SDPA
            out = F.scaled_dot_product_attention(q, k, v)
            return out  # (2, 4, 8)

    model = Model()
    x = torch.randn(2, 4, 8)
    run_luminal_and_compare(model, x, atol=1e-4)


def test_sdpa_multihead():
    """Test SDPA with multi-head pattern"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(16, 48, bias=False)
            with torch.no_grad():
                self.qkv.weight.fill_(0.1)

        def forward(self, x):
            # x shape: (2, 4, 16) - (batch, seq, features)
            batch_size, seq_len, _ = x.shape
            num_heads = 4
            head_dim = 4

            qkv = self.qkv(x)  # (2, 4, 48)
            q, k, v = qkv.split(16, dim=-1)  # Each (2, 4, 16)

            # Reshape for multi-head: (batch, num_heads, seq, head_dim)
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

            # Apply SDPA
            out = F.scaled_dot_product_attention(q, k, v)  # (2, 4, 4, 4)

            # Reshape back
            out = out.transpose(1, 2).view(batch_size, seq_len, -1)  # (2, 4, 16)
            return out

    model = Model()
    x = torch.randn(2, 4, 16)
    run_luminal_and_compare(model, x, atol=1e-4)


def test_sdpa_single_head():
    """Test SDPA with single head (no reshaping)"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False)
            self.k_proj = nn.Linear(8, 8, bias=False)
            self.v_proj = nn.Linear(8, 8, bias=False)
            with torch.no_grad():
                self.q_proj.weight.fill_(0.1)
                self.k_proj.weight.fill_(0.1)
                self.v_proj.weight.fill_(0.1)

        def forward(self, x):
            # x shape: (2, 4, 8)
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            # Add head dimension: (batch, 1, seq, features)
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)

            # Apply SDPA
            out = F.scaled_dot_product_attention(q, k, v)

            # Remove head dimension
            out = out.squeeze(1)
            return out

    model = Model()
    x = torch.randn(2, 4, 8)
    run_luminal_and_compare(model, x, atol=1e-4)


def test_sdpa_different_seq_lengths():
    """Test SDPA where key/value have different sequence length"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False)
            self.kv_proj = nn.Linear(8, 16, bias=False)
            with torch.no_grad():
                self.q_proj.weight.fill_(0.1)
                self.kv_proj.weight.fill_(0.1)

        def forward(self, x_q, x_kv):
            # x_q shape: (2, 4, 8) - query has seq_len=4
            # x_kv shape: (2, 6, 8) - key/value has seq_len=6
            q = self.q_proj(x_q).unsqueeze(1)  # (2, 1, 4, 8)

            kv = self.kv_proj(x_kv)  # (2, 6, 16)
            k, v = kv.split(8, dim=-1)
            k = k.unsqueeze(1)  # (2, 1, 6, 8)
            v = v.unsqueeze(1)  # (2, 1, 6, 8)

            # Apply SDPA
            out = F.scaled_dot_product_attention(q, k, v)
            return out.squeeze(1)  # (2, 4, 8)

    model = Model()
    x_q = torch.randn(2, 4, 8)
    x_kv = torch.randn(2, 6, 8)

    with torch.no_grad():
        pytorch_output = model(x_q, x_kv)

    nodes, inputs = extract_graph_and_inputs(model, (x_q, x_kv))

    import pytest
    luminal_native = pytest.importorskip("luminal_native")
    outputs = luminal_native.compile(nodes, inputs, verbose=False)

    output_key = next(iter(outputs.keys()))
    luminal_output = torch.tensor(outputs[output_key]).reshape(pytorch_output.shape)

    torch.testing.assert_close(luminal_output, pytorch_output, atol=1e-4, rtol=0.0)


def test_sdpa_with_layer():
    """Test SDPA followed by a linear layer"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(8, 24, bias=False)
            self.proj = nn.Linear(8, 8, bias=False)
            with torch.no_grad():
                self.qkv.weight.fill_(0.1)
                self.proj.weight.fill_(0.2)

        def forward(self, x):
            qkv = self.qkv(x)
            q, k, v = qkv.split(8, dim=-1)

            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)

            attn_out = F.scaled_dot_product_attention(q, k, v)
            attn_out = attn_out.squeeze(1)

            return self.proj(attn_out)

    model = Model()
    x = torch.randn(2, 4, 8)
    run_luminal_and_compare(model, x, atol=1e-4)


def test_sdpa_small_dimensions():
    """Test SDPA with small dimensions"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(4, 12, bias=False)
            with torch.no_grad():
                self.qkv.weight.fill_(0.1)

        def forward(self, x):
            qkv = self.qkv(x)
            q, k, v = qkv.split(4, dim=-1)

            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)

            out = F.scaled_dot_product_attention(q, k, v)
            return out.squeeze(1)

    model = Model()
    x = torch.randn(2, 3, 4)
    run_luminal_and_compare(model, x, atol=1e-4)


def test_sdpa_larger_batch():
    """Test SDPA with larger batch size"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(8, 24, bias=False)
            with torch.no_grad():
                self.qkv.weight.fill_(0.1)

        def forward(self, x):
            qkv = self.qkv(x)
            q, k, v = qkv.split(8, dim=-1)

            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)

            out = F.scaled_dot_product_attention(q, k, v)
            return out.squeeze(1)

    model = Model()
    x = torch.randn(8, 4, 8)  # batch_size=8
    run_luminal_and_compare(model, x, atol=1e-4)


def test_sdpa_with_residual():
    """Test SDPA with residual connection"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(8, 24, bias=False)
            with torch.no_grad():
                self.qkv.weight.fill_(0.1)

        def forward(self, x):
            qkv = self.qkv(x)
            q, k, v = qkv.split(8, dim=-1)

            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)

            attn_out = F.scaled_dot_product_attention(q, k, v)
            attn_out = attn_out.squeeze(1)

            # Residual connection
            return x + attn_out

    model = Model()
    x = torch.randn(2, 4, 8)
    run_luminal_and_compare(model, x, atol=1e-4)


def test_sdpa_two_heads():
    """Test SDPA with 2 heads"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(8, 24, bias=False)
            with torch.no_grad():
                self.qkv.weight.fill_(0.1)

        def forward(self, x):
            batch_size, seq_len, _ = x.shape
            num_heads = 2
            head_dim = 4

            qkv = self.qkv(x)
            q, k, v = qkv.split(8, dim=-1)

            # Reshape for 2 heads
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

            out = F.scaled_dot_product_attention(q, k, v)

            # Reshape back
            out = out.transpose(1, 2).view(batch_size, seq_len, -1)
            return out

    model = Model()
    x = torch.randn(2, 4, 8)
    run_luminal_and_compare(model, x, atol=1e-4)


def test_sdpa_self_attention():
    """Test full self-attention block"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(16, 48, bias=True)
            self.out_proj = nn.Linear(16, 16, bias=True)
            with torch.no_grad():
                self.qkv.weight.fill_(0.1)
                self.qkv.bias.fill_(0.01)
                self.out_proj.weight.fill_(0.1)
                self.out_proj.bias.fill_(0.01)

        def forward(self, x):
            batch_size, seq_len, embed_dim = x.shape
            num_heads = 4
            head_dim = embed_dim // num_heads

            qkv = self.qkv(x)
            q, k, v = qkv.split(embed_dim, dim=-1)

            # Multi-head reshape
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

            # Attention
            attn_out = F.scaled_dot_product_attention(q, k, v)

            # Reshape and project
            attn_out = attn_out.transpose(1, 2).contiguous()
            attn_out = attn_out.view(batch_size, seq_len, embed_dim)

            return self.out_proj(attn_out)

    model = Model()
    x = torch.randn(2, 8, 16)
    run_luminal_and_compare(model, x, atol=1e-4)
