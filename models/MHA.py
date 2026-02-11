import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _build_rope_cache(seq_len: int, head_dim: int, device, dtype, base: float = 10000.0):
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
    # inv_freq: [head_dim/2]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    # freqs: [seq_len, head_dim/2]
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().to(dtype=dtype)[None, None, :, :]  # [1, 1, seq, head_dim/2]
    sin = freqs.sin().to(dtype=dtype)[None, None, :, :]  # [1, 1, seq, head_dim/2]
    return cos, sin

def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [batch, heads, seq, head_dim]
    x_even = x[..., ::2]   # [batch, heads, seq, head_dim/2]
    x_odd = x[..., 1::2]   # [batch, heads, seq, head_dim/2]
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., ::2] = rot_even
    out[..., 1::2] = rot_odd
    return out

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.head_dim = hidden_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even to use RoPE")
        # Project to all heads at once: [*, hidden_dim] -> [*, num_heads * head_dim] == [*, hidden_dim]
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, input):
        batch_size, seq_len, hidden_dim = input.shape
        q = self.W_q(input).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(input).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(input).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = _build_rope_cache(seq_len=seq_len, head_dim=self.head_dim, device=input.device, dtype=q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [batch, heads, seq, head_dim]
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)  # [batch, seq, hidden_dim]
        return self.W_o(attn)

if __name__ == "__main__":

    model = MultiHeadAttention(hidden_dim=10, num_heads=2)
    print(model)