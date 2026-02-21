import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

class CannonMerge(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=2):
        super(CannonMerge, self).__init__()
        self.kernel_size = int(kernel_size)
        self.cannon_layer = nn.Conv1d(
            input_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, input):
        # input: [batch, seq, channels] -> Conv1d expects [batch, channels, seq]
        x = input.transpose(-1, -2)
        # With stride == kernel_size, Conv1d will drop any "remainder" tokens that
        # don't fill a full window. Pad on the right so seq_out = ceil(seq / k).
        seq_len = x.shape[-1]
        rem = seq_len % self.kernel_size
        if rem != 0:
            pad_right = self.kernel_size - rem
            x = F.pad(x, (0, pad_right))
        x = self.cannon_layer(x)  # [batch, hidden_dim, seq_out]
        return x.transpose(-1, -2)  # [batch, seq_out, hidden_dim]

class CannonExpand(nn.Module):
    def __init__(self, kernel_size=2):
        super(CannonExpand, self).__init__()
        # This module is a pure "token copier": it expands the sequence by repeating
        # each token `kernel_size` times. No convolution, no mixing across tokens.
        #
        # Example: [b, 100, h] with kernel_size=2 -> [b, 200, h]
        self.kernel_size = int(kernel_size)

    def forward(self, input):
        # input: [batch, seq, channels] -> [batch, seq * kernel_size, channels]
        # Pad the leftmost token to shift the sequence right, then drop the last token
        # Fast shift-right on GPU: allocate once and do two slice copies.
        output = input.new_empty(input.shape)
        output[:, :1, :] = input[:, :1, :]      # keep leftmost token
        output[:, 1:, :] = input[:, :-1, :]     # shift right by 1
        output = output.repeat_interleave(self.kernel_size, dim=1)
        
        return output


class CoarseAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=2, num_heads=4):
        super(CoarseAttention, self).__init__()
        self.cannon = CannonMerge(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size)
        self.expand = CannonExpand(kernel_size=kernel_size)
        self.num_heads = num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.head_dim = hidden_dim // num_heads
        # Project to all heads at once: [*, hidden_dim] -> [*, num_heads * head_dim] == [*, hidden_dim]
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, input):
        x = self.cannon(input)
        batch_size, seq_len, hidden_dim = x.shape
        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)  # [batch, heads, seq, head_dim]
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)  # [batch, seq, hidden_dim]
        attn = self.expand(attn)
        return self.W_o(attn)

if __name__ == "__main__":

    #things to do:
    # we need to maket the expander aware of the padding so that it doesn't expand the padding tokens
    batch_size = 10
    seq_len = 14
    hidden_dim = 16
    input = torch.randn(batch_size, seq_len, hidden_dim)
    print(f"input shape: {input.shape}")
    kernel_size = 7
    farsight = CannonMerge(input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=kernel_size)
    output = farsight(input)
    print(f"cannon merge shape: {output.shape}")
    expander = CannonExpand(kernel_size=kernel_size)
    expanded = expander(output)
    print(f"expanded shape: {expanded.shape}")
    coarse_attention = CoarseAttention(input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=kernel_size)
    output = coarse_attention(input)
    print(f"coarse attention shape: {output.shape}")