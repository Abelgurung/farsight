from MLP import MLP_cannon
from MHA import MultiHeadAttention
import torch.nn as nn
import torch

class TransformerCannonBlock(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, kernel_size=7, num_heads=4):
        super(TransformerCannonBlock, self).__init__()
        self.attention = MultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_heads)
        self.pre_attention_norm = nn.RMSNorm(hidden_dim)
        self.post_attention_norm = nn.RMSNorm(hidden_dim)
        self.mlp = MLP_cannon(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, kernel_size=kernel_size)
        self.pre_mlp_norm = nn.RMSNorm(hidden_dim)
        self.post_mlp_norm = nn.RMSNorm(hidden_dim)

    def forward(self, x):
        identity = x
        x = self.pre_attention_norm(x)
        x = self.attention(x)
        x = x + identity
        x = self.post_attention_norm(x)
        identity = x
        x = self.pre_mlp_norm(x)
        x = self.mlp(x)
        x = x + identity
        x = self.post_mlp_norm(x)
        return x

class TransformerCannonModel(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, kernel_size=7, num_heads=4, num_layers=6, vocab_size=512):
        super(TransformerCannonModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([TransformerCannonBlock(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, kernel_size=kernel_size, num_heads=num_heads) for _ in range(num_layers)])
        self.norm = nn.RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.lm_head(x)
        return x

if __name__ == "__main__":
    model = TransformerCannonModel(hidden_dim=16, intermediate_dim=16*4, kernel_size=7, num_heads=4, num_layers=6, vocab_size=512)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters in millions: {model_params / 1000000}")
    batch_size = 32
    seq_len = 20
    vocab_size = 512
    input = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(input)
    print(output.shape)