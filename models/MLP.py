import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.conv as conv

class MLP(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super(MLP, self).__init__()
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim)

    def forward(self, x):
        x = F.relu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return x

class CannonLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=7):
        super(CannonLayer, self).__init__()
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, "Kernel size must be odd -- for padding"
        self.cannon_layer = nn.Conv1d(
            input_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding="same",
        )

    def forward(self, input):
        # input: [batch, seq, channels] -> Conv1d expects [batch, channels, seq]
        x = input.transpose(-1, -2)
        x = self.cannon_layer(x)  # [batch, hidden_dim, seq_out]
        return x.transpose(-1, -2)  # [batch, seq_out, hidden_dim]

class MLP_cannon(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, kernel_size=7):
        super(MLP_cannon, self).__init__()
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim)
        self.up_cannon = CannonLayer(hidden_dim, intermediate_dim, kernel_size=kernel_size)
        self.down_cannon = CannonLayer(intermediate_dim, hidden_dim, kernel_size=kernel_size)

    def forward(self, x):
        x = F.relu(self.gate_proj(x)) * self.up_proj(x)
        x = self.up_cannon(x)
        x = self.down_proj(x)
        x = self.down_cannon(x)
        return x

if __name__ == "__main__":
    input = torch.randn(32, 20, 16)
    model = MLP_cannon(hidden_dim=16, intermediate_dim=16, kernel_size=7)
    output = model(input)
    print(output.shape)