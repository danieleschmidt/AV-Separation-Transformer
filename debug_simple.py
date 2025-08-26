import torch
import torch.nn as nn

# Minimal debug version
class DebugModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(80, 256)
        self.out = nn.Linear(256, 80)
    
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.embed(x)
        print(f"After embed: {x.shape}")
        return self.out(x)

# Test
model = DebugModel()
dummy_input = torch.randn(1, 200, 80)  # [batch, seq, features]
print("Testing with shape:", dummy_input.shape)
output = model(dummy_input)
print("Output shape:", output.shape)