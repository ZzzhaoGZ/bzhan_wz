import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.LayerNorm_1 = nn.Sequential(
            nn.Conv2d(),
            nn.ReLU(),
            nn.Conv2d(),
            nn.ReLU(),
            nn.MaxPool2d(),
        )

    def forward(self,x):
        x = self.LayerNorm_1(x)
        x = torch.flatten(x,start_dim=1)
        return x