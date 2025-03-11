import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=64, output_dim=10):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x
