import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=[64 for _ in range(2)]):
        super(Actor, self).__init__()

        self.layers = nn.ModuleList()
        input_dims = [input_dim] + hidden
        output_dims = hidden + [output_dim]

        for in_dim, out_dim in zip(input_dims[:-1], output_dims[:-1]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            nn.init.orthogonal_(self.layers[-1].weight)
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(input_dims[-1], output_dims[-1]))
        nn.init.orthogonal_(self.layers[-1].weight)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.gumbel_softmax(x, dim=-1, hard=True)


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=[64 for _ in range(2)]):
        super(Critic, self).__init__()

        self.layers = nn.ModuleList()
        input_dims = [input_dim] + hidden
        output_dims = hidden + [output_dim]

        for in_dim, out_dim in zip(input_dims[:-1], output_dims[:-1]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            nn.init.orthogonal_(self.layers[-1].weight)
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(input_dims[-1], output_dims[-1]))
        nn.init.orthogonal_(self.layers[-1].weight)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x