import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque


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


class ReplayMemory:
    def __init__(self, agents, capacity, device):
        self.agents = agents
        self.memory = deque(maxlen=capacity)
        self.device = device

    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)

        states, actions, next_states, rewards, dones = zip(*transitions)

        states = {agent: torch.from_numpy(np.vstack([state[agent] for state in states])).float().to(self.device) for agent in self.agents}
        actions = {agent: torch.from_numpy(np.vstack([action[agent] for action in actions])).float().to(self.device) for agent in self.agents}
        next_states = {agent: torch.from_numpy(np.vstack([next_state[agent] for next_state in next_states])).float().to(self.device) for agent in self.agents}
        rewards = {agent: torch.from_numpy(np.array([reward[agent] for reward in rewards]).reshape(-1, 1)).float().to(self.device) for agent in self.agents}
        dones = {agent: torch.from_numpy(np.array([done[agent] for done in dones]).reshape(-1, 1)).float().to(self.device) for agent in self.agents}

        return states, actions, next_states, rewards, dones