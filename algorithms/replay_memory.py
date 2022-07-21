import random

import torch
import numpy as np
from collections import deque


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
        actions = {agent: torch.from_numpy(np.vstack([action[agent] for action in actions])).long().to(self.device) for agent in self.agents}
        next_states = {agent: torch.from_numpy(np.vstack([next_state[agent] for next_state in next_states])).float().to(self.device) for agent in self.agents}
        rewards = {agent: torch.from_numpy(np.array([reward[agent] for reward in rewards]).reshape(-1, 1)).float().to(self.device) for agent in self.agents}
        dones = {agent: torch.from_numpy(np.array([done[agent] for done in dones]).reshape(-1, 1)).float().to(self.device) for agent in self.agents}

        return states, actions, next_states, rewards, dones