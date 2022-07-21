import random

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from .utils import update_model
from .actor_critic import Actor, Critic
from .replay_memory import ReplayMemory

class MADDPGAgent(nn.Module):
    def __init__(self, device, agents, obs_dims, action_dims, lr,
                 gamma=0.99, eps=1.0, eps_decay=0.9, eps_min=0.05, batch_size=800, tau=0.001):
        super(MADDPGAgent, self).__init__()

        self.device = device
        self.agents = agents
        self.n_agents = len(self.agents)
        self.obs_dims = obs_dims
        self.action_dims = action_dims

        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.tau = tau
        self.step = 0
        self.decay_step = 2000

        self.actor = nn.ModuleDict({agent: Actor(self.obs_dims[agent], self.action_dims[agent]).to(self.device) for agent in self.agents})
        self.target_actor = nn.ModuleDict({agent: Actor(self.obs_dims[agent], self.action_dims[agent]).to(self.device) for agent in self.agents})
        for agent in self.agents:
            update_model(self.actor[agent], self.target_actor[agent], tau=1.0)

        self.critic = nn.ModuleDict({agent: Critic(self.obs_dims[agent] * self.n_agents + self.action_dims[agent] * self.n_agents, 1).to(self.device) for agent in self.agents})
        self.target_critic = nn.ModuleDict({agent: Critic(self.obs_dims[agent] * self.n_agents + self.action_dims[agent] * self.n_agents, 1).to(self.device) for agent in self.agents})
        for agent in self.agents:
            update_model(self.critic[agent], self.target_critic[agent], tau=1.0)

        self.mse_loss = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.memory = ReplayMemory(self.agents, capacity=125000, device=self.device)

    def select_action(self, state):
        actions, action_norms = {}, {}
        for agent in self.agents:
            obs = torch.from_numpy(state[agent]).float().unsqueeze(0).to(self.device)
            if random.random() < self.eps:
                action = np.zeros(self.action_dims[agent])
                action[random.randint(0, self.action_dims[agent]-1)] = 1.0
            else:
                action = self.actor[agent](obs).detach().cpu().numpy().squeeze(0)
            action_norm = action.argmax()
            
            actions[agent] = action
            action_norms[agent] = action_norm
        return actions, action_norms

    def push(self, transition):
        self.memory.push(transition)

    def train_start(self):
        return len(self.memory) >= self.batch_size

    def train(self):
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)

        total_value_loss = 0.0
        total_policy_loss = 0.0

        for target_agent in self.agents:
            joint_state = torch.cat([states[agent] for agent in self.agents], dim=-1)
            joint_actions = torch.cat([actions[agent] for agent in self.agents], dim=-1)
            current_q_values = self.critic[target_agent](torch.cat([joint_state, joint_actions], dim=-1))

            joint_next_state = torch.cat([next_states[agent] for agent in self.agents], dim=-1)
            joint_next_actions = torch.cat([self.target_actor[agent](next_states[agent]) for agent in self.agents], dim=-1)
            next_q_values = self.target_critic[target_agent](torch.cat([joint_next_state, joint_next_actions], dim=-1)).detach()
            target_q_values = rewards[target_agent] + self.gamma * next_q_values * (1 - dones[target_agent])
            # Add Normalization
            target_q_values = (target_q_values - target_q_values.mean())  / target_q_values.std()

            value_loss = self.mse_loss(target_q_values, current_q_values)
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            clip_grad_norm_(self.critic.parameters(), 10.0)
            self.critic_optimizer.step()

            joint_actions = []
            for agent in self.agents:
                if agent == target_agent:
                    joint_actions.append(self.actor[agent](states[agent]))
                else:
                    joint_actions.append(actions[agent])
            joint_actions = torch.cat(joint_actions, dim=-1)

            policy_loss = -self.critic[target_agent](torch.cat([joint_state, joint_actions], dim=1)).mean()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()

            update_model(self.actor[target_agent], self.target_actor[target_agent], tau=self.tau)
            update_model(self.critic[target_agent], self.target_critic[target_agent], tau=self.tau)

            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()

        if self.step % self.decay_step == 0:
            if self.eps > self.eps_min:
                self.eps *= self.eps_decay
            else:
                self.eps = self.eps_min
        self.step += 1

        return total_policy_loss / self.n_agents, total_value_loss / self.n_agents, self.eps

    def write(self, reward, policy_loss, value_loss):
        wandb.log({'Reward': reward,
                   'Actor Loss': policy_loss, 'Critic Loss': value_loss})

    def __str__(self):
        return "MADDPG"