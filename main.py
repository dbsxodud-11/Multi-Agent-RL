import json

import wandb
import torch
import numpy as np
from tqdm import tqdm

from pettingzoo.mpe import simple_v2, simple_spread_v2
from algorithms.MADDPG import MADDPGAgent

with open("./config/simple_spread_maddpg.json", "r") as f:
    config = json.load(f)

env_config = config["env"]
N = env_config["N"]
max_cycles = env_config["max_cycles"]
continuous_actions = True if env_config["continuous_actions"] == "True" else False
env = simple_spread_v2.parallel_env(N=N, max_cycles=max_cycles, continuous_actions=continuous_actions)
env.reset()

obs_dims = {agent: env.observation_space(agent).shape[0] for agent in env.agents}
action_dims = {agent: env.action_space(agent).n for agent in env.agents}

train_config = config["training"]
num_episodes = train_config["num_episodes"]
lr = train_config["lr"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = config["model_name"]
if model_name == "MADDPG":
    model = MADDPGAgent(device, env.agents, obs_dims, action_dims, lr=lr)

wandb.init(project="MARL in Multi-Particle Environment", name=str(model))
for episode in tqdm(range(num_episodes)):
    episode_reward = 0
    state = env.reset()

    for _ in range(max_cycles):
        actions, action_norms = model.select_action(state)
        next_state, reward, done, _ = env.step(action_norms)
        # print(reward)
        episode_reward += sum(reward.values())
        model.push((state, actions, next_state, reward, done))
        state = next_state

    if model.train_start():
        if model_name == "MADDPG":
            policy_loss, value_loss, epsilon = model.train()
        elif model_name == "QMIX":
            loss, epsilon = model.train()

    if model.train_start():
        if model_name == "MADDPG":
            wandb.log({"Reward": episode_reward, "Episode": episode, "Policy Loss": policy_loss, "Value Loss": value_loss, "Epsilon": epsilon})
        elif model_name == "QMIX":
            wandb.log({"Reward": episode_reward, "Episode": episode, "Loss": loss, "Epsilon": epsilon})
wandb.finish()