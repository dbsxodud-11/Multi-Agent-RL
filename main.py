import argparse

import wandb
import torch
import numpy as np
from tqdm import tqdm

from envs.mpe.scenarios import load
from envs.mpe.environment import MultiAgentEnv
from algorithms.MADDPG import MADDPGAgent

scenario = load("simple_spread.py").Scenario()
episode_length = 25
# create world
world = scenario.make_world(episode_length=episode_length, num_agents=3, num_landmarks=3)
# create multiagent environment
env = MultiAgentEnv(world, scenario.reset_world,
                    scenario.reward, scenario.observation, scenario.info)

obs_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n

num_episodes = 100000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MADDPGAgent(device, [agent for agent in range(env.n)], obs_dim, action_dim)

wandb.init(project="MARL in Multi-Particle Environment",
           name=str(model))
for episode in tqdm(range(num_episodes)):
    episode_reward = 0
    state = env.reset()

    for _ in range(episode_length):
        actions, action_norms = model.select_action(state)
        next_state, reward, done, _ = env.step(action_norms)
        # print(reward)
        episode_reward += np.mean([r[0] for r in reward])
        model.push((state, actions, next_state, reward, done))
        state = next_state

    if model.train_start():
        policy_loss, value_loss, epsilon = model.train()

    if model.train_start():
        wandb.log({"Reward": episode_reward, "Episode": episode, "Policy Loss": policy_loss, "Value Loss": value_loss, "Epsilon": epsilon})
wandb.finish()