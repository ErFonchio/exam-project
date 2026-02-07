import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame
import gymnasium as gym
import argparse


def evaluate(env=None, n_episodes=100 , render=False):
    agent = Policy()
    agent.load()

    env = gym.make('HalfCheetah-v5', ctrl_cost_weight=0.1)

    if render:
        env = gym.make('HalfCheetah-v5', ctrl_cost_weight=0.1, render_mode='human')

    rewards = []
    for episode in range(n_episodes):
        print(f"ep n {episode}", "\r")
        total_reward = 0
        done = False
        s, _ = env.reset()
        while not done:
            action = agent.act(s)
            
            s, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
    
        