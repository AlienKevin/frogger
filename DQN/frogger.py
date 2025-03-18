import gymnasium as gym
import pygame
import numpy as np
import ale_py
from opto.trace import node, bundle
import autogen
from opto.optimizers import OptoPrime
from opto import trace
from opto.trace.utils import render_opt_step
from collections import deque
import matplotlib.pyplot as plt
# Initialize the environment
env = gym.make("ALE/Frogger-v5", render_mode="rgb_array", frameskip=4, repeat_action_probability=0, mode=0)

obs, info = env.reset()
for i in range(110):
    obs, reward, terminated, truncated, info = env.step(0)
    if i > 105:
        img = env.render()
        plt.imsave(f"frogger_frames/frogger_{i}.png", img)

timestep = i

running = True
while running:
    img = env.render()
    plt.imsave(f"frogger_frames/frogger_{timestep}.png", img)
    action = input("Enter action: ")
    obs, reward, terminated, truncated, info = env.step(int(action))
    timestep += 1
    if terminated or truncated:
        break
