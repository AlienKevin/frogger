import gymnasium as gym
import ale_py  # Ensure Atari environments work
import matplotlib.pyplot as plt
from IPython.display import display, Image, clear_output
import numpy as np
import collections
import random
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import re

env = gym.make("ALE/Frogger-v5", render_mode="human", frameskip=4, repeat_action_probability=0, mode=0)

state, info = env.reset()
