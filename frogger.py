import gymnasium as gym
import pygame
import numpy as np
import ale_py
import time
import pickle

# Initialize the environment
env = gym.make("ALE/Pitfall-v5", render_mode="rgb_array", frameskip=4, repeat_action_probability=0, mode=0)

# Initialize Pygame
pygame.init()
width, height = 210*2, 160*2  # Display size
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Play Frogger with Gymnasium and Pygame")

# Mapping of Pygame keys to Atari actions
key_mapping = {
    pygame.K_UP: 1,     # Move Up
    pygame.K_DOWN: 4,   # Move Down
    pygame.K_LEFT: 3,   # Move Left
    pygame.K_RIGHT: 2,  # Move Right
}

# Reset the environment
obs, info = env.reset()

for i in range(109):
    env.step(0)

running = True
trace = []
num_traces = 0

while running:
    screen.fill((0, 0, 0))  # Clear screen
    frame = env.render()    # Get the game frame

    # Convert the frame to a Pygame surface
    surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    surf = pygame.transform.scale(surf, (width, height))  # Resize
    screen.blit(surf, (0, 0))  # Draw to screen
    pygame.display.flip()  # Update display

    action = 0  # Default action (do nothing)

    # Handle user input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in key_mapping:
                action = key_mapping[event.key]

    # Step the environment
    state = obs
    obs, reward, terminated, truncated, info = env.step(action)
    trace.append({'state': state, 'action': action, 'reward': reward, 'next_state': obs, 'done': terminated})
    time.sleep(0.1)

    if terminated or truncated:
        with open(f'trace_{num_traces}.pkl', 'wb+') as f:
            pickle.dump(trace, f)
        trace = []
        num_traces += 1
        obs, info = env.reset()  # Restart game when done
        for i in range(109):
            env.step(0)

# Cleanup
env.close()
pygame.quit()
