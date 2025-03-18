# Plot the performance of the agent
import matplotlib.pyplot as plt
import numpy as np
# import the reward data
timesteps = np.arange(5000)
raw_reward_data = np.loadtxt("frogger_dqn_rewards.txt")
improved_reward_data = np.loadtxt("frogger_dqn_rewards_improved.txt")

# Define smoothing function
def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Apply moving average smoothing
smoothed_rewards = moving_average(raw_reward_data, window_size=100)
smoothed_imrpved_rewards = moving_average(improved_reward_data, window_size=100)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(timesteps, raw_reward_data, alpha=0.2, label="Raw DQN Rewards", color='red')
plt.plot(timesteps, improved_reward_data, alpha=0.3, label="Raw LLM Guided DQN Rewards", color='blue')
plt.plot(timesteps[:len(smoothed_rewards)], smoothed_rewards, label="DQN Training Rewards per 100 Iterations", color='red')
plt.plot(timesteps[:len(smoothed_imrpved_rewards)], smoothed_imrpved_rewards, label="LLM-DQN Training Rewards per 100 Iterations", color='blue')

categories = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
values1 = [1, 15.17, 20.02, 20.4, 20.81, 19.82, 23.25, 14.69, 18.83, 21.32, 23.76]
values2 = [0.05, 16.29, 12.52, 15.72, 15.57, 13.22, 13.99, 17.64, 12.23, 12.4, 15.78]

width = 200  # Adjust width for better visualization
x = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
plt.bar(x - width/2, values2, width=width, label="DQN Model Rewards", color='red', alpha=0.3)
plt.bar(x + width/2, values1, width=width, label="LLM-DQN Model Rewards", color='blue', alpha=0.3)





plt.xlabel("Iterations")
plt.ylabel("Rewards")
plt.title("DQN vs LLM-DQN")
plt.legend()
plt.show()

# plot the reward data


# plt.plot(reward_data)
# plt.show()