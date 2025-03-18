from utils import get_env
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from scipy.stats import linregress

fig, axes = plt.subplots(1, 3, figsize=(16, 8))
fig.suptitle('Episodic Rewards over 5 Lives', fontsize=26)
plot_x = np.arange(1, 6, 1)

rewards_vs_completion_tokens = []

for plot_index, reasoning_effort in enumerate(['low', 'medium', 'high', 'default']):
    if reasoning_effort != 'default':
        axes[plot_index].set_title(f'Reasoning effort {reasoning_effort}', fontsize=20)
        axes[plot_index].set_xlabel('Lives', fontsize=16)
        axes[plot_index].set_ylabel('Cumulative Rewards', fontsize=16)
        axes[plot_index].tick_params(axis='both', labelsize=16)
        axes[plot_index].set_xticks(plot_x)
        axes[plot_index].set_ylim(0, 50)
    max_steps=1000
    seed = 1
    for model in ['o3-mini-2025-01-31', 'qwen-qwq-32b']:
        for past_steps in [0, 3, 'all']:
            for show_rewards in [False, True]:
                for exploration in [False, True]:
                    for feedback in [False, True]:
                        for temperature in [1.0, 0.6]:
                            trace_path = f'traces/{model}_{reasoning_effort}_past_{past_steps}_rewards_{"show" if show_rewards else "hide"}_seed_{seed}_temp_{temperature}{"_explore" if exploration else ""}{"_round_2" if feedback else ""}.json'
                            if not os.path.exists(trace_path):
                                continue
                            with open(trace_path) as f:
                                config_name = f"{model} Reasoning: {reasoning_effort}, Past steps: {past_steps}, Rewards: {'show' if show_rewards else 'hide'}{', Explore' if exploration else ''}{', Round 2' if feedback else ''}"
                                print(f"# {config_name}")
                                trace = json.load(f)[1:]
                                env = get_env(oc=True, oc_obs_mode='ori', framestack=1, repeat_action_probability=0)
                                env.reset()
                                rewards = 0
                                completion_tokens = []
                                deadpoints = []
                                lives = env._env.unwrapped.ale.lives()
                                llm_step = 0
                                for i, t in enumerate(trace):
                                    if 'llm_output' in t:
                                        completion_tokens.append(t['llm_output']['usage']['completion_tokens'])
                                    if not t['enforced_noop']:
                                        llm_step += 1
                                    if t['done']:
                                        rewards += t['reward']
                                        deadpoints.append((llm_step+1, rewards, trace[i-2], trace[i-1], t))
                                        break
                                    obs, reward, terminated, truncated, info = env.step(t['action'])
                                    assert(reward == t['reward'])
                                    rewards += t['reward']
                                    new_lives = env._env.unwrapped.ale.lives()
                                    if new_lives < lives:
                                        deadpoints.append((llm_step+1, rewards, trace[i-2], trace[i-1], t))
                                        lives = new_lives
                                
                                median = np.median(completion_tokens)
                                rewards_vs_completion_tokens.append((rewards, reasoning_effort, median, median - np.percentile(completion_tokens, 25), np.percentile(completion_tokens, 75) - median))
                                # print(rewards_vs_completion_tokens)
                                
                                if reasoning_effort != 'default':
                                    axes[plot_index].plot(plot_x, [d[1] for d in deadpoints], label=f'Past steps: {past_steps}, Rewards: {"show" if show_rewards else "hide"}')
                                    axes[plot_index].legend(loc='upper left')

                                for i, t in enumerate(deadpoints):
                                    print(f"## Life {i+1} ({config_name})")
                                    llm_step, rewards, t1, t2, t3 = t
                                    if t3['enforced_noop']:
                                        t = t2
                                        tp = t1
                                    else:
                                        t = t3
                                        tp = t2
                                    # print(f"* Step: {llm_step}")
                                    # print(f"* Previous and current states:\n    <p float='left'><img src='data:image/png;base64,{tp['frame']}' width='300'><img src='data:image/png;base64,{t['frame']}' width='300'></p>")
                                    # print(f"* Action: {index_to_action(t['action'])}")
                                    # print(f"* Reward for this life: {rewards - deadpoints[i-1][1] if i > 0 else rewards}")
                                    # # print(f"* LLM Prompt: {t['llm_input']}")
                                    # print(f"* LLM Reasoning: {json.loads(t['llm_output']['choices'][0]['message']['content'])['reasoning']}")
                                    # print(f"* LLM Prompt Tokens: {t['llm_output']['usage']['prompt_tokens']}")
                                    # if reasoning_effort != 'default':
                                    #     print(f"* LLM Reasoning Tokens: {t['llm_output']['usage']['completion_tokens_details']['reasoning_tokens']}")
                                    #     print(f"* LLM Completion Tokens: {t['llm_output']['usage']['completion_tokens'] - t['llm_output']['usage']['completion_tokens_details']['reasoning_tokens']}")
                                    # else:
                                    #     print(f"* LLM Completion Tokens: {t['llm_output']['usage']['completion_tokens']}")
                                    # print('<div style="page-break-after: always;"></div>\n')

fig.savefig('life_rewards.png')        

rewards, reasoning_efforts, median_completion_tokens, down_errors, up_errors = zip(*rewards_vs_completion_tokens)
def reasoning_effort_to_color(e):
    return 'tab:red' if e == 'low' else 'tab:blue' if e == 'medium' else 'tab:green' if e == 'high' else 'tab:purple'

reg_result = linregress(median_completion_tokens, rewards)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Episodic Rewards vs Number of Completion Tokens')
ax.set_xlabel('Number of Completion Tokens (Q1, Median, Q3)')
ax.set_ylabel('Episodic Rewards')
ax.errorbar(x=median_completion_tokens, y=rewards, ecolor='gray', xerr=[down_errors, up_errors], fmt='none', capsize=5)
for reasoning_effort in ['low', 'medium', 'high', 'default']:
    x, y = zip(*[(m, r) for (m, r, e) in zip(median_completion_tokens, rewards, reasoning_efforts) if e == reasoning_effort])
    ax.scatter(x=x, y=y, color=reasoning_effort_to_color(reasoning_effort), label='QwQ-32B' if reasoning_effort == 'default' else f'o3-mini {reasoning_effort}', s=80)
ax.plot(median_completion_tokens, reg_result.intercept + reg_result.slope*np.array(median_completion_tokens), color='black', linestyle='--', label=f'Fitted line (r={reg_result.rvalue:.3f})')
ax.legend()
ax.grid(True)
fig.savefig('rewards_vs_completion_tokens.png')
