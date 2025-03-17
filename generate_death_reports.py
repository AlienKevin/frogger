model='o3-mini-2025-01-31'
temperature=1.0
max_steps=1000
from utils import get_env, index_to_action

import json

results = {}

for reasoning_effort in ['low', 'medium', 'high']:
    for past_steps in [0, 3, 'all']:
        for show_rewards in [False, True]:
            for seed in [1]:
                trace_path = f'traces/{model}_{reasoning_effort}_past_{past_steps}_rewards_{"show" if show_rewards else "hide"}_seed_{seed}_temp_{temperature}.json'
                with open(trace_path) as f:
                    print(f"# Reasoning: {reasoning_effort}, Past steps: {past_steps}, Rewards: {'show' if show_rewards else 'hide'}")
                    trace = json.load(f)[1:]
                    steps = []
                    env = get_env(oc=True, oc_obs_mode='ori', framestack=1, repeat_action_probability=0)
                    env.reset()
                    rewards = 0
                    deadpoints = []
                    lives = env._env.unwrapped.ale.lives()
                    for i, t in enumerate(trace):
                        if t['done']:
                            rewards += t['reward']
                            deadpoints.append((rewards, trace[i-2], trace[i-1], t))
                            break
                        obs, reward, terminated, truncated, info = env.step(t['action'])
                        assert(reward == t['reward'])
                        rewards += t['reward']
                        new_lives = env._env.unwrapped.ale.lives()
                        if new_lives < lives:
                            deadpoints.append((rewards, trace[i-2], trace[i-1], t))
                            lives = new_lives
                        if not t['enforced_noop']:
                            steps.append((t))
                    # print(trace_path, rewards, len(deadpoints))
                    for i, t in enumerate(deadpoints):
                        print(f"## Death {i} (Reasoning: {reasoning_effort}, Past steps: {past_steps}, Rewards: {'show' if show_rewards else 'hide'})")
                        rewards, t1, t2, t3 = t
                        if t3['enforced_noop']:
                            t = t2
                            tp = t1
                        else:
                            t = t3
                            tp = t2
                        print(f"* Step: {t['step']}")
                        print(f"* Previous and current states:\n    <p float='left'><img src='data:image/png;base64,{tp['frame']}' width='300'><img src='data:image/png;base64,{t['frame']}' width='300'></p>")
                        print(f"* Action: {index_to_action(t['action'])}")
                        print(f"* Reward for this life: {rewards - deadpoints[i-1][0] if i > 0 else rewards}")
                        # print(f"* LLM Prompt: {t['llm_input']}")
                        print(f"* LLM Reasoning: {json.loads(t['llm_output']['choices'][0]['message']['content'])['reasoning']}")
                        print(f"* LLM Prompt Tokens: {t['llm_output']['usage']['prompt_tokens']}")
                        print(f"* LLM Reasoning Tokens: {t['llm_output']['usage']['completion_tokens_details']['reasoning_tokens']}")
                        print(f"* LLM Completion Tokens: {t['llm_output']['usage']['completion_tokens'] - t['llm_output']['usage']['completion_tokens_details']['reasoning_tokens']}")
                        print('<div style="page-break-after: always;"></div>\n')

                    # if trace_path == "traces/o3-mini-2025-01-31_high_past_0_rewards_show_seed_1_temp_1.0.json":
                    #     s = []
                    #     for i, step in enumerate(steps):
                    #         objs = objs_to_str(step['objs'])
                    #         if 'llm_output' in step:
                    #             out = step['llm_output']["choices"][0]["message"]["content"]
                    #         s.append(f'Step: {i}{f", reasoning: {out}" if "llm_output" in step else ""}, action: {step["action"]}, reward: {step["reward"]}, game objects: {objs}')
                    #     print('\n'.join(s))
