import matplotlib.pyplot as plt
import numpy as np
from ocatari.vision.game_objects import NoObject
from dotenv import load_dotenv
import os
from tenacity import retry, stop_after_attempt, wait_fixed
from utils import get_env, numpy_to_base64, frames_to_video, base64_to_numpy
import json
import os
import concurrent.futures
from openai import OpenAI
import traceback

@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_openai(model, reasoning_effort, prompt, step, trace, past_steps, show_rewards, seed, temperature, print_log):
    content = []

    steps_content = []
    for s in range(step-1, 0, -1):
        if trace[s]['enforced_noop']:
            continue
        # Only include past k steps
        if past_steps != 'all' and len(steps_content) >= past_steps + 1:
            break
        a = index_to_action(trace[s]["action"])
        r = trace[s-1]['reward'] + trace[s]['reward']
        objs = objs_to_str(trace[s]["objs"])
        llm_step = sum(0 if t['enforced_noop'] else 1 for t in trace[:s+1])
        steps_content.append(f'Step: {llm_step}, action: {a}{f", reward: {r}" if show_rewards else ""}, game objects: {objs}')
    steps_content.reverse()
    content.append({"type": "text", "text": '\n'.join(steps_content) + '\n\n' + prompt})

    if print_log:
        for c in content:
            if c['type'] == 'text':
                print(c['text'])
            elif c['type'] == 'image_url':
                import base64
                import io
                from matplotlib import pyplot as plt
                import matplotlib.image as mpimg
                i = base64.b64decode(c['image_url']['url'].removeprefix('data:image/png;base64,'))
                i = io.BytesIO(i)
                i = mpimg.imread(i, format='PNG')
                plt.figure()
                plt.imshow(i, interpolation='nearest')
                plt.show()

    message_list = [
        {
            "role": 'user',
            "content": content
        }
    ]
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    completion = client.chat.completions.create(
        model=model,
        reasoning_effort=reasoning_effort,
        messages=message_list,
        seed=seed,
        temperature=temperature,
        response_format={ "type": "json_schema", "json_schema": {
            "name": "pick_next_action",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "game_state": { "type": "string" },
                    "reasoning": { "type": "string" },
                    "action": {
                        "type": "string",
                        "enum": ["NOOP", "UP", "RIGHT", "LEFT", "DOWN"]
                    }
                },
                "additionalProperties": False,
                "required": ["game_state", "reasoning", "action"]
            }
        }}
    )
    res = completion.choices[0].message.content
    res = json.loads(res)
    return res['action'], message_list, completion.model_dump()


def extract_objs(env):
    return [{'name': obj.__class__.__name__, 'x': obj.x, 'y': obj.y, 'w': obj.w, 'h': obj.h} for obj in env.objects if obj != NoObject() and obj.visible]

def objs_to_str(objs):
    return ', '.join(f"{obj['name']} at ({obj['x']}, {obj['y']}) size ({obj['w']}, {obj['h']})" for obj in objs)

def action_to_index(action: str) -> int:
    actions_dict = { "NOOP": 0, "UP": 1, "RIGHT": 2, "LEFT": 3, "DOWN": 4 }
    return actions_dict[action]

def index_to_action(index: int) -> str:
    actions_dict = { "NOOP": 0, "UP": 1, "RIGHT": 2, "LEFT": 3, "DOWN": 4 }
    return list(actions_dict.keys())[index]

def render_frame(env):
    env.detect_objects()
    objs = extract_objs(env)
    frame = np.flip(np.rot90(env.render(), k=-1), axis=1)
    return frame, objs


def main(model, temperature, print_log, max_steps, initial_pause_steps):
    load_dotenv()
    traces_folder = 'traces'
    os.makedirs(traces_folder, exist_ok=True)

    def run_experiment(model, reasoning_effort, past_steps, show_rewards, seed, temperature, max_steps, initial_pause_steps, feedback):
        env = get_env(oc=True, oc_obs_mode='ori', framestack=1, repeat_action_probability=0)
        experiment_name = f'{model}_{reasoning_effort}_past_{past_steps}_rewards_{"show" if show_rewards else "hide"}_seed_{seed}_temp_{temperature}'
        if feedback:
            experiment_name += "_round_2"
        output_file_name = f'{traces_folder}/{experiment_name}.json'
        trace = []

        def reset_env():
            nonlocal step, env, frames_to_skip
            state, info = env.reset()
            frame, objs = render_frame(env)
            trace.append({"step": step, "action": 0, "state": numpy_to_base64(state), "frame": numpy_to_base64(frame), "objs": objs, "reward": 0, "done": False, "enforced_noop": False})
            frames_to_skip = initial_pause_steps
            step += 1
            return frame, objs

        step = 0
        frame, objs = reset_env()
        
        while True:
            if frames_to_skip > 0:
                state, reward, terminated, truncated, _ = env.step(0)
                frame, objs = render_frame(env)
                trace.append({'step': step, 'action': 0, 'state': numpy_to_base64(state), 'frame': numpy_to_base64(frame), 'objs': objs, 'reward': 0,
                    'done': terminated or truncated, 'enforced_noop': (initial_pause_steps-frames_to_skip) % 2 == 1})
                frames_to_skip -= 1
                step += 1
                if step >= max_steps:
                    break
                continue

            llm_step = sum(0 if t['enforced_noop'] else 1 for t in trace[:step])

            # You can play at most {max_steps} steps in the game and you must maximize the cumulative reward of all steps.
            prompt = f"""You are an expert gamer and your goal is to choose the best action to beat the game.{feedback}
The game objects are given by their top-left corner's (x, y) positions followed by their width and height in (w, h).
Think about all possible actions and why each action is or is not the best action to take. You are at step {llm_step} and the potential actions you can take are NOOP, UP, RIGHT, LEFT, DOWN.
Output in this JSON format: {{game_state: describe the current game state in detail, reasoning: reasoning for choosing an action, action: the chosen action}}"""

            action, llm_input, llm_output = call_openai(model, reasoning_effort, prompt, step, trace,
                past_steps=past_steps, show_rewards=show_rewards, seed=seed, temperature=temperature, print_log=print_log)
            action = action_to_index(action)

            state, reward, terminated, truncated, _ = env.step(action)
            frame, objs = render_frame(env)
            trace.append({'step': step, 'llm_input': llm_input, 'llm_output': llm_output,
                          'action': action, 'state': numpy_to_base64(state), 'frame': numpy_to_base64(frame), 'objs': objs, 'reward': reward,
                          'done': terminated or truncated, "enforced_noop": False})
            if print_log:
                plt.figure()
                plt.imshow(frame, interpolation='nearest')
                plt.show()
            step += 1
            if step >= max_steps:
                break

            if terminated or truncated:
                # frame, objs = reset_env()
                # continue
                break

            with open(output_file_name, 'w+') as f:
                f.write(json.dumps(trace))

            # Enforced No-op after each action
            state, reward, terminated, truncated, _ = env.step(0)
            frame, objs = render_frame(env)
            trace.append({'step': step, 'action': 0, 'state': numpy_to_base64(state), 'frame': numpy_to_base64(frame), 'objs': objs, 'reward': reward,
                          'done': terminated or truncated, 'enforced_noop': True})
            step += 1
            if step >= max_steps:
                break

            if terminated or truncated:
                # frame, objs = reset_env()
                # continue
                break
        
        with open(output_file_name, 'w+') as f:
            f.write(json.dumps(trace))
        
        env.close()

    def safe_run_experiment(args):
        model, reasoning_effort, past_steps, show_rewards, seed, temperature, max_steps, initial_pause_steps, feedback = args
        try:
            run_experiment(model, reasoning_effort, past_steps, show_rewards, seed, temperature, max_steps, initial_pause_steps, feedback)
        except Exception as e:
            print(f'Exception in (model={model}, reasoning_effort={reasoning_effort}, past_steps={past_steps}, show_rewards={show_rewards}')
            print(e)
            print(traceback.format_exc())
            return None

    configs = []
    for reasoning_effort in ['high']:
        for past_steps in [0]:
            for show_rewards in [True]:
                for seed in [1]:
                    with open('feedback_round_1.txt') as f:
                        feedback = f'\nBelow are a game analyzer\'s observations and advise for your previous episode of game play: {f.read()}'
                    configs.append((model, reasoning_effort, past_steps, show_rewards, seed, temperature, max_steps, initial_pause_steps, feedback))

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(safe_run_experiment, configs))

if __name__ == "__main__":
    main(model='o3-mini-2025-01-31', temperature=1.0, print_log=True, max_steps=1000, initial_pause_steps=108)
