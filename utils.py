import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, RecordVideo, RecordEpisodeStatistics
import numpy as np
import torch
from collections import deque
from gymnasium import spaces
import base64
from io import BytesIO
from PIL import Image
from ocatari.core import OCAtari
from ocatari.vision.game_objects import NoObject
import os
import pickle
import cv2
import time
import json

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self, seed=None, options=None):
        ob, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), info

    def step(self, action):
        ob, reward, done, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.stack(self._frames)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


class FrameStackOC():
    def __init__(self, env, k, knn):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        self.env = env
        self.k = k
        self.knn = knn
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)
        self.action_space = env.action_space
    
    def detect_objects(self):
        result = self.env.detect_objects()
        self.objects = self.env.objects
        return result
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()

    def reset(self, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        ob = extract_objs(self.env, return_tensor=True, knn=self.knn)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), info

    def step(self, action):
        _, reward, done, truncated, info = self.env.step(action)
        ob = extract_objs(self.env, return_tensor=True, knn=self.knn)
        self.frames.append(ob)
        return self._get_ob(), reward, done, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFramesOC(list(self.frames))


class LazyFramesOC(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = torch.cat([torch.cat((torch.full(frame.size()[:-1] + (1,), i), frame), dim=-1) for i, frame in enumerate(self._frames)], dim=0)
            self._frames = None
        return self._out
    
    def to_tensor(self):
        return self._force()

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]



class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


def wrap_recording(env, video_folder, episode_trigger, name_prefix):
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=episode_trigger, name_prefix=name_prefix)
    env = RecordEpisodeStatistics(env, buffer_length=1)
    return env

def get_env(oc=False, oc_obs_mode='obj', hud=False, repeat_action_probability=0.25, framestack=4, episodic=False):
    if oc:
        env = OCAtari("ALE/Frogger-v5", mode="vision", render_mode="rgb_array", obs_mode=oc_obs_mode, hud=hud,
                      render_oc_overlay=True, frameskip=4, repeat_action_probability=repeat_action_probability, buffer_window_size=framestack)
        if episodic:
            env._env = EpisodicLifeEnv(env._env)
    else:
        env = gym.make("ALE/Frogger-v5", render_mode="rgb_array", frameskip=4, repeat_action_probability=repeat_action_probability)
        env = GrayscaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, (84, 84))
        if episodic:
            env = EpisodicLifeEnv(env)
    if framestack > 1 and not oc:
        env = FrameStack(env, framestack)
    return env


def image_to_base64(path):
    with open(path, "rb") as image_file:
        binary_data = image_file.read()
        base_64_encoded_data = base64.b64encode(binary_data)
        base64_string = base_64_encoded_data.decode('utf-8')
        return base64_string

def numpy_to_base64(numpy_array, resize_to=None):
    """
    Converts a NumPy array to a Base64 encoded PNG image.

    Args:
        numpy_array (numpy.ndarray): The NumPy array to convert.

    Returns:
        str: A Base64 encoded string of the PNG image.
    """
    image = Image.fromarray(numpy_array)
    if resize_to:
        image = image.resize(resize_to)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def base64_to_numpy(base64_string):
    """
    Decodes a base64 string to a NumPy array.

    Args:
        base64_string: The base64 encoded string.

    Returns:
        A NumPy array representing the decoded data.
    """
    try:
        decoded_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(decoded_bytes))
        return np.asarray(image)
    except Exception as e:
        print(f"Error decoding base64 string: {e}")
        return None



def load_demonstrations(ids=[2], stack_frames=False, process=True):
    # load expert demonstrations
    trace = []
    for i in ids:
        with open(f'traces/trace_{i}.pkl', 'rb') as f:
            trace += pickle.load(f)

    def process_frame(frame):
        if process:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(
                frame, (84, 84), interpolation=cv2.INTER_AREA
            )
        return frame

    demonstrations = []
    for obj in trace:
        demonstrations.append((process_frame(obj['state']), obj['action'], obj['reward'], process_frame(obj['next_state']), obj['done']))

    if stack_frames:
        demonstrations = [(
            np.stack([demonstrations[j][0] for j in range(i-3,i+1)]),
            demonstrations[i][1],
            demonstrations[i][2],
            np.stack([demonstrations[j][3] for j in range(i-3,i+1)]),
            demonstrations[i][4])
            for i in range(4, len(demonstrations))]
    
    return demonstrations


def load_llm_demonstrations(oc=False, trace_path='traces/o3-mini-2025-01-31_high_past_3_rewards_show_seed_1_temp_1.0.json'):
    # load expert demonstrations
    with open(trace_path, 'r') as f:
        trace = json.load(f)[1:]
        actions = []
        for t in trace:
            if t['done']:
                break
            actions.append(t['action'])
    env = get_env(oc=oc, repeat_action_probability=0)
    state, _ = env.reset()
    buffer = []
    total_reward = 0
    for i,(action, t) in enumerate(zip(actions, trace)):
        next_state, reward, terminated, truncated, _ = env.step(action)
        assert reward == t['reward']
        done = terminated or truncated
        buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        if done:
            break
    print('total actions:', len(actions))
    print('total reward:', total_reward)
    print('total length:', len(buffer))
    return buffer

def record_video(env, select_action, video_folder, episode_trigger, name_prefix):
    if isinstance(env, gym.Env):
        env = wrap_recording(env, video_folder=video_folder, episode_trigger=episode_trigger, name_prefix=name_prefix)
    else:
        env._env = wrap_recording(env._env, video_folder=video_folder, episode_trigger=episode_trigger, name_prefix=name_prefix)
    state, info = env.reset()
    while True:
        action = select_action(env=env, state=state)
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        if terminated or truncated:
            break
    env.close()
    if isinstance(env, gym.Env):
        e = env
    else:
        e = env._env
    return e.return_queue[0], e.length_queue[0], e.time_queue[0]

def frames_to_video(video_folder, video_name, frames, fps):
    os.makedirs(video_folder, exist_ok=True)
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=fps)
    path = os.path.join(video_folder, f"{video_name}.mp4")
    clip.write_videofile(path)

def get_obj_classes():
    classes = ["Frog", "Log", "Alligator", "Turtle", "LadyFrog", "Snake", "HappyFrog", "AlligatorHead", "Fly", "Car"]
    return classes

def extract_objs(env, return_tensor=False, knn=None):
    env.detect_objects()
    objs = [{'name': obj.__class__.__name__, 'x': obj.x, 'y': obj.y, 'w': obj.w, 'h': obj.h} for obj in env.objects if obj != NoObject() and obj.visible]
    if return_tensor:
        classes = get_obj_classes()
        x = []
        if knn is not None:
            for obj in objs:
                if obj['name'] == 'Frog':
                    frog = obj
                    break
            obj_indices = np.argsort([abs(obj['x'] - frog['x']) + abs(obj['y'] - frog['y']) for obj in objs])[:knn]
            objs = [objs[i] for i in obj_indices]
            # print(objs)
        for obj in objs:
            # +1 to account for first padding class
            obj_class = float(classes.index(obj['name']) + 1)
            x.append(torch.tensor([obj_class, obj['x'], obj['y'], obj['w'], obj['h']]))
        return torch.stack(x)
    else:
        return objs

def objs_to_str(objs):
    return ', '.join(f"{obj['name']} at ({obj['x']}, {obj['y']}) size ({obj['w']}, {obj['h']})" for obj in objs)

def action_to_index(action: str) -> int:
    actions_dict = { "NOOP": 0, "UP": 1, "RIGHT": 2, "LEFT": 3, "DOWN": 4 }
    return actions_dict[action]

def index_to_action(index: int) -> str:
    actions_dict = { "NOOP": 0, "UP": 1, "RIGHT": 2, "LEFT": 3, "DOWN": 4 }
    return list(actions_dict.keys())[index]

def render_frame(env, return_tensor=False, knn=None):
    objs = extract_objs(env, return_tensor, knn)
    frame = np.flip(np.rot90(env.render(), k=-1), axis=1)
    return frame, objs

def record_video_oc(select_action, video_folder, video_name, max_length=2000, knn=None):
    env = get_env(process=False, oc=True, knn=knn)
    state, info = env.reset()
    frame, _ = render_frame(env)
    frames = [frame]
    total_reward = 0
    total_length = 0
    start_time = time.time()
    
    while total_length <= max_length:
        action = select_action(env=env, state_objs=state)
        print('iteration', total_length, 'action', action)
        next_state, reward, terminated, truncated, info = env.step(action)
        frame, _ = render_frame(env)
        frames.append(frame)
        state = next_state
        total_reward += reward
        total_length += 1
        if terminated or truncated:
            break
    total_time = time.time() - start_time
    frames_to_video(video_folder=video_folder, video_name=video_name, frames=frames, fps=15)
    env.close()
    return total_reward, total_length, total_time

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
