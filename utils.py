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


def wrap_deepmind(env):
    """Configure environment for DeepMind-style Atari.
    """
    env = FrameStack(env, 4)
    return env

def wrap_recording(env, video_folder, episode_trigger, name_prefix):
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=episode_trigger, name_prefix=name_prefix)
    env = RecordEpisodeStatistics(env, buffer_length=1)
    return env

def get_env(process=True, oc=False, hud=False, repeat_action_probability=0.25):
    if oc:
        env = OCAtari("ALE/Frogger-v5", mode="vision", render_mode="rgb_array", obs_mode="ori", hud=hud, render_oc_overlay=True, frameskip=4, repeat_action_probability=repeat_action_probability)
    else:
        env = gym.make("ALE/Frogger-v5", render_mode="rgb_array", frameskip=4, repeat_action_probability=repeat_action_probability)
    if process:
        env = GrayscaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, (84, 84))
    if not oc:
        env = wrap_deepmind(env)
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

def record_video(select_action, video_folder, episode_trigger, name_prefix):
    env = get_env()
    env = wrap_recording(env, video_folder=video_folder, episode_trigger=episode_trigger, name_prefix=name_prefix)
    state, info = env.reset()
    while True:
        action = select_action(env=env, state=state)
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        if done:
            break
    env.close()
    return env.return_queue[0], env.length_queue[0], env.time_queue[0]

def frames_to_video(video_folder, video_name, frames, fps):
    os.makedirs(video_folder, exist_ok=True)
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=fps)
    path = os.path.join(video_folder, f"{video_name}.mp4")
    clip.write_videofile(path)

def get_obj_classes():
    classes = ["Frog", "Log", "Alligator", "Turtle", "LadyFrog", "Snake", "HappyFrog", "AlligatorHead", "Fly", "Car"]
    return classes

def extract_objs(env, return_tensor=False):
    env.detect_objects()
    objs = [{'name': obj.__class__.__name__, 'x': obj.x, 'y': obj.y, 'w': obj.w, 'h': obj.h} for obj in env.objects if obj != NoObject() and obj.visible]
    if return_tensor:
        classes = get_obj_classes()
        x = []
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

def render_frame(env, return_tensor=False):
    objs = extract_objs(env, return_tensor)
    frame = np.flip(np.rot90(env.render(), k=-1), axis=1)
    return frame, objs

def record_video_oc(select_action, video_folder, video_name, max_length=2000):
    env = get_env(process=False, oc=True)
    state, info = env.reset()
    frame, state_objs = render_frame(env, return_tensor=True)
    frames = [frame]
    total_reward = 0
    total_length = 0
    start_time = time.time()
    
    while total_length <= max_length:
        action = select_action(env=env, state_objs=state_objs)
        print('iteration', total_length, 'action', action)
        next_state, reward, terminated, truncated, info = env.step(action)
        frame, state_objs = render_frame(env, return_tensor=True)
        frames.append(frame)
        total_reward += reward
        total_length += 1
        if terminated or truncated:
            break
    total_time = time.time() - start_time
    frames_to_video(video_folder=video_folder, video_name=video_name, frames=frames, fps=15)
    env.close()
    return total_reward, total_length, total_time
