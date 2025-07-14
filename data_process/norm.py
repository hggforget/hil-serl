import numpy as np
import cv2
import argparse
import pickle as pkl
from tqdm import tqdm
from utils import find_all_hdf5, load_hdf5, decompress_image, decompress_images, relative_pose
from episodes import Episode
import numpy as np
import cv2
import pathlib
import sys
directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
from examples.experiments.mappings import CONFIG_MAPPING
from examples.experiments.config import DefaultTrainingConfig
from gymnasium.spaces import flatten


if __name__ == '__main__':
    task_name = 'take_cup'
    config: DefaultTrainingConfig = CONFIG_MAPPING[task_name]()
    assert task_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=True,
    )
    with open(f'/mnt/hil-serl/tasks/take_cup/init/demos/2025-02-11_15-51-36/episodes.pickle', "rb") as f:
        episodes = pkl.load(f)
    states, actions = [], []
    for episode in tqdm(episodes):
        state = {key: getattr(episode, key) for key in env.proprio_space.spaces.keys()}
        state = {k: v[:,:6] if '_hand_' else v for k, v in state.items()}
        state = {k: v / env.state_config[k]['scale'] for k, v in state.items()}
        episode_len = len(episode)
        state_len = sum(v.shape[-1] for v in state.values())
        state_array = np.ones((state_len, episode_len), dtype=np.float32)
        index = 0
        for key in env.proprio_space.spaces.keys():
            v = state[key]
            size = v.shape[-1]
            state_array[index: index + size] = v.T
            index += size
        states.append(state_array.T)
        action = {key: episode.action[key] for key in env.action_dict_space.spaces.keys()}
        action = {k: v[:,:6] if '_hand_' else v for k, v in action.items()}
        action = {k: v / env.action_config[k]['scale'] for k, v in action.items()}
        action_len = sum(v.shape[-1] for v in action.values())
        action_array = np.zeros((action_len, episode_len), dtype=np.float32)
        index = 0
        for key in env.action_dict_space.spaces.keys():
            v = action[key]
            size = v.shape[-1]
            action_array[index: index + size] = v.T
            index += size
        actions.append(action_array.T)
        
    states = np.concatenate(states, 0)
    actions = np.concatenate(actions, 0)
    states_min, states_max = np.min(states, 0), np.max(states, 0)
    actions_min, actions_max = np.min(actions, 0), np.max(actions, 0)
    print('states',states_min, states_max)
    print('actions',actions_min, actions_max)

    
