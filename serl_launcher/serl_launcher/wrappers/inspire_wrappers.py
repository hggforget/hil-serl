import gymnasium as gym
import numpy as np
import queue
import cv2
import threading

from gymnasium.spaces import flatten_space, flatten
from franka_env.utils.transformations import decompress_image

class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [v for k, v in img_array.items()], axis=0
            )
            cv2.imshow(self.name, frame)
            cv2.waitKey(1)

class InsprieWrapper(gym.ObservationWrapper, gym.ActionWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        self.display_image = self.config.DISPLAY_IMAGE
        self.proprio_keys = proprio_keys
        self.cameras_config = self.config.CAMERAS
        self.state_config = self.config.STATE
        self.action_config = self.config.ACTION
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )
        
        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space["images"]),
            }
        )
        
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.url)
            self.displayer.start()
        
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(self.action(action))
        if 'action' in info:
            info['action'] = self.reverse_process_action(info['action'])
        return self.observation(observation), reward, terminated, truncated, info

    def observation(self, obs):
        images = {k: obs[v['alias']] for k, v in self.cameras_config.items()}
        images = self.process_images(images)
        state = {k: obs[v['alias']] for k, v in self.state_config.items()}
        state = self.process_state(state)
        obs = {
            "state": state,
            **images,
        }
        if self.display_image:
            self.img_queue.put(images)
        return obs

    def action(self, action):
        return self.process_action(action)

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info
    
    def process_images(self, images):
        return {name: cv2.resize(decompress_image(image_array), tuple(reversed(self.cameras_config[name]['shape'][:2]))) for name, image_array in images.items()}
            
    def process_state(self, state):
        state = {key: state[key] for key in self.proprio_keys}
        state = {k: v[:6] if '_hand_angle' else v for k, v in state.items()}
        state = {k: v / self.state_config[k]['scale'] for k, v in state.items()}
        state = flatten(self.proprio_space, state)
        state = (state - self.config.NORM_STATS['state_min']) / (self.config.NORM_STATS['state_max'] - self.config.NORM_STATS['state_min']) * 2 - 1
        return state
    
    @staticmethod
    def array2dict(array, space):
        index = 0
        result = {}
        for k, v in space.spaces.items():
            if isinstance(v, gym.spaces.Dict):
                raise Exception('Nested dicts are not supported')
            size = v.shape[0]
            result[k] = array[index:index+size]
            index += size
        return result

    def process_action(self, action):
        action = (action + 1) / 2 * (self.config.NORM_STATS['action_max'] - self.config.NORM_STATS['action_min'] + 1e-5) + self.config.NORM_STATS['action_min']
        action = self.array2dict(action, self.action_dict_space)
        action = {self.action_config[k]['alias']: v * self.action_config[k]['scale'] for k, v in action.items()} 
        return action

    def reverse_process_action(self, action):
        action = {key: action[key] for key in self.action_dict_space.spaces.keys()}
        action = {k: v[:6] if '_hand_angle' else v for k, v in action.items()}
        action = {k: v / self.action_config[k]['scale'] for k, v in action.items()}
        action = flatten(self.action_dict_space, action)
        action = (action - self.config.NORM_STATS['action_min']) / (self.config.NORM_STATS['action_max'] - self.config.NORM_STATS['action_min']) * 2 - 1
        return action
    
    def close(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()