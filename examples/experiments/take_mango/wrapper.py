import copy
import time
from franka_env.utils.rotations import euler_2_quat
from scipy.spatial.transform import Rotation as R
import numpy as np
import requests
# from pynput import keyboard

from franka_env.envs.insprie_env import InsprieEnv

class MangoEnv(InsprieEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)