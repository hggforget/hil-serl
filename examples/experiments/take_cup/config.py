import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    DualQuat2EulerWrapper,
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.insprie_env import DefaultInsprieEnvConfig
from serl_launcher.wrappers.inspire_wrappers import InsprieWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.take_cup.wrapper import CupEnv

class EnvConfig(DefaultInsprieEnvConfig):
    SERVER_URL = "http://192.168.10.53:50055/"
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 300
    CAMERAS = {
    'cam_body': 
        {
            'shape': (480, 640, 3),
            'alias': 'image_body'
        },
    'cam_head_right': 
        {
            'shape': (360, 640, 3),
            'alias': 'image_head_right'
        }
    }
    ACTION = {
        'relative_arm_right': {
            'scale': 1.0,
            'alias': 'right_arm_action'
        },
        'relative_hand_right': {
            'scale': 1000.0,
            'alias': 'right_hand_action'
        }
    }
    STATE = {
        'relative_arm_right': {
            'scale': 1.0,
            'alias': 'right_arm_pose'
        },
        'relative_hand_right': {
            'scale': 2000.0,
            'alias': 'right_hand_angle'
        }
    }
    POSE_LIMIT_LOW = np.array([-0.0,-0.1,-0.4])
    POSE_LIMIT_HIGH = np.array([0.,0.1,0.0])
    ANGLE_LIMIT_LOW = np.zeros(())
    ANGLE_LIMIT_HIGH = np.ones(()) * np.pi / 6.0
    RELATIVE_POSE_LIMIT_LOW = np.array([-0.01,-0.01,-0.01])
    RELATIVE_POSE_LIMIT_HIGH = np.array([0.01,0.01,0.01])
    RELATIVE_ANGLE_LIMIT_LOW = np.zeros(())
    RELATIVE_ANGLE_LIMIT_HIGH = np.ones(()) * np.pi / 18.0
    RELATIVE_HAND_ANGLE_LIMIT_LOW = np.array([-100, -100, -100, -100, -100, -100])
    RELATIVE_HAND_ANGLE_LIMIT_HIGH = np.array([100, 100, 100, 100, 100, 100])
    STATE_MEAN = np.array([-0.01195743,  0.05904795, -0.15037453, -0.4224188 ,  0.33944982,
       -0.04863422,  0.45423675,  0.11292544,  0.08474214,  0.1670834 ,
        0.33722413,  0.809032  ], dtype=np.float32)
    STATE_STD = np.array([0.03729864, 0.03876481, 0.06630935, 0.1981131 , 0.18376806,
       0.09266762, 0.05690856, 0.07314496, 0.0782358 , 0.14691487,
       0.18487382, 0.01991429], dtype=np.float32)
    ACTION_MEAN = np.array([-0.0136115 ,  0.06203553, -0.15717955, -0.447127  ,  0.35129535,
       -0.04741895,  0.69626546,  0.7581536 ,  0.9178515 ,  0.8621609 ,
        0.41211852,  0.14999607], dtype=np.float32)
    ACTION_STD = np.array([3.9725382e-02, 3.8907744e-02, 5.9021130e-02, 1.8735009e-01,
       1.8075813e-01, 9.7725756e-02, 2.0460993e-01, 3.0272129e-01,
       1.3507724e-01, 1.6023266e-01, 1.7862214e-01, 3.9339066e-06],
      dtype=np.float32)
    STATE_MIN = np.array([-0.044400003, -0.071961015, -0.08462399, -0.1457818, -0.0050920933, -0.41122085, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    STATE_MAX = np.array([0.034888998, 0.053503007, 0.057633013, 0.20218806, 0.7957855, 0.0011863295, 0.4235, 0.656, 0.4915, 0.713, 0.6765, 0.9235], dtype=np.float32)
    ACTION_MIN = np.array([-0.060302943, -0.099265695, -0.06536576, -0.15862897, -0.017979145, -0.41128758, 0.334, 0.27, 0.328, 0.319, 0.503, 0.0], dtype=np.float32)
    ACTION_MAX = np.array([0.05008167, 0.066430956, 0.056927115, 0.20228474, 0.8178695, 0.022086907, 1.0, 1.0, 0.94, 1.0, 1.0, 0.32], dtype=np.float32)
    
    NORM_STATS = {
        'state_mean': STATE_MEAN,
        'state_std': STATE_STD,
        'action_mean': ACTION_MEAN,
        'action_std': ACTION_STD,
        'state_min': STATE_MIN,
        'state_max': STATE_MAX,
        'action_min': ACTION_MIN,
        'action_max': ACTION_MAX,
    }



class TrainConfig(DefaultTrainingConfig):
    image_keys = ["cam_body", "cam_head_right"]
    classifier_keys = ["cam_body"]
    proprio_keys = ["relative_arm_right", "relative_hand_right"]
    replay_buffer_capacity: int = 100000
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 300
    batch_size = 32
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-insprie"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = CupEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        # env = GripperCloseEnv(env)
        # if not fake_env:
        #     env = SpacemouseIntervention(env)
        # env = RelativeFrame(env)
        # env = Quat2EulerWrapper(env)
        env = InsprieWrapper(env, proprio_keys=self.proprio_keys)
        # env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            print(os.path.abspath("tasks/take_cup/grasp_classifier_ckpt/"))
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("tasks/take_cup/grasp_classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                out = sigmoid(classifier(obs))
                print(f'reward: {out}')
                # added check for z position to further robustify classifier, but should work without as well
                return int((out > 0.5)[0])

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env