"""Gym Interface for Franka"""
import os
import numpy as np
import gymnasium as gym
import cv2
import time
import requests
from datetime import datetime

from franka_env.utils.transformations import absolute_pose, relative_pose
from gymnasium.spaces import flatten_space


from franka_env.grpc import RobotControlClient

from datetime import datetime

class DefaultInsprieEnvConfig:
    """Default configuration for FrankaEnv. Fill in the values below."""

    SERVER_URL: str = "http://192.168.10.53:50055/"
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
            },
        'cam_head_left': 
            {
                'shape': (360, 640, 3),
                'alias': 'image_head_left'
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
    DISPLAY_IMAGE: bool = True
    POSE_LIMIT_HIGH = np.zeros((3,))
    POSE_LIMIT_LOW = np.zeros((3,))
    ANGLE_LIMIT_LOW = np.zeros(())
    ANGLE_LIMIT_HIGH = np.zeros(())
    RELATIVE_POSE_LIMIT_HIGH = np.zeros((3,))
    RELATIVE_POSE_LIMIT_LOW = np.zeros((3,))
    RELATIVE_ANGLE_LIMIT_LOW = np.zeros(())
    RELATIVE_ANGLE_LIMIT_HIGH = np.zeros(())
    RELATIVE_HAND_ANGLE_LIMIT_LOW = np.zeros((6,))
    RELATIVE_HAND_ANGLE_LIMIT_HIGH = np.zeros((6,))
    MAX_EPISODE_LENGTH: int = 100
    ARM_TYPE: str = 'right'


##############################################################################


class InsprieEnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultInsprieEnvConfig = None,
        set_load=False,
    ):
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.arm_type = config.ARM_TYPE
        self.lastsent = time.time()
 
        self.hz = hz

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.POSE_LIMIT_LOW,
            config.POSE_LIMIT_HIGH,
            dtype=np.float64,
        )
        self.xyz_relative_bounding_box = gym.spaces.Box(
            config.RELATIVE_POSE_LIMIT_LOW,
            config.RELATIVE_POSE_LIMIT_HIGH,
            dtype=np.float64,
        )
        self.angle_bounding_box = gym.spaces.Box(
            config.ANGLE_LIMIT_LOW,
            config.ANGLE_LIMIT_HIGH,
            dtype=np.float64,
        )
        self.angle_relative_bounding_box = gym.spaces.Box(
            config.RELATIVE_ANGLE_LIMIT_LOW,
            config.RELATIVE_ANGLE_LIMIT_HIGH,
            dtype=np.float64,
        )
        self.hand_relative_bounding_box = gym.spaces.Box(
            config.RELATIVE_HAND_ANGLE_LIMIT_LOW,
            config.RELATIVE_HAND_ANGLE_LIMIT_HIGH,
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_dict_space = gym.spaces.Dict(
                    {
                        "relative_arm_right": gym.spaces.Box(
                            -np.inf, np.inf, shape=(6,)
                        ),  # xyz + quat
                        "relative_hand_right": gym.spaces.Box(0, 1000.0, shape=(6,)),
                    }
                )
        self.action_space = flatten_space(self.action_dict_space)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "relative_arm_right": gym.spaces.Box(
                            -np.inf, np.inf, shape=(6,)
                        ),  # xyz + quat
                        "relative_hand_right": gym.spaces.Box(0, np.inf, shape=(6,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=value['shape'], dtype=np.uint8) 
                                for key, value in config.CAMERAS.items()}
                ),
            }
        )
        self.cycle_count = 0

        if fake_env:
            return
        
        
        self.RobotControlClient = RobotControlClient()

        print("Initialized Insprie")
        
        


    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        new_pose = pose.copy()
        new_pose[:3] = np.clip(
            new_pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        axis_angle = new_pose[3:]
        angle = np.linalg.norm(axis_angle)
        axis = axis_angle / angle
        angle = np.clip(angle, self.angle_bounding_box.low, self.angle_bounding_box.high)
        new_pose[3:] =  axis * angle
        return new_pose

    def clip_relative_safety_box(self, pose: np.ndarray) -> np.ndarray:
        new_pose = pose.copy()
        new_pose[:3] = np.clip(
            new_pose[:3], self.xyz_relative_bounding_box.low, self.xyz_relative_bounding_box.high
        )
        axis_angle = new_pose[3:]
        angle = np.linalg.norm(axis_angle)
        axis = axis_angle / angle
        angle = np.clip(angle, self.angle_relative_bounding_box.low, self.angle_relative_bounding_box.high)
        new_pose[3:] =  axis * angle
        return new_pose
    
    def clip_relative_hand(self, hand_angle: np.ndarray) -> np.ndarray:
        relative_hand_angle = np.clip(
            hand_angle - self.curr_hand_angle, self.hand_relative_bounding_box.low, self.hand_relative_bounding_box.high
        )
        return relative_hand_angle + self.curr_hand_angle

    def step(self, action: dict) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        
        arm_action = np.clip(action['right_arm_action'], self.action_dict_space['relative_arm_right'].low,  self.action_dict_space['relative_arm_right'].high)     
        hand_action = np.clip(action['right_hand_action'], self.action_dict_space['relative_hand_right'].low,  self.action_dict_space['relative_hand_right'].high)
        # [-0.004780866, -0.015523747, -0.004656203, 0.02139324, 0.4007417, -0.19339906]
        # [0.09822413325309753, 0.4802522510290146, 0.3468308076262474, -0.47605710946168894, 0.4283055290199461, 0.5096435121402851, 0.5746194330817926]
        # 采取action后的绝对位姿
        nextpos = absolute_pose(self.initial_arm_pos, arm_action)
        obs = self._send_command(np.concatenate([nextpos, hand_action], -1))
        print("send_action_time:", time.time()-start_time)
        self.curr_path_length += 1
        # time.sleep(1)
        obs = self._get_obs(obs)
        print("action:" , action)
        print("initial_arm_pos", self.initial_arm_pos)
        print("initial_hand_angle", self.initial_hand_angle)
        print("curr_arm_pos", self.curr_arm_pos)
        print("curr_hand_angle", self.curr_hand_angle)
        print("abs_arm_pos:" , nextpos)
        print("abs_hand_angle:" , hand_action)
        reward = self.compute_reward(obs)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        return obs, int(reward), done, False, {"succeed": reward, "action": {'relative_arm_right':arm_action,'relative_hand_right':hand_action}}

    
    # def step(self, action: np.ndarray) -> tuple:
    #     """standard gym step function."""
    #     start_time = time.time()
        
    #     action = np.clip(action, self.action_space.low, self.action_space.high)
        
    #     arm_action = action[:6]
    #     hand_action = action[6:]
        
    #     # 采取action后 相对于当前状态的安全相对位姿
    #     safe_action = self.clip_relative_safety_box(arm_action)
        
    #     # 采取action后的绝对位姿
    #     abs_action = relative_pose(self.initial_arm_pos, absolute_pose(self.curr_arm_pos, safe_action))
    #     # 最终的安全action
    #     safe_abs_action = self.clip_safety_box(abs_action)
    #     nextpos  = absolute_pose(self.initial_arm_pos, safe_abs_action)
    #     scaled_hand_action = self.clip_relative_hand(hand_action * self.action_scale)
    #     self._send_arm_command(nextpos)
    #     self._send_hand_command(scaled_hand_action)
    #     min_coords = self.xyz_bounding_box.low
    #     max_coords = self.xyz_bounding_box.high
    #     penalty = np.any(np.concatenate([(min_coords >= abs_action[:3]), (abs_action[:3] >= max_coords)], -1))
        
    #     safe_relative_curr_action = relative_pose(self.curr_arm_pos, nextpos)
    #     # # 采取action后的绝对位姿
    #     # unsafe_next_pos = absolute_pose(self.initial_arm_pos, arm_action)
    #     # # 采取action后 相对于当前状态的安全相对位姿
    #     # safe_relative_pose = self.clip_relative_safety_box(relative_pose(self.curr_arm_pos, unsafe_next_pos))
    #     # # 相对当前帧的安全action
    #     # safe_relative_action = relative_pose(self.initial_arm_pos, absolute_pose(self.curr_arm_pos, safe_relative_pose))
    #     # # 最终的安全action
    #     # safe_action = self.clip_safety_box(safe_relative_action)
    #     # self.nextpos  = absolute_pose(self.initial_arm_pos, safe_action)
    #     # scaled_hand_action = self.clip_relative_hand(hand_action * self.action_scale)
    #     # self._send_arm_command(self.nextpos)
    #     # self._send_hand_command(scaled_hand_action)

    #     self.curr_path_length += 1
    #     dt = time.time() - start_time
    #     time.sleep(max(0, (1.0 / self.hz) - dt))
        
    #     ob = self._get_obs()
    #     print("action:" , action)
    #     print("arm_action:" , arm_action)
    #     print("safe_action:", safe_action)
    #     print("abs_action:", abs_action)
    #     print("safe_abs_action:", safe_abs_action)
    #     print("initial_arm_pos", self.initial_arm_pos)
    #     print("abs_arm_pos:" , nextpos)
    #     print("abs_hand_angle:" , scaled_hand_action)
    #     reward = self.compute_reward(ob)
    #     done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
    #     print("reward+penalty:", int(reward) - int(penalty) * 0.05)
    #     return ob, int(reward) - int(penalty) * 0.05, done, False, {"succeed": reward, 'action': np.concatenate([safe_relative_curr_action, hand_action], -1)}

    def compute_reward(self, obs) -> bool:
        
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {self._REWARD_THRESHOLD}')
        return False


    def reset(self, **kwargs):
        if self.save_video:
            self.save_video_recording()

        obs = self.RobotControlClient.reset()
        self.initial_arm_pos = np.array(obs[f"{self.arm_type}_arm_pose"])
        self.initial_hand_angle = np.array(obs[f"{self.arm_type}_hand_angle"][:6])
        self.curr_arm_pos = self.initial_arm_pos
        self.curr_hand_angle = self.initial_hand_angle
        self.curr_path_length = 0
        self.terminate = False
        return self._get_obs(obs), {"succeed": False}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.recording_frames[0].keys():
                    if self.url == "http://192.168.10.53:50055/":
                        video_path = f'./videos/left_{camera_key}_{timestamp}.mp4'
                    else:
                        video_path = f'./videos/right_{camera_key}_{timestamp}.mp4'
                    
                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    
                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")
                
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def _send_command(self, action):
        """Internal function to send position command to the robot."""
        obs = self.RobotControlClient.step(right_action=action)
        self.curr_arm_pos = np.array(obs[f"{self.arm_type}_arm_pose"])
        self.curr_hand_angle = np.array(obs[f"{self.arm_type}_hand_angle"][:6])
        return obs
    
    def _get_obs(self, obs) -> dict:
        state_obs = {
            f"{self.arm_type}_arm_pose": relative_pose(self.initial_arm_pos, np.array(obs[f"{self.arm_type}_arm_pose"])),
            f"{self.arm_type}_hand_angle": np.array(obs[f"{self.arm_type}_hand_angle"][:6])
        }
        obs.update(state_obs)
        return obs


if __name__ == "__main__":
    
    InsprieEnv()