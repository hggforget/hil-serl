import numpy as np
import cv2
import ast
import os
import sys
import argparse
import pickle as pkl
from tqdm import tqdm
from utils import find_all_hdf5, load_hdf5, decompress_images, create_demos, annotate
from episodes import Episode
from datetime import datetime
from functools import partial

import numpy as np
import cv2
import pathlib
directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
from examples.experiments.mappings import CONFIG_MAPPING
from examples.experiments.config import DefaultTrainingConfig
    
def post_process(self, args):
    setattr(self, 'episode_reward', self.reward)
    len = list(self.action.values())[0].shape[0]
    setattr(self, 'length', len)

def main(args):
    load_dataset_path = args.load_dataset_path
    all_hdf5_list = find_all_hdf5(load_dataset_path)
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f'tasks/{args.task_name}/{args.actions_type}/{args.save_dir}/{date}'
    if not os.path.exists(save_dir):
         os.makedirs(save_dir)
    episodes = []
    config: DefaultTrainingConfig = CONFIG_MAPPING[args.task_name]()
    assert args.task_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=True,
    )
    if args.reward_data:
        with open(args.reward_data, "rb") as f:
            reward_data = pkl.load(f)
        reward_data = [epi.step_rewards for epi in reward_data]
        
    for index, hdf5_one in tqdm(enumerate(all_hdf5_list)):
        episode_idx = hdf5_one.split('.hdf5')[0].split('episode_')[1]
        dataset_name = f'episode_{episode_idx}'
        episode_dict = load_hdf5(load_dataset_path, dataset_name)
        episode = Episode(data_dict=episode_dict, post_process=partial(post_process, args=args), data_keys=ast.literal_eval(args.data_keys))

        if args.reward_data:
            setattr(episode, 'step_rewards', reward_data[index])
        else:
            decompressed_images = np.concatenate(
                [decompress_images(getattr(episode, key)) for key in ast.literal_eval(args.annotate_cameras)], axis=1
            )
            rewards = annotate(decompressed_images)
            setattr(episode, 'step_rewards', rewards)

        episodes.append(episode)
        print('hdf5 loaded!!', episode_idx)
        
    cv2.destroyAllWindows()

    with open(f'{save_dir}/episodes.pickle', "wb") as f:
        pkl.dump(episodes, f)
    if args.save_demos:
        demos = create_demos(env, episodes)
        with open(f'{save_dir}/rlpd_demos.pickle', "wb") as f:
            pkl.dump(demos, f)
        if args.bc:
            pos_episodes = list(filter(lambda x: x.reward, episodes))
            bc_demos = create_demos(env, pos_episodes)
            with open(f'{save_dir}/bc_demos.pickle', "wb") as f:
                pkl.dump(bc_demos, f)
            dev_episodes = list(filter(lambda x: not x.reward, episodes))
            dev_demos = create_demos(env, dev_episodes)
            with open(f'{save_dir}/dev_demos.pickle', "wb") as f:
                pkl.dump(dev_demos, f)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dataset_path', default="/mnt/hil-serl/datasets/data_take_cup_new_rl/pre_process_init", type=str, required=False, help='load dataset directory')
    parser.add_argument('--annotate_cameras', default="['cam_body']", type=str, required=False, help='cameras for annotation')
    parser.add_argument('--data_keys', default="['/reward', '/observations/relative_arm_right', '/observations/relative_hand_right', '/observations/relative_arm_left', '/observations/relative_hand_left', '/observations/images/cam_head_left', '/observations/images/cam_body', '/observations/images/cam_head_right', '/action']", type=str, required=False, help='episode data needed')
    parser.add_argument('--task_name', default="take_cup", type=str, required=False, help='Name of task')
    parser.add_argument('--actions_type', default="init", type=str, required=False, help='Type of actions')
    parser.add_argument('--bc', default=True, type=bool, required=False, help='whether to save bc demos')
    parser.add_argument('--save_dir', default="demos", type=str, required=False, help='dirctionary for saving data')
    parser.add_argument('--save_demos', default=True, type=bool, required=False, help='whether to save demos')
    parser.add_argument('--reward_data', default=None, type=str, required=False, help='load reward data')
    main(parser.parse_args())