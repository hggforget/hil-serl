#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.agents.continuous.bc import BCAgent
import sys
import pathlib
directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.joinpath('data_process')))
from serl_launcher.utils.launcher import (
    make_bc_agent,
    make_trainer_config,
    make_wandb_logger,
    make_tensorboard_logger,
)
from serl_launcher.utils.train_utils import(
    tensorstats
)
from serl_launcher.data.data_store import ReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING
from experiments.config import DefaultTrainingConfig
from gymnasium.spaces import flatten
FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("bc_checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("train_steps", 40_000, "Number of pretraining steps.")
flags.DEFINE_bool("save_video", False, "Save video of the evaluation.")
flags.DEFINE_string("eval_mode", 'eval', "eval mode.")


flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


##############################################################################

def eval(
    env,
    bc_agent: BCAgent,
    sampling_rng,
):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    success_counter = 0
    time_list = []
    for episode in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset()
        done = False
        start_time = time.time()
        while not done:
            step_start_time = time.time()
            rng, key = jax.random.split(sampling_rng)
            normed_actions = bc_agent.sample_actions(observations=obs, argmax=True)
            normed_actions = np.asarray(jax.device_get(normed_actions / 100.0))
            print(f"infer: {time.time()-step_start_time}")
            next_obs, reward, done, truncated, info = env.step(normed_actions)
            print(f"step: {time.time()-step_start_time}")
            obs = next_obs
            if done:
                if reward:
                    dt = time.time() - start_time
                    time_list.append(dt)
                    print(dt)
                success_counter += reward
                print(reward)
                print(f"{success_counter}/{episode + 1}")

    print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
    print(f"average time: {np.mean(time_list)}")
    
    
##############################################################################

def eval_dataset(
    bc_replay_buffer,
    bc_agent: BCAgent,
    config: DefaultTrainingConfig,
    env,
    sampling_rng,
):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    # bc_replay_iterator = bc_replay_buffer.get_iterator(
    #     sample_args={
    #         "batch_size": 1,
    #     },
    #     device=sharding.replicate(),
    # )
    with open('/mnt/hil-serl/tasks/take_cup/init/demos/2025-02-11_20-20-49/episodes.pickle', "rb") as f:
        episodes_class = pkl.load(f)
    episodes_first = []
    for episode in episodes_class:
        observation_data = dict(
        state=env.process_state({key: getattr(episode, key)[0] for key in env.proprio_space.spaces.keys()}),
        **env.process_images({key: getattr(episode, key)[0] for key in env.cameras_config.keys()}),
        )
        actions = env.reverse_process_action({key: episode.action[key][0] for key in env.action_dict_space.spaces.keys()})
        episodes_first.append(dict(
            observations=observation_data, 
            actions=actions,
        ))
    # Pretrain BC policy to get started
    for batch in tqdm.tqdm(
        episodes_first,
        dynamic_ncols=True,
        desc="eval_dataset",
    ):
        obs = batch['observations']
        batch_actions = batch['actions']
        # sampling_rng, key = jax.random.split(sampling_rng)
        pi_actions = bc_agent.sample_actions(observations=obs, argmax=True)
        pi_actions = np.asarray(jax.device_get(pi_actions / 100.0))
        pi_actions = env.process_action(pi_actions[0])
        batch_actions = env.process_action(batch_actions[0])
        mse_fun = lambda x, y: ((x - y) ** 2).mean(-1)
        keys = batch_actions.keys()
        mse = {k: np.round(mse_fun(pi_actions[k], batch_actions[k]), decimals=4) for k in keys}
        print('mse', mse)
        print('pi_actions', {key: np.round(action, decimals=4) for key, action in pi_actions.items()})
        print('batch_actions', {key: np.round(action, decimals=4) for key, action in batch_actions.items()})

    print_green("bc eval dataset done")


##############################################################################


def train(
    bc_agent: BCAgent,
    bc_replay_buffer,
    config: DefaultTrainingConfig,
    logger=None,
):

    bc_replay_iterator = bc_replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
        },
        device=sharding.replicate(),
    )
    rng = jax.random.PRNGKey(FLAGS.seed)
    sampling_rng = jax.device_put(rng, sharding.replicate())
    # Pretrain BC policy to get started
    for step in tqdm.tqdm(
        range(FLAGS.train_steps),
        dynamic_ncols=True,
        desc="bc_pretraining",
    ):
        batch = next(bc_replay_iterator)
        obs = batch['observations']
        batch = batch.copy(
            add_or_replace={
                'actions': batch['actions'] * 100.0
            }
        )
        bc_agent, bc_update_info = bc_agent.update(batch)
        sampling_rng, key = jax.random.split(sampling_rng)
        pi_actions = bc_agent.sample_actions(observations=obs, seed=key)
        mse = ((pi_actions - batch['actions']) ** 2).mean(-1)
        if (step + 1) % config.log_period == 0 and logger:
            logger.log({"bc": bc_update_info}, step=step)
            logger.log({"dev": tensorstats(mse, prefix='mse')}, step=step)
        if (step + 1) > FLAGS.train_steps - 100 and (step + 1) % 10 == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.bc_checkpoint_path), bc_agent.state, step=step, keep=5
            )
    print_green("bc pretraining done and saved checkpoint")


##############################################################################


def main(_):
    config: DefaultTrainingConfig = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    eval_mode = FLAGS.eval_n_trajs > 0
    env = config.get_environment(
        fake_env=not eval_mode or FLAGS.eval_mode == 'eval_dataset',
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    bc_agent: BCAgent = make_bc_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    bc_agent: BCAgent = jax.device_put(
        jax.tree_map(jnp.array, bc_agent), sharding.replicate()
    )

    if not eval_mode or FLAGS.eval_mode == 'eval_dataset':
        assert not os.path.isdir(
            os.path.join(FLAGS.bc_checkpoint_path, f"checkpoint_{FLAGS.train_steps}")
        )

        bc_replay_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
        )

        # set up wandb and logging
        # logger = make_wandb_logger(
        #     project="hil-serl",
        #     description=FLAGS.exp_name,
        #     debug=FLAGS.debug,
        # )
        
        assert FLAGS.log_dir is not None
        
        logger = make_tensorboard_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
            logdir=FLAGS.log_dir,
        )
        assert FLAGS.demo_path is not None

        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if np.linalg.norm(transition['actions']) > 0.0:
                        bc_replay_buffer.insert(transition)
        print(f"bc replay buffer size: {len(bc_replay_buffer)}")

        if FLAGS.eval_mode == 'eval_dataset':
            rng = jax.random.PRNGKey(FLAGS.seed)
            sampling_rng = jax.device_put(rng, sharding.replicate())

            bc_ckpt = checkpoints.restore_checkpoint(
                FLAGS.bc_checkpoint_path,
                bc_agent.state,
            )
            bc_agent = bc_agent.replace(state=bc_ckpt)
            eval_dataset(
                bc_agent=bc_agent,
                bc_replay_buffer=bc_replay_buffer,
                config=config,            
                sampling_rng=sampling_rng,
                env=env
            )
        else:
            # learner loop
            print_green("starting learner loop")
            train(
                bc_agent=bc_agent,
                bc_replay_buffer=bc_replay_buffer,
                logger=logger,
                config=config,
            )

    else:
        rng = jax.random.PRNGKey(FLAGS.seed)
        sampling_rng = jax.device_put(rng, sharding.replicate())

        bc_ckpt = checkpoints.restore_checkpoint(
            FLAGS.bc_checkpoint_path,
            bc_agent.state,
        )
        bc_agent = bc_agent.replace(state=bc_ckpt)

        print_green("starting actor loop")
        eval(
            env=env,
            bc_agent=bc_agent,
            sampling_rng=sampling_rng,
        )


if __name__ == "__main__":
    app.run(main)
