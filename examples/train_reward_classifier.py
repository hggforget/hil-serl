import glob
import os
import pickle as pkl
import jax
from jax import numpy as jnp
from jax.tree_util import tree_map
import flax.linen as nn
from flax.training import checkpoints
from flax.core import frozen_dict
from typing import Any, Iterator, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import optax
from tqdm import tqdm
from absl import app, flags

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier

from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    
    def _init_dev_dict(obs_space: gym.Space) -> dict:
        if isinstance(obs_space, gym.spaces.Box):
            return []
        elif isinstance(obs_space, gym.spaces.Dict):
            data_dict = {}
            for k, v in obs_space.spaces.items():
                data_dict[k] = _init_dev_dict(v)
            return data_dict
        else:
            raise TypeError()
    
    dev_batch = {'observations': _init_dev_dict(env.observation_space), 'labels': []}
    dev_paths = glob.glob(os.path.join(os.getcwd(), f"tasks/{FLAGS.exp_name}/classifier_data", "*dev*.pkl"))
    for path in dev_paths:
        dev_data = pkl.load(open(path, "rb"))
        for trans in dev_data:
            if "images" in trans['observations'].keys():
                continue
            dev_batch['labels'].append(trans["rewards"])
            tree_map(lambda x, y: y.append(x), trans['observations'], dev_batch['observations'])
    
    def _stack(dataset):
        if isinstance(dataset, list):
            return np.stack(dataset, 0)
        elif isinstance(dataset, dict):
            batch = {}
            for k, v in dataset.items():
                batch[k] = _stack(v)
            return batch
        else:
            raise TypeError("Unsupported type.")
    
    dev_batch =  {'observations': _stack(dev_batch['observations']), 'labels': np.stack(dev_batch['labels'], 0)}
    dev_batch = jax.device_put(frozen_dict.freeze(dev_batch), device=sharding.replicate()) 
            
    # Create buffer for positive transitions
    pos_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=20000,
        include_label=True,
    )
    # buffer = ReplayBuffer(
    #     env.observation_space,
    #     env.action_space,
    #     capacity=20000,
    #     include_label=True,
    # )

    success_paths = glob.glob(os.path.join(os.getcwd(), f"tasks/{FLAGS.exp_name}/classifier_data", "*success*.pkl"))
    for path in success_paths:
        success_data = pkl.load(open(path, "rb"))
        for trans in success_data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 1
            trans['actions'] = env.action_space.sample()
            pos_buffer.insert(trans)
            
    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )
    
    # Create buffer for negative transitions
    neg_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=50000,
        include_label=True,
    )
    failure_paths = glob.glob(os.path.join(os.getcwd(), f"tasks/{FLAGS.exp_name}/classifier_data", "*failure*.pkl"))
    for path in failure_paths:
        failure_data = pkl.load(
            open(path, "rb")
        )
        for trans in failure_data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 0
            trans['actions'] = env.action_space.sample()
            neg_buffer.insert(trans)
            
    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    print(f"failed buffer size: {len(neg_buffer)}")
    print(f"success buffer size: {len(pos_buffer)}")

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    
    # iterator = buffer.get_iterator(
    #     sample_args={
    #         "batch_size": FLAGS.batch_size,
    #     },
    #     device=sharding.replicate(),
    # )
    sample = concat_batches(pos_sample, neg_sample, axis=0)
    # sample = next(iterator)
    rng, key = jax.random.split(rng)
    classifier = create_classifier(key, 
                                   sample["observations"], 
                                   config.classifier_keys,
                                   )

    def data_augmentation_fn(rng, observations):
        for pixel_key in config.classifier_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=1
                    )
                }
            )
        return observations

    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, batch["observations"], rngs={"dropout": key}, train=True
            )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn(
            {"params": state.params}, batch["observations"], train=False, rngs={"dropout": key}
        )
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    @jax.jit
    def dev_step(state, dev_batch, rng):
        rng, key = jax.random.split(rng)
        obs = dev_batch["observations"]
        # obs = data_augmentation_fn(key, dev_batch["observations"])
        batch = dev_batch.copy(
            add_or_replace={
                "observations": obs,
                "labels": dev_batch["labels"][..., None],
            }
        )
        rng, key = jax.random.split(rng)
        logits = state.apply_fn(
            {"params": state.params}, batch["observations"], train=False, rngs={"dropout": key}
        )
        prediction = (nn.sigmoid(logits) >= 0.5) == batch["labels"]

        return rng, prediction
    
    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        # Merge and create labels
        batch = concat_batches(
            pos_sample, neg_sample, axis=0
        )
        # batch = next(iterator)
        rng, key = jax.random.split(rng)
        obs = data_augmentation_fn(key, batch["observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "labels": batch["labels"][..., None],
            }
        )
            
        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)
        
        rng, dev_accuracy = dev_step(classifier, dev_batch, rng)
        pos_accuracy = np.mean(dev_accuracy[np.where(dev_batch['labels'] == 1)])
        neg_accuracy = np.mean(dev_accuracy[np.where(dev_batch['labels'] == 0)])
        dev_accuracy = np.mean(dev_accuracy)
        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Dev Accuracy: {dev_accuracy:.4f}, Dev pos Accuracy: {pos_accuracy:.4f}, Dev neg Accuracy: {neg_accuracy:.4f}"
        )

    checkpoints.save_checkpoint(
        os.path.join(os.getcwd(), f"tasks/{FLAGS.exp_name}/classifier_ckpt/"),
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )
    

if __name__ == "__main__":
    app.run(main)