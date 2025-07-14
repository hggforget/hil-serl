import datetime
import tempfile
from copy import copy
from socket import gethostname

import absl.flags as flags
import ml_collections
import wandb
import collections
import numpy as np


def _recursive_flatten_dict(d: dict):
    keys, values = [], []
    for key, value in d.items():
        if isinstance(value, dict):
            sub_keys, sub_values = _recursive_flatten_dict(value)
            keys += [f"{key}/{k}" for k in sub_keys]
            values += sub_values
        else:
            keys.append(key)
            values.append(value)
    return keys, values


class WandBLogger(object):
    @staticmethod
    def get_default_config():
        config = ml_collections.ConfigDict()
        config.project = "serl_launcher"  # WandB Project Name
        config.entity = ml_collections.config_dict.FieldReference(None, field_type=str)
        # Which entity to log as (default: your own user)
        config.exp_descriptor = ""  # Run name (doesn't have to be unique)
        # Unique identifier for run (will be automatically generated unless
        # provided)
        config.unique_identifier = ""
        config.group = None
        return config

    def __init__(
        self,
        wandb_config,
        variant,
        wandb_output_dir=None,
        debug=False,
    ):
        self.config = wandb_config
        if self.config.unique_identifier == "":
            self.config.unique_identifier = datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

        self.config.experiment_id = (
            self.experiment_id
        ) = f"{self.config.exp_descriptor}_{self.config.unique_identifier}"  # NOQA

        print(self.config)

        if wandb_output_dir is None:
            wandb_output_dir = tempfile.mkdtemp()

        self._variant = copy(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if debug:
            mode = "disabled"
        else:
            mode = "online"

        self.run = wandb.init(
            config=self._variant,
            project=self.config.project,
            entity=self.config.entity,
            group=self.config.group,
            tags=self.config.tag,
            dir=wandb_output_dir,
            id=self.config.experiment_id,
            save_code=True,
            mode=mode,
        )

        if flags.FLAGS.is_parsed():
            flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS}
        else:
            flag_dict = {}
        for k in flag_dict:
            if isinstance(flag_dict[k], ml_collections.ConfigDict):
                flag_dict[k] = flag_dict[k].to_dict()
        wandb.config.update(flag_dict)
        self._wandb = wandb

    def log(self, data: dict, step: int = None):
        data_flat = _recursive_flatten_dict(data)
        data = {k: v for k, v in zip(*data_flat)}
        wandb = self._wandb
        metrics = collections.defaultdict(dict)
        for name, value in data.items():
            if len(value.shape) == 0 and self._pattern.search(name):
                metrics[name] = float(value)
            elif len(value.shape) == 1:
                metrics[name] = wandb.Histogram(value)
            elif len(value.shape) == 2:
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
                value = np.transpose(value, [2, 0, 1])
                metrics[name] = wandb.Image(value)
            elif len(value.shape) == 3:
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
                value = np.transpose(value, [2, 0, 1])
                metrics[name] = wandb.Image(value)
            elif len(value.shape) == 4:
                # Sanity check that the channeld dimension is last
                assert value.shape[3] in [1, 3, 4], f"Invalid shape: {value.shape}"
                value = np.transpose(value, [0, 3, 1, 2])
                # If the video is a float, convert it to uint8
                if np.issubdtype(value.dtype, np.floating):
                    value = np.clip(255 * value, 0, 255).astype(np.uint8)
                metrics[name] = wandb.Video(value, fps=40)
        self._wandb.log(metrics, step=step)