import pathlib
import sys
directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
from experiments.take_cup.config import TrainConfig as TakeCupTrainConfig
from experiments.take_mango.config import TrainConfig as TakeMangoTrainConfig
from experiments.usb_pickup_insertion.config import TrainConfig as USBPickupInsertionTrainConfig
from experiments.object_handover.config import TrainConfig as ObjectHandoverTrainConfig
from experiments.egg_flip.config import TrainConfig as EggFlipTrainConfig

CONFIG_MAPPING = {
                "ram_insertion": RAMInsertionTrainConfig,
                "usb_pickup_insertion": USBPickupInsertionTrainConfig,
                "object_handover": ObjectHandoverTrainConfig,
                "egg_flip": EggFlipTrainConfig,
                "take_mango": TakeMangoTrainConfig,
                "take_cup": TakeCupTrainConfig,
               }