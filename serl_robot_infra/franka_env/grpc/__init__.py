import pathlib
import sys
directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory))

from .client_rl import RobotControlClient
