import yaml

from os import getcwd
from os.path import dirname, join
import sys
ROOT_DIR = dirname(getcwd())
sys.path.append(ROOT_DIR)

with open(join(ROOT_DIR,"config/config.yaml"), "r") as f:
    config = yaml.safe_load(f)


PROJECT_NAME = config['project_name']
VERSION = config['version']