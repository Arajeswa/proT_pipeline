import yaml
from os.path import dirname, abspath, join
import sys
parent_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_path)



with open(join(parent_path,"config/config.yaml"), "r") as f:
    config = yaml.safe_load(f)

def get_folders(key:str="local"):

    options = {"local"  : "base_dir_local",
               "cluster" : "base_dir_cluster"}
    
    BASE_DIR = config[options[key]]
    DATA_DIR = join(BASE_DIR,"data")
    INPUT_DIR = join(DATA_DIR,"input")
    OUTPUT_DIR = join(DATA_DIR,"output")
    INTERMEDIATE_DIR = join(DATA_DIR,"intermediate")
    return DATA_DIR,INPUT_DIR,OUTPUT_DIR,INTERMEDIATE_DIR


PROJECT_NAME = config['project_name']
VERSION = config['version']