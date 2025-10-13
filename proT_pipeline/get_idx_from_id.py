import numpy as np
from os.path import dirname, join, abspath
import sys
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)
from proT_pipeline.core.labels import *
import json


def get_idx_from_id(dataset_id: str, id_sel_filename: str, idx_sel_filename: str):
    """
    Get indices of the dataset corresponding to selected id

    Args:
        dataset_id (str): builds directory
        id_sel_filename (str): selected id file name
        idx_sel_filename (str): file name to save selected indices
    """
    
    
    # define directories
    ROOT_DIR = ROOT_DIR = dirname(dirname(abspath(__file__)))
    sys.path.append(ROOT_DIR)
    _,OUTPUT_DIR,CONTROL_DIR = get_dirs(ROOT_DIR, dataset_id)
    filepath_target = join(CONTROL_DIR, target_filename)
    
    id_sel = np.load(join(CONTROL_DIR,id_sel_filename),allow_pickle=True)
    
    if id_sel.dtype == "object":
        
        with open(join(OUTPUT_DIR,batch_dict_filename), 'r') as file:
            batch_dict_json = file.read()
        
        batch_dict = json.loads(batch_dict_json)
        id_selected =  np.array([batch_dict[id] for id in id_sel])
    
    else:
        id_selected = id_sel
    
    dataset_name = f"ds_{dataset_id}"
    dataset_dir = join(OUTPUT_DIR, dataset_name)
    
    X = np.load(join(dataset_dir,input_ds_label))
    Y = np.load(join(dataset_dir,trg_ds_label))
    
    ids = X[:, 0, 0]
    mask = np.isin(ids, id_selected)
    sel_ids = np.where(mask)[0]
    
    
    # TODO: check that also Y is ok
    
    np.save(join(dataset_dir,idx_sel_filename), sel_ids)


if __name__ == "__main__":
    idx_sel_filename = "test_ds_idx.npy"
    dataset_id = "dx_250406_max_len_200_mean"
    id_sel_filename = "selected_id.npy"
    get_idx_from_id(dataset_id, id_sel_filename, idx_sel_filename)