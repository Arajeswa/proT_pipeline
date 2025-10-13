import pandas as pd
import numpy as np
import logging
from os.path import dirname, join, abspath
from os import mkdir
import sys
from core.modules import explode_time_components, pandas_to_numpy_ds
import json
sys.path.append(dirname(dirname(abspath(__file__))))
from proT_pipeline.core.labels import *
from labels import *


def generate_dataset(dataset_id):
    """
    Generate the dataset.
    The pandas dataframe from previous steps are converted into numpy dataset. 
    Plus, this method
    - selects the features to include 
    - generates the variable/process/position vocabularies
    - maps vocabulary entries to embedding index
    
    Args:
        dataset_id (str), working dataset folder
    """
    
    # define directories
    ROOT_DIR = ROOT_DIR = dirname(dirname(abspath(__file__)))
    sys.path.append(ROOT_DIR)
    _,OUTPUT_DIR,CONTROL_DIR = get_dirs(ROOT_DIR, dataset_id)
    filepath_target = join(CONTROL_DIR, target_filename)
    
    # import input and target dataframes
    df_trg = pd.read_csv(filepath_target, sep=target_sep)
    df_trg,_ = explode_time_components(df_trg, trans_date_label)
    df_trg[trans_process_label] = "IST"
    
    # read input file
    df_input = pd.read_parquet(join(OUTPUT_DIR,trans_df_input))
    
    # generate batch dictionary
    group_array = df_input[trans_group_id].unique()
    group_dict = {batch:j for j,batch in enumerate(group_array)}
    logging.info(f"batch dictionary size: {len(group_dict)}")
    with open(join(OUTPUT_DIR, batch_dict_filename), "w") as f:
        json.dump(group_dict, f, indent=4)
    
    
    # generate processes dictionary ---------------------------------------------------------------------------
    # input
    pro_array_input = df_input[trans_process_label].unique()
    pro_dict_input = {pro:j for j,pro in enumerate(pro_array_input, start=1)}
    logging.info(f"processes dictionary size: {len(pro_dict_input)}")
    with open(join(OUTPUT_DIR, process_dict_filename), "w") as f:
        json.dump(pro_dict_input, f, indent=4)
    
    # target
    pro_array_trg = df_trg[trans_process_label].unique()
    pro_dict_trg = {pro:j for j,pro in enumerate(pro_array_trg, start=1)}
    logging.info(f"processes dictionary size: {len(pro_dict_trg)}")
    with open(join(OUTPUT_DIR, process_dict_filename), "w") as f:
        json.dump(pro_dict_trg, f, indent=4)
    
    
    # generate variables dictionary ---------------------------------------------------------------------------
    # input
    var_array_input = df_input[trans_variable_label].unique()
    var_dict_input = {var:i for i,var in enumerate(var_array_input, start=1)}
    logging.info(f"variables dictionary size: {len(var_dict_input)}")
    with open(join(OUTPUT_DIR, var_dict_filename + "_input"), "w") as f:
        json.dump(var_dict_input, f, indent=4)
    
    # target
    var_array_trg = df_trg[trans_variable_label].unique()
    
    # if int64, following maps won't work, convert to int
    if var_array_trg.dtype=="int64":
        var_array_trg = var_array_trg.astype(str)
        df_trg[trans_variable_label] = df_trg[trans_variable_label].astype(str)
        
    var_dict_trg = {str(var):i for i,var in enumerate(var_array_trg, start=1)}
    logging.info(f"variables dictionary size: {len(var_dict_trg)}")
    with open(join(OUTPUT_DIR, var_dict_filename + "_trg"), "w") as f:
        json.dump(var_dict_trg, f, indent=4)
    
    
    # generate positions dictionary ---------------------------------------------------------------------------
    # input
    pos_array_input = df_input[trans_position_label].sort_values().unique()
    pos_dict_input = {pos:k for k,pos in enumerate(pos_array_input, start=1)}
    logging.info(f"positions dictionary size: {len(pos_dict_input)}")
    with open(join(OUTPUT_DIR, pos_dict_filename), "w") as f:
        json.dump(pos_dict_input, f, indent=4)
    
    # map processes/variables/positions to their vocabulary value (to have numbers instead of objects)
    df_trg[trans_process_label]    = df_trg[trans_process_label].map(pro_dict_trg)
    df_input[trans_process_label]  = df_input[trans_process_label].map(pro_dict_input)
    df_trg[trans_variable_label]   = df_trg[trans_variable_label].map(var_dict_trg)
    df_input[trans_variable_label] = df_input[trans_variable_label].map(var_dict_input)
    df_input[trans_position_label] = df_input[trans_position_label].map(pos_dict_input)
    df_input[trans_group_id] = df_input[trans_group_id].map(group_dict)
    df_trg[trans_group_id] = df_trg[trans_group_id].map(group_dict)
    
    # get the unique ids from the mapped group column of the target
    id_samples = df_input[trans_group_id].unique()
    
    # define features 
    input_features = [trans_process_label, trans_variable_label, trans_position_label, trans_value_norm_label, trans_order_label]
    input_features.extend(time_components_labels) # include exploded time
    trg_features = [trans_process_label, trans_variable_label, trans_position_label, trans_value_label]
    trg_features.extend(time_components_labels) # include exploded time
    
    # assemble the features dictionary to document which features are on which index
    feat_dict = {
        "input" : {key+1:val for key, val in enumerate(input_features)},
        "target": {key+1:val for key, val in enumerate(trg_features)}}
    
    feat_dict["input"][0]=trans_group_id
    feat_dict["target"][0]=trans_group_id
    
    feat_dict["input"] = {key: feat_dict["input"][key] for key in sorted(feat_dict["input"].keys())}
    feat_dict["target"] = {key: feat_dict["target"][key] for key in sorted(feat_dict["target"].keys())}
    
    with open(join(OUTPUT_DIR, feat_dict_filename), "w") as f:
        json.dump(feat_dict, f, indent=4)
    
    # convert pandas to numpy dataset
    logging.info("Flattening input sequence")
    array_input = pandas_to_numpy_ds(id_samples,df_input,input_features,trans_group_id,2000)
    logging.info("Flattening target sequence")
    array_trg = pandas_to_numpy_ds(id_samples,df_trg,trg_features,trans_group_id,3000)
    
    # save dataset
    dataset_name = f"ds_{dataset_id}"
    dataset_dir = join(OUTPUT_DIR, dataset_name)
    if not exists(dataset_dir):
        mkdir(dataset_dir)
    
    np.save(join(dataset_dir,input_ds_label),array_input)
    np.save(join(dataset_dir,trg_ds_label),array_trg)
    
    
if __name__ == "__main__":
    generate_dataset(dataset_id = "dx_250602_200_mean_clip")

