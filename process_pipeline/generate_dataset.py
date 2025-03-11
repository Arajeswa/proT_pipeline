import pandas as pd
import numpy as np
from labels import *
from os.path import dirname, join, abspath
import sys
from core.modules import explode_time_components, pandas_to_numpy_ds
import json


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
    _,OUTPUT_DIR,_,CONTROL_DIR = get_dirs(ROOT_DIR, dataset_id)
    filepath_target = join(CONTROL_DIR, target_filename)
    
    # import input and target dataframes
    df_trg = pd.read_csv(filepath_target, sep=target_sep)
    df_trg,_ = explode_time_components(df_trg, trans_date_label)
    df_trg[trans_process_label] = "IST"
    df_input = pd.read_parquet(join(OUTPUT_DIR,trans_df_input))
    
    # generate process dictionary
    pro_array = np.concatenate([df_input[trans_process_label].unique(),df_trg[trans_process_label].unique()])
    pro_dict = {pro:j for j,pro in enumerate(pro_array)}
    with open(join(OUTPUT_DIR, process_dict_filename), "w") as f:
        json.dump(pro_dict, f, indent=4)
    
    # generate variables dictionary
    var_array = np.concatenate([df_input[trans_variable_label].unique(),df_trg[trans_variable_label].unique()])
    var_dict = {var:i for i,var in enumerate(var_array)}
    with open(join(OUTPUT_DIR, var_dict_filename), "w") as f:
        json.dump(var_dict, f, indent=4)
        
    # generate position dictionary (for the input)
    pos_array = df_input[trans_position_label].sort_values().unique()
    pos_dict = {pos:k for k,pos in enumerate(pos_array)}
    with open(join(OUTPUT_DIR, pos_dict_filename), "w") as f:
        json.dump(pos_dict, f, indent=4)
        
    # map processes/variables/positions to their embedding index
    df_trg[trans_process_label] = df_trg[trans_process_label].map(pro_dict)
    df_input[trans_process_label] = df_input[trans_process_label].map(pro_dict)
    df_trg[trans_variable_label] = df_trg[trans_variable_label].map(var_dict)
    df_input[trans_variable_label] = df_input[trans_variable_label].map(var_dict)
    df_input[trans_position_label] = df_input[trans_position_label].map(pos_dict)
    
    # get samples ID
    id_samples = df_input[trans_id_label].unique()
    
    # define features 
    input_features = [trans_process_label, trans_variable_label, trans_position_label, trans_value_norm_label]
    input_features.extend(time_components_labels) # include exploded time
    trg_features = [trans_process_label, trans_variable_label, trans_position_label, trans_value_label]
    trg_features.extend(time_components_labels) # include exploded time
    
    # convert pandas to numpy dataset 
    array_input = pandas_to_numpy_ds(id_samples,df_input,input_features,trans_id_label,2000)
    array_trg = pandas_to_numpy_ds(id_samples,df_trg,trg_features,trans_id_label,2400)
    
    # save dataset
    np.save(join(OUTPUT_DIR,input_ds_label),array_input)
    np.save(join(OUTPUT_DIR,trg_ds_label),array_trg)
    
    
if __name__ == "__main__":
    generate_dataset(dataset_id = "dyconex_252901")

