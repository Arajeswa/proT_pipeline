import pandas as pd
import numpy as np
import logging
import sys
from data_loader import get_processes
from proT_pipeline.core.labels import *
from proT_pipeline.core.modules import split_queries_by_keys
from labels import *
import json
import re
from os.path import dirname, join, abspath, exists
from os import makedirs
import sys
from typing import Tuple, List



def assemble_raw(dataset_id, grouping_method, grouping_column, debug=False)->None:
    
    """
    Assembles a raw dataframe containing process data from the single
    process files, according to control files which select
    - which variables for each process to add
    - which step (PaPos) to include
    
    The raw dataframe is finally saved.
    
    Args:
    dataset_id (str), working dataset folder 
    """
    
    # define directories
    ROOT_DIR = ROOT_DIR = dirname(dirname(abspath(__file__)))
    sys.path.append(ROOT_DIR)
    INPUT_DIR,OUTPUT_DIR,CONTROL_DIR = get_dirs(ROOT_DIR, dataset_id)
    filepath_selected = join(CONTROL_DIR, selected_filename)
    filepath_target = join(CONTROL_DIR, target_filename)
    

    #load processes
    _, processes = get_processes(INPUT_DIR,filepath_selected, grouping_method, grouping_column)

    # read target (IST) file
    df_trg = pd.read_csv(filepath_target, sep=target_sep)
    query_groups = df_trg[trans_group_id].unique().tolist()
    
    if debug:
        query_groups = df_trg[trans_group_id].unique()[:100].tolist()
        

    # import control file for process selection
    df_steps_sel = pd.read_excel(join(CONTROL_DIR,selected_process_filename))
    steps_sel = np.array(df_steps_sel[df_steps_sel['Select']]["Step"])

    missing_groups_dic = {}
    df_raw = None
    
    df_list = []
    for pro in processes:

        
        
        # import control file (lookup table) for current process
        df_lookup = pd.read_excel(filepath_selected,sheet_name=pro.process_label)
        date_labels = [i for i in df_lookup["index"] if i in pro.date_label]        # date labels
        parameters = df_lookup[df_lookup["Select"]]["index"].tolist()              # selected parameters
        variables = df_lookup[df_lookup["Select"]][trans_variable_label].tolist()  # and relative variables

        assert len(parameters)==len(variables)                                     # create a dict with parameters (key) and variable name (values)
        params_vars = {parameters[i]:variables[i] for i in range(len(parameters))}

        df_cp = pro.df
        
        # address mismatch between query and key batches
        keys_groups = df_cp[trans_group_id].unique().tolist()
        missing_groups, available_groups = split_queries_by_keys(query_groups, keys_groups)
        missing_groups_dic[pro.process_label] = missing_groups
        
        if pro.process_label in ["Multibond", "Microetch"]:
            df_cp = group_expand_dataframe(df_cp,available_groups,keys_groups)
            
            
            
            
        
        print(pro.process_label)
        
        df_cp = df_cp.set_index(trans_group_id).loc[available_groups].reset_index() # select available batches 
        
        # fix datetime
        datetime_list = []
        for date_label in date_labels:
            try:
                date_time_col = pd.to_datetime(df_cp[date_label],format=pro.date_format)
            except:
                date_time_col = pd.to_datetime(df_cp[date_label],format="mixed")
            datetime_list.append(date_time_col)
        
        if len(datetime_list)>1:
            print(f"Process {pro.process_label} has more than one date label...taking one of them")
            logging.info(f"From \"assemble_raw\" in process \"{pro.process_label}\" found more than one date label")
                    
        df_cp[trans_date_label] = datetime_list[0]
        
        
        if len(pro.missing_columns) != 0:
            parameters = [p for p in parameters if p not in pro.missing_columns]
        
        # melt dataframe
        df_cp = df_cp.melt(
            id_vars=[trans_group_id,pro.PaPos_label,trans_date_label],
            value_vars=parameters,
            var_name=trans_parameter_label,
            value_name=trans_value_label)
        
        # add variables and process label
        df_cp[trans_variable_label] = df_cp[trans_parameter_label].map(params_vars)
        df_cp[trans_process_label] = pro.process_label
        
        # rename transversal columns
        df_cp = df_cp.rename(columns={
            pro.PaPos_label: trans_position_label,
            })
        
        # append process dataframe to list
        if not df_cp.empty:
            df_list.append(df_cp)
        
    # concatenate process dataframes
    if len(df_list) == 0:
        raise ValueError("Zero process dataframe found, check your queries!")
    elif len(df_list) == 1:
        df_raw = df_list[0]
    else:
        df_raw = pd.concat(df_list,ignore_index=True)
    
    # select steps from control file
    df_raw = df_raw[df_raw[trans_position_label].isin(steps_sel)]
    
    if df_raw.empty:
        raise ValueError("Selected steps produced empty dataframe. If debug mode, try increasing slice.")
    
    # check uniqueness of map position --> process
    df_unique_pairs = df_raw[[trans_position_label,trans_process_label]].drop_duplicates().sort_values(by=trans_position_label)
    count_process_per_position = df_unique_pairs.groupby(trans_position_label)[trans_process_label].nunique()
    df_check = count_process_per_position[count_process_per_position > 1]
    assert len(df_check)==0, AssertionError("Action needed! Some position ID is used for > 1 process")
    
    # export
    if not exists(OUTPUT_DIR):
        makedirs(OUTPUT_DIR)
    
    df_raw.to_csv(join(OUTPUT_DIR, trans_df_process_raw))
    with open(join(OUTPUT_DIR, trans_missing_batches), "w") as f:
        json.dump(missing_groups_dic, f, indent=4)






def group_expand_dataframe(df_cp,available_groups,key_groups):
    df_ = df_cp.copy()
    df_list = []
    for av_group in available_groups:
        
        av_group_unexpanded = av_group.split("_")[0]+"_*"
        df_temp = df_.set_index(trans_group_id).loc[[av_group_unexpanded]].reset_index()
        df_temp["new_group_temp"] = av_group
        df_list.append(df_temp)
    
    df_long = pd.concat(df_list)
    df_long.reset_index(inplace=True)
    df_long[trans_group_id]=df_long["new_group_temp"]
    
    return df_long




if __name__ == "__main__":
    assemble_raw(dataset_id = "test")