import pandas as pd
import numpy as np
from os.path import join,dirname
from typing import List

import sys
ROOT_DIR = dirname(dirname(__file__))
sys.path.append(ROOT_DIR)
from process_pipeline.utils import nested_dict_from_pandas
from core.modules import Process
from labels import *


def get_template(
    df: pd.DataFrame,
    processes = List[Process],
    # input_design_label: str = "SAP",
    # input_version_label: str = "Version",
    # input_variable_label: str = "Variable",
    # input_process_label: str = "Process",
    # input_batch_label: str = "WA",
    # input_step_label: str = "PaPos",
    # input_id_label: str = "id",
    # input_abs_pos_label: str = "AbsPos",
    ):
    
    # create a nested dictionary from Y (IST) queries
    multi_idx = [input_design_label,input_version_label,input_batch_label,input_id_label,input_step_label]
    d = nested_dict_from_pandas(df.set_index(multi_idx))

    # get the reference batches
    outer_dict = {}

    for design in d.keys():
        middle_dict = {}

        for version in d[design].keys():
            inner_dict = {}

            # scan all possible batch/id
            #----------------------------------------------------------------------------------------------------------------------
            for batch in d[design][version].keys():

                for id in d[design][version][batch].keys():
                    # steps = df.set_index(multi_idx).loc[design,version,batch,id][step_label].unique()
                    
                    for step in d[design][version][batch][id]:
                        if step not in inner_dict.keys():
                            process = df.set_index(multi_idx).loc[design,version,batch,id,step][input_process_label].value_counts().index[0]
                            var_list = [pro.variables_list for pro in processes if pro.process_label == process][0]
                            inner_dict[step] = {input_process_label:process,input_variable_label:var_list,input_abs_pos_label:0}
            #----------------------------------------------------------------------------------------------------------------------
            
            for i, key in enumerate(inner_dict.keys()):
                var_dict = dict()
                for var in inner_dict[key][input_variable_label]:
                    var_dict[var] = {
                        input_abs_pos_label:i,
                        input_process_label:inner_dict[key][input_process_label]}
                inner_dict[key] = var_dict
                
            middle_dict[version] = inner_dict
        
        outer_dict[design] = middle_dict
    
    return outer_dict


def level_sequences(
    df: pd.DataFrame,
    processes: List[Process],
    # input_design_label: str = "SAP",
    # input_version_label: str = "Version",
    # input_variable_label: str = "Variable",
    # input_process_label: str = "Process",
    # input_batch_label: str = "WA",
    # input_step_label: str = "PaPos",
    # input_id_label: str = "id",
    # input_abs_pos_label: str = "AbsPos",
    # input_given_label: str = "Given",
    # input_value_label: str = "Value",
    ):
    
    # create a nested dictionary from Y (IST) queries
    multi_idx = [input_design_label,input_version_label,input_batch_label,input_id_label]
    d = nested_dict_from_pandas(df.set_index(multi_idx))

    
    templates = get_template(
        df=df,
        processes=processes,
        # input_design_label = input_design_label,
        # input_version_label = input_version_label,
        # input_variable_label = input_variable_label,
        # input_process_label = input_process_label,
        # input_batch_label = input_batch_label,
        # input_step_label = input_step_label,
        # input_id_label = input_id_label,
        # input_abs_pos_label = input_abs_pos_label
        )
    print("Template assembled")
    
    # get absolute position in the templates
    df_lev = None
    max_seq_len = 0
    
    for design in d.keys():

        for version in d[design].keys():
            sel_template = templates[design][version]

            df_template = pd.DataFrame.from_dict(
                {(i,j): sel_template[i][j] 
                for i in sel_template.keys() 
                for j in sel_template[i].keys()},orient="index")
            
            if len(df_template)>max_seq_len:
                max_seq_len = len(df_template)
                
            for batch in d[design][version].keys():

                for id in d[design][version][batch]:
                    _df = df.set_index([input_design_label,input_version_label,input_batch_label,input_id_label]).loc[design,version,batch,id].reset_index()
                    _df = _df.drop_duplicates(subset=[input_step_label,input_variable_label])
                    _df = _df.set_index([input_step_label,input_variable_label])
                    df_lev_temp = pd.concat([df_template,_df],axis=1)
                    df_lev_temp = df_lev_temp.reset_index().rename(columns={'level_0':input_step_label, "level_1":input_variable_label})
                    
                    df_lev_temp[input_design_label] = design
                    df_lev_temp[input_version_label] = version
                    df_lev_temp[input_batch_label] = batch
                    df_lev_temp[input_id_label] = id
                    
                    if df_lev is None:
                        df_lev = df_lev_temp
                    else:
                        df_lev = pd.concat([df_lev,df_lev_temp],ignore_index=True)
                
                        

    df_lev[input_given_label] = df_lev[input_value_label].notna().astype(int)
    
    return df_lev,max_seq_len