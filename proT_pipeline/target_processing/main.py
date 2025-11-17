import pandas as pd
import logging
from datetime import datetime
from os.path import abspath, dirname, join, exists
from os import mkdir
import sys
ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(ROOT_DIR)
from proT_pipeline.target_processing.modules import *
from proT_pipeline.labels import *


def get_df_num_unique(df:pd.DataFrame, column:str):
    assert column in df.columns, AssertionError(f"Column {column} not in the dataframe")
    return len(df[column].unique())


def main(
    build_id: str,
    grouping_method: str,
    grouping_column: str,
    max_len: float,
    filter_type: str="C",     # canary "C" or product "P"
    uni_method: str=None, 
    max_len_mode :str=None, 
    mean_bool: bool=False, 
    std_bool: bool=False):
    
    # set up folders
    ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
    INPUT_DIR, BUILDS_DIR = get_target_dirs(ROOT_DIR)
    EXP_DIR = join(BUILDS_DIR, build_id)
    
    if not exists(EXP_DIR):
        mkdir(EXP_DIR)
    
    # set up logging
    log_filename = join(EXP_DIR,"ist_builds.log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    
	# open ist data
    df_ist = pd.read_csv(join(INPUT_DIR, target_filename_input), sep=target_original_sep, low_memory=False)
        
    # process ist data
    print("Processing ist dataframe...")
    
    df_ist = process_ist_dataframe(
        df_ist, 
        grouping_method=grouping_method,
        grouping_column=grouping_column, 
        filter_type=filter_type)
    
    logging.info(f"{trans_batch_label} uniques: {get_df_num_unique(df_ist, target_original_batch_label)}")
    
    
    # define ist features to export
    features = [
        trans_group_id, 
        target_type_label, 
        target_original_id_label, 
        target_original_batch_label, 
        target_original_pos_label, 
        target_original_time_label,
        target_original_design_label,
        target_original_version_label
        ]
    values_cols = [target_delta_A_label,target_delta_B_label]
    df_trg = df_ist[features+values_cols].copy()

    # normalize values, assumption max = 10
    max_val = 10
    print("Normalizing values...")
    df_trg[values_cols] = df_trg[values_cols].apply(lambda x:x/max_val)
    logging.info(f"Values max-normalized by {max_val}")
    logging.info(f"{trans_batch_label} uniques: {get_df_num_unique(df_trg, target_original_batch_label)}")

    # convert date to datetime
    df_trg[target_original_time_label] = pd.to_datetime(df_trg[target_original_time_label], format=target_time_format)
    
    # rename columns for trans compatibility
    df_trg = df_trg.rename(columns={
            target_original_id_label        : trans_id_label,
            target_original_batch_label     : trans_batch_label,
            target_original_pos_label       : trans_position_label,
            target_original_time_label      : trans_date_label,
            target_original_design_label    : trans_design_label,
            target_original_version_label   : trans_version_label,
            target_delta_A_label            : target_norm_delta_A_label,
            target_delta_B_label            : target_norm_delta_B_label,
            })
    
    # check if the grouping groups more than one ids
    id_counts = df_trg.groupby(trans_group_id)[trans_id_label].nunique()
    multiple_ids = id_counts[id_counts > 1]

    # multiple IDs
    if multiple_ids.empty:
        print("Each group has exactly one unique id.")
    else:
        print("Some groups have multiple ids")
        print(multiple_ids)
        if mean_bool==False:
            print("Since mean_bool=False, proceed selecting the dominating one")
            df_trg = filter_dominant_ids(
                df=df_trg, 
                group_col=trans_group_id, 
                id_col=trans_id_label, 
                value_col=target_norm_delta_B_label)
            
            # check if the grouping groups more than one ids
            id_counts = df_trg.groupby(trans_group_id)[trans_id_label].nunique()
            multiple_ids = id_counts[id_counts > 1]
            if multiple_ids.empty:
                print("Each group has exactly one unique id now.")
            else:
                raise ValueError("Still multiple IDs are present for a given group!")
            
    group_cols = [trans_group_id,trans_position_label]
    variable_cols =[target_norm_delta_A_label,target_norm_delta_B_label]
    
    # get mean and std
    assert not std_bool or mean_bool, AssertionError("Invalid selection! Set `mean_bool=True`")
    
    if mean_bool:
        
        # get curves with same lengths within batches for smoother statistics
        if uni_method is not None:
            logging.info(f"Uniform function selected with {uni_method} method")
            df_trg = get_ist_uniform_length(df=df_trg, method=uni_method, variables=variable_cols)
        
        if std_bool:
            agg_spec = ({v: ["mean", "std"] for v in variable_cols} | {trans_date_label: "first"})
            df_trg = df_trg.groupby(group_cols).agg(agg_spec).reset_index()

            df_trg.columns = ["_".join([str(x) for x in col if x not in (None, "", " ", "first")]) for col in df_trg.columns.to_flat_index()]
            value_cols = [f"{v}_{stat}" for v in variable_cols for stat in ("mean", "std")]
            id_vars = group_cols + [trans_date_label]
            
        else:
            agg_spec = ({v: "mean" for v in variable_cols} | {trans_date_label: "first"})
            df_trg = df_trg.groupby(group_cols).agg(agg_spec).reset_index()
            #df_trg.columns = ["_".join([str(x) for x in col if x not in (None, "", " ", "first")]) for col in df_trg.columns.to_flat_index()]
            #value_cols = [f"{v}_mean" for v in variable_cols]
            value_cols = variable_cols
            id_vars = group_cols + [trans_date_label]
    
    
    else:
        value_cols = variable_cols
        id_vars = group_cols + [target_id_label, trans_date_label, trans_design_label, trans_version_label]
    
    logging.info(f"Mean/std calculated; {trans_batch_label} uniques: {get_df_num_unique(df_trg, trans_group_id)}")
    
    
    df_trg = df_trg.melt(
        id_vars   = id_vars,
        value_vars= value_cols,
        var_name  = trans_variable_label,
        value_name= trans_value_label,
        )
    
    logging.info(f"Dataframe melted; {trans_batch_label} uniques: {get_df_num_unique(df_trg, trans_group_id)}")
    

    # apply maximum length
    if max_len is not None:
        df_trg = filter_df_max_len(df_trg, max_len, max_len_mode)   # remove too long
        df_trg = pad_df_to_max_len(df_trg, max_len)                 # pad too short
        
    logging.info(f"Length filtered; {trans_batch_label} uniques: {get_df_num_unique(df_trg, trans_group_id)}")
    
    
    # export
    df_trg.to_csv(join(EXP_DIR,target_filename_df))
    print(f"Target filtered dataframe assembled: index unique: {df_ist.index.is_unique}")
    logging.info(f"Target filtered dataframe assembled: index unique: {df_ist.index.is_unique}")
    logging.info(f"File {target_filename_df} generated")





if __name__ == "__main__":
    
    build_id = "debug_design_version"              # output directory name
    grouping_method="panel"                     # "column" or "panel"
    grouping_column= None                       # column if grouping_method=="column" usually `target_original_batch_label`, else set None
    max_len = 200                               # maximum sequence length
    filter_type="C"                             # C = canary P = product
    uni_method = "clip"                         # clipping option when calculating moments
    max_len_mode = "clip"                       # clipping option to select maximum length
    mean_bool = False                           # flag to calculate first moment
    std_bool = False                            # flag to calculate second moment
    
    main(
        build_id,
        grouping_method=grouping_method,
        grouping_column=grouping_column,
        max_len=max_len,
        filter_type=filter_type,
        uni_method=uni_method, 
        max_len_mode=max_len_mode, 
        mean_bool=mean_bool, 
        std_bool=std_bool )
