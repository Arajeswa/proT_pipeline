import pandas as pd
import sys
from os.path import dirname, join, abspath
import sys

ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)
from proT_pipeline.core.labels import *
from labels import *
from proT_pipeline.core.modules import explode_time_components, filter_vars_max_missing



def process_raw(dataset_id: str, missing_threshold: float = None)->None:
    """
    Process raw process data. Operations are:
    - Normalization
    - Take the mean of multiple measurements
    - Add temporal order column
    - Explode time components
    - Filters missing values to a max % per variable defined by `threshold`

    Args:
        dataset_id (str): working directory
        missing_threshold (float): threshold for max % of missing values per variable
    """
    
    # define directories
    ROOT_DIR = ROOT_DIR = dirname(dirname(abspath(__file__)))
    sys.path.append(ROOT_DIR)
    _,OUTPUT_DIR,_ = get_dirs(ROOT_DIR, dataset_id)


    df_raw = pd.read_csv(join(OUTPUT_DIR,trans_df_process_raw))

    # normalization function(s)
    def max_normalizer(df:pd.DataFrame,var_label, val_label):
        max_map = df.groupby(var_label)[val_label].max()
        df[trans_value_norm_label] = df[val_label]/df[var_label].map(max_map) 
        return df
    
    
    # take mean of multiple variable measurements
    grouping_cols = [trans_group_id, trans_position_label, trans_process_label, trans_variable_label]
    df_input_short = df_raw.groupby(grouping_cols).agg({
        trans_value_label: "mean",
        trans_date_label: "first",
        trans_parameter_label: "first"
        }).reset_index()
    
    
    # normalize
    df_input_short = max_normalizer(df_input_short, var_label=trans_variable_label, val_label=trans_value_label)
    
    # add temporal order
    df_input_short[trans_order_label] = (
    df_input_short.groupby(trans_group_id)[trans_date_label].rank(
        method='dense',  
        ascending=True,          
        na_option='keep'
        ).astype('Int64'))
    
    # convert to float (it handles nan) for next numpy conversion
    df_input_short[trans_order_label] = df_input_short[trans_order_label].astype('float64')

    # explode time
    df_input_short,_ = explode_time_components(df_input_short,trans_date_label)
    
    
    # filter max missing values per variable
    if missing_threshold is not None:
        df_input_short = filter_vars_max_missing(df_input_short, missing_threshold)
    
    # export
    # TODO: remove short one
    df_input_short.to_parquet(join(OUTPUT_DIR, trans_df_input_short))
    
    df_input_short.to_parquet(join(OUTPUT_DIR, trans_df_input))
    
    
if __name__ == "__main__":
    process_raw(dataset_id = "dyconex_test")