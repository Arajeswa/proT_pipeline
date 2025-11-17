import pandas as pd
import numpy as np
import logging
from typing import List
from functools import partial
from proT_pipeline.labels import *



def filter_dominant_ids(df: pd.DataFrame, group_col :str, id_col: str, value_col :str)->pd.DataFrame:
    """
    If a group results in multiple id, this function discards the dominated ones and keep the dominant.
    Dominance condition is defined by summation.
    Args:
        df (pd.DataFrame): dataframe with multiple ids per group
        group_col (str): name of the group column, usually `target_group_id`
        id_col (str): name of the id column, usually `trans_id_label`
        value_col (str): name of the value column, user defined, can be delta resistance A/B

    Returns:
        pd.DataFrame: filtered dataset with dominant id
    """
    # Step 1: Aggregate by group + id using only the sum
    agg_df = df.groupby([group_col, id_col]).agg(
        total_value=(value_col, 'sum')
    ).reset_index()

    # Step 2: Sort only by total_value
    dominant_ids = (
        agg_df.sort_values('total_value', ascending=False)
        .groupby(group_col)
        .first()
        .reset_index()
    )

    # Step 3: Merge back to filter the original DataFrame
    df_filtered = df.merge(
        dominant_ids[[group_col, id_col]],
        on=[group_col, id_col],
        how='inner'
    )

    return df_filtered





def fix_format_columns(df: pd.DataFrame)->pd.DataFrame:
    """
    Remove repeating strings in column names.
    Example: (col_name col_name) --> col_name 

    Args:
        df (pd.DataFrame): dataframe with corrupted column names

    Returns:
        pd.DataFrame: dataframe with fixed column names
    """
    df = df.copy()
    df.columns = [df.columns[col].replace(" ","_")[:len(df.columns[col])//2] for col in range(len(df.columns))] #remove redundant column names
    return df



def fix_duplicate_columns(df: pd.DataFrame)->pd.DataFrame:
    """
    Modifies identical column names to make them distinguishable
    Args:
        df (pd.DataFrame): dataframe with identical column names

    Returns:
        pd.DataFrame: dataframe with distinguishable column names
    """
    df = df.copy()
    columns = df.columns
    seen = []
    counter = np.ones(len(columns))
    
    for i,col in enumerate(columns):
        
        if col in seen:
            counter[i] = counter[i]+1
        else:
            seen.append(col)
            
    df_col = pd.DataFrame({"Columns": columns,"Counter": counter})
    new_cols = np.array([])
    
    for i in df_col.index:
        
        if df_col.at[i,"Counter"] == 1:
            new_cols = np.append(new_cols,df_col.at[i,"Columns"])
        else:
            new_cols = np.append(new_cols,df_col.at[i,"Columns"]+"_"+str(int(df_col.at[i,"Counter"])))
    
    df.columns = new_cols
    return df
    
    
    
def get_delta(df: pd.DataFrame, id_label: str, columns: List[str])->pd.DataFrame:
    """
    Calculates the percentage increment (from the initial value) of the absolute values measurements
    --> delta = (absolute_value/initial_value-1) X 100
    Args:
        df (pd.DataFrame): IST dataframe
        id_label (str): id column label
        columns (List[str]): value columns with absolute values

    Returns:
        pd.DataFrame: _description_
    """
    df = df.copy()
    for column in columns:
        df["mask_first"] = (df[id_label] != df[id_label].shift(periods=1, axis=0) ) #get bool column when a new device starts
        df["ref"] = df[df["mask_first"]][column]                                    #place reference in its column
        df["ref"] = df["ref"].ffill()                                               #fill the columns with the reference
        df[target_delta_prefix+column] =  (df[column]/df["ref"]-1)*100              #calculate shift
        df.pop("ref")   
        df.pop("mask_first")                                                          
    
    return df



def get_type(df: pd.DataFrame):
    """
    Returns whether a coupon is product or canary according to the following rule:
    For design 453828, letter L -> product, letter K -> canary
    For other designs, letter K -> product, letter L -> canary 

    Args:
        df (pd.DataFrame): IST dataframe
    """
    
    def get_type_fun(row):
        design = row[target_original_design_label]
        letter = row[target_letter_label]
        coupon_type = None
        if design == 453828:
            if letter == "L":
                coupon_type = target_type_product_value
            
            elif letter == "K":
                coupon_type = target__type_canary_value       
        else:
            if letter == "L":
                coupon_type = target__type_canary_value
            
            elif letter == "K":
                coupon_type = target_type_product_value
        return coupon_type
    
    df = df.copy()
    df[[target_panel_label, target_letter_label]] = df[target_original_name_label].str.split("_", expand=True)
    df[target_panel_label] = df[target_panel_label]
    df[target_type_label] = df.apply(get_type_fun, axis=1)
    
    return df


def get_group_id(df: pd.DataFrame, grouping_method: str, grouping_column: str=None):
    """
    Adds the grouping grouping_columns according to the `grouping_method` or `grouping_column` specified

    Args:
        df (pd.DataFrame): _description_
        grouping_method (str): _description_
        grouping_column (str, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    assert grouping_method in ["panel", "column"], AssertionError(f"Invalid grouping_method! Got {grouping_method}")
    
    df_ = df.copy()
    
    if grouping_method == "panel":
        df_['numeric_part'] = df_[target_original_id_name_label].str.extract(r'(\d+)')[0].astype(int)
        df_[trans_group_id] = df_[target_original_batch_label] + '_' + df_['numeric_part'].astype(str)
        
    elif grouping_method == "column" and grouping_column is not None:
        df_[trans_group_id] = df_[grouping_column]
        
    else:
        raise ValueError(f"Invalid grouping option! grouping_method={grouping_method}, grouping_column={grouping_column}")
    
    return df_



def process_ist_dataframe(df: pd.DataFrame, grouping_method: str, grouping_column: str, filter_type:str, max_len: float=None)->pd.DataFrame:
    """
    Process the IST dataframe performing the following actions:
    
    1) Fixes columns names which could be unnecessary repeated and with a wrong format
    2) Filters out cycles which are not numeric and convert the rest to float
    3) Select between LOW/HIGH temperature measurement
    4) For each cycle, select a single resistance value, the maximum
    5) calculate the delta R
    6) calculate the coupon type (product or canary)
    7) filter coupon type
    8) define group_id according to instructions
    9) set delta R values > 10 to nan 

    Args:
        df (pd.DataFrame): IST dataframe

    Returns:
        pd.DataFrame: processed dataframe
    """
    
    df = fix_format_columns(df)
    df = fix_duplicate_columns(df)
    df = df[pd.to_numeric(df[target_original_pos_label], errors='coerce').notnull()]
    df = df.set_index(target_temperature_label).loc[target_temperature_select_label].reset_index()  #select temperature LOW/HIGH
    logging.info(f"Selected temperature: {target_temperature_select_label}")
    df[target_original_pos_label] = df[target_original_pos_label].astype(float) # convert position to float
    df = df.loc[df.groupby([target_original_id_label,target_original_pos_label])[target_original_sense_A_label].idxmax()] # make cycles:-->resistance bijective
    df = df.sort_values(by = [target_original_id_label,target_original_pos_label])
    df = df.reset_index()
    df = get_delta(df,target_original_id_label,target_senses_list) # calculate delta R
    df = get_type(df)
    df = df[df[target_type_label]==filter_type].reset_index()
    df = get_group_id(df,grouping_method, grouping_column)
    # set values > 10 to nan:
    # the IST test is interrupted at 10 and any values above
    # might me the result of numerical artifacts
    df.loc[df[target_delta_A_label].abs() > 10,target_delta_A_label] = np.nan
    df.loc[df[target_delta_B_label].abs() > 10,target_delta_B_label] = np.nan
    return df


# TODO: remove
# def map_df(row,df,columns,id_label):
#         return df.loc[row[id_label]][columns].item()


# TODO: remove
# def calculate_snr(df: pd.DataFrame,columns: List[str])->None:
#     df = df.copy()
#     snr = pd.DataFrame(df.groupby(target_id_label)[columns].mean()/df.groupby(target_id_label)[columns].std())
#     n_cycles = pd.DataFrame(df.groupby(target_id_label)[target_number_cycles_label].median())
#     sample_type = pd.DataFrame(df.groupby(target_id_label)[target_type_label].apply(lambda x: x.iloc[0]))
    
#     if len(columns)>1:
#         snr = snr.apply(np.mean,axis=1)
    
#     df_info = pd.concat([snr, n_cycles, sample_type],axis=1).rename(columns={0:SNR_label})
    
#     insert_snr = partial(map_df,df=df_info, columns=SNR_label, id_label=target_id_label)
    
#     df[SNR_label] = df.apply(insert_snr,axis=1)
    
#     return df_info, df

# TODO: remove
# def filter_df(df: pd.DataFrame, settings: str)->pd.DataFrame:
#     df = df.copy()
#     assert settings in filter_settings, f"Settings available {[s for s in filter_settings]}"
#     snr = filter_settings[settings][SNR_label]
#     n_cycles = filter_settings[settings][number_cycles_label]
#     df = df[df[SNR_label] > snr]
    
#     if n_cycles is not None:
#         df = df[df[target_number_cycles_label] == n_cycles]
    
#     return df.reset_index()


def filter_df_max_len(df: pd.DataFrame, max_len: float, mode: str)->pd.DataFrame:
    """
    Filters the IST dataframe with an upper bound of the maximum length
    Args:
        df (pd.DataFrame): unfiltered dataframe
        max_len (float): maximum length
        mode (str): filtering method, 
            'remove' removes all non conformal IST;
            'clip' cuts IST to conformal length

    Returns:
        pd.DataFrame: filtered dataframe
    """
    df = df.copy()
    assert mode in ["remove", "clip"], AssertionError(f"Mode {mode} not valid!")
    
    id_label = trans_group_id
    
    if mode == "remove":
    
        count_condition_1 = df.groupby([id_label,trans_variable_label])[trans_position_label].max() <= max_len

        valid_groups = count_condition_1 
        valid_groups = valid_groups[valid_groups].index

        df_out = df[df.set_index([id_label, trans_variable_label]).index.isin(valid_groups)]
        
    elif mode == "clip":
        
        df.loc[df[trans_position_label] > max_len, trans_value_label] = np.nan
        df_out = df.dropna()
        
        
        
    logging.info(f"Max length {max_len} set with method {mode}, number of curves: {len(df_out[id_label].unique())}")
    
    return df_out.reset_index()


def pad_df_to_max_len(df: pd.DataFrame, max_len: float) -> pd.DataFrame:
    """
    Pads sequences in the IST dataframe to reach the specified max_len by adding 
    placeholder rows with NaN values for positions that don't exist.
    
    Args:
        df (pd.DataFrame): dataframe with sequences that may be shorter than max_len
        max_len (float): target length to pad sequences to

    Returns:
        pd.DataFrame: dataframe with sequences padded to max_len using NaN placeholders
    """
    df = df.copy()
    
    id_label = trans_group_id
    
    # Find the maximum position for each group-variable combination
    max_positions = df.groupby([id_label, trans_variable_label])[trans_position_label].max()
    
    # Identify which group-variable combinations need padding
    groups_to_pad = max_positions[max_positions < max_len]
    
    if len(groups_to_pad) == 0:
        logging.info(f"No sequences need padding to max_len {max_len}")
        return df.reset_index(drop=True)
    
    # Create padding rows
    padding_rows = []
    
    for (group_id, variable), current_max_pos in groups_to_pad.items():
        # Get a sample row from this group-variable to copy metadata
        sample_row = df[(df[id_label] == group_id) & (df[trans_variable_label] == variable)].iloc[0]
        
        # Create padding positions from current_max_pos + 1 to max_len
        padding_positions = range(int(current_max_pos) + 1, int(max_len) + 1)
        
        for pos in padding_positions:
            padding_row = sample_row.copy()
            padding_row[trans_position_label] = pos
            padding_row[trans_value_label] = np.nan
            padding_rows.append(padding_row)
    
    if padding_rows:
        # Convert list of Series to DataFrame
        padding_df = pd.DataFrame(padding_rows)
        
        # Combine original and padding data
        df_padded = pd.concat([df, padding_df], ignore_index=True)
        
        # Sort by group, variable, and position
        df_padded = df_padded.sort_values([id_label, trans_variable_label, trans_position_label])
        
        logging.info(f"Padded {len(groups_to_pad)} group-variable combinations to max_len {max_len}, added {len(padding_rows)} padding rows")
    else:
        df_padded = df
        logging.info(f"No padding needed for max_len {max_len}")
    
    return df_padded.reset_index(drop=True)


def get_ist_uniform_length(df: pd.DataFrame, method: str, variables: List)->pd.DataFrame:
    """
    get batches with uniform length ist curves, useful to take smooth expectation.
    Tip: place this function BEFORE calculating batch statistics 
    
    two methods are available:
    1) 'exclude': will remove batches with non uniform ist lengths
    2) 'clip': will clip all ist curves to the shortest (except zero length, they
    will be excluded)

    Args:
        df (pd.DataFrame): non-uniform lengths ist DataFrame
        method (str): method to use, options 'exclude', 'clip'

    Returns:
        pd.DataFrame: uniform lengths ist DataFrame
    """
    

    df = df.copy()
    uniform_batches = []

    assert method in ["exclude", "clip"], AssertionError(f"Method {method} not supported! should be one of 'exclude', or 'clip'")
    assert trans_id_label in df.columns, AssertionError(f"{trans_id_label} not in dataframe columns, still needed for this function!")

    if method == "exclude":
        groups = df[trans_group_id].unique()
        
        for group in groups:
            df_ = df.set_index(trans_group_id).loc[group]
            df_group = df_.groupby([trans_id_label, trans_variable_label])[trans_value_label].agg("count").reset_index()
            max_len = df_group[trans_value_label].max()
            df_group["Filter"] = df_group[trans_value_label] < max_len
            total_id = len(df_group)
            excluded_id = df_group["Filter"].sum()
            
            logging.info(f"{group} excluded {excluded_id}/{total_id}")
            
            if excluded_id == 0:
                uniform_batches.append(group)
            
        df_filter = df[df[trans_group_id].isin(uniform_batches)]
        df_out = df_filter.copy()
        logging.info(f"Total batches: {len(groups)}, Filtered batches: {len(uniform_batches)}")


    if method == "clip":

        df_reconstruct = None
        groups = df[trans_group_id].unique()

        for group in groups:
            df_ = df.set_index(trans_group_id).loc[group] # select group id
            
            for var in variables:          
                df_group = df_.groupby(trans_id_label)[var].agg("count").reset_index()
                min_len_var = np.array([count for count in df_group[var] if count != 0]).min()
                df_.loc[df_[trans_position_label] > min_len_var, var] = np.nan
            
            # append group dataframe
            if df_reconstruct is None:
                df_reconstruct = df_.reset_index()
            else:
                df_reconstruct = pd.concat([df_reconstruct, df_.reset_index()], axis=0)

        df_reconstruct = df_reconstruct.reset_index(drop=True)
        df_out = df_reconstruct.copy()
        
    return df_out
