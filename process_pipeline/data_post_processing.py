import numpy as np
import pandas as pd
import concurrent.futures
from typing import List

from core.modules import explode_time_components
from core.sequencer import Sequencer

def data_post_processing(
    df:pd.DataFrame,
    time_label: str,
    id_label: str,
    sort_label: str,
    features: List[str],
    max_seq_len: int,
    cluster: bool):
    
    # create time features
    df_tok,time_cmp = explode_time_components(df,time_label)
    
    features.extend(time_cmp)
    
    sequencer = Sequencer(
        df=df_tok, 
        features=features, 
        id_label = id_label, 
        sort_label=sort_label, 
        max_seq_len=max_seq_len)
    
    seq_list = []
    
    if cluster:
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for result in executor.map(sequencer.get_seq, sequencer.get_ids()):
                    seq_list.append(result)
        except Exception as e:
            print(f"Error occurred {e}")
        
    else:
        try:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for result in executor.map(sequencer.get_seq, sequencer.get_ids()):
                    seq_list.append(result)
        except Exception as e:
            print(f"Error occurred {e}")
    
    arr_np = np.array(seq_list)
    
    return arr_np



