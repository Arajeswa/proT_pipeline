import numpy as np
import pandas as pd
import logging


def safe_read_csv(filepath, chunksize_fallback=50000, **kwargs):
    """
    Memory-safe wrapper for pd.read_csv.
    
    First attempts to read with low_memory=False for optimal dtype inference.
    If a MemoryError occurs (e.g., on low-RAM laptops), falls back to chunked
    reading which uses less memory at the cost of slower performance.
    
    Args:
        filepath: Path to CSV file
        chunksize_fallback: Number of rows per chunk when using fallback mode.
            Default: 50000 rows per chunk.
        **kwargs: All other arguments passed to pd.read_csv (sep, header, skiprows, etc.)
    
    Returns:
        pd.DataFrame: The loaded dataframe
    
    Example:
        >>> df = safe_read_csv("data.csv", sep=";", header=5)
        >>> df = safe_read_csv("large_file.csv", chunksize_fallback=10000)
    """
    logger = logging.getLogger(__name__)
    
    # Remove low_memory from kwargs if present (we control it)
    kwargs.pop('low_memory', None)
    
    try:
        # First try with low_memory=False for best dtype inference
        return pd.read_csv(filepath, low_memory=False, **kwargs)
    except MemoryError:
        logger.warning(
            f"MemoryError loading {filepath}. "
            f"Falling back to chunked reading with chunksize={chunksize_fallback}. "
            f"This may be slower but uses less memory."
        )
        chunks = pd.read_csv(filepath, chunksize=chunksize_fallback, **kwargs)
        return pd.concat(chunks, ignore_index=True)


def fix_duplicate_columns(df:pd.DataFrame)->None:
    """Fixes n-times repeated names from a pandas DataFrame adding a suffix "_n"
    
    Example: df.columns = Time, Value, Time, Variable, Time
    ---> new columns = Time_1, Value, Time_2, Variable, Time_3

    Args:
        df (pandas DataFrame): DataFrame containing multiple columns with the same name
    """
    columns = df.columns
    seen = np.array([])
    counter = np.ones(len(columns))
    
    for i,col in enumerate(columns):
        if col in seen:
            counter[i] = counter[i]+1
        else:
            seen = np.append(seen,col)
    
    df_col = pd.DataFrame({"Columns": columns,"Counter": counter})
    new_cols = np.array([])
    
    for i in df_col.index:
        if df_col.at[i,"Counter"] == 1:
            new_cols = np.append(new_cols,df_col.at[i,"Columns"])
        else:
            new_cols = np.append(new_cols,df_col.at[i,"Columns"]+"_"+str(df_col.at[i,"Counter"]))
    
    df.columns = new_cols
    
def fix_format_columns(df:pd.DataFrame)->pd.DataFrame:
    """Fix repeated column names of a pandas DataFrame
    
    Example: df.columns = Time Time, Value Value, First Variable First Variable
    ---> new columns = Time, Value, First Variable

    Args:
        df (pandas DataFrame): DataFrame containing column names formed by repeated identical words
        
    Returns:
        pandas DataFrame: DataFrame with fixed columns names
    """
    df.columns = ["_".join(df.columns[col].split()[0:int(0.5*len(df.columns[col].split()))]) for col in range(len(df.columns))]
    
    return df

def nested_dict_from_pandas(df:pd.DataFrame)->dict:
    """Generates a nested dictionary from the pandas DataFrame MultiIndex

    Args:
        df (pandas DataFrame): its MultiIndex must be set before with .set_index([Index1, Index2,...])

    Returns:
        dict : nested dictionary
    """
    
    if (df.index.nlevels==1):
        return df.index.unique().tolist()
    dict_f = {}
    for level in df.index.levels[0]:
        if (level in df.index):
            dict_f[level] = nested_dict_from_pandas(df.xs(level))
    
    return dict_f