"""
IST Placeholder Generator for Prediction Mode.

This module provides functionality to generate placeholder target dataframes
(df_trg.csv) for prediction mode, where actual IST (resistance test) data 
is not yet available.

The generated placeholders have the same structure as real IST data but with
zero values, allowing the transformer model to generate predictions.
"""

import pandas as pd
import numpy as np
from os.path import join, exists, splitext
from os import makedirs
from typing import Union, List, Optional
from datetime import datetime


def generate_ist_placeholders(
    group_ids: Union[List[str], str],
    max_len: int = 200,
    num_vars: int = 2,
    date: Optional[str] = None,
    output_path: Optional[str] = None,
    id_file_column: str = "group",
    save_index: bool = False
) -> pd.DataFrame:
    """
    Generate placeholder IST target dataframe for prediction mode.
    
    Creates a df_trg.csv compatible dataframe with the correct structure
    but placeholder values (zeros) for the IST resistance measurements.
    This allows the input pipeline to process manufacturing data for
    samples that don't yet have IST test results.
    
    Args:
        group_ids: Sample identifiers. Can be:
            - List of strings: ["CYDH_01", "CYDH_05", "CYEI_04", ...]
            - Path to CSV file: "path/to/sample_ids.csv"
            - Path to NPY file: "path/to/sample_ids.npy"
            - Path to TXT file: "path/to/sample_ids.txt" (one ID per line)
        max_len: Maximum sequence length (must match training configuration).
            Default: 200 (standard IST cycle count)
        num_vars: Number of target variables. Default: 2 (Sense A and Sense B)
        date: Placeholder date string. Default: current date in ISO format
        output_path: If provided, saves the dataframe to this path.
            Can be a directory (saves as df_trg.csv) or full file path.
        id_file_column: Column name to use when loading IDs from CSV file.
            Default: "group"
        save_index: Whether to include index when saving to CSV.
            Default: False
    
    Returns:
        pd.DataFrame: Target dataframe in df_trg.csv format with columns:
            - group: Sample identifier
            - position: Cycle number (1 to max_len)
            - date: Placeholder date
            - variable: Variable ID (1, 2, ... num_vars)
            - value: Placeholder value (0.0)
    
    Examples:
        >>> # From list
        >>> df = generate_ist_placeholders(["CYDH_01", "CYDH_05"])
        
        >>> # From file with auto-save
        >>> df = generate_ist_placeholders(
        ...     "sample_ids.csv",
        ...     max_len=200,
        ...     output_path="data/builds/my_build/control/"
        ... )
        
        >>> # Custom configuration
        >>> df = generate_ist_placeholders(
        ...     group_ids=["BATCH_A_1", "BATCH_A_2"],
        ...     max_len=150,
        ...     num_vars=2,
        ...     date="2025-01-01 08:00:00"
        ... )
    
    Notes:
        - The variable encoding uses integers (1, 2, ...) to match the 
          pipeline's internal representation
        - Values are set to 0.0 (not NaN) as the model expects numeric values
        - The date field is a placeholder and not used for prediction
    """
    
    # Load group IDs from file if path is provided
    ids = _load_group_ids(group_ids, id_file_column)
    
    if len(ids) == 0:
        raise ValueError("No group IDs provided or loaded from file")
    
    # Set default date if not provided
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate placeholder dataframe
    df = _create_placeholder_dataframe(ids, max_len, num_vars, date)
    
    # Save if output path provided
    if output_path is not None:
        _save_dataframe(df, output_path, save_index)
    
    return df


def _load_group_ids(
    group_ids: Union[List[str], str], 
    id_file_column: str
) -> List[str]:
    """
    Load group IDs from various input formats.
    
    Args:
        group_ids: List of IDs or path to file
        id_file_column: Column name for CSV files
    
    Returns:
        List of group ID strings
    """
    # If already a list, return as-is
    if isinstance(group_ids, (list, tuple, np.ndarray)):
        return [str(id_) for id_ in group_ids]
    
    # If string, treat as file path
    if isinstance(group_ids, str):
        if not exists(group_ids):
            raise FileNotFoundError(f"ID file not found: {group_ids}")
        
        _, ext = splitext(group_ids.lower())
        
        if ext == '.csv':
            df = pd.read_csv(group_ids)
            if id_file_column not in df.columns:
                # Try first column if specified column doesn't exist
                id_file_column = df.columns[0]
            return df[id_file_column].astype(str).tolist()
        
        elif ext == '.npy':
            arr = np.load(group_ids, allow_pickle=True)
            return [str(id_) for id_ in arr.flatten()]
        
        elif ext == '.txt':
            with open(group_ids, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        
        elif ext == '.xlsx':
            df = pd.read_excel(group_ids)
            if id_file_column not in df.columns:
                id_file_column = df.columns[0]
            return df[id_file_column].astype(str).tolist()
        
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported: .csv, .npy, .txt, .xlsx"
            )
    
    raise TypeError(
        f"group_ids must be a list or file path, got {type(group_ids)}"
    )


def _create_placeholder_dataframe(
    ids: List[str],
    max_len: int,
    num_vars: int,
    date: str
) -> pd.DataFrame:
    """
    Create the placeholder dataframe structure.
    
    Args:
        ids: List of group identifiers
        max_len: Sequence length
        num_vars: Number of variables
        date: Placeholder date string
    
    Returns:
        DataFrame with placeholder structure
    """
    # Pre-calculate total rows for efficiency
    total_rows = len(ids) * num_vars * max_len
    
    # Build arrays
    groups = []
    positions = []
    variables = []
    
    for id_ in ids:
        for var in range(1, num_vars + 1):
            for pos in range(1, max_len + 1):
                groups.append(id_)
                positions.append(pos)
                variables.append(var)
    
    # Create dataframe
    df = pd.DataFrame({
        "group": groups,
        "position": positions,
        "date": date,
        "variable": variables,
        "value": np.zeros(total_rows)
    })
    
    return df


def _save_dataframe(
    df: pd.DataFrame, 
    output_path: str, 
    save_index: bool
) -> None:
    """
    Save dataframe to the specified path.
    
    Args:
        df: DataFrame to save
        output_path: Directory or full file path
        save_index: Whether to include index in output
    """
    from proT_pipeline.labels import target_filename_df
    
    # Determine full path
    if output_path.endswith('.csv'):
        full_path = output_path
        dir_path = splitext(output_path)[0]
    else:
        # Treat as directory
        dir_path = output_path
        full_path = join(output_path, target_filename_df)
    
    # Create directory if needed
    if dir_path and not exists(dir_path):
        makedirs(dir_path, exist_ok=True)
    
    # Save
    df.to_csv(full_path, index=save_index)
    print(f"Saved placeholder target to: {full_path}")


def load_group_ids_from_process_data(
    dataset_id: str,
    limit: Optional[int] = None
) -> List[str]:
    """
    Extract group IDs from an existing processed dataset.
    
    Useful when you want to generate predictions for the same samples
    that were used in a training dataset.
    
    Args:
        dataset_id: Dataset build identifier
        limit: Optional limit on number of IDs to return
    
    Returns:
        List of group ID strings
    
    Example:
        >>> ids = load_group_ids_from_process_data("dyconex_251117")
        >>> df = generate_ist_placeholders(ids, max_len=200)
    """
    from proT_pipeline.labels import get_root_dir, get_input_dirs, trans_df_input
    
    ROOT = get_root_dir()
    _, OUTPUT_DIR, _ = get_input_dirs(ROOT, dataset_id)
    
    input_file = join(OUTPUT_DIR, trans_df_input)
    
    if not exists(input_file):
        raise FileNotFoundError(
            f"Processed input file not found: {input_file}\n"
            f"Make sure the dataset '{dataset_id}' has been processed."
        )
    
    df = pd.read_parquet(input_file)
    ids = df["group"].unique().tolist()
    
    if limit is not None:
        ids = ids[:limit]
    
    return ids
