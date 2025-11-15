"""
Sample metrics computation utilities for proT_pipeline dataset generation.

This module provides functions to compute various metrics for samples in a dataset.
These metrics can be used for:
- Stratified train-test splitting
- Dataset analysis and visualization
- Quality control and monitoring

The module is organized into two categories:
1. **Metric Functions**: Compute sample-level metrics (e.g., last value, NaN fraction)
2. **Rarity Functions**: Convert metrics into rarity scores for stratification

Rarity scores are computed by:
1. Computing a metric value for each sample
2. Binning metric values into histogram bins
3. Computing inverse frequency (1/bin_count) as rarity
4. Normalizing to [0, 1] range

Higher rarity scores indicate rarer samples.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Callable, Dict, Any


def compute_rarity(values: np.ndarray, n_bins: int = 50) -> np.ndarray:
    """
    Compute rarity scores based on histogram bin frequencies.
    
    Values in less frequent bins receive higher rarity scores.
    Rarity is computed as 1 / bin_frequency, normalized to [0, 1].
    
    This is the core rarity function ported from the Ishigami dataset.
    
    Parameters
    ----------
    values : np.ndarray
        1D array of target values to compute rarity for
    n_bins : int, optional
        Number of bins for histogram (default: 50)
    
    Returns
    -------
    np.ndarray
        Rarity score for each value, normalized to [0, 1] where
        1.0 indicates the rarest values and 0.0 the most common
    
    Examples
    --------
    >>> values = np.array([1, 1, 1, 2, 2, 3, 10, 10])
    >>> rarity = compute_rarity(values, n_bins=4)
    >>> # Values 1 (3 occurrences) will have lower rarity than 10 (2 occurrences)
    """
    # Compute histogram
    counts, bin_edges = np.histogram(values, bins=n_bins)
    
    # Assign each value to a bin
    bin_indices = np.digitize(values, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Compute rarity as inverse frequency
    bin_frequencies = counts[bin_indices]
    rarity = 1.0 / (bin_frequencies + 1)  # +1 to avoid division by zero
    
    # Normalize to [0, 1]
    rarity = (rarity - rarity.min()) / (rarity.max() - rarity.min() + 1e-10)
    
    return rarity


def compute_rarity_last_value(
    df: pd.DataFrame,
    group_id_label: str,
    variable_label: str,
    value_label: str,
    position_label: str,
    n_bins: int = 50,
    keep_intermediate: bool = False
) -> pd.DataFrame:
    """
    Compute rarity based on average of last non-NaN values across variables.
    
    This metric is particularly useful for time series or sequential data where
    the final outcome (last value) is most relevant for stratification.
    
    Processing steps:
    1. For each sample (group_id), for each variable:
       - Sort by position to ensure correct sequence order
       - Extract the last non-NaN value
    2. Average these last values across all variables for the sample
    3. Apply histogram-based rarity scoring to these averaged values
    
    Parameters
    ----------
    df : pd.DataFrame
        Generic dataframe with sequential data. Can be target or input dataframe.
        Must contain columns specified by the label parameters.
    group_id_label : str
        Column name for sample/group identifier (e.g., 'group', 'sample_id')
    variable_label : str
        Column name for variable identifier (e.g., 'variable', 'Sense_A/B')
    value_label : str
        Column name for the numeric values
    position_label : str
        Column name for position/sequence order (e.g., 'position', 'Zyklus')
    n_bins : int, optional
        Number of bins for histogram-based rarity calculation (default: 50)
    keep_intermediate : bool, optional
        If True, keep the 'avg_last_value' column in output (default: False)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - {group_id_label}: Sample identifier
        - 'rarity_last_value': Rarity score [0, 1]
        - 'avg_last_value': (optional) The averaged last value if keep_intermediate=True
    
    Examples
    --------
    >>> rarity_df = compute_rarity_last_value(
    ...     df=df_target,
    ...     group_id_label='group',
    ...     variable_label='variable',
    ...     value_label='value',
    ...     position_label='position',
    ...     n_bins=50
    ... )
    >>> print(rarity_df.head())
       group  rarity_last_value
    0      0               0.23
    1      1               0.87
    2      2               0.45
    
    Notes
    -----
    - Handles variable number of variables per sample (not hardcoded to 2)
    - Samples with no valid last values are excluded from the result
    - Uses position column to determine sequence order, not DataFrame row order
    """
    logger = logging.getLogger(__name__)
    
    last_values_per_sample = []
    skipped_samples = []
    
    # Get unique sample IDs
    sample_ids = df[group_id_label].unique()
    logger.info(f"Computing rarity_last_value for {len(sample_ids)} samples")
    
    for sample_id in sample_ids:
        # Get data for this sample
        sample_df = df[df[group_id_label] == sample_id]
        
        # Get unique variables for this sample
        variables = sample_df[variable_label].unique()
        
        last_values = []
        
        # For each variable, get last non-NaN value
        for var in variables:
            var_df = sample_df[sample_df[variable_label] == var].copy()
            
            # Sort by position to ensure correct order
            var_df = var_df.sort_values(by=position_label)
            
            # Get non-NaN values
            non_nan_values = var_df[value_label].dropna()
            
            if len(non_nan_values) > 0:
                last_value = non_nan_values.iloc[-1]  # Last element
                last_values.append(last_value)
        
        # Average across all variables
        if len(last_values) > 0:
            avg_last_value = np.mean(last_values)
            last_values_per_sample.append({
                group_id_label: sample_id,
                'avg_last_value': avg_last_value
            })
        else:
            skipped_samples.append(sample_id)
    
    if len(skipped_samples) > 0:
        logger.warning(
            f"Skipped {len(skipped_samples)} samples with no valid last values: "
            f"{skipped_samples[:5]}{'...' if len(skipped_samples) > 5 else ''}"
        )
    
    # Create DataFrame
    result_df = pd.DataFrame(last_values_per_sample)
    
    if len(result_df) == 0:
        raise ValueError("No valid samples found for rarity calculation")
    
    logger.info(f"Successfully computed last values for {len(result_df)} samples")
    
    # Compute rarity scores using histogram-based binning
    rarity_scores = compute_rarity(result_df['avg_last_value'].values, n_bins=n_bins)
    result_df['rarity_last_value'] = rarity_scores
    
    # Log statistics
    logger.info(f"Rarity statistics:")
    logger.info(f"  avg_last_value: min={result_df['avg_last_value'].min():.3f}, "
                f"max={result_df['avg_last_value'].max():.3f}, "
                f"mean={result_df['avg_last_value'].mean():.3f}")
    logger.info(f"  rarity_last_value: min={result_df['rarity_last_value'].min():.3f}, "
                f"max={result_df['rarity_last_value'].max():.3f}, "
                f"mean={result_df['rarity_last_value'].mean():.3f}")
    
    # Select output columns
    if keep_intermediate:
        result_df = result_df[[group_id_label, 'avg_last_value', 'rarity_last_value']]
    else:
        result_df = result_df[[group_id_label, 'rarity_last_value']]
    
    return result_df


def apply_metrics(
    metric_functions: List[Tuple[Callable, Dict[str, Any]]]
) -> pd.DataFrame:
    """
    Generic orchestrator to apply multiple metric functions and collect results.
    
    This function allows computing multiple sample-level metrics at once and
    combining them into a single DataFrame. Each metric function adds one or
    more columns to the result.
    
    The metrics DataFrame is initialized with the first function's output,
    then subsequent functions add columns via merge operations.
    
    This is a generic function that works with any metric computation function,
    including rarity functions, raw metrics, or any other sample-level statistics.
    
    Parameters
    ----------
    metric_functions : List[Tuple[Callable, Dict[str, Any]]]
        List of tuples where each tuple contains:
        - function: The metric computation function to call
        - kwargs: Dictionary of keyword arguments to pass to the function
        
        Each function must:
        - Accept 'group_id_label' as a parameter
        - Return a DataFrame with the group_id column and one or more metric columns
        
        The group_id_label must be consistent across all functions.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with group_id column and one or more metric columns.
        Column names depend on the metric functions used.
        Example: [group_id, rarity_last_value, nan_fraction, seq_length, ...]
    
    Raises
    ------
    ValueError
        If metric_functions is empty or if group_id_label is inconsistent
    
    Examples
    --------
    >>> from proT_pipeline.labels import *
    >>> 
    >>> # Single metric function
    >>> metrics_df = apply_metrics([
    ...     (compute_rarity_last_value, {
    ...         'df': df_target,
    ...         'group_id_label': trans_group_id,
    ...         'variable_label': trans_variable_label,
    ...         'value_label': trans_value_label,
    ...         'position_label': trans_position_label,
    ...         'n_bins': 50
    ...     })
    ... ])
    >>> 
    >>> # Multiple metrics from different sources
    >>> metrics_df = apply_metrics([
    ...     (compute_rarity_last_value, {...}),      # Rarity from target
    ...     (compute_rarity_nan_fraction, {...}),    # Rarity from input
    ...     (compute_mean_value, {...}),             # Raw metric (future)
    ... ])
    
    Notes
    -----
    - First function initializes the DataFrame
    - Subsequent functions add columns via outer merge on group_id
    - All functions must use the same group_id_label parameter
    - This function is agnostic to the type of metrics computed
    """
    logger = logging.getLogger(__name__)
    
    if len(metric_functions) == 0:
        raise ValueError("Must provide at least one metric function")
    
    logger.info(f"Applying {len(metric_functions)} metric function(s)")
    
    # Initialize with first function
    first_func, first_kwargs = metric_functions[0]
    group_id_label = first_kwargs.get('group_id_label')
    
    if group_id_label is None:
        raise ValueError("First function kwargs must include 'group_id_label'")
    
    logger.info(f"Initializing metrics DataFrame with {first_func.__name__}")
    metrics_df = first_func(**first_kwargs)
    
    # Add columns from subsequent functions
    for i, (func, func_kwargs) in enumerate(metric_functions[1:], start=2):
        # Verify consistent group_id_label
        func_group_id = func_kwargs.get('group_id_label')
        if func_group_id != group_id_label:
            raise ValueError(
                f"Inconsistent group_id_label: expected '{group_id_label}', "
                f"got '{func_group_id}' in function {i}"
            )
        
        logger.info(f"Adding metric from {func.__name__}")
        result_df = func(**func_kwargs)
        
        # Merge on group_id, adding new metric column(s)
        # Use outer join to preserve all samples
        metrics_df = metrics_df.merge(
            result_df, 
            on=group_id_label, 
            how='outer'
        )
    
    logger.info(f"Final metrics DataFrame shape: {metrics_df.shape}")
    logger.info(f"Columns: {list(metrics_df.columns)}")
    
    return metrics_df


# Backward compatibility alias
def apply_rarity(rarity_functions: List[Tuple[Callable, Dict[str, Any]]]) -> pd.DataFrame:
    """
    Backward compatibility wrapper for apply_metrics.
    
    Deprecated: Use apply_metrics() instead. This alias will be removed in future versions.
    """
    logger = logging.getLogger(__name__)
    logger.warning("apply_rarity() is deprecated. Use apply_metrics() instead.")
    return apply_metrics(rarity_functions)


# ============================================================================
# Additional Rarity Functions
# ============================================================================


def compute_rarity_nan_fraction(
    df: pd.DataFrame,
    group_id_label: str,
    value_label: str,
    n_bins: int = 50,
    keep_intermediate: bool = False
) -> pd.DataFrame:
    """
    Compute rarity based on the fraction of NaN values in the sequence.
    
    This metric measures data completeness/sparsity. Samples with unusual
    proportions of missing data (very high or very low) receive higher rarity scores.
    This is particularly useful for input data where the amount of missing data
    can indicate rare data collection patterns or edge cases.
    
    Processing steps:
    1. For each sample (group_id):
       - Count total values (NaN + non-NaN)
       - Count NaN values
       - Compute fraction: nan_count / total_count
    2. Apply histogram-based rarity scoring to these fractions
    
    Parameters
    ----------
    df : pd.DataFrame
        Generic dataframe with sequential data. Can be input or target dataframe.
        Must contain columns specified by the label parameters.
    group_id_label : str
        Column name for sample/group identifier (e.g., 'group', 'sample_id')
    value_label : str
        Column name for the numeric values to check for NaNs
    n_bins : int, optional
        Number of bins for histogram-based rarity calculation (default: 50)
    keep_intermediate : bool, optional
        If True, keep the 'nan_fraction' column in output (default: False)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - {group_id_label}: Sample identifier
        - 'rarity_nan_fraction': Rarity score [0, 1]
        - 'nan_fraction': (optional) The fraction of NaN values if keep_intermediate=True
    
    Examples
    --------
    >>> # Compute rarity based on missing data in input
    >>> rarity_df = compute_rarity_nan_fraction(
    ...     df=df_input,
    ...     group_id_label='group',
    ...     value_label='value',
    ...     n_bins=50
    ... )
    >>> print(rarity_df.head())
       group  rarity_nan_fraction
    0      0                 0.45
    1      1                 0.12
    2      2                 0.89
    
    Notes
    -----
    - Fraction is computed as: nan_count / total_count
    - Samples with 0% or 100% NaN can be rare depending on the dataset
    - Useful for identifying samples with unusual data completeness patterns
    """
    logger = logging.getLogger(__name__)
    
    nan_fractions_per_sample = []
    
    # Get unique sample IDs
    sample_ids = df[group_id_label].unique()
    logger.info(f"Computing rarity_nan_fraction for {len(sample_ids)} samples")
    
    for sample_id in sample_ids:
        # Get data for this sample
        sample_df = df[df[group_id_label] == sample_id]
        
        # Count total and NaN values
        total_count = len(sample_df)
        nan_count = sample_df[value_label].isna().sum()
        
        # Compute fraction
        nan_fraction = nan_count / total_count if total_count > 0 else 0.0
        
        nan_fractions_per_sample.append({
            group_id_label: sample_id,
            'nan_fraction': nan_fraction
        })
    
    # Create DataFrame
    result_df = pd.DataFrame(nan_fractions_per_sample)
    
    logger.info(f"Successfully computed NaN fractions for {len(result_df)} samples")
    
    # Compute rarity scores using histogram-based binning
    rarity_scores = compute_rarity(result_df['nan_fraction'].values, n_bins=n_bins)
    result_df['rarity_nan_fraction'] = rarity_scores
    
    # Log statistics
    logger.info(f"Rarity statistics:")
    logger.info(f"  nan_fraction: min={result_df['nan_fraction'].min():.3f}, "
                f"max={result_df['nan_fraction'].max():.3f}, "
                f"mean={result_df['nan_fraction'].mean():.3f}")
    logger.info(f"  rarity_nan_fraction: min={result_df['rarity_nan_fraction'].min():.3f}, "
                f"max={result_df['rarity_nan_fraction'].max():.3f}, "
                f"mean={result_df['rarity_nan_fraction'].mean():.3f}")
    
    # Select output columns
    if keep_intermediate:
        result_df = result_df[[group_id_label, 'nan_fraction', 'rarity_nan_fraction']]
    else:
        result_df = result_df[[group_id_label, 'rarity_nan_fraction']]
    
    return result_df
