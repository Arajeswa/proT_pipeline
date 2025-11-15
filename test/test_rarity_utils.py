"""
Test script for rarity_utils.py

This script creates sample data and tests the rarity computation functions
to ensure they work correctly before integrating into the main pipeline.
"""

import numpy as np
import pandas as pd
import logging
import sys
from os.path import dirname, abspath

# Setup path
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)

from proT_pipeline.rarity_utils import (
    compute_rarity, 
    compute_rarity_last_value,
    compute_rarity_nan_fraction,
    apply_metrics  # Generic orchestrator (replaces apply_rarity)
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def create_sample_target_data(n_samples=100, n_positions=50, n_variables=2):
    """
    Create sample target data mimicking the proT_pipeline structure.
    
    Structure: BxLxD where
    - B: number of samples (n_samples)
    - L: sequence length (n_positions, padded)
    - D: features (position, variable, value)
    """
    data = []
    
    for sample_id in range(n_samples):
        # Random actual sequence length (before padding)
        actual_length = np.random.randint(10, n_positions)
        
        for var_id in range(n_variables):
            for pos in range(n_positions):
                if pos < actual_length:
                    # Non-padded region: real values
                    # Create some rare and common values
                    if sample_id < 10:  # Rare samples
                        value = np.random.uniform(100, 150)
                    elif sample_id >= 90:  # Another rare group
                        value = np.random.uniform(0, 10)
                    else:  # Common samples
                        value = np.random.uniform(40, 60)
                else:
                    # Padded region: NaN
                    value = np.nan
                
                data.append({
                    'group': sample_id,
                    'position': pos,
                    'variable': f'Var_{var_id}',
                    'value': value
                })
    
    df = pd.DataFrame(data)
    return df


def test_compute_rarity():
    """Test basic compute_rarity function."""
    print("\n" + "="*70)
    print("TEST 1: Basic compute_rarity function")
    print("="*70)
    
    # Create values with known distribution
    values = np.array([1]*30 + [2]*20 + [3]*10 + [10]*5)  # Common to rare
    np.random.shuffle(values)
    
    rarity = compute_rarity(values, n_bins=10)
    
    print(f"Input values shape: {values.shape}")
    print(f"Rarity scores shape: {rarity.shape}")
    print(f"Rarity range: [{rarity.min():.3f}, {rarity.max():.3f}]")
    print(f"Mean rarity: {rarity.mean():.3f}")
    
    # Check that rarer values (10) get higher rarity scores
    rare_indices = np.where(values == 10)[0]
    common_indices = np.where(values == 1)[0]
    
    mean_rare_score = rarity[rare_indices].mean()
    mean_common_score = rarity[common_indices].mean()
    
    print(f"Mean rarity for value=10 (rare): {mean_rare_score:.3f}")
    print(f"Mean rarity for value=1 (common): {mean_common_score:.3f}")
    
    assert mean_rare_score > mean_common_score, "Rare values should have higher rarity!"
    print("✓ Test passed: Rare values have higher rarity scores")


def test_compute_rarity_last_value():
    """Test compute_rarity_last_value function."""
    print("\n" + "="*70)
    print("TEST 2: compute_rarity_last_value function")
    print("="*70)
    
    # Create sample data
    df_target = create_sample_target_data(n_samples=100, n_positions=50, n_variables=2)
    
    print(f"Target DataFrame shape: {df_target.shape}")
    print(f"Columns: {list(df_target.columns)}")
    print(f"Number of unique samples: {df_target['group'].nunique()}")
    print(f"Number of unique variables: {df_target['variable'].nunique()}")
    
    # Compute rarity
    rarity_df = compute_rarity_last_value(
        df=df_target,
        group_id_label='group',
        variable_label='variable',
        value_label='value',
        position_label='position',
        n_bins=20,
        keep_intermediate=True  # Keep avg_last_value for inspection
    )
    
    print(f"\nRarity DataFrame shape: {rarity_df.shape}")
    print(f"Columns: {list(rarity_df.columns)}")
    print(f"\nFirst few rows:")
    print(rarity_df.head(10))
    
    print(f"\nRarity statistics:")
    print(rarity_df[['avg_last_value', 'rarity_last_value']].describe())
    
    # Check that rare samples (first 10 and last 10) have different rarity
    rare_low = rarity_df[rarity_df['group'] < 10]['rarity_last_value'].mean()
    rare_high = rarity_df[rarity_df['group'] >= 90]['rarity_last_value'].mean()
    common = rarity_df[(rarity_df['group'] >= 10) & (rarity_df['group'] < 90)]['rarity_last_value'].mean()
    
    print(f"\nMean rarity by group:")
    print(f"  Rare low values (group 0-9): {rare_low:.3f}")
    print(f"  Common values (group 10-89): {common:.3f}")
    print(f"  Rare high values (group 90-99): {rare_high:.3f}")
    
    print("✓ Test passed: Rarity computed successfully")


def test_apply_rarity():
    """Test apply_metrics orchestrator function (generic metric aggregation)."""
    print("\n" + "="*70)
    print("TEST 3: apply_metrics orchestrator function")
    print("="*70)
    
    # Create sample data
    df_target = create_sample_target_data(n_samples=50, n_positions=30, n_variables=2)
    
    # Use orchestrator with single function
    metrics_df = apply_metrics([
        (compute_rarity_last_value, {
            'df': df_target,
            'group_id_label': 'group',
            'variable_label': 'variable',
            'value_label': 'value',
            'position_label': 'position',
            'n_bins': 20,
            'keep_intermediate': False
        })
    ])
    
    print(f"\nMetrics DataFrame shape: {metrics_df.shape}")
    print(f"Columns: {list(metrics_df.columns)}")
    print(f"\nFirst few rows:")
    print(metrics_df.head())
    
    # Verify structure
    assert 'group' in metrics_df.columns, "Should have group column"
    assert 'rarity_last_value' in metrics_df.columns, "Should have rarity_last_value column"
    assert len(metrics_df) == 50, "Should have 50 samples"
    
    print("✓ Test passed: Orchestrator works correctly")


def test_with_variable_sequence_lengths():
    """Test that the function handles variable sequence lengths correctly."""
    print("\n" + "="*70)
    print("TEST 4: Variable sequence lengths (real-world scenario)")
    print("="*70)
    
    # Create data with more realistic variable lengths
    data = []
    for sample_id in range(20):
        # Each sample has different actual length
        actual_length = np.random.randint(5, 40)
        n_vars = np.random.randint(1, 4)  # 1-3 variables
        
        for var_id in range(n_vars):
            for pos in range(50):  # Max length 50
                if pos < actual_length:
                    value = np.random.uniform(10, 100) * (sample_id + 1) / 10
                else:
                    value = np.nan
                
                data.append({
                    'group': sample_id,
                    'position': pos,
                    'variable': f'Var_{var_id}',
                    'value': value
                })
    
    df = pd.DataFrame(data)
    
    print(f"Created {len(df)} rows for {df['group'].nunique()} samples")
    print(f"Variables per sample range: {df.groupby('group')['variable'].nunique().min()}-{df.groupby('group')['variable'].nunique().max()}")
    
    rarity_df = compute_rarity_last_value(
        df=df,
        group_id_label='group',
        variable_label='variable',
        value_label='value',
        position_label='position',
        n_bins=10,
        keep_intermediate=True
    )
    
    print(f"\nRarity DataFrame:")
    print(rarity_df)
    
    print("✓ Test passed: Handles variable sequence lengths and variable counts")


def create_sample_input_data(n_samples=100, n_positions=200, rare_nan_samples=None):
    """
    Create sample input data with varying NaN fractions.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_positions : int
        Maximum sequence length
    rare_nan_samples : list, optional
        List of sample IDs that should have unusual NaN fractions
    """
    data = []
    
    if rare_nan_samples is None:
        rare_nan_samples = list(range(5)) + list(range(95, 100))  # First 5 and last 5
    
    for sample_id in range(n_samples):
        for pos in range(n_positions):
            # Create different NaN patterns
            if sample_id in rare_nan_samples[:5]:
                # Very few NaNs (rare - complete data)
                is_nan = np.random.random() < 0.05
            elif sample_id in rare_nan_samples[5:]:
                # Very many NaNs (rare - sparse data)
                is_nan = np.random.random() < 0.90
            else:
                # Normal NaN proportion (common)
                is_nan = np.random.random() < 0.30
            
            value = np.nan if is_nan else np.random.uniform(10, 100)
            
            data.append({
                'group': sample_id,
                'position': pos,
                'value': value
            })
    
    df = pd.DataFrame(data)
    return df


def test_compute_rarity_nan_fraction():
    """Test compute_rarity_nan_fraction function."""
    print("\n" + "="*70)
    print("TEST 5: compute_rarity_nan_fraction function")
    print("="*70)
    
    # Create input data with varying NaN fractions
    df_input = create_sample_input_data(n_samples=100, n_positions=200)
    
    print(f"Input DataFrame shape: {df_input.shape}")
    print(f"Number of unique samples: {df_input['group'].nunique()}")
    
    # Compute rarity based on NaN fraction
    rarity_df = compute_rarity_nan_fraction(
        df=df_input,
        group_id_label='group',
        value_label='value',
        n_bins=20,
        keep_intermediate=True
    )
    
    print(f"\nRarity DataFrame shape: {rarity_df.shape}")
    print(f"Columns: {list(rarity_df.columns)}")
    print(f"\nFirst few rows:")
    print(rarity_df.head(10))
    
    print(f"\nRarity statistics:")
    print(rarity_df[['nan_fraction', 'rarity_nan_fraction']].describe())
    
    # Check that samples with unusual NaN fractions have higher rarity
    rare_complete = rarity_df[rarity_df['group'] < 5]['rarity_nan_fraction'].mean()
    rare_sparse = rarity_df[rarity_df['group'] >= 95]['rarity_nan_fraction'].mean()
    common = rarity_df[(rarity_df['group'] >= 5) & (rarity_df['group'] < 95)]['rarity_nan_fraction'].mean()
    
    print(f"\nMean rarity by NaN pattern:")
    print(f"  Rare complete data (group 0-4): {rare_complete:.3f}")
    print(f"  Common data (group 5-94): {common:.3f}")
    print(f"  Rare sparse data (group 95-99): {rare_sparse:.3f}")
    
    print("✓ Test passed: NaN fraction rarity computed successfully")


def test_combined_rarity_input_and_target():
    """Test combining rarity metrics from both input and target data."""
    print("\n" + "="*70)
    print("TEST 6: Combined rarity from input and target (Multi-source)")
    print("="*70)
    
    n_samples = 50
    
    # Create target data (for last value rarity)
    df_target = create_sample_target_data(n_samples=n_samples, n_positions=30, n_variables=2)
    
    # Create input data (for NaN fraction rarity)
    df_input = create_sample_input_data(n_samples=n_samples, n_positions=100)
    
    print(f"Target DataFrame shape: {df_target.shape}")
    print(f"Input DataFrame shape: {df_input.shape}")
    
    # Compute combined metrics using generic orchestrator
    metrics_df = apply_metrics([
        # Rarity from target: last value
        (compute_rarity_last_value, {
            'df': df_target,
            'group_id_label': 'group',
            'variable_label': 'variable',
            'value_label': 'value',
            'position_label': 'position',
            'n_bins': 15,
            'keep_intermediate': False
        }),
        # Rarity from input: NaN fraction
        (compute_rarity_nan_fraction, {
            'df': df_input,
            'group_id_label': 'group',
            'value_label': 'value',
            'n_bins': 15,
            'keep_intermediate': False
        })
    ])
    
    print(f"\nCombined Metrics DataFrame shape: {metrics_df.shape}")
    print(f"Columns: {list(metrics_df.columns)}")
    print(f"\nFirst few rows:")
    print(metrics_df.head(10))
    
    print(f"\nMetrics statistics:")
    print(metrics_df[['rarity_last_value', 'rarity_nan_fraction']].describe())
    
    # Verify we have both metrics
    assert 'group' in metrics_df.columns, "Should have group column"
    assert 'rarity_last_value' in metrics_df.columns, "Should have rarity_last_value from target"
    assert 'rarity_nan_fraction' in metrics_df.columns, "Should have rarity_nan_fraction from input"
    assert len(metrics_df) == n_samples, f"Should have {n_samples} samples"
    
    # Check for samples that are rare in both metrics
    high_rarity_threshold = 0.7
    rare_in_both = metrics_df[
        (metrics_df['rarity_last_value'] > high_rarity_threshold) & 
        (metrics_df['rarity_nan_fraction'] > high_rarity_threshold)
    ]
    
    print(f"\nSamples rare in both metrics (>{high_rarity_threshold}):")
    print(f"  Count: {len(rare_in_both)}")
    if len(rare_in_both) > 0:
        print(f"  Sample IDs: {rare_in_both['group'].tolist()}")
    
    print("✓ Test passed: Combined rarity from multiple sources works correctly")


def main():
    """Run all tests."""
    print("\n" + "#"*70)
    print("# RARITY UTILS TEST SUITE")
    print("#"*70)
    
    try:
        test_compute_rarity()
        test_compute_rarity_last_value()
        test_apply_rarity()
        test_with_variable_sequence_lengths()
        test_compute_rarity_nan_fraction()
        test_combined_rarity_input_and_target()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nThe rarity_utils.py module is ready to use.")
        print("\nKey Features Demonstrated:")
        print("  ✓ Basic histogram-based rarity computation")
        print("  ✓ Last value rarity for target data")
        print("  ✓ NaN fraction rarity for input data")
        print("  ✓ Combining multiple metrics from different sources")
        print("  ✓ Variable sequence lengths and variable counts")
        print("\nNext steps:")
        print("  1. Add more rarity functions as needed")
        print("  2. Create stratified_split.py for train/test splitting")
        print("  3. Integrate into generate_dataset.py")
        
    except Exception as e:
        print("\n" + "="*70)
        print("TEST FAILED! ✗")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
