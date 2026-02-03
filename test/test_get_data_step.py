"""
Test script for get_data_step functionality.

Tests retrieving data for specific process steps.
"""

import pandas as pd
import sys
from os.path import dirname, abspath, join, exists

# Setup path
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)

from proT_pipeline.core.modules import get_data_step
from proT_pipeline.input_processing.data_loader import get_processes
from proT_pipeline.utils import nested_dict_from_pandas


def test_get_data_step(input_data_path: str, control_path: str):
    """
    Test retrieving data for specific process steps.
    
    Args:
        input_data_path: Path to input data folder
        control_path: Path to control folder with lookup files
    """
    print("=" * 60)
    print("GET DATA STEP TEST")
    print("=" * 60)
    
    filepath_sel = join(control_path, "lookup_selected.xlsx")
    
    # Load processes
    print("\nLoading processes...")
    _, processes = get_processes(
        input_data_path, 
        filepath_sel,
        grouping_method="panel",
        grouping_column=None
    )
    print(f"Loaded {len(processes)} processes")
    
    # Test getting data for a specific step
    # Note: You'll need to adjust these values based on your actual data
    test_wa = None
    test_step = None
    
    # Try to find a valid WA and step from the loaded processes
    for process in processes:
        if len(process.df) > 0:
            test_wa = process.df[process.WA_label].iloc[0]
            test_step = process.df[process.PaPos_label].iloc[0]
            break
    
    if test_wa and test_step:
        print(f"\nTesting with WA={test_wa}, step={test_step}")
        
        try:
            df, _ = get_data_step(test_wa, test_step, processes, filepath_sel)
            print(f"  ✓ Retrieved {len(df)} rows")
            if len(df) > 0:
                print(f"  Found processes: {df['Process'].unique().tolist()}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    else:
        print("\nNo test data available - skipping step test")
    
    print("\nTest complete!")


if __name__ == "__main__":
    # Configuration - update these paths for your setup
    TEST_INPUT_PATH = join(ROOT_DIR, "data", "input", "Prozessdaten_MSEI_01_01_2022-07_07_2025_csv")
    TEST_CONTROL_PATH = join(ROOT_DIR, "data", "builds", "dyconex_251117", "control")
    
    if exists(TEST_INPUT_PATH) and exists(TEST_CONTROL_PATH):
        test_get_data_step(TEST_INPUT_PATH, TEST_CONTROL_PATH)
    else:
        print("Test data not found. Please ensure data files are available.")
        print(f"  Expected input path: {TEST_INPUT_PATH}")
        print(f"  Expected control path: {TEST_CONTROL_PATH}")
        print("\nSkipping test - data files required.")
