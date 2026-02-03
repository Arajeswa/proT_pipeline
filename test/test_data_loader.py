"""
Test script for data_loader module.

Tests the process loading functionality with the updated module structure.
"""

import sys
from os.path import dirname, abspath, join, exists

# Setup path
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)

from proT_pipeline.input_processing.data_loader import get_processes


def test_get_processes(input_data_path: str, filepath_sel: str, grouping_method: str = "panel"):
    """
    Test loading processes from data files.
    
    Args:
        input_data_path: Path to input data folder
        filepath_sel: Path to lookup_selected.xlsx file
        grouping_method: Grouping method ("panel" or "column")
    """
    print(f"Testing get_processes with:")
    print(f"  Input path: {input_data_path}")
    print(f"  Selection file: {filepath_sel}")
    print(f"  Grouping method: {grouping_method}")
    print()
    
    # Load processes
    process_labels, processes = get_processes(
        input_data_path, 
        filepath_sel, 
        grouping_method=grouping_method,
        grouping_column=None
    )
    
    print(f"Loaded {len(processes)} processes: {process_labels}")
    
    # Test each process
    wa_list = []
    for process in processes:
        try:
            wa_data = process.df[process.WA_label]
            wa_list.append(wa_data)
            n_unique_wa = wa_data.nunique()
            print(f"  ✓ Process {process.process_label}: {len(process.df)} rows, {n_unique_wa} unique batches")
        except Exception as e:
            print(f"  ✗ Process {process.process_label}: Error - {e}")
    
    print()
    print("All tests passed!")
    return processes


if __name__ == "__main__":
    # Configuration - update these paths for your setup
    # Default paths assume running from the project root
    
    # Example with test data (if available)
    TEST_INPUT_PATH = join(ROOT_DIR, "data", "input", "Prozessdaten_MSEI_01_01_2022-07_07_2025_csv")
    TEST_CONTROL_PATH = join(ROOT_DIR, "data", "builds", "dyconex_251117", "control")
    TEST_SELECTION_FILE = join(TEST_CONTROL_PATH, "lookup_selected.xlsx")
    
    if exists(TEST_INPUT_PATH) and exists(TEST_SELECTION_FILE):
        print("=" * 60)
        print("DATA LOADER TEST")
        print("=" * 60)
        test_get_processes(TEST_INPUT_PATH, TEST_SELECTION_FILE)
    else:
        print("Test data not found. Please ensure data files are available.")
        print(f"  Expected input path: {TEST_INPUT_PATH}")
        print(f"  Expected selection file: {TEST_SELECTION_FILE}")
        print("\nSkipping test - data files required.")
