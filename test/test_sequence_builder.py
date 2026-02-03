"""
Test script for sequence_builder module.

NOTE: This test references the deprecated sequence_builder module which has been 
refactored into the new pipeline structure. The functionality is now part of
the input_processing module.

This test file is kept for reference but may not work with the current 
module structure.
"""

import pandas as pd
import sys
from os.path import dirname, abspath, join, exists

# Setup path
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)


def test_sequence_builder():
    """
    Test sequence builder functionality.
    
    NOTE: The original sequence_builder module has been deprecated.
    This functionality is now handled differently in the pipeline.
    """
    print("=" * 60)
    print("SEQUENCE BUILDER TEST")
    print("=" * 60)
    print()
    print("NOTE: This test references deprecated functionality.")
    print("The sequence_builder module has been refactored.")
    print()
    print("For current sequence building, use:")
    print("  - proT_pipeline.input_processing.assemble_raw")
    print("  - proT_pipeline.input_processing.process_raw")
    print("  - proT_pipeline.input_processing.generate_dataset")
    print()
    print("Test skipped - module deprecated.")


if __name__ == "__main__":
    test_sequence_builder()
