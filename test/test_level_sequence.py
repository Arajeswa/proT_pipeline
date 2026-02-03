"""
Test script for level_sequences module.

NOTE: This test references the deprecated level_sequences module which has been 
refactored into the new pipeline structure. The functionality is now part of
the input_processing and core modules.

This test file is kept for reference but may not work with the current 
module structure.
"""

import pandas as pd
import sys
from os.path import dirname, abspath, join, exists

# Setup path
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)


def test_level_sequences():
    """
    Test level sequence functionality.
    
    NOTE: The original level_sequences module has been deprecated.
    This functionality is now handled differently in the pipeline.
    """
    print("=" * 60)
    print("LEVEL SEQUENCES TEST")
    print("=" * 60)
    print()
    print("NOTE: This test references deprecated functionality.")
    print("The level_sequences module has been refactored.")
    print()
    print("For current sequence processing, use:")
    print("  - proT_pipeline.input_processing.assemble_raw")
    print("  - proT_pipeline.input_processing.process_raw")
    print("  - proT_pipeline.core.sequencer")
    print()
    print("Test skipped - module deprecated.")


if __name__ == "__main__":
    test_level_sequences()
