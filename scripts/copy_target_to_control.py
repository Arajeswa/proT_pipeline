"""
Helper script to copy df_trg.csv from target builds to input control folder.

This script facilitates the workflow where:
1. Target pipeline generates df_trg.csv in data/target/builds/{build_id}/
2. This script copies it to data/builds/{dataset_id}/control/
3. Input pipeline can then use it for processing
"""

import sys
import shutil
from os.path import dirname, abspath, join, exists
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from proT_pipeline.labels import *


def copy_target_to_control(build_id: str, dataset_id: str):
    """
    Copy df_trg.csv from target build to input control folder.
    
    Args:
        build_id (str): Target build identifier
        dataset_id (str): Input dataset identifier
    """
    ROOT_DIR = dirname(dirname(abspath(__file__)))
    
    # Source: target build
    INPUT_DIR, BUILDS_DIR = get_target_dirs(ROOT_DIR)
    source_file = join(BUILDS_DIR, build_id, target_filename_df)
    
    # Destination: input control
    _, _, CONTROL_DIR = get_input_dirs(ROOT_DIR, dataset_id)
    dest_file = join(CONTROL_DIR, target_filename_df)
    
    # Check source exists
    if not exists(source_file):
        raise FileNotFoundError(f"Source file not found: {source_file}")
    
    # Copy file
    print(f"Copying target file:")
    print(f"  From: {source_file}")
    print(f"  To:   {dest_file}")
    
    shutil.copy2(source_file, dest_file)
    print(f"✓ Copy complete!")


if __name__ == "__main__":
    # Configuration
    build_id = "example_build"        # Target build ID
    dataset_id = "example_dataset"    # Input dataset ID
    
    print("=" * 80)
    print("COPY TARGET TO CONTROL")
    print("=" * 80)
    
    copy_target_to_control(build_id=build_id, dataset_id=dataset_id)
    
    print("=" * 80)
