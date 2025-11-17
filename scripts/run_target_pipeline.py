"""
Script to run the target (IST) processing pipeline.

This script processes IST resistance data to generate df_trg.csv files
that can be used by the input processing pipeline.
"""

import sys
from os.path import dirname, abspath, join
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from proT_pipeline.target_processing.main import main


if __name__ == "__main__":
    # Configuration
    build_id = "example_build"                  # Output directory name
    grouping_method = "panel"                   # "column" or "panel"
    grouping_column = None                      # Column if grouping_method=="column"
    max_len = 200                               # Maximum sequence length
    filter_type = "C"                           # C = canary, P = product
    uni_method = "clip"                         # Clipping option when calculating moments
    max_len_mode = "clip"                       # Clipping option to select maximum length
    mean_bool = False                           # Flag to calculate first moment
    std_bool = False                            # Flag to calculate second moment
    
    print("=" * 80)
    print("TARGET (IST) PROCESSING PIPELINE")
    print("=" * 80)
    print(f"Build ID: {build_id}")
    print(f"Grouping method: {grouping_method}")
    print(f"Max length: {max_len}")
    print(f"Filter type: {filter_type}")
    print("=" * 80)
    
    main(
        build_id=build_id,
        grouping_method=grouping_method,
        grouping_column=grouping_column,
        max_len=max_len,
        filter_type=filter_type,
        uni_method=uni_method,
        max_len_mode=max_len_mode,
        mean_bool=mean_bool,
        std_bool=std_bool
    )
    
    print("=" * 80)
    print("TARGET PIPELINE COMPLETE")
    print(f"Output: data/target/builds/{build_id}/df_trg.csv")
    print("=" * 80)
