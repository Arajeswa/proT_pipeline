"""
Script to run the input (process data) processing pipeline.

This script processes manufacturing process data and generates final datasets (X.npy, Y.npy).
It expects df_trg.csv to be present in the control folder for the specified dataset_id.
"""

import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from proT_pipeline.main import main


if __name__ == "__main__":
    # Configuration
    dataset_id = "example_dataset"
    missing_threshold = 30
    use_stratified_split = True
    stratified_metric = 'rarity_last_value'
    train_ratio = 0.8
    n_bins = 50
    split_shuffle = False
    split_seed = 42
    grouping_method = 'panel'
    grouping_column = None
    debug = False
    
    print("=" * 80)
    print("INPUT (PROCESS DATA) PROCESSING PIPELINE")
    print("=" * 80)
    print(f"Dataset ID: {dataset_id}")
    print(f"Missing threshold: {missing_threshold}")
    print(f"Stratified split: {use_stratified_split}")
    print(f"Grouping method: {grouping_method}")
    print("=" * 80)
    
    main(
        dataset_id=dataset_id,
        missing_threshold=missing_threshold,
        select_test=False,
        use_stratified_split=use_stratified_split,
        stratified_metric=stratified_metric,
        train_ratio=train_ratio,
        n_bins=n_bins,
        split_shuffle=split_shuffle,
        split_seed=split_seed,
        grouping_method=grouping_method,
        grouping_column=grouping_column,
        debug=debug
    )
    
    print("=" * 80)
    print("INPUT PIPELINE COMPLETE")
    print(f"Output: data/builds/{dataset_id}/output/")
    print("=" * 80)
