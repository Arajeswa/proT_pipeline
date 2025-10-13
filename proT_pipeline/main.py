import logging
from os.path import dirname, join, abspath, exists
from os import makedirs
import sys
from omegaconf import OmegaConf
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)
from proT_pipeline.labels import *
from proT_pipeline.assemble_raw import assemble_raw
from proT_pipeline.process_raw import process_raw
from proT_pipeline.generate_dataset import generate_dataset
from proT_pipeline.get_idx_from_id import get_idx_from_id



def main(
    dataset_id: str, 
    missing_threshold: float, 
    select_test: bool, 
    grouping_method: str, 
    grouping_column: str,
    debug: bool = False):
    """
    Dyconex dataset assembly according to the control files
    Folder structure:
    data
        |__input              | process files here
        |__builds             |
            |__dataset_id     | must be created beforehand!
                |__control    | control files here
                |__output     | output files here
    
    Args:
        dataset_id (str): name of dataset folder, must be created before!
        missing_threshold (float)
        select_test (bool):
        debug (bool, optional): if True assembles a slice of target. Defaults to False.
    """
    
    
    ROOT_DIR = dirname(dirname(abspath(__file__)))
    sys.path.append(ROOT_DIR)
    _,OUTPUT_DIR,_ = get_dirs(ROOT_DIR, dataset_id)
    
    if not(exists(OUTPUT_DIR)):
        makedirs(OUTPUT_DIR)
    
    
    
    
    log_filename = join(OUTPUT_DIR, "process_chain_build.log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",)
    
    
    
    print("Assembling raw process dataframe...")
    assemble_raw(
        dataset_id=dataset_id, 
        grouping_method=grouping_method, 
        grouping_column=grouping_column,
        debug=debug)
    print("Done!")

    print("Processing raw dataframe...")
    process_raw(dataset_id=dataset_id, missing_threshold=missing_threshold )
    print("Done!")

    print("Generate dataset...")
    generate_dataset(dataset_id=dataset_id)
    print("Done!")
    
    
    print("Exporting selected indices...")
    if select_test:
        get_idx_from_id(dataset_id=dataset_id,
                        id_sel_filename="selected_id.npy", 
                        idx_sel_filename="test_ds_idx.npy")
    print("Done!")



if __name__ == "__main__":
    main(
        dataset_id = "dx_250806_panel_200_pad",
        missing_threshold=30,
        select_test = True,
        grouping_method="panel", # column or panel
        grouping_column=None, # if column, specify which one, available [`batch`]
        debug=False)