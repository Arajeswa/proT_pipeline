from assemble_raw import assemble_raw
from process_raw import process_raw
from match_target import match_target
from generate_dataset import generate_dataset




def main(dataset_id: str, debug: bool = False):
    """_summary_

    Args:
        dataset_id (str): _description_
        debug (bool, optional): _description_. Defaults to False.
    """
    
    print("Assembling raw process dataframe...")
    assemble_raw(dataset_id=dataset_id, debug=debug)
    print("Done!")

    print("Processing raw dataframe...")
    process_raw(dataset_id=dataset_id)
    print("Done!")

    print("Matching the target data...")
    match_target(dataset_id=dataset_id)
    print("Done!")

    print("Generate dataset...")
    generate_dataset(dataset_id)
    print("Done!")




if __name__ == "__main__":
    main(
        dataset_id = "dyconex_252901",
        debug = False)