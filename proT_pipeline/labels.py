
from os.path import join, exists, dirname, abspath
from os import makedirs
from omegaconf import OmegaConf
import sys

# DIRECTORIES

def get_root_dir():
    """
    Get the root directory of the project.
    
    Returns:
        str: Absolute path to the project root directory
    """
    return dirname(dirname(abspath(__file__)))


def get_dirs(root: str, dataset_id: str):
    
    DATA_DIR = join(root,"data")
    BASE_DIR = join(DATA_DIR,"builds",dataset_id)
    assert exists(BASE_DIR), AssertionError(f"Builds directory \"{dataset_id}\" doesn't exist")
    
    OUTPUT_DIR = join(BASE_DIR,"output")
    CONTROL_DIR = join(BASE_DIR,"control")
    
    config = None
    try:
        config = OmegaConf.load(join(CONTROL_DIR,"config.yaml"))
    except:
        print("Configuration file not found")
    
    if config is not None:
        INPUT_DIR = join(DATA_DIR, "input", config["dataset"])
    else:
        INPUT_DIR = join(DATA_DIR,"input")
        
    return INPUT_DIR, OUTPUT_DIR, CONTROL_DIR

# TARGET
target_filename = "df_trg.csv"
target_trimmed_filename = "y_trimmed.csv"
target_sep = ","
target_design_label = "SapNummer"
target_version_label = "Version"
target_design_version_label = "SAP_Version"
target_batch_label = "WA"
target_id_label = "id"
target_value_label = "Value"
target_time_label = "CreateDate"
target_pos_label = "Zyklus"


# INPUT
input_design_label = "SAP"
input_version_label = "Version"
input_variable_label = "Variable"
input_process_label = "Process"
input_batch_label = "batch"
input_step_label = "PaPos"
input_id_label = "id"
input_abs_pos_label = "AbsPos"
input_given_label = "Given"
input_value_label = "Value"
input_time_label = "Time"

# OCCURRENCE FILE LABELS
occurrence_position_label = "Pos"
occurrence_layer_label = "Lage"


# CONTROL FILES
selected_filename = "lookup_selected.xlsx"
selected_process_filename = "steps_selected.xlsx"
lookup_filename = "lookup.xlsx"
input_raw_filename = "x_prochain.csv"
input_leveled_filename = "x_prochain_lev.csv"
booking_missing_filename = "booking_missing.csv"
process_missing_filename = "process_missing.csv"
booking_design_label = "SAP"
booking_version_label = "SAP_Version"
booking_batch_label = "WA"
booking_step_label = "PaPosNumber"

# TEMPLATES
templates_filename = "templates.csv"
templates_design_label = input_design_label
templates_version_label = input_version_label
templates_step_label = input_step_label
templates_variable_label = input_variable_label

# GENERAL
standard_sep = ","

# TRANSVERSAL LABELS
trans_parameter_label = "parameter"
trans_value_label = "value"
trans_value_norm_label = "value_norm"
trans_position_label = "position"
trans_date_label = "date"
trans_batch_label = "batch"
trans_process_label = "process"
trans_variable_label = "variable"
trans_order_label = "order"
trans_id_label = "id"
trans_group_id = "group"
trans_design_label = "design"
trans_version_label = "version"
trans_design_version_label = "design_version"
trans_occurrence_label = "occurrence"
trans_step_label = "step"
time_components_labels = ["year","month","day","hour","minute"]

# TRANSVERSAL FILES
trans_missing_batches = "missing_batches.json"
var_dict_filename = "variables_vocabulary.json"
process_dict_filename = "process_vocabulary.json"
pos_dict_filename = "position_vocabulary.json"
group_dict_filename = "group_vocabulary.json"
step_dict_filename = "step_vocabulary.json"
batch_dict_filename = "batch_vocabulary.json"
feat_dict_filename = "features_dict"
trans_df_process_raw = "df_process_raw.csv"
trans_df_input_short = "df_input_short.parquet"
trans_df_input = "df_input.parquet"



# OUTPUT DATASET FILES
input_ds_label = "X.npy"
trg_ds_label = "Y.npy"
# Compressed dataset files (npz format)
full_data_label = "data.npz"
train_data_label = "train_data.npz"
test_data_label = "test_data.npz"
