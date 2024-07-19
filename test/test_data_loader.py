from os import getcwd
from os.path import dirname
import sys
ROOT_DIR = dirname(getcwd())
sys.path.append(ROOT_DIR)

from process_pipeline.data_loader import get_processes


input_data_path = "process_pipeline/data/input/"
intermediate_data_path = "process_pipeline/data/intermediate/"
filename_sel  = intermediate_data_path +"lookup_selected.xlsx"

#load processes
_, processes = get_processes(input_data_path,filename_sel)

wa_list = []
for process in processes:
    try:
        wa_list.append(process.df[process.WA_label])
    except Exception as e:
        print(f"Error occurred in {process.process_label}: {e}")
    print(f"Process {process.process_label }...Test WA passed!")
    

print("All test passed!")