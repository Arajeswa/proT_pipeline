import pandas as pd
from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

from process_pipeline.data_loader import get_processes,get_booking
from process_pipeline.sequence_builder import sequence_builder


input_data_path = "process_pipeline/data/input/"
intermediate_data_path = "process_pipeline/data/intermediate/"
output_data_path = "process_pipeline/data/output/"

filename_sel  = intermediate_data_path +"lookup_selected.xlsx"
filename_look = intermediate_data_path +"lookup.xlsx"


# load processes
_, processes = get_processes(input_data_path,filename_sel)

# read the keys file (Y target: IST)
df_ist = pd.read_csv(intermediate_data_path + "y_ist.csv", sep=",")
df_ist = df_ist.iloc[:100000]

# get the booking file
df_book = get_booking(input_data_path)

_,_,_ = sequence_builder(df_query=df_book.copy(), df_keys=df_ist.copy(), keys_branches=["SapNummer","Version","WA","id"],
                        processes=processes, saving_path = intermediate_data_path, filename_sel=filename_sel)

print("All test passed!")