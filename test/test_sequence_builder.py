import pandas as pd

from os import getcwd
from os.path import dirname, join
import sys
ROOT_DIR = dirname(getcwd())
sys.path.append(ROOT_DIR)

from process_pipeline.data_loader import get_processes,get_booking
from process_pipeline._old.sequence_builder import sequence_builder
from process_pipeline.labels import *

input_data_path = join(ROOT_DIR,"data/input/")
intermediate_data_path = join(ROOT_DIR,"data/intermediate/")
output_data_path = join(ROOT_DIR,"data/output/")

filepath_sel  = intermediate_data_path + selected_filename
filepath_look = intermediate_data_path + lookup_filename


# load processes
_, processes = get_processes(input_data_path,filepath_sel)

# read the keys file (Y target: IST)
df_ist = pd.read_csv(intermediate_data_path + "y_ist.csv", sep=",")
df_ist = df_ist.iloc[:100000]

# get the booking file
df_book = get_booking(input_data_path)

df_pc, df_book_mis,df_pro_mis = sequence_builder(df_key=df_book.copy(), df_query=df_ist.copy(), query_branches=["SapNummer","Version","WA","id"],
                        processes=processes, saving_path = intermediate_data_path, selected_filename=filepath_sel)

print("All test passed!")