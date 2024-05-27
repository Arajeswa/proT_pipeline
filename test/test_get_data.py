import pandas as pd

from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

from process_pipeline.core.modules import get_data_step
from process_pipeline.data_loader import get_processes, get_booking
from process_pipeline.utils import nested_dict_from_pandas

format = "%m/%d/%y %I:%M %p"
input_data_path = "process_pipeline/data/input/"
intermediate_data_path = "process_pipeline/data/intermediate/"
output_data_path = "process_pipeline/data/output/"

filename_sel  = intermediate_data_path +"lookup_selected.xlsx"
filename_look = intermediate_data_path +"lookup.xlsx"



#load booking
df_book = get_booking(input_data_path)

#load processes
_, processes = get_processes(input_data_path,filename_sel)



# read Y (IST) file
df_ist = pd.read_csv(intermediate_data_path + "y_ist.csv", sep=",")
df_ist = df_ist.iloc[:100000]

# create a nested dictionary from Y (IST) queries
d = nested_dict_from_pandas(df_ist.set_index(["SapNummer","Version","WA","id"]))
wa = df_book.loc[0]["WA"]
step = df_book.loc[0]["PaPosNumber"]

print(f"Testing batch {wa}, step {step}...")

df,_ = get_data_step(wa,step,processes,filename_sel)

if type(df)==pd.DataFrame:
    print("The dataframe was successfully assembled!")