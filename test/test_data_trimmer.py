import pandas as pd
from os.path import dirname, abspath
import sys
parent_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_path)
from process_pipeline.data_trimmer import data_trimmer

intermediate_data_path = "process_pipeline/data/intermediate/"

df_pc = pd.read_csv(parent_path + "/data/test/x_prochain.csv")
df_ist = pd.read_csv(parent_path + "/data/test/y_ist.csv")
df_book_mis = pd.read_csv(parent_path + "/data/test/booking_missing.csv")

y_trimmed = data_trimmer(df_x=df_pc, df_y=df_ist, df_miss=df_book_mis, save_path = intermediate_data_path)

print("All tests passed!")
