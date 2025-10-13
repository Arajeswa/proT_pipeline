import pandas as pd

from os import getcwd
from os.path import dirname
import sys
ROOT_DIR = dirname(getcwd())
sys.path.append(ROOT_DIR)

from proT_pipeline.data_trimmer import data_trimmer

intermediate_data_path = "process_pipeline/data/intermediate/"

df_pc = pd.read_csv(ROOT_DIR + "/data/test/x_prochain.csv")
df_ist = pd.read_csv(ROOT_DIR + "/data/test/y_ist.csv")
df_book_mis = pd.read_csv(ROOT_DIR + "/data/test/booking_missing.csv")

y_trimmed = data_trimmer(df_x=df_pc, df_y=df_ist, df_miss=df_book_mis, save_path = intermediate_data_path)

print("All tests passed!")
