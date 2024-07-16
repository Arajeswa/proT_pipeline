import pandas as pd

class Process():
    def __init__(
        self,
        process_label: str,
        hidden_label : str,
        machine_label: str,
        WA_label: str,
        PaPos_label:str,
        date_label:list,
        date_format : str,
        prefix : str,
        filename:str,
        sep: str,
        header: int
        ):
        self.process_label = process_label
        self.hidden_label = hidden_label
        self.machine_label = machine_label
        self.WA_label = WA_label
        self.PaPos_label = PaPos_label
        self.date_label = date_label
        self.date_format = date_format
        self.prefix = prefix
        self.filename = filename
        self.sep = sep
        self.header = header
        self.flag = 0
        
    def get_df(self,input_data_path)->None:
        self.df = pd.read_csv(input_data_path+self.filename, sep=self.sep,header=self.header,low_memory=False)
        self.flag = 1
        
    def normalize_df(self,filename_sel):
        if self.flag == 0:
            raise ValueError("First call get_df to initialize the daraframe!")
        
        self.df_lookup = pd.read_excel(filename_sel,sheet_name=self.process_label)
        self.parameters = self.df_lookup[self.df_lookup["Select"]]["index"]

        for p in self.parameters:
            try:
                self.df[p] = (self.df[p]-self.df[p].min())/(self.df[p].max()-self.df[p].min()+1E-6)
            except Exception as e:
                print(f"Error occurred {e}")
            
    def convert_timestamp(self):
        if self.flag == 0:
            raise ValueError("First call get_df to initialize the daraframe!")
        
        for d in self.date_label:
            self.df[d] = pd.to_datetime(self.df[d],format=self.date_format)
            
    def get_variables_list(self, filename_sel)->None:
        self.df_lookup = pd.read_excel(filename_sel,sheet_name=self.process_label)
        self.variables_list = self.df_lookup[self.df_lookup["Select"]]["variable"].tolist()
            
            