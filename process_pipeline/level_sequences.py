import pandas as pd
import numpy as np
from os.path import join,dirname
from typing import List

import sys
ROOT_DIR = dirname(dirname(__file__))
sys.path.append(ROOT_DIR)
from process_pipeline.utils import nested_dict_from_pandas
from core.modules import Process

# def level_sequences_OLD(
#     df: pd.DataFrame,
#     design_label: str = "SAP",
#     version_label: str = "Version",
#     variable_label: str = "Variable",
#     process_label: str = "Process",
#     batch_label: str = "WA",
#     step_label: str = "PaPos",
#     id_label: str = "id",
#     abs_pos_label: str = "AbsPos",
#     given_label: str = "Given",
#     value_label: str = "Value",
#     ):
    
#     # create a nested dictionary from Y (IST) queries
#     multi_idx = [design_label,version_label,batch_label,id_label]
#     d = nested_dict_from_pandas(df.set_index(multi_idx))

#     # get the reference batches
#     ref_batches = []
#     len_list = []

#     for design in d.keys():

#         for version in d[design].keys():
#             max_len = 0
#             step_list = []

#             for batch in d[design][version].keys():

#                 for id in d[design][version][batch]:
#                     steps = df.set_index(multi_idx).loc[design,version,batch][step_label].unique()
#                     for step in steps:
#                         if step not in step_list:
#                             step_list.append(step)
                            
#                     # l = len(df.set_index(multi_idx).loc[design,version,batch][step_label].unique())
                    
#                     # if l >= max_len:
#                     #     max_len = l
#                     #     ref_batch = batch
#                     #     ref_id = id

#             ref_batches.append((design,version,ref_batch,step_list))
#             len_list.append((design,version,ref_batch,step_list))

#     # assemble the hash DataFrame
#     df_hash = None
#     df_template = None
#     template_cols = [design_label,version_label,step_label,abs_pos_label,variable_label,process_label]
#     h_multi_idx = [design_label,version_label,step_label]

#     for ref_batch in ref_batches:
#         df_hash_temp = df.set_index(multi_idx).loc[ref_batch][step_label].sort_values()
#         df_hash_temp = df_hash_temp.drop_duplicates().reset_index()
#         df_hash_temp["AbsPos"] = np.arange(len(df_hash_temp))
#         df_template_temp = df.set_index(multi_idx).loc[ref_batch].sort_values(by=[step_label,variable_label])
#         df_template_temp = df_template_temp.drop_duplicates(subset=[step_label,variable_label]).reset_index()

#         if df_hash is not None:
#             df_hash = pd.concat([df_hash,df_hash_temp],ignore_index=True)

#         if df_template is not None:
#             df_template = pd.concat([df_template,df_template_temp],ignore_index=True)

#         if df_hash is None:
#             df_hash = df_hash_temp

#         if df_template is None:
#             df_template = df_template_temp
            

#     print("Template assembled")
    
#     # get absolute position in the templates
#     abs_pos_list = []

#     for ix in df_template.index:
#             coordinate = tuple(df_template.iloc[ix][h_multi_idx])
#             abs_pos = df_hash.set_index(h_multi_idx).loc[coordinate][abs_pos_label]
#             abs_pos_list.append(abs_pos)

#     df_template[abs_pos_label] = abs_pos_list
#     df_template = df_template[template_cols]


#     # level all sequences
#     df_lev = None
#     non_unique_index_list = [] 

#     for design in d.keys():

#         for version in d[design].keys():
#             template = df_template.set_index([design_label,version_label,step_label,variable_label]).loc[design,version]


#             for batch in d[design][version].keys():

#                 for id in d[design][version][batch]:
#                     # _df = df.set_index([design_label,version_label,batch_label,id_label,step_label,variable_label]).sort_index().loc[design,version,batch,id]
#                     _df = df.set_index([design_label,version_label,batch_label,id_label]).loc[design,version,batch,id]
#                     _df = _df.drop_duplicates(subset=[step_label,variable_label])
#                     _df = _df.set_index([step_label,variable_label]).sort_index()
#                     _df = _df.drop([process_label], axis=1)
                    
#                     if _df.index.is_unique:
#                         df_lev_temp = pd.concat([template,_df],axis=1)
#                         # place back information
#                         df_lev_temp = df_lev_temp.reset_index()
#                         df_lev_temp[design_label] = design
#                         df_lev_temp[version_label] = version
#                         df_lev_temp[batch_label] = batch
#                         df_lev_temp[id_label] = id
                        
#                         if df_lev is None:
#                             df_lev = df_lev_temp
#                         else:
#                             df_lev = pd.concat([df_lev,df_lev_temp],ignore_index=True)
#                     else:
#                         non_unique_index_list.append(id)
                        

#     df_lev[given_label] = df_lev[value_label].notna().astype(int)
    
#     return df_lev,len_list



def get_template(
    df: pd.DataFrame,
    processes = List[Process],
    design_label: str = "SAP",
    version_label: str = "Version",
    variable_label: str = "Variable",
    process_label: str = "Process",
    batch_label: str = "WA",
    step_label: str = "PaPos",
    id_label: str = "id",
    abs_pos_label: str = "AbsPos",
    given_label: str = "Given",
    value_label: str = "Value",
    ):
    
    # create a nested dictionary from Y (IST) queries
    multi_idx = [design_label,version_label,batch_label,id_label,step_label]
    d = nested_dict_from_pandas(df.set_index(multi_idx))

    # get the reference batches
    outer_dict = {}
    #middle_dict = {}
    #inner_dict = {}


    for design in d.keys():
        middle_dict = {}

        for version in d[design].keys():
            #max_len = 0
            #step_list = []
            inner_dict = {}

            # scan all possible batch/id
            #----------------------------------------------------------------------------------------------------------------------
            for batch in d[design][version].keys():

                for id in d[design][version][batch].keys():
                    # steps = df.set_index(multi_idx).loc[design,version,batch,id][step_label].unique()
                    
                    for step in d[design][version][batch][id]:
                        if step not in inner_dict.keys():
                            process = df.set_index(multi_idx).loc[design,version,batch,id,step][process_label].value_counts().index[0]
                            
                            var_list = [pro.variables_list for pro in processes if pro.process_label == process][0]
                            
                            #var_list = df.set_index(multi_idx).loc[design,version,batch,id,step][variable_label].unique().tolist()
                            #step_dic = {step:var_list}
                            #step_list.append(step_dic)
                            #inner_dict[step] = var_list
                            inner_dict[step] = {process_label:process,variable_label:var_list,abs_pos_label:0}
            #----------------------------------------------------------------------------------------------------------------------
            
            #abs_pos = np.arange(len(inner_dict.keys()))
            for i, key in enumerate(inner_dict.keys()):
                var_dict = dict()
                for var in inner_dict[key][variable_label]:
                    var_dict[var] = {
                        abs_pos_label:i,
                        process_label:inner_dict[key][process_label]}
                inner_dict[key] = var_dict
                
            middle_dict[version] = inner_dict
        
        outer_dict[design] = middle_dict
    
    return outer_dict


# def get_level(
#     df: pd.DataFrame,
#     design_label: str = "SAP",
#     version_label: str = "Version",
#     variable_label: str = "Variable",
#     process_label: str = "Process",
#     batch_label: str = "WA",
#     step_label: str = "PaPos",
#     id_label: str = "id",
#     abs_pos_label: str = "AbsPos",
#     given_label: str = "Given",
#     value_label: str = "Value",
#     ):    
    
#     # get absolute position in the templates
#     abs_pos_list = []

#     for ix in df_template.index:
#             coordinate = tuple(df_template.iloc[ix][h_multi_idx])
#             abs_pos = df_hash.set_index(h_multi_idx).loc[coordinate][abs_pos_label]
#             abs_pos_list.append(abs_pos)

#     df_template[abs_pos_label] = abs_pos_list
#     df_template = df_template[template_cols]


#     # level all sequences
#     df_lev = None
#     non_unique_index_list = [] 

#     for design in d.keys():

#         for version in d[design].keys():
#             template = df_template.set_index([design_label,version_label,step_label,variable_label]).loc[design,version]


#             for batch in d[design][version].keys():

#                 for id in d[design][version][batch]:
#                     _df = df.set_index([design_label,version_label,batch_label,id_label,step_label,variable_label]).sort_index().loc[design,version,batch,id]
#                     _df = _df.drop([process_label], axis=1)
                    
#                     if _df.index.is_unique:
#                         df_lev_temp = pd.concat([template,_df],axis=1)
#                         # place back information
#                         df_lev_temp = df_lev_temp.reset_index()
#                         df_lev_temp[design_label] = design
#                         df_lev_temp[version_label] = version
#                         df_lev_temp[batch_label] = batch
#                         df_lev_temp[id_label] = id
                        
#                         if df_lev is None:
#                             df_lev = df_lev_temp
#                         else:
#                             df_lev = pd.concat([df_lev,df_lev_temp],ignore_index=True)
#                     else:
#                         non_unique_index_list.append(id)
                        

#     df_lev[given_label] = df_lev[value_label].notna().astype(int)
    
#     return df_lev,non_unique_index_list


def level_sequences(
    df: pd.DataFrame,
    processes: List[Process],
    design_label: str = "SAP",
    version_label: str = "Version",
    variable_label: str = "Variable",
    process_label: str = "Process",
    batch_label: str = "WA",
    step_label: str = "PaPos",
    id_label: str = "id",
    abs_pos_label: str = "AbsPos",
    given_label: str = "Given",
    value_label: str = "Value",
    ):
    
    # create a nested dictionary from Y (IST) queries
    multi_idx = [design_label,version_label,batch_label,id_label]
    d = nested_dict_from_pandas(df.set_index(multi_idx))

    
    templates = get_template(df=df,processes=processes)
    print("Template assembled")
    
    # get absolute position in the templates
    df_lev = None
    max_seq_len = 0
    
    for design in d.keys():

        for version in d[design].keys():
            sel_template = templates[design][version]

            df_template = pd.DataFrame.from_dict({(i,j): sel_template[i][j] 
                                       for i in sel_template.keys() 
                                       for j in sel_template[i].keys()},orient="index")
            
            if len(df_template)>max_seq_len:
                max_seq_len = len(df_template)
                
            for batch in d[design][version].keys():

                for id in d[design][version][batch]:
                    # _df = df.set_index([design_label,version_label,batch_label,id_label,step_label,variable_label]).sort_index().loc[design,version,batch,id]
                    _df = df.set_index([design_label,version_label,batch_label,id_label]).loc[design,version,batch,id].reset_index()
                    _df = _df.drop_duplicates(subset=[step_label,variable_label])
                    _df = _df.set_index([step_label,variable_label])
                    df_lev_temp = pd.concat([df_template,_df],axis=1)
                    df_lev_temp = df_lev_temp.reset_index().rename(columns={'level_0':step_label, "level_1":variable_label})
                    
                    df_lev_temp[design_label] = design
                    df_lev_temp[version_label] = version
                    df_lev_temp[batch_label] = batch
                    df_lev_temp[id_label] = id
                    
                    if df_lev is None:
                        df_lev = df_lev_temp
                    else:
                        df_lev = pd.concat([df_lev,df_lev_temp],ignore_index=True)
                
                        

    df_lev[given_label] = df_lev[value_label].notna().astype(int)
    
    return df_lev,max_seq_len