'''
objective: utility functions for ADEPT package
'''
import numpy as np
import json
import pandas as pd
import itertools
from pathlib import Path

def xlsx_to_json(uploaded_file: str) -> tuple:
    """
    returns 4D data block in json format from input xls file
    
    arguments:
        uploaded_file [str]: name of file (including file extension)
    return:
        fluid [str]: fluid name
        dict [json]: json file
    """
    
    # remove file prefixes/suffixes
    fluid = uploaded_file.name.replace('.xlsx', '')
    fluid = fluid.replace('lookup_', '')
    
    # create excel object and create dataFrame
    xls = pd.ExcelFile(uploaded_file)
    df_master = pd.read_excel(xls, sheet_name="master")
    df_lookup = pd.read_excel(xls, sheet_name=None, header=None)

    # get columns headings from df_master
    df_cols = df_master.columns.values.tolist()

    # extract master properties
    # write each col of df_master to a list
    df_str = [df_master[col].tolist() for col in df_cols]

    # extract CSV into lists
    df_list = [col[0].split(", ") for col in df_str]

    # convert coordinate vals from string to float
    idx_coord = df_cols.index("coord_label")    #idx of coord_label
    str_coord = df_list[idx_coord]
    nCoord = len(str_coord)
    nX = [None] * nCoord
    for i,col in enumerate(str_coord):
        idx = df_cols.index(col)
        df_list[idx] = [float(item) for item in df_list[idx]]
        nX[i] = len(df_list[idx])

    # count num props
    idx_prop = df_cols.index("prop_label")  #idx of prop_label
    nProp = len(df_list[idx_prop])

    # preallocate `prop_table` (4D data block containing all properties as a function of the coordinates)
    prop_shape = tuple([nProp] + nX)
    prop_table = np.zeros(prop_shape, dtype=float)

    # loop through sheets ('TPF' only) and generate slice corresponding to each sheet
    i = 0
    for (sht_name, sht_df) in df_lookup.items():
        if "TPF" in sht_name:
            df = pd.read_excel(xls, sheet_name=sht_name, header=None)
            prop_TP = df.to_numpy()
            #prop_TP = sht_df.to_numpy()
            
            # each iteration is a slice of coord[1], coord[2] at current coord[0]
            for iX1, iX2 in itertools.product(range(nX[1]), range(nX[2])):
                # split each string and convert to a list of floats
                prop_str = prop_TP[iX1][iX2]
                prop_val = prop_str.split(", ")
                prop_val = [float(item) for item in prop_val]
                
                # populate prop_table
                prop_table[:,i,iX1,iX2] = prop_val

            # next coord[0] slice     
            i += 1

    # close xls file
    xls.close
    
    # put prop_table in df_list
    idx_prop = df_cols.index("prop_table")  #idx of prop_table
    df_list[idx_prop] = prop_table.tolist()
    prop_dict = {'fluid': fluid}

    for i in range(len(df_cols)):
        prop_dict[df_cols[i]] = df_list[i]

    # create json object from dict
    return fluid, json.dumps(prop_dict)

def json_to_stdUnits(LUT: dict or json or Path or str) -> dict:
    return get_dict_from_input(LUT)

def get_dict_from_input(LUT: dict or json or Path or str)->dict:
    '''convert input (dict, json, Path, str) -> dict'''
    if isinstance(LUT, dict):
        return LUT        
    elif isinstance(LUT, str):
        if not Path(LUT).is_file():
            return json.loads(LUT)
        with open(LUT, 'r') as json_file:
            json_string = json_file.read()
            return json.loads(json_string)
    else:
        with open(LUT, 'r') as json_file:
            return json.load(json_file)
        
# TEST
file_name = r'C:\Users\cjsis\Documents\Github\ENNOVA\ADEPT\data\Fluid-LUT\PC003-PR78.json'
data = json_to_stdUnits(file_name)

print(type(data))