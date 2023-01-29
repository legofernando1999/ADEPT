'''
objective: driver file for calling ADEPT
author: @cjsisco
'''

# import
from ADEPT import ADEPT_Run1
import json
import pandas as pd

# define inputs
t = 8640000 #s
mFlow = 10.25838 #kg/s
L = 5438.586 #m
R = 0.059309 #m
T_prof = (357.0388889, 323.15)  #K
P_prof = (704.692178, 260.4629664)  #bar
GOR = 584.5186  #scf/stb
json_FLUT = 'data/Fluid-LUT/S14-SAFT-MILA2020.json'
json_KLUT = 'data/Kinetics-LUT/ME15-PR76-MILA2020.json'

# get json files
with open(json_FLUT, 'r') as file:
    FLUT_dict = json.load(file)
with open(json_KLUT, 'r') as file:
    KLUT_dict = json.load(file)

# run calculation
var = ADEPT_Run1(t, mFlow, L, R, T_prof, P_prof, GOR, FLUT_dict, KLUT_dict)
print(var)