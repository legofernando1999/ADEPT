'''
objective: driver file for calling ADEPT
author: @cjsisco
'''

# import
from ADEPT import ADEPT_Run1
import json
import pandas as pd
from config import ROOT_DIR, DATA_DIR

# define inputs
t = 8640000 #s
mFlow = 10.25838 #kg/s
L = 5438.586 #m
R = 0.059309 #m
T_prof = (357.0388889, 323.15)  #K
P_prof = (704.692178, 260.4629664)  #bar
GOR = 584.5186  #scf/stb
json_FLUT = DATA_DIR / 'Fluid-LUT' / 'S14-SAFT-MILA2020.json'
json_KLUT = DATA_DIR / 'Kinetics-LUT' / 'ME15-PR76-MILA2020.json'

# read json files
with open(json_FLUT, 'r') as file:
    FLUT_dict = json.load(file)
with open(json_KLUT, 'r') as file:
    KLUT_dict = json.load(file)

# write inputs to dict
keys = ['t', 'mFlow', 'L', 'R', 'T_prof', 'P_prof', 'GOR', 'FLUT', 'KLUT']
vals = [t, mFlow, L, R, T_prof, P_prof, GOR, json_FLUT, json_KLUT]
data = dict.fromkeys(keys, vals)

# run calculation
var = ADEPT_Run1(t, mFlow, L, R, T_prof, P_prof, GOR, FLUT_dict, KLUT_dict)
print(var)

# NOTE: we should have one method that takes only the `data` dict as input 
# so that if we want to run a series of simulations, then we can populate a 
# tuple of dicts (or jsons) and loop through them to perform all simulations.

#sims = ('p1.json', 'p2.json', 'p3.json')