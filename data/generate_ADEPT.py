'''
objective: generator file for creating ADEPT input json file
'''

# import packages
import json
from pathlib import Path
from config import ROOT_DIR, DATA_DIR

# define inputs
t = 8640000 #s
mFlow = 10.25838 #kg/s
L = 5438.586 #m
R = 0.059309 #m
T_prof = (357.0388889, 323.15)  #K
P_prof = (704.692178, 260.4629664)  #bar
GOR = 584.5186  #scf/stb
json_FLUT = DATA_DIR / 'Fluid-LUT' / 'S14-SAFT-MILA2020.json' #'data/Fluid-LUT/S14-SAFT-MILA2020.json'
json_KLUT = 'data/Kinetics-LUT/ME15-PR76-MILA2020.json'

# read json files
with open(json_FLUT, 'r') as file:
    FLUT_dict = json.load(file)
with open(json_KLUT, 'r') as file:
    KLUT_dict = json.load(file)

# write inputs to dict
keys = ['t', 'mFlow', 'L', 'R', 'T_prof', 'P_prof', 'GOR', 'FLUT', 'KLUT']
vals = [t, mFlow, L, R, T_prof, P_prof, GOR, json_FLUT, json_KLUT]
data = dict.fromkeys(keys, vals)
data = dict(t=t, mFlow=mFlow, L=L, R=R, T_prof=T_prof, P_prof=P_prof, GOR=GOR, 
    FLUT=json_FLUT, KLUT=json_KLUT)

# save to json
path = Path(__file__).parent.resolve()
file_name = path / 'p1.json'
with open(file_name, 'w') as file:
    json.dump(data, file)