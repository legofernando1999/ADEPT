# import packages
import numpy as np
from ADEPT_python import pipe, sim, depo
import json

# create objects
mypipe = pipe(5438.58579701311, 5.93091186182372e-2)
mysim = sim(8640000., 10.2583786168161, True)
mix_phase = ["volume", "volume", "sum"]
with open(r'C:\Users\Satellite\Documents\ADEPT\data\Fluid-LUT\PC003-PR78.json') as json_file:
    TLUT_json = json.load(json_file)
KLUT_dict = {
    "kP_param": [0.360826555152274, 41.1242235495591, 0., 0.],
    "kP_model": "t-sp",
    "kAg_param": [1.87085201371982e-03, 0, 0, 0],
    "kAg_model": "default",
    "kD": 0.03,
    "kD_scale": "wb",
    "SR_param": [5, 0.6, 0.16],
    "SR_model": "default"
}
mydepo = depo(mypipe, mysim, TLUT_json, KLUT_dict, mix_phase)

# convert T, P profile
T_prof = np.array((357.0388888888889, 323.15))
P_prof = np.array((10206., 3763.))*0.0689475729

# run ADEPT
with open(r'C:\Users\Satellite\Downloads\temp_json.json') as json_file:
    VBA_json = json.load(json_file)
output = mydepo.ADEPT_Solver(T_prof, P_prof, 0., VBA_json)
output_dict = vars(output)
with open("output_json.json", "w") as json_file:
    json.dump(output_dict, json_file)