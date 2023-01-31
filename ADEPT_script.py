# import packages
import numpy as np
from ADEPT_python import pipe, sim, depo
import json

# unit conversion
def convert_TF_to_TK(TF):
    return (5/9)*(TF-32)+273.15

def convert_Ppsi_to_Pbar(Ppsi):
    return Ppsi * 0.0689475729

# create objects
mypipe = pipe(5438.59, 0.05931)
mysim = sim(8640000., 10.2584, True)
mix_phase = ["volume", "volume", "sum"]
with open(r'C:\Users\Satellite\Documents\ADEPT\data\Fluid-LUT\PC003-PR78.json') as json_file:
    TLUT_json = json.load(json_file)
KLUT_dict = {
    "kP_param": [0.360827, 41.12422],
    "kP_model": "t-sp",
    "kAg_param": 0.001871,
    "kAg_model": "default",
    "kD": 0.03,
    "kD_scale": "wb",
    "SR_param": [5, 0.6, 0.16],
    "SR_model": "default"
}
mydepo = depo(mypipe, mysim, TLUT_json, KLUT_dict, mix_phase)

# convert T, P profile
T_prof = np.array((183., 122.), dtype=float)
T_prof = (5/9)*(T_prof-32)+273.15
P_prof = np.array((10206., 3763.))*0.0689475729

# run ADEPT
output = mydepo.ADEPT_Solver(T_prof, P_prof, 0.)