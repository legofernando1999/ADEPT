# import packages
import numpy as np
from ADEPT_python import pipe, sim, depo
import json
from itertools import product

L = 22000       #ft
ID = 4.5        #in
R = 0.5*4.5     #in  

mypipe = pipe(L*0.3048, R*0.0254)

mix_phase = ["volume", "volume", "sum"]

with open(r'"C:\Users\Fernando\Downloads\MP-562_SI.json"') as json_file:
    FLUT_json = json.load(json_file)

# T, P profile (Bottomhole, Wellhead)
T_prof = np.array((0, 0))
P_prof = np.array((0, 0))

volFlow_tuple = (4000., 6000., 9000., 12000.)     #STB/day
kD_tuple = (0.1, 1., 10.)*0.00213   #1/s
t_sim_tuple = (30, 60, 90)*86400    #seconds

i = 0
for vFlow, kD_us,  in product(volFlow_tuple, kD_tuple):
    i += 1
    KLUT_dict = {
        "kP_param": [0.769222508, 100, 0., 0.],
        "kP_model": "t-sp",
        "kAg_param": [0.004429599, 0, 0, 0],
        "kAg_model": "default",
        "kD": kD_us,
        "kD_scale": "cap",
        "SR_param": [5., 0., 0.16],
        "SR_model": "default"
    }

    mysim = sim(8640000., vFlow, True)

    mydepo = depo(mypipe, mysim, FLUT_json, KLUT_dict, mix_phase)

    # run ADEPT
    output = mydepo.ADEPT_Solver(T_prof, P_prof, 0.)
    output_dict = vars(output)
    with open(f"output_{i}_.json", "w") as json_file:
        json.dump(output_dict, json_file)