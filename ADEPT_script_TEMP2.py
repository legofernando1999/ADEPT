# import packages
import numpy as np
from ADEPT_python_TEMP2 import pipe, sim, depo
import json

# create objects
mypipe = pipe(0.3048, 0.000508*0.5)
mysim = sim(227520., 1.1111e-9, True)
mix_phase = ["volume", "volume", "sum"]
KLUT_dict = {
    "kP": 1.45e-3,
    'kAg': 5.07e-3,
    'kD': 1.31e-2,
    'kDiss': 0.
}

mydepo = depo(mypipe, mysim, KLUT_dict, mix_phase)

# run ADEPT
Ceq = 0.7
output = mydepo.ADEPT_Solver(Ceq)
output_dict = vars(output)
with open("output_json.json", "w") as json_file:
    json.dump(output_dict, json_file)