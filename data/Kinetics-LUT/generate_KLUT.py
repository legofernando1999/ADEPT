'''
objective: generator file for creating KLUT json file
'''

# import packages
import json
from pathlib import Path
from config import ROOT_DIR, DATA_DIR

# ADEPT kinetic parameters
kP_param = (0.360827, 41.12422)
kP_model = 't-sp'
kAg_param = (0.001871)
kAg_model = 'default'
kD_param = (0.03)
kD_scale = 'wb'
SR_param = (5, 0.6, 0.16)
SR_model = 'default'

# create dict
data = {'kP_param': kP_param, 'kP_model': kP_model, 'kAg_param': kAg_param, 'kAg_model': kAg_model,
          'kD_param': kD_param, 'kD_scale': kD_scale, 'SR_param': SR_param, 'SR_model': SR_model}

# save to json
path = Path(__file__).parent.resolve()
file_name = path / 'sample.json'
with open(file_name, 'w') as file:
    json.dump(data, file)
