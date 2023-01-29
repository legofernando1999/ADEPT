'''
The `ADEPT` module provides a simple API for calculating the time-evolution of asphaltene deposition in a pipe.

Notes: 
 - this implementation of ADEPT considers ...
    - 1D flow in the axial direction (no radial diffusion)
    - single-phase averaged flow (gas and liquid are averaged to calculate fluid density, viscosity, velocity along the axial direction)
    - dynamic P, T profiles that update as the simulation marches forward in time 
 - if the input `Fluid_LUT` includes failed flashes (detected if sum(phase_wtf=0)), then the `Fluid_LUT` will be corrected by interpolation.

References:
 - Naseri et al. (2020); link: http://www.sciencedirect.com/science/article/pii/S0920410520306240
 - Rajan Babu et al. (2019); link: http://pubs.acs.org/doi/10.1021/acs.energyfuels.8b03239
 
Examples:
```Python

    
```

'''

# TODO: validate against ADEPT-VBA (file: `ADEPT(2023-01-17).xlsm`)
# TODO: the construction of the FLUT (the 4D numpy array) should probably occur in the API file or a separate file called by the API. The FLUT should exist before entering `ADEPT_Python.py`.
# TODO: fix flash calculation failures in FLUT. This functionality should be in the same module as where the FLUT is constructed (either from an input json, dict, or numpy array) as a post-processing step to fix failures.
# TODO: Release 2.0:
# add multiphase flow simulator to capture velocity and pressure gradients. There are open source packages we should investigate (OpenFOAM, pyFoam, etc.)


# import packages
import json
import numpy as np
from pathlib import Path

# TODO: write the functions below that represent the API that the user will interact with.
# we should be able to use common names and then polymorphism will work to find the correct entry-point to the `ADEPT_python.py` file

def ADEPT_Run1(t: float, mFlow: float, L: float, R: float, T_tuple: tuple or np.ndarray, P_tuple: tuple or np.ndarray, GOR: float, 
               FLUT: dict or json or Path or np.ndarray, KLUT: dict or json or Path):
    '''
    
    Parameters
    ----------
    t : float
        time (s)
    mFlow : float
        mass flow rate (kg/s)
    L : float
        pipe length (m)
    R : float
        pipe radius (m)
    T_tuple : tuple[float]
        (BHT, WHT) .= must be same unit as LUT
    P_tuple : tuple[float]
        (BHP, WHP) .= must be same unit as LUT
    GOR : float
        gas-oil ratio .= must be same unit as LUT
    FLUT : dict or json or Path or np.ndarray
        fluid lookup table
    KLUT : dict or json or Path
        kinetics lookup table
    
    Return
    ------
    depo_return : dict

    '''
    
    # prepare vars for `ADEPT_Solver`
    T = read_TP(T_tuple)        # T (K)
    P = read_TP(P_tuple)        # P (bar)
    FLUT_np = read_FLUT(FLUT)   # get dict: `fluid` LUT
    KLUT_dict = read_KLUT(KLUT) # get dict: `kinetics` LUT

    #return = ADEPT_Solver(T, P, GOR) 
    return t, mFlow

def ADEPT_Run(input_file: dict or json or Path):
    dict_ = get_dict_from_input(input_file)

# TODO: `sim`, `pipe`, and `fluid` class initialization should probably happen in the API or a helper file called by the API.


# reader functions
def read_TP(x)->np.ndarray:
    '''convert input -> np.ndarray'''
    if isinstance(x, (tuple, list)):
        return np.array(x, dtype=np.float64)
    elif isinstance(x, np.ndarray):
        return x
    
def read_FLUT(LUT)->dict:
    '''convert input -> dict: `Fluid` LUT'''
    return get_dict_from_input(LUT)
    # if isinstance(LUT, np.ndarray):
    #     return LUT
    # dict_ = get_dict_from_input(LUT)
    # return np.array(list(dict_.values()), dtype=dict)
        
def read_KLUT(LUT)->dict:
    '''convert input -> dict: `Kinetics` LUT'''
    return get_dict_from_input(LUT)

def get_dict_from_input(LUT: dict or json or Path or str)->dict:
    '''convert input (dict, json, Path, str) -> dict'''
    if isinstance(LUT, dict):
        return LUT
    elif isinstance(LUT, json):
        with open(LUT, 'r') as json_file:
            return json.load(json_file)
    elif isinstance(LUT, str):
        if not Path(LUT).is_file():
            return json.loads(LUT)
        with open(LUT, 'r') as json_file:
            json_string = json_file.read()
            return json.loads(json_string)
