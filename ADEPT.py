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

# TODO: write the functions below that represent the API that the user will interact with.
# we should be able to use common names and then polymorphism will work to find the correct entry-point to the `ADEPT_python.py` file

def ADEPT_Run(mFlow, T_tuple, P_tuple, GOR_tuple, FLUT: np.array):
    pass

def ADEPT_Run(json_ADEPT: json):
    pass

# TODO: `sim`, `pipe`, and `fluid` class initialization should probably happen in the API or a separate file called by the API.