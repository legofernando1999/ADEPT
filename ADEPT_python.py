# import packages
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import RegularGridInterpolator, griddata
import itertools
from scipy import ndimage as nd

# TODO: add a module level docstring (https://numpydoc.readthedocs.io/en/latest/format.html#documenting-modules) to this file
# FIXME: the whole code!

# define constants
Rg = 8.314          # gas constant (J/mol.K)
kB = 1.3806504e-23  # Boltzmann constant (J/K)
#Nav = 6.022e23      # Avogadro number (1/mol)
Pi = np.pi

# <-- REFERENCES -->
# `NRB 2019a`: Rajan Babu, N. (2019) "Mechanistic Investigation and Modeling of Asphaltene Deposition," Rice University, Houston, Texas, USA, 2019. link: https://www.proquest.com/docview/2568024745?fromopenview=true&pq-origsite=gscholar
# `NRB 2019b`: Rajan Babu, N. et al. (2019). "Systematic Investigation of Asphaltene Deposition in the Wellbore and Near-Wellbore Region of a Deepwater Oil Reservoir under Gas Injection. Part 2: Computational Fluid Dynamics Modeling of Asphaltene Deposition." doi: https://doi.org/10.1021/acs.energyfuels.8b03239.


class pipe(object):
    
    def __init__(self, L: float, R: float, T_ext: float=0, del_Asp: np.ndarray=None, del_DM: np.ndarray=None) -> None:
        '''
        L: length of pipe (m)
        R: radius of pipe (m)
        T_ext: temperature of pipe
        del_Asp0: initial deposit thickness (m), assuming only asphaltenes deposit
        del_DM0: initial deposit thickness (m), assuming aromatics and resins also deposit
        
        NOTE: 
        + del_Asp0 and del_DM0 are arrays with discrete values of deposit thickness corresponding to the array of `z`. 
        In case the array of `z` changes from one simulation to the next (which I don't know why it would), 
        we could fit a polynomial to del_Asp0 and then use this to translate del_Asp0 from one discretization scheme to another. 
        + added variable types for `del_Asp0` and `del_DM0`. if `del_Asp0` is None, then we will initialize an array of zeros.
        '''
        self.L = L
        self.R = R
        self.T_ext = T_ext
        self.del_Asp = del_Asp
        self.del_DM = del_DM


class sim(object):
    
    def __init__(self, t_sim: float, vFlow: float, isTransient: bool=True) -> None:
        '''
        t_sim: simulation time (s)
        mFlow: mass flow rate (kg/s)
        vFlow: volume flow rate (STB/day)
        isTransient: {True=transient, False=steady-state}
        '''
        self.t_sim = t_sim
        self.vFlow = vFlow*0.0000018401307  #(m3/s)
        self.isTransient = isTransient


class fluid(object):

    def __init__(self) -> None:
        pass


class depo_return(object):
    
    def __init__(self, flow_status, t_noFlow, del_Asp, del_DM, J_Asp, C, mass_Asp):
        '''
        object containing:
            flow_status: `flowing`, `shutdown`, `plugged`
            t @ noFlow:
            thickness: del_Asp(t,z), del_DM(t,z)
            flux: J_Asp(t,z)
            conc(t,z): C (primary particle)
            mass_Asp(t,z) 
            pressure drop: dP(t)
            temperature profile: dT(t)
        '''
        self.flow_status = flow_status
        self.t_noFlow = t_noFlow
        self.thickness = [del_Asp.tolist(), del_DM.tolist()]
        self.flux = J_Asp.tolist()
        self.conc = C.tolist()
        self.mass = mass_Asp.tolist()


class depo(object):

    def __init__(self, pipe: pipe, sim: sim, file: json or dict, KLUT: dict, mix_phase: list) -> None:
        '''
        file: file containing thermodynamic information about fluid
        KLUT: Kinetic Lookup Table
            kP_param:   [array] scaling parameters for kP correlation: [a0, a1]
            kP_model:   [str] model for kP correlation: {'T', 'T-SP', 'NRB'}
            kAg_param:  [array] scaling parameters for kAg correlation: [c0]
            kAg_model:  [str] model for kAg correlation: {'IL', 'NRB'}
            kD_param:   [array] scaling parameters for kD correlation: [mFlow, R, rho, mu]
            kD_scale:   [str] scaling correlation to apply to kD0 (unscaled kD) {'wb', 'cap', 'pb'}, where 'wb'=wellbore (no scaling); 'cap'=capillary; 'pb'=packed bed
            SR_param:   [array] scaling parameters for shear removal (SR) correlation: [tau0, k, n], where tau0=critical shear stress (Pa), k=pre-factor, n=exponent
            SR_model:   [str] model for SR correlation: {'default'}    
        mix_phase
            dens: density mixing rule (default="volume")
            visco: viscosity mixing rule (default="volume")
            velocity: velocity mixing rule (default="sum")
        '''
        self.file = file
        self.pipe = pipe
        self.sim = sim
        self.mix_phase = mix_phase

        # instance of fluid object
        fl = fluid()

        # asphaltene rate parameters
        fl.kD_us = KLUT["kD"]
        fl.kP_param = KLUT["kP_param"]
        fl.kP_model = KLUT["kP_model"]
        fl.kAg_param = KLUT["kAg_param"]
        fl.kAg_model = KLUT["kAg_model"]
        fl.kD_scale = KLUT["kD_scale"]
        fl.SR_param = KLUT["SR_param"]
        fl.SR_model = KLUT["SR_model"]

        self.fl = fl

        file_tuple = self.FLUT_loader(file)
        self.FLUT, self.prop_label, self.prop_unit, self.GOR_range, self.P_range, self.T_range = file_tuple

    def FLUT_loader(self, file):
        '''
        FLUT: Thermodynamic Lookup Table, 4D array (Prop, GOR, P, T)
            Ceq: asphaltene solubility (gAsp[L1]/gAsp[T])
            wtFrac[V, L1, L2]: phase frac, mass (g[k]/g)
            volFrac[V, L1, L2]: phase frac, volume (m3[k]/m3)
            dens[V, L1, L2, Asp]: phase density (kg/m3)
            visco[V, L1, L2, Asp]: phase viscosity (Pa.s)
            SP[V, L1, L2, Asp]: solubility parameter (MPa^0.5)
            yAsp[V, L1, L2]: asphaltene composition (gAsp[k] / g[k])
        '''
        prop_dict = json.load(file) if type(file) != dict else file

        FLUT = np.array(prop_dict['prop_table'])
        prop_label = prop_dict['prop_label']
        prop_unit = prop_dict['prop_unit']
        coord_label = prop_dict['coord_label']
        GOR_range = np.array(prop_dict[coord_label[0]])
        P_range = np.array(prop_dict[coord_label[1]])       # not to be confused with P_prof, T_prof
        T_range = np.array(prop_dict[coord_label[2]])
        crange = (GOR_range, P_range, T_range)

        # preprocessing 
        # FLUT = self.preproc(FLUT, crange, prop_label)

        # unit conversion
        # P_range = P_range*0.0689475729
        # T_range = (5/9)*(T_range-32.)+273.15

        return (FLUT, prop_label, prop_unit, GOR_range, P_range, T_range)

    def f_founder(self, arr):
        c1 = nd.uniform_filter(arr, size=3)
        c2 = nd.uniform_filter(arr*arr, size=3)
        var = c2 - c1*c1
        var = np.where(var < 1e-7, 1e-7, var)
        std = np.sqrt(var)
        mu = c1
        z = np.abs((arr - mu)/std)      # z-score with abs
        np.nan_to_num(z, copy=False)
        return np.where(z >= 1)

    def preproc(self, ptable, crange, plabel):
        x = range(len(plabel))
        y = range(len(crange[0]))
        err = []
        for X1 in y:
            arr = ptable[0, X1]     # get the failures of the first prop only, all props fail at the same places
            err.append(self.f_founder(arr))
        err = tuple(err)
        for X0, X1 in itertools.product(x, y):
            arr = ptable[X0, X1]
            e = err[X1]
            arr[e] = np.nan
            arr = np.ma.masked_invalid(arr)
            xx, yy = np.meshgrid(crange[2], crange[1])
            # get only the valid values
            x1 = xx[~arr.mask]
            y1 = yy[~arr.mask]
            newarr = arr[~arr.mask]
            newarr = griddata((x1, y1), newarr.ravel(), (xx, yy), method='cubic')
            # https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python
            ptable[X0, X1] = newarr
        return ptable

    # NOTE: FYI, I think the convention for a function like this is to have it in a `utils.py` file. Don't worry about moving it, just wanted to make you aware.
    def VolCylinder(self, h, Ro, Ri=0):
        '''
        Computes volume of cylinder
        input:
            h: height
            Ro: outer radius
            Ri: inner radius
        output:
            volume of hollow cylinder
        '''
        return Pi*(Ro**2 - Ri**2)*h

    def FLUT_interpolator(self, Coord: tuple, Prop: str or tuple) -> np.array:
        '''
        input:
            Coord: tuple containing GOR, P and T profiles (GOR, P, T are ndarrays)
            Prop: string or tuple of strings of the properties that the user wants to interpolate
        output:
            if user inputs 1 string, then one numpy array is returned. 
            if user inputs n strings in the tuple, then a 2D numpy array is returned where columns correspond to property index.
        '''
        if type(Prop) == str:
            # get index of property 
            idx = self.prop_label.index(f'{Prop}')
            # define an interpolating function for the given property
            interp = RegularGridInterpolator((self.GOR_range, self.P_range, self.T_range), self.FLUT[idx])
            # points for interpolation
            pts = np.column_stack(Coord)
            return interp(pts)
        else:
            Prop_list = []
            for p in Prop:
                # get index of property 
                idx = self.prop_label.index(f'{p}')
                # define an interpolating function for the given property
                interp = RegularGridInterpolator((self.GOR_range, self.P_range, self.T_range), self.FLUT[idx])
                # points for interpolation
                pts = np.column_stack(Coord)
                Prop_list.append(interp(pts))

            return np.column_stack(Prop_list)

    def FLUT_extractor(self) -> None:
        ''' add data from the Fluid LUT to the `fluid` object.'''
        Coord = (self.fl.GOR, self.fl.P, self.fl.T)
        
        self.fl.Ceq = self.FLUT_interpolator(Coord, 'Ceq')
        self.fl.yAsp = self.FLUT_interpolator(Coord, ('yAsp_V', 'yAsp_L1', 'yAsp_L2'))
        self.fl.wtFrac = self.FLUT_interpolator(Coord, ('wtFrac_V', 'wtFrac_L1', 'wtFrac_L2'))
        self.fl.volFrac = self.FLUT_interpolator(Coord, ('volFrac_V', 'volFrac_L1', 'volFrac_L2'))
        self.fl.dens = self.FLUT_interpolator(Coord, ('dens_V', 'dens_L1', 'dens_L2'))
        self.fl.dens_Asp = self.FLUT_interpolator(Coord, 'dens_Asp')
        self.fl.SP = self.FLUT_interpolator(Coord, ('SP_V', 'SP_L1', 'SP_L2'))
        self.fl.SP_Asp = self.FLUT_interpolator(Coord, 'SP_Asp')
        self.fl.visco = self.FLUT_interpolator(Coord, ('visco_V', 'visco_L1', 'visco_L2'))

        # unit conversion
        self.fl.dens = self.fl.dens*1000
        self.fl.dens_Asp = self.fl.dens_Asp*1000
        self.fl.visco = self.fl.visco/1000

    def Phase_velo(self) -> None:
        ''' calculate the axial velocity (m/s) of all phases =f(mFlow, A)'''
        a = Pi*self.pipe.R_eff**2      # m2
        A = a.reshape(a.size, 1)
        volFlow = np.where(self.fl.wtFrac > 0., self.sim.mFlow*self.fl.wtFrac/self.fl.dens, 0.)   # m3/s
        self.fl.velo = volFlow / A

    def _mix_aider(self, Frac_V, Frac_L1):
        ''' for methods `mix_Dens`, `mix_Visco`, `mix_Uz` '''
        f_sum = Frac_V + Frac_L1
        f_V = Frac_V/f_sum
        f_L1 = Frac_L1/f_sum
        return np.column_stack((f_V, f_L1))

    def mix_Dens(self) -> None:
        '''
        calculate averaged total density by method: "rho_mix"
        '''      
        rho_mix = self.mix_phase[0]
        dens = self.fl.dens
        if rho_mix == "mass":
            # mass-averaged dens
            wtf = self._mix_aider(self.fl.wtFrac[:, 0], self.fl.wtFrac[:, 1])        # mass fraction
            self.fl.rho = 1 / np.sum(wtf/dens[:, :2], axis=1)
        elif rho_mix == "volume":
            # vol-averaged dens
            volf = self._mix_aider(self.fl.volFrac[:, 0], self.fl.volFrac[:, 1])     # volume fraction
            self.fl.rho = np.sum(dens[:, :2]*volf, axis=1)
        else:
            # none (total dens = L1 dens)
            self.fl.rho = dens[:, 1]  

    def mix_Visco(self) -> None:
        '''
        calculate averaged total viscosity by method: "mu_mix"
        '''
        mu_mix = self.mix_phase[1]
        wtFrac = self.fl.wtFrac
        visco = self.fl.visco
        if mu_mix == 'mass':
            # mass-averaged visco
            wtf = self._mix_aider(wtFrac[:, 0], wtFrac[:, 1])        # mass fraction
            mu_ = np.nan_to_num(wtf/visco[:, :2])
            self.fl.mu = np.sum(mu_, axis=1)
        elif mu_mix == 'volume':
            # volume-averaged visco
            wtf = self._mix_aider(wtFrac[:, 0], wtFrac[:, 1])        # mass fraction
            mu_ = np.nan_to_num(wtf*visco[:, :2]/self.fl.dens[:, :2])
            self.fl.mu = self.fl.rho*np.sum(mu_, axis=1)
        else:
            # none (total visco = L1 visco)
            self.fl.mu = visco[:, 1]

        np.nan_to_num(self.fl.mu, copy=False)

    def mix_Uz(self) -> None:
        '''
        calculate averaged total velocity by method: "uz_mix"
        '''
        uz_mix = self.mix_phase[2]
        self.fl.uz = np.sum(self.fl.velo[:, :2], axis=1) if uz_mix == 'sum' else self.fl.velo[:, 1]

    def ADEPT_Diffusivity(self, model='const', Ds=2.e-9) -> float or np.array:
        '''
        particle diffusivity (m2/s) from Stokes-Einstein equation
        correlation:
            if 'const' then Dm = Ds
            if 'SE' then use Stokes-Einstein equation to calculate Dm
        '''
        return Ds if model == 'const' else kB*self.fl.T/(3*Pi*self.fl.mu*Ds)

    def _colebrook(self, fricFactor0=0.01, e=0.000001):
        '''
        Help function for fricFactor()

        Parameters
        ----------
        fricFactor0 : float
            Initial guess for friction factor.
        e : float
            pipe roughness (must be same unit as radius)
        
        Return
        ------
        fricFactor : float
            converged friction factor (solves Colebrook-White equation)
        '''
        Re = self.fl.avg_Re
        D = 2*self.pipe.avg_R
        def colebrook(x, e):
            return (1./np.sqrt(x)) + 2.*np.log10(2.51/(Re*np.sqrt(x)) + e/(3.71*D))

        return fsolve(colebrook, fricFactor0, args=(e))

    def fricFactor(self, correlation='blasius'):
        '''
        Calculate averaged friction factor along the pipe.

        Parameters
        ----------
        correlation : str, optional
            Optional input specifying the friction factor correlation.

        Notes
        -----
        More information about the Colebrook-White equation (in Spanish):
        https://es.wikipedia.org/wiki/Ecuaci%C3%B3n_de_Colebrook-White
        '''
        Re = self.fl.Re
        # Blasius (turbulent) and Colebrook-White friction factors
        # else_condition = 0.316/Re**0.25 if correlation == 'blasius' else self._colebrook()
        self.pipe.fricFactor = np.where(Re < 2400., 64/Re, 0.316/Re**0.25)
        np.nan_to_num(self.pipe.fricFactor, copy=False)

    def ADEPT_kP(self):
        '''
        precipitation kinetics (kP) parameter in the ADEPT model
        '''
        T = self.fl.T
        A = self.fl.kP_param
        if self.fl.kP_model == 'nrb':
            # NRB 2019
            dSP2 = self.fl.del_SP**2
            lnkP = -(A[0]*np.exp(-A[1]/T) + (A[2]*np.exp(-A[3]/T)/dSP2))
        elif self.fl.kP_model == 't-sp':
            dSP2 = self.fl.del_SP**2
            lnkP = np.log(A[0]) - A[1]*1000/(T*dSP2)
        else:   # kP_model = "default"
            dSP2 = 1.
            lnkP = np.log(A[0]) - A[1]*1000/(T*dSP2)

        self.fl.kP = np.exp(lnkP)

    def ADEPT_kDiss(self, model='default'):
        '''
        Kinetics of redissolution
        '''
        self.fl.kDiss = 0.01 if model == 'default' else 0.01/self.fl.kP

    def ADEPT_kAg(self):
        '''
        aggregation kinetics (kAg) parameter in the ADEPT model
        '''
        T = self.fl.T
        A = self.fl.kAg_param
        if self.fl.kAg_model == 'nrb':
            # NRB 2019a
            dSP2 = self.fl.del_SP**2
            lnkAg = -(A[0]*np.exp(-A[1]/T) + (A[2]*np.exp(-A[3]/T)/dSP2))
        else:   # kAg_model = "default"
            # a[1] represents the collision efficiency
            RT = Rg*T
            lnkAg = np.log(0.0666667/750. * RT/(self.fl.mu*1000) * A[0])

        self.fl.kAg = np.exp(lnkAg)*self.fl.c0_Asp

    def ADEPT_kD(self, delta, Dm):
        '''
        deposition kinetics (kD) parameter in the ADEPT model
        '''
        kD_us = self.fl.kD_us
        phi = Dm/(delta**2*kD_us)
        ScF = (2*delta/self.pipe.R)*(phi/(phi + 1))
        self.fl.kD = ScF*kD_us

    def ADEPT_rSR(self, tau):
        '''
        Shear removal rate.
        '''
        # extract SR parameters
        tau0, k, n = self.fl.SR_param
        
        if tau0 < 1.e-12:
            tau0 = 5.       # critical shear stress at wall (Pa), default
   
        self.fl.rSR = np.where(tau > tau0, k*np.abs(tau/tau0 - 1)**n, 0.)

    def Damkohler(self):
        '''
        Calculate scaling factor (ScF) & Damkohler numbers (Da) at each spatial point
        Da_P:  precipitation
        Da_Ag: aggregation
        Da_D:  deposition
        '''
        R_eff = self.pipe.R_eff
        L = self.pipe.L

        # density and viscosity averaging (="none" for L1 only, "mass", or "volume")
        # velocity averaging (="none" for L1 only, or "sum")
        self.mix_Dens()
        self.mix_Visco()
        self.mix_Uz()

        rho = self.fl.rho
        mu = self.fl.mu
        uz = self.fl.uz

        # residence time (s)
        self.fl.t_res = L/uz

        # axial dispersion, Peclet number
        Dm = self.ADEPT_Diffusivity()
        self.fl.Re = 2*R_eff*uz*rho/mu       # Reynolds number
        self.fl.avg_Re = np.average(self.fl.Re)                             # averaged Reynolds over the length of the pipe
        self.fl.Dax = np.where(self.fl.Re < 2500, Dm + (R_eff*uz)**2/(48*Dm), 0.4*R_eff*uz)
        self.fl.Pe = L*uz/self.fl.Dax                      # Peclet number

        # boundary layer calculation
        self.fricFactor()   # friction factor
        tau = 0.125*self.fl.rho*self.pipe.fricFactor*self.fl.uz**2          # shear stress at wall, Pa
        uf = np.sqrt(tau/self.fl.rho)                                       # friction velocity, m/s
        del_wall = self.fl.mu/(self.fl.rho*uf)                              # wall layer thickness, m
        del_lam = 5*del_wall                                                # laminar boundary layer, m
        del_mom = 125.4*self.pipe.R_eff*self.fl.Re**(-0.875)                # momentum boundary layer, m

        # choose boundary layer
        delta = del_lam

        # ADEPT rate parameters
        self.fl.del_SP = self.fl.SP_Asp - self.fl.SP[:, 1]

        #-- precipitation
        self.ADEPT_kP()
        
        #-- redissolution
        self.ADEPT_kDiss()

        #-- aggregation
        self.ADEPT_kAg()

        #-- deposition
        self.ADEPT_kD(delta, Dm)

        #-- shear removal rate, rSR
        self.ADEPT_rSR(tau)

        # Damkohler numbers
        self.fl.Da_P = self.fl.kP*self.fl.t_res
        self.fl.Da_Ag = self.fl.kAg*self.fl.t_res
        self.fl.Da_D = self.fl.kD*(1 - self.fl.rSR)*self.fl.t_res

        # Dimensionless time step
        self.sim.dtau = self.sim.dt/self.fl.t_res

    def ADEPT_Solver_Cf(self, u_old, nZ, BC):
        '''
        Solves PDE for dissolved asphaltene concentration (gAsp_L1 / gAsp_Ovr).

        Parameters
        ----------
        u_old : ndarray 
            Cf at jth step in tau.
        nZ : int 
            Number of Z-axis steps.
        dtau : ndarray
            Dimensionless time step size.
        dZ : float
            Z-axis step size (= 1/(nZ-1)).
        BC : float 
            Boundary condition at inlet (Z=0).
        isTransient : bool 
            Transient or steady-state (dCf/dtau = 0).

        Returns
        -------
        u_new : ndarray 
            Cf at (j+1)st step in tau.

        Notes
        -----
        Solves the PDE:
        dCf/dtau + dCf/dZ + Da_P*Cf = Da_P*Ceq

        Subject to the conditions:
        IC: Cf(Z, tau=0) = 1    (all Asp dissolved at tau=0)
        BC: Cf(Z=0, tau) = 1    (all Asp dissolved at Z=0)

        Using a Crank-Nicolson scheme (unconditionally stable). 
        For more information read the markdown file "PDE_solutions.md"
        '''
        if self.sim.isTransient:
            alpha = 0.5*self.sim.dtau/self.sim.dZ
            beta = 0.5*self.sim.dtau*self.fl.Da_P
            gamma = self.sim.dtau*self.fl.Da_P*self.fl.Ceq

            A = np.diag(1. + alpha[1:] + beta[1:]) - np.diag(alpha[2:], k=-1)
            B = np.diag(1. - alpha[1:] - beta[1:]) + np.diag(alpha[2:], k=-1)
            
            w_old = u_old[1:]
            w0_old = u_old[0]
            w0_new = BC
            b = gamma[1:]
            b[0] = alpha[1]*(w0_new + w0_old) + gamma[1]
            c = np.matmul(B, w_old) + b

            # Solves linear system A*w_new = B*w_old + b = c
            w_new = np.linalg.solve(A, c)

            u_new = np.zeros(nZ)
            u_new[1:] = w_new
            u_new[0] = BC
        else:
            u_new = u_old

        # z = np.linspace(0, 1, nZ)
        # plt.plot(z, u_new)
        # plt.show()
        return u_new

    def ADEPT_Solver_C(self, u_old, v_old, v_new, nZ, BC1):
        '''
        Solves PDE for primary particle concentration

        Parameters
        ----------
        u_old : ndarray 
            C at jth step in tau.
        v_old : ndarray 
            Cf at jth step in tau.
        v_new : ndarray 
            Cf at (j+1)st step in tau.
        nZ : int 
            Number of Z-axis steps.
        dtau : ndarray
            Dimensionless time step size.
        dZ : float
            Z-axis step size (= 1/(nZ-1)).
        BC1 : float 
            Boundary condition at inlet (Z=0).
        isTransient : bool 
            Transient or steady-state (dCf/dtau = 0).

        Returns
        -------
        u_new : ndarray 
            C at (j+1)st step in tau.

        Notes
        -----
        Solves the PDE:
        dC/dtau = 1/Pe*d^2C/dZ^2 - dC/dz + Da_p*rp - Da_ag*C^2 - Da_d*C
        rp = Cf-Ceq     for Cf >= Ceq (precipitation)
        rp = -kDiss*C   for Cf < Ceq (redissolution)

        Subject to the conditions:
        IC: C(Z, 0) = 0         (no PP at t=0)
        BC1: C(0, tau) = 0      (no PP at z=0) 
        BC2: dC/dZ(1, tau) = 0  (no gradient at end of pipe)

        Using a Crank-Nicolson scheme (unconditionally stable). 
        For more information read the markdown file "PDE_solutions.md"
        '''
        if self.sim.isTransient: 
            # driving force
            F_new = v_new - self.fl.Ceq
            F_old = v_old - self.fl.Ceq

            # define coefficients of node equations
            alpha = 0.5*self.sim.dtau/(self.fl.Pe*self.sim.dZ**2)
            beta = 0.5*self.sim.dtau/self.sim.dZ
            gamma = 0.5*self.sim.dtau*self.fl.Da_P
            delta = 0.5*self.sim.dtau*self.fl.Da_Ag
            epsilon = 0.5*self.sim.dtau*self.fl.Da_D

            w_old = u_old[1:nZ-1]
            w0_old = BC1
            w0_new = BC1

            A = np.diag(1 + 2*alpha[1:nZ-1] + beta[1:nZ-1] + gamma[1:nZ-1]*self.fl.kDiss*np.heaviside(-F_new[1:nZ-1], 0.) + epsilon[1:nZ-1]) - np.diag(alpha[1:nZ-2], k=1) - np.diag(alpha[2:nZ-1] + beta[2:nZ-1], k=-1)
            B = np.diag(-1 + 2*alpha[1:nZ-1] + beta[1:nZ-1] + gamma[1:nZ-1]*self.fl.kDiss*np.heaviside(-F_old[1:nZ-1], 0.) + epsilon[1:nZ-1]) - np.diag(alpha[1:nZ-2], k=1) - np.diag(alpha[2:nZ-1] + beta[2:nZ-1], k=-1)
            f = -gamma[1:nZ-1]*(F_new[1:nZ-1]*np.heaviside(F_new[1:nZ-1], 1.) + F_old[1:nZ-1]*np.heaviside(F_old[1:nZ-1], 1.)) + delta[1:nZ-1]*w_old**2
            f[0] += -(alpha[1] + beta[1])*(w0_new + w0_old)
            Bw_old = np.matmul(B, w_old)

            def func(w_new):
                wlast_old = u_old[-1]
                wlast_new = w_new[-1]   # forward-difference approximation
                f[-1] = -gamma[-2]*(F_new[-2]*np.heaviside(F_new[-2], 1.) + F_old[-2]*np.heaviside(F_old[-2], 1.)) + delta[-2]*w_old[-1]**2
                f[-1] += -alpha[-2]*(wlast_new + wlast_old)     # BC2
                return np.matmul(A, w_new) + Bw_old + delta[1:nZ-1]*w_new**2 + f

            # finds the roots of func(u) = 0
            w_new = fsolve(func, w_old)
            u_new = np.zeros(nZ)
            u_new[0] = w0_new
            u_new[1:nZ-1] = w_new
            u_new[-1] = u_new[-2]

        # z = np.linspace(0, 1, nZ)
        # plt.plot(z, u_new)
        # plt.show()
        return u_new

    # def energy_balance(self, T):
    #     '''calculates T=f(z) given updated `del`'''
    #     pass

    # def pressure_drop(self, dz, nz, dPf, dPg, dP):
    #     '''calculates P=f(z) given updated "del"'''
    #     fricFactor = self.fl.fricFactor
    #     uz = self.fl.uz
    #     rho = self.fl.rho
    #     R_eff = self.fl.R_eff
    #     for i in range(nz):
    #         # friction pressure drop
    #         dPf_i = fricFactor[i]*dz*uz[i]**2*rho[i]/(R_eff[i]*4)   # Darcy-Weisbach formula (Pa)
    #         dPf += dPf_i
            
    #         # gravity pressure drop
    #         dPg_i = dz*9.81*rho[i]      # Pa
    #         dPg += dPg_i
            
    #         # sum pressure drop in segment
    #         dP_i = dPf_i + dPg_i
            
    #         # cumulative dp
    #         dP += dP_i
    #     return dP

    def ADEPT_Solver(self, T_prof: tuple, P_prof: tuple, GOR: float):
        '''
        Solves asphaltene deposition problem =f(t,z) using the ADEPT formulation

        Parameters
        ----------
            T : np.ndarray(float)
                temperature profile (K)
            P : np.ndarray(float)
                pressure profile (bar)
            GOR : np.ndarray(float)
                Gas-Oil Ratio
            nz : int
                number of z-axis points
            nt : int
                number of time points
        Return
        ------
            `depo_return` object
            
        Notes
        -----
            + Currently using a Crank-Nicolson scheme to solve PDEs
            + deposition model
                + ADEPT formulation: primary particle (PP) material balance [1-3]
                + eqn: dC/dt = 
                    + advection (transport of PP)
                    + precipitation (source of PP)
                    + aggregation (sink of PP)
                    + deposition (sink of PP)        
            + dT model (enthalpy change):
                + model description
                + relevant assumptions
            + dP model (pressure drop):  
                + Darcy-Weisbach
                + friction factor: Colebrook-White equation
        
        References
        ----------
        [1] Rajan Babu, N. (2019) "Mechanistic Investigation and Modeling of Asphaltene Deposition," Rice University, Houston, Texas, USA, 2019. 
        link: https://www.proquest.com/docview/2568024745?fromopenview=true&pq-origsite=gscholar
        [2] Rajan Babu, N. et al. (2019). "Systematic Investigation of Asphaltene Deposition in the Wellbore and Near-Wellbore Region of a Deepwater Oil Reservoir under Gas Injection. 
        Part 2: Computational Fluid Dynamics Modeling of Asphaltene Deposition." doi: https://doi.org/10.1021/acs.energyfuels.8b03239.
        [3] Naseri et al. (2020) "A new multiphase and dynamic asphaltene deposition tool (MAD-ADEPT) to predict the deposition of asphaltene particles on tubing wall."
        doi: https://doi.org/10.1016/j.petrol.2020.107553

        '''
        R = self.pipe.R
        L = self.pipe.L
        t_sim = self.sim.t_sim

        self.pipe.V = self.VolCylinder(L, R)    # total tubing volume
        C_in = 0.                               # concentration of primary particles (C) at inlet (z=0)
        Cf_in = 1.                              # concentration of dissolved asphaltenes (Cf) at inlet (z=0)
        # BHP = 1.e6                            # dummy value of bottom-hole P
        kgOil = 0.
        kgAsp = 0.
        C_tol = 1.e-10
        t_noFlow = None
        # dT = 0.
        # dPf, dPg = 0.

        nZ = 200
        # nt = 100.
        
        # (Bottomhole -> Wellhead)
        z = np.linspace(0., 1, nZ)
        xp = [0, 1]
        T = np.interp(z, xp, fp=T_prof)
        P = np.interp(z, xp, fp=P_prof)

        # pass input thermo properties to `fluid` object
        self.fl.GOR = GOR*np.ones(nZ)
        self.fl.P = P
        self.fl.T = T

        # z-axis step size
        self.sim.dZ = 1./(nZ - 1.)

        # time-step size (s)
        # TODO: (next version) `dt` should be calculated from the characteristic time of deposit formation
        self.sim.dt = 3600.

        # initial conditions
        Cf0 = np.ones(nZ)     # assumes all asphaltenes are soluble at t=0
        C0 = np.zeros(nZ)     # assumes no primary particles (PP) at t=0

        # initial deposition thickness (del) along the pipe
        if self.pipe.del_Asp is None or self.pipe.del_DM is None:
            del_Asp = np.zeros(nZ)
            del_DM = np.zeros(nZ)
        else:
            del_Asp = self.pipe.del_Asp
            del_DM = self.pipe.del_DM

        # extract thermo and transport properties from FLUT at each spatial point
        self.FLUT_extractor()

        #--- time-march loop ---
        flow_status = 'flowing'     # will exit time-march loop when flow_status != 'flowing'
        t = 0.
        itr = 0
        while t < t_sim and flow_status == 'flowing':
            itr += 1
            t += self.sim.dt
            
            # update radius profile due to restriction (R_eff: effective radius)
            R_eff = R - del_DM
            self.pipe.R_eff = R_eff
            self.pipe.R_eff_avg = np.average(self.pipe.R_eff)

            # extract thermo and transport properties from FLUT at each spatial point
            # self.FLUT_extractor()

            # phase velocity =f(mFlow, R_eff)
            self.Phase_velo()

            #zAsp: asphaltene composition (g[Asp]/g[Ovr]) at inlet
            self.fl.zAsp = np.sum(self.fl.wtFrac*self.fl.yAsp, axis=1)[0]

            #c0_Asp: asphaltene concentration (kg[Asp]/m3[Ovr]) at inlet  
            self.fl.c0_Asp = self.fl.zAsp*self.fl.dens[0, 1]

            # mass flows (kg), oil and Asp
            mFlow_Asp = self.sim.mFlow*self.fl.zAsp     # kgAsp/s
            kgOil += self.sim.mFlow*self.sim.dt         # kg Oil (cumulative; added at each dt)
            kgAsp += mFlow_Asp*self.sim.dt              # kg Asp (cumulative; added at each dt)

            # calculate scaling factor & Damkohler numbers
            self.Damkohler()

            #----- solve governing PDEs ----------------------------------------------         
            # solve for conc dissolved asphaltenes: Cf(t,z)
            Cf = self.ADEPT_Solver_Cf(Cf0, nZ, Cf_in)

            # solve for conc primary particles: C(t,z)
            C = self.ADEPT_Solver_C(C0, Cf0, Cf, nZ, C_in)
            #-------------------------------------------------------------------------

            # update initial concentrations for next time step
            C_df = Cf - self.fl.Ceq     # update concentration driving force
            Cf = np.where(Cf < C_tol, 0., Cf)
            C = np.where(C < C_tol, 0., C)
            C_df = np.where(C_df < C_tol, 0., C_df)
            Cf0 = Cf    # soluble asphaltenes
            C0 = C      # primary particles

            #=========================================================
            #--- calculate deposition profile, flux, and other important outputs
            # post-processing step after PDEs are solved
            #=========================================================

            # rate of asphaltene deposition (kg/m3/s)
            rD = self.fl.kD*C*self.fl.c0_Asp*(1 - self.fl.rSR)

            # deposition flux (kg/m2/s). might need to revisit to account for V_bl not V_cell.
            J_Asp = 0.5*rD*self.pipe.R_eff

            # thickness of asphaltenes and DM deposit (m) (Assumes R >> del which is a good approx for small dt)
            # TODO: I want to write the thickness calculation such that it applies to R ~ del
            rho_Asp = np.average(self.fl.dens_Asp*self.fl.yAsp[:, 2])
            rho_DM = np.average(self.fl.dens[:, 2]*self.fl.yAsp[:, 2])
            del_Asp_z = self.sim.dt*J_Asp/rho_Asp
            del_DM_z = self.sim.dt*J_Asp/rho_DM

            del_Asp += del_Asp_z
            del_DM += del_DM_z

            # set maximum deposit thickness (del) to pipe radius (R)
            del_Asp = np.where(del_Asp >= R, R, del_Asp)
            del_DM = np.where(del_DM >= R, R, del_DM)

            # deposit thickness frac (r/R)
            delf_Asp = del_Asp / R
            delf_DM = del_DM / R
            delf_max = delf_Asp.max()
            
            # check if pipe is plugged
            if delf_max > 0.999:
                t_noFlow = t
                flow_status = 'shutdown; cross-section is plugged'
                
            #-----/ depo flux (J) and thickness (del) calculation -----

            #----- deposited Asp mass ----
                
            # volume (m3) and mass (kg) of deposit 
            V_cyl = self.VolCylinder(L*self.sim.dZ, R, R - del_Asp)
            mass_Asp = V_cyl*self.fl.dens_Asp

            #----- update `T,P` profiles -----
            # store dT, dP results from prev iteration
            # dT0 = dT
            # dPf0, dPg0 = dPf, dPg
            
            # # TODO: add energy balance model to calculate change in T profile as deposit builds; should consider heat transfer across pipe wall, solid deposit, and fluid.
            # # dT = self.energy_balance(self.fl)
            
            # # dP @ t; dP=(dPf=friction loss, dPg=gravity loss)
            # dPf, dPg = self.pressure_drop()
            
            # # WHP, update wellhead pressure
            # WHP = BHP - (dPf + dPg)
        
            # # determine if well still flowing (fluid must be able to overcome pressure drop)
            # # exit loop if no flow, else update T,P profiles
            # if WHP < WHP_min:
            #     t_noFlow = t
            #     flow_status = 'shutdown; P drop too large'
            # else:          
            #     # dPf/dt, friction pressure drop wrt time       
            #     dPf_dt = (dPf0 - dPf) / dt

                # update T and P profiles
                # if T_prof != 'cst':
                #     self.fl.T = self.energy_model(T, FLUT)
                # if P_prof != 'cst':
                #     self.fl.P = self.well_model(P, FLUT)

        #--/ time loop

        # vol flows (m3), oil and Asp
        # m3Oil = kgOil/self.fl.dens[0, 1]
        # m3Asp = kgAsp/self.fl.dens_Asp[0]

        return depo_return(flow_status, t_noFlow, delf_Asp, delf_DM, J_Asp, C, mass_Asp)