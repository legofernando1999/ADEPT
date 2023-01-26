# import packages
import numpy as np
import json
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.interpolate import RegularGridInterpolator

# TODO: validate against ADEPT-VBA (file: `ADEPT(2023-01-17).xlsm`)
# TODO: add energy balance model to calculate change in T profile as deposit builds.
# TODO: add multiphase flow simulator to capture velocity and pressure gradients.
# TODO: fix flash calculation failures in FLUT
# FIXME: the whole code!

# define constants
Rg = 8.314          # gas constant [J/mol.K]
kB = 1.3806504e-23  # Boltzmann constant [J/K]
#Nav = 6.022e23      # Avogadro number [1/mol]
Pi = np.pi

# <-- REFERENCES -->
# `NRB 2019a`: Rajan Babu, N. (2019) "Mechanistic Investigation and Modeling of Asphaltene Deposition," Rice University, Houston, Texas, USA, 2019. link: https://www.proquest.com/docview/2568024745?fromopenview=true&pq-origsite=gscholar
# `NRB 2019b`: Rajan Babu, N. et al. (2019). "Systematic Investigation of Asphaltene Deposition in the Wellbore and Near-Wellbore Region of a Deepwater Oil Reservoir under Gas Injection. Part 2: Computational Fluid Dynamics Modeling of Asphaltene Deposition." doi: https://doi.org/10.1021/acs.energyfuels.8b03239.


class pipe(object):
    
    def __init__(self, L: float, R: float, T_ext: float, del_Asp0: np.array(float)=None, del_DM0: np.array(float)=None) -> None:
        '''
        L: length of pipe (m)
        R: radius of pipe (m)
        T_ext: temperature of pipe
        del_Asp0: initial deposit thickness (m), assuming only asphaltenes deposit
        del_DM0: initial deposit thickness (m), assuming aromatics and resins also deposit
        
        NOTE: del_Asp0 and del_DM0 are arrays with discrete values of deposit thickness corresponding to the array of `z`. 
        In case the array of `z` changes from one simulation to the next (which I don't know why it would), 
        we could fit a polynomial to del_Asp0 and then use this to translate del_Asp0 from one discretization scheme to another. 
        
        NOTE: added variable types for `del_Asp0` and `del_DM0`. if `del_Asp0` is None, then we will initialize an array of zeros.
        '''
        self.L = L
        self.R = R
        self.T_ext = T_ext  
        self.del_Asp0 = del_Asp0
        self.del_DM0 = del_DM0


class sim(object):
    
    def __init__(self, t_sim: float, mFlow: float, isTransient: bool=True) -> None:
        '''
        t_sim: simulation time (s)
        mFlow: mass flow rate (kg/s)
        isTransient: {True=transient, False=steady-state}
        '''
        self.t_sim = t_sim
        self.mFlow = mFlow
        self.isTransient = isTransient


class fluid(object):

    def __init__(self) -> None:
        pass


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
        df = fluid()

        # asphaltene rate parameters
        df.kP = KLUT["kP"]
        df.kAg = KLUT["kAg"]
        df.kD = KLUT["kD"]
        df.kP_param = KLUT["kP_param"]
        df.kP_model = KLUT["kP_model"]
        df.kAg_param = KLUT["kAg_param"]
        df.kAg_model = KLUT["kAg_model"]
        df.kD_param = KLUT["kD_param"]
        df.kD_scale = KLUT["kD_scale"]
        df.SR_param = KLUT["SR_param"]
        df.SR_model = KLUT["SR_model"]

        self.df = df

        file_tuple = self.FLUT_loader(file)
        self.FLUT, self.prop_label, self.prop_unit, self.GOR_range, self.P_range, self.T_range = file_tuple

    def FLUT_loader(self, file):
        '''
        FLUT: Thermodynamic Lookup Table, 4D array (Prop, GOR, P, T)
            Ceq: asphaltene solubility (gAsp[L1]/gAsp[T])
            wtFrac[V, L1, L2]: phase frac, mass (g[k]/g)
            volFrac[V, L1, L2]: phase frac, volume (m3[k]/m3)
            dens[V, L1, L2, Asp]: phase density (kg/m3)
            visco[V, L1, L2, Asp]: phase viscosity (cP)
            SP[V, L1, L2, Asp]: solubility parameter (MPa^0.5)
            yAsp[V, L1, L2]: asphaltene composition (gAsp[k] / g[k])
        '''
        prop_dict = json.load(file) if type(file) != dict else file

        FLUT = prop_dict['prop_table']
        prop_label = prop_dict['prop_label']
        prop_unit = prop_dict['prop_unit']
        GOR_range = np.array(prop_dict['GOR'])
        P_range = np.array(prop_dict['P'])
        T_range = np.array(prop_dict['T'])

        return (FLUT, prop_label, prop_unit, GOR_range, P_range, T_range)

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
            Coord: tuple containing GOR, P and T profiles
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
        Coord = (self.df.GOR, self.df.P, self.df.T)
                
        self.df.Ceq = self.FLUT_prop(Coord, 'Ceq')
        self.df.yAsp = self.FLUT_prop(Coord, ('yAsp_V', 'yAsp_L1', 'yAsp_L2'))
        self.df.wtFrac = self.FLUT_prop(Coord, ('wtFrac_V', 'wtFrac_L1', 'wtFrac_L2'))
        self.df.volFrac = self.FLUT_prop(Coord, ('volFrac_V', 'volFrac_L1', 'volFrac_L2'))
        self.df.dens = self.FLUT_prop(Coord, ('dens_V', 'dens_L1', 'dens_L2'))
        self.df.dens_Asp = self.FLUT_prop(Coord, 'dens_Asp')
        self.df.SP = self.FLUT_prop(Coord, ('SP_V', 'SP_L1', 'SP_L2'))
        self.df.SP_Asp = self.FLUT_prop(Coord, 'SP_Asp')
        self.df.visco = self.FLUT_prop(Coord, ('visco_V', 'visco_L1', 'visco_L2'))

    def Phase_velo(self) -> None:
        a = Pi*self.pipe.Rz**2      # m2
        A = a.reshape(a.size, 1)
        volFlow = self.sim.mFlow*self.df.wtFrac/self.df.dens   # m3/s
        self.df.velo = volFlow / A

    def _mix_aider(self, Frac_V, Frac_L1):
        '''
        for methods `mix_Dens`, `mix_Visco`, `mix_Velo`
        '''
        f_sum = Frac_V + Frac_L1
        f_V = Frac_V/f_sum
        f_L1 = Frac_L1/f_sum
        return np.column_stack((f_V, f_L1))

    def mix_Dens(self) -> None:
        '''
        calculate averaged total density by method: "rho_mix"
        '''      
        rho_mix = self.mix_phase[0]
        if rho_mix == "mass":
            # mass-averaged dens
            wtf = self._mix_aider(self.df.wtFrac[:, 0], self.df.wtFrac[:, 1])        # mass fraction
            self.df.rho = 1 / np.sum(wtf/self.df.dens[:, :2], axis=1)
        elif rho_mix == "volume":
            # vol-averaged dens
            volf = self._mix_aider(self.df.volFrac[:, 0], self.df.volFrac[:, 1])     # volume fraction
            self.df.rho = np.sum(self.df.dens[:, :2]*volf, axis=1)
        else:
            # none (total dens = L1 dens)
            self.df.rho = self.df.dens[:, 1]  

    def mix_Visco(self) -> None:
        '''
        calculate averaged total viscosity by method: "mu_mix"
        '''
        mu_mix = self.mix_phase[1]
        if mu_mix == 'mass':
            # mass-averaged visco
            wtf = self._mix_aider(self.df.wtFrac[:, 0], self.df.wtFrac[:, 1])        # mass fraction
            self.df.mu = np.sum(wtf/self.df.visco[:, :2], axis=1)
        elif mu_mix == 'volume':
            # volume-averaged visco
            wtf = self._mix_aider(self.df.wtFrac[:, 0], self.df.wtFrac[:, 1])        # mass fraction
            self.df.mu = self.df.rho*np.sum(wtf*self.df.visco[:, :2]/self.df.dens[:, :2], axis=1)
        else:
            # none (total visco = L1 visco)
            self.df.mu = self.df.visco[:, 1]

    def mix_Velo(self) -> None:
        '''
        calculate averaged total velocity by method: "uz_mix"
        '''
        uz_mix = self.mix_phase[2]
        self.df.uz = np.sum(self.df.velo, axis=1) if uz_mix == 'sum' else self.df.velo[:, 1]

    def ADEPT_Diffusivity(self, model='const', Ds=2.e-9) -> float or np.array:
        '''
        particle diffusivity (m2/s) from Stokes-Einstein equation
        correlation:
            if 'const' then Dm = Ds
            if 'SE' then use Stokes-Einstein equation to calculate Dm
        '''
        return Ds if model == 'const' else kB*self.df.T/(3*Pi*self.df.mu*Ds)

    def _colebrook(self, fricfactor0=0.01, e=0.000001):
        '''
        Help function for fricfactor()

        Parameters
        ----------
        fricfactor0 : float
            Initial guess for friction factor.
        e : float
            Roughness
        '''
        Re = self.df.avg_Re
        D = 2*self.pipe.avg_R
        def colebrook(x, e):
            return (1./np.sqrt(x)) + 2.*np.log10(2.51/(Re*np.sqrt(x)) + e/(3.71*D))

        return fsolve(colebrook, fricfactor0, args=(e))

    def fricfactor(self, correlation='Blasius'):
        '''
        Calculate averaged friction factor along the pipe.

        Parameters
        ----------.
        correlation : str, optional
            Optional input specifying the friction factor correlation.

        Notes
        -----
        More information about the Colebrook-White equation (in Spanish):
        https://es.wikipedia.org/wiki/Ecuaci%C3%B3n_de_Colebrook-White
        '''
        if self.df.avg_Re < 1.e-12:
            self.pipe.fricfactor = 0.
        elif self.df.avg_Re < 2400:
            self.pipe.fricfactor = 64/self.df.avg_Re            # laminar friction factor
        else:
            # Blasius (turbulent) and Colebrook-White friction factors
            self.pipe.fricfactor = 0.316/self.df.avg_Re**0.25 if correlation == 'Blasius' else self._colebrook()
 
    def ADEPT_kP(self):
        '''
        Calculate the precipitation kinetics (kP) parameter in the ADEPT model
        '''
        T = self.df.T
        A = self.df.kP_param
        if self.df.kP_model in ['nrb', 'narmi']:
            # NRB 2019
            dSP2 = self.df.del_SP**2
            lnkP = -(A[0]*np.exp(-A[1]/T) + (A[2]*np.exp(-A[3]/T)/dSP2))
        elif self.df.kP_model in ['t-sp', 'il-sp']:
            dSP2 = self.df.del_SP**2
            lnkP = np.log(A[0]) - A[1]*1000/(T*dSP2)
        else:   # kP_model = "default"
            dSP2 = 1.
            lnkP = np.log(A[0]) - A[1]*1000/(T*dSP2)

        self.df.kP = np.exp(lnkP)

    def ADEPT_kDiss(self, model='default'):
        '''
        Kinetics of redissolution
        '''
        self.df.kDiss = 0.01 if model == 'default' else 0.01/self.df.kP

    def ADEPT_kAg(self):
        '''
        Calculates the aggregation kinetics (kAg) parameter in the ADEPT model
        '''
        T = self.df.T
        A = self.df.kAg_param
        if self.df.kAg_model == 'nrb':
            # NRB 2019a
            dSP2 = self.df.del_SP**2
            lnkAg = -(A[0]*np.exp(-A[1]/T) + (A[2]*np.exp(-A[3]/T)/dSP2))
        else:   # kAg_model = "default"
            # a[1] represents the collision efficiency
            RT = Rg*T
            lnkAg = np.log((RT/self.df.visco)*A[0]/11250)

        self.df.kAg = np.exp(lnkAg)*self.df.c0_Asp

    def ADEPT_kD(self, delta, Dm):
        kD_us = self.df.kD
        phi = Dm/(delta**2*kD_us)
        ScF = (2*delta/self.pipe.R)*(phi/(phi + 1))
        self.df.kD = ScF*kD_us

    def ADEPT_rSR(self, A, tau):
        '''
        Shear removal rate.
        '''
        # extract SR parameters
        tau0, k, n = self.df.SR_param
        
        if tau0 < 1.e-12:
            tau0 = 5.       # critical shear stress at wall (Pa), default
   
        self.df.rSR = np.where(tau > tau0, k*(tau/tau0 - 1)**n, 0.)

    def Damkohler(self):
        '''
        Calculate scaling factor (ScF) & Damkohler numbers (Da, reaction parameters) at each spatial point
        Da_P:  precipitation
        Da_Ag: aggregation
        Da_D:  deposition
        '''
        # density and viscosity averaging (="none" for L1 only, "mass", or "volume")
        # velocity averaging (="none" for L1 only, or "sum")
        self.mix_Dens()
        self.mix_Visco()
        self.mix_Velo()

        # residence time (s)
        self.df.t_res = self.pipe.L/self.df.uz

        # axial dispersion, Peclet number
        Dm = self.ADEPT_Diffusivity()
        self.df.Re = 2*self.pipe.Rz*self.df.uz*self.df.rho/self.df.mu       # Reynolds number
        self.df.avg_Re = np.average(self.df.Re)                             # averaged Reynolds over the length of the pipe
        Dax = np.where(self.df.Re < 2500, Dm + (self.pipe.Rz*self.df.uz)**2/(48*Dm), 0.4*self.pipe.Rz*self.df.uz)
        self.df.Pe = self.pipe.L*self.df.uz/Dax                             # Peclet number

        # boundary layer calculation
        self.fricfactor()   # friction factor
        tau = 0.125*self.df.rho*self.pipe.fricfactor*self.df.uz**2          # shear stress at wall, Pa
        uf = np.sqrt(tau/self.df.rho)                                       # friction velocity, m/s
        del_wall = self.df.mu/(self.df.rho*uf)                              # wall layer thickness, m
        del_lam = 5*del_wall                                                # laminar boundary layer, m
        del_mom = 125.4*self.pipe.Rz*self.df.Re**(-0.875)                   # momentum boundary layer, m

        # choose boundary layer
        delta = del_lam

        # ADEPT rate parameters
        self.df.del_SP = self.df.SP_Asp - self.df.SP[:, 1]

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
        self.df.Da_P = self.df.kP*self.df.t_res
        self.df.Da_Ag = self.df.kAg*self.df.t_res
        self.df.Da_D = self.df.kD*(1 - self.df.rSR)*self.df.t_res

    def ADEPT_Solver_Cf(self, u0, m, k, h, V, F, BC, isTransient: bool=True):
        '''
        Solves PDE for dissolved asphaltene concentration (gAsp_L1 / gAsp_Ovr)
        dCf/dt = -dCf/dz - Da_p*(Cf-Ceq)
        IC: Cf(z, t=0) = 1    (all Asp dissolved at t=0)
        BC: Cf(z=0, t) = 1    (all Asp dissolved at z=0)

        input
            u0: Cf at j-th step in t
            m: number of z-axis steps
            k: time step size (=T/N)
            h: z-axis step size (=L/m)
            V: Da_p at j-th step in t
            F: Ceq at j-th step in t
            BC: boundary condition at inlet (z=0)
            isTransient: transient or steady-state (dCf/dt = 0)
        output
            u: Cf at (j+1)st step in t
        '''
        if isTransient:
            alpha = 0.25*k/h

            # A = np.zeros((m - 1, m - 1))
            # B = np.zeros_like(A)

            # np.fill_diagonal(A, 1 + 0.5*k*v[1:])
            # np.fill_diagonal(B, 1 - 0.5*k*v[1:])

            # for i in range(m - 2):
            #     A[i, i+1] = alpha
            #     A[i+1, i] = -alpha
            #     B[i, i+1] = -alpha
            #     B[i+1, i] = alpha

            A = np.diag(1 + 0.5*k*v[1:]) + np.diag(alpha*np.ones(m-2), k=1) + np.diag(-alpha*np.ones(m-2), k=-1)
            B = np.diag(1 - 0.5*k*v[1:]) + np.diag(-alpha*np.ones(m-2), k=1) + np.diag(alpha*np.ones(m-2), k=-1)

            v = V[1:]
            f = v*F[1:]
            w0 = u0[1:]

            C = np.matmul(B, w0) + k*f

            # Solves linear system Aw = C
            w = np.linalg.solve(A, C)

            u = np.empty(m)
            u[0] = BC
            u[1:] = w

        else:
            u = u0

        return u

    def ADEPT_Solver_C(self, u0, m, k, h, A1, A2, A3, A4, A5, V0, V, F, BC, isTransient: bool=True):
        '''
        Solves PDE for primary particle concentration:
        dC/dt = 1/Pe*d2C/dz2 - dC/dz + rp - Da_ag*C^2 - Da_d*C
        rp = Da_p*(Cf-Ceq)      for Cf > Ceq (precipitation)
        rp = -kdiss.Da_p.C      for Cf < Ceq (redissolution)
        IC: C(z, t=0) = 0       (no PP at t=0)
        BC1: C(z=0, t) = 0      (no PP at z=0) 
        BC2: dC/dz(z=1) = 0     (no gradient at end of pipe)?? NOT IMPLEMENTED YET!!

        input
            u0: C at j-th step in t
            m: number of z-axis steps
            k: time step size (=T/N)
            h: z-axis step size (=L/m)
            A1: Pe, Peclet number
            A2: Da_Ag at j-th step in t
            A3: Da_D at j-th step in t
            A4: Da_P at j-th step in t
            A5: kDiss at j-th step in t
            V0: Cf at j-th step in t
            V: Cf at (j+1)st step in t
            F: Ceq at j-th step in t
            BC: boundary condition at inlet (z=0)
            isTransient: transient or steady-state (dCf/dt = 0)
        output
            u: C at (j+1)st step in t
        '''
        if isTransient:
            alpha = 0.5*k/h**2
            beta = 0.25*k/h

            a1 = 1/A1[1:]
            a2 = A2[1:]
            a3 = A3[1:]
            a4 = A4[1:]
            a5 = A5[1:]
            w0 = u0[1:]

            v0 = V0[1:]
            v = V[1:]
            f = F[1:]

            R0 = np.where(v0 < f, 1., 0.)
            R = np.where(v < f, 1., 0.)

            A = np.diag(1 + 2*alpha*a1 + 0.5*k*a3 + 0.5*k*a5*a4*R) + np.diag(-alpha*a1[:m-2] + beta, k=1) + np.diag(-alpha*a1[1:] - beta, k=-1)
            B = np.diag(-1 + 2*alpha*a1 + 0.5*k*a3 + 0.5*k*a5*a4*R0) + np.diag(-alpha*a1[:m-2] + beta, k=1) + np.diag(-alpha*a1[1:] - beta, k=-1)

            r0 = np.where(v0 >= f, 1., 0.)
            r = np.where(v >= f, 1., 0.)

            def func(w):
                return np.matmul(A, w) + 0.5*k*a2*w**2 - 0.5*k*a4*(v - f)*r + np.matmul(B, w0) + 0.5*k*a2*w0**2 - 0.5*k*a4*(v0 - f)*r0

            w = fsolve(func, w0)
            u = np.empty(m)
            u[0] = BC
            u[1:] = w

        elif not isTransient:
            u = u0

        return u

    def energy_balance(self, T):
        '''calculates T=f(z) given updated `del`'''
        pass

    def pressure_drop(self, P):
        '''calculates P=f(z) given updated `del`'''
        pass

    def ADEPT_Solver(self, T: np.array, P: np.array, GOR: np.array, nz, nt, WHP_min=1.):
        '''
        Solves asphaltene material balance
        
        Notes:
            1.- Should we choose the time step according to the characteristic time of formation of deposit?
            2.- Currently using a Crank-Nicolson scheme to solve PDEs
            3.- Review ADEPT_Diffusivity function with Paco
            4.- Use Colebrook-White equation to calculate the friction factor

        arguments:
            T: temperature profile
            P: pressure profile
            GOR: Gas to Oil ratio
            nz: number of z-axis points
            nt: number of time points
        return:
            depo_return object
        '''
        R = self.pipe.R
        L = self.pipe.L
        isTransient = self.sim.isTransient

        self.pipe.V = self.VolCylinder(L, R)    # total tubing volume
        C_in = 0.                               # concentration of primary particles (C) at inlet (z=0)
        Cf_in = 1.                              # concentration of dissolved asphaltenes (Cf) at inlet (z=0)
        BHP = 1.e6                              # dummy value of bottom-hole P
        kgOil = 0.
        kgAsp = 0.
        C_tol = 1.e-10
        dT = 0.
        dPf = 0.
        dPg = 0.

        self.df.GOR = GOR
        self.df.P = P
        self.df.T = T
            
        # discretize pipe into nz points
        z = np.linspace(0, 1, nz)

        # z-axis step size (=1/(nz-1))
        dz = z[1] - z[0] 

        # time-step size [sec]
        dt = self.sim.t_sim/(nt - 1)

        # populate array of depth (m). required for dP model
        depth = np.linspace(L, 0, nz)

        # initial conditions
        Cf0 = np.ones(nz)     # assumes all asphaltenes are soluble at t=0
        C0 = np.zeros(nz)     # assumes no primary particles (PP) at t=0

        # initial deposition thickness along the pipe
        if self.pipe.del_Asp0 is None or self.pipe.del_DM0 is None:
            del_Asp = np.zeros(nz)
            del_DM = np.zeros(nz)
        else:
            del_Asp = self.pipe.del_Asp0
            del_DM = self.pipe.del_DM0

        #--- time-march loop ---
        flow_status = 'flowing'     # will exit time-march loop when flow_status != 'flowing'
        t = 0.
        while t < self.sim.t_sim and flow_status == 'flowing':
            t += dt
            
            # update radius profile due to restriction
            # Rz_Asp = R - del_Asp
            Rz_DM = R - del_DM
            self.pipe.Rz = Rz_DM
            self.pipe.avg_Rz = np.average(self.pipe.Rz)

            # extract thermo and transport properties from FLUT at each spatial point
            self.FLUT_extractor()

            # calculate phase velocity at each point from mass flows
            self.Phase_velo()

            #zAsp: asphaltene composition (g[Asp]/g[T]) at inlet
            self.df.zAsp = np.sum(self.df.wtFrac*self.df.yAsp, axis=1)[0]

            #c0_Asp: asphaltene concentration (kg[Asp]/m3[T]) at inlet  
            self.df.c0_Asp = self.df.zAsp*self.df.dens[0, 1]

            # mass flows (kg), oil and Asp
            mFlow_Asp = self.sim.mFlow*self.df.zAsp     # kgAsp/s
            kgOil += self.sim.mFlow*dt                  # kg Oil (cumulative; added at each dt)
            kgAsp += mFlow_Asp*dt                       # kg Asp (cumulative; added at each dt)

            # calculate scaling factor & Damkohler numbers
            self.Damkohler()

            #----- solve governing PDEs ----------------------------------------------         
            # solve for conc dissolved asphaltenes: Cf(t,z)
            Cf = self.ADEPT_Solver_Cf(Cf0, nz, dt, dz, self.df, Cf_in, isTransient)

            # solve for conc primary particles: C(t,z)
            C = self.ADEPT_Solver_C(C0, nz, dt, dz, self.df, Cf0, Cf, C_in, isTransient)
            #-------------------------------------------------------------------------

            # update initial concentrations for next time step
            C_df = Cf - self.df.Ceq     # update concentration driving force
            Cf = np.where(Cf < C_tol, 0., Cf)
            C = np.where(C < C_tol, 0., C)
            C = np.where(C_df < C_tol, 0., C)
            Cf0 = Cf    # soluble asphaltenes
            C0 = C      # primary particles

            #=========================================================
            #--- calculate deposition profile, flux, and other important outputs
            # post-processing step after PDEs are solved
            #=========================================================

            # rate of asphaltene deposition [g/cc/s]
            rD = self.df.kD*C*self.df.c0_Asp*(1 - self.df.rSR)

            # deposition flux [kg/m2/s]. might need to revisit to account for V_bl not V_cell.
            J_Asp = 0.5*rD*self.pipe.Rz

            # thickness of asphaltenes and DM deposit (m) (Assumes R >> del which is a good approx for small dt)
            del_Asp_z = dt*J_Asp/self.df.dens_Asp
            del_DM_z = dt*J_Asp/(self.df.dens[:, 2]*self.df.yAsp[:, 2])

            del_Asp = np.where(np.abs(J_Asp) > 0., del_Asp + del_Asp_z, del_Asp)
            del_DM = np.where(np.abs(J_Asp) > 0., del_DM + del_DM_z, del_DM)

            # set maximum deposit thickness to R_pipe
            del_Asp = np.where(del_Asp >= R, R, del_Asp)
            del_DM = np.where(del_DM >= R, R, del_DM)

            # deposit thickness frac (r/R)
            delf_Asp = del_Asp / R
            delf_DM = del_DM / R
            delf_max = delf_Asp.max()
            
            if delf_max > 0.999:
                t_noFlow = t
                flow_status = 'shutdown; cross-section is plugged'
                
            #-----/ depo flux (J) and thickness (del) calculation -----

            #----- calculate deposited Asp mass -----
            # element length (m)
            dL_m = L*dz
                
            # volume of deposit (m3)
            V_cyl = self.VolCylinder(dL_m, R, R - del_Asp)
                
            # mass deposit (kg)
            mass_Asp = V_cyl*self.df.dens_Asp

            #----- calculate pressure drop (dP) if at (or near) t_out -----
            # store dT, dP results from prev iteration
            # dT0 = dT
            dPf0, dPg0 = dPf, dPg
            
            # dT @ t; NOTE: to be defined
            # dT = self.energy_balance(depofluid)
            
            # dP @ t; dP=(dPf=friction loss, dPg=gravity loss)
            dPf, dPg = self.pressure_drop(self.sim.mFlow, L, R, del_DM, df)
            
            # WHP, update wellhead pressure
            WHP = BHP - (dPf + dPg)
        
            # determine if well still flowing (fluid must be able to overcome pressure drop)
            # exit loop if no flow, else update T,P profiles
            if WHP < WHP_min:
                t_noFlow = t
                flow_status = 'shutdown; P drop too large'
            else:          
                # dPf/dt, friction pressure drop wrt time       
                dPf_dt = (dPf0 - dPf) / dt

                # update T and P profiles
                # if T_prof != 'cst':
                #     self.df.T = self.energy_model(T, FLUT)
                # if P_prof != 'cst':
                #     self.df.P = self.well_model(P, FLUT)

        #--/ time loop

        # vol flows (m3), oil and Asp
        m3Oil = kgOil/self.df.dens[0, 1]
        m3Asp = kgAsp/self.df.dens_Asp[0]

        return depo_return()


class depo_return(object):
    
    def __init__(self, flow_status, t_noFlow, t_cuts, del_Asp, del_DM, J_Asp, C, Cag, Cd, mass_Asp, dP, dT):
        '''
        object containing:
            flow_status: `flowing`, `shutdown`, `plugged`
            t @ noflow:
            t: time cuts
            thickness: del_Asp(t,z), del_DM(t,z)
            flux: J_Asp(t,z)
            conc(t,z): C (primary particle), Cag (aggregates), Cd (deposit))
            mass_Asp(t,z) 
            pressure drop: dP(t)
            temperature profile: dT(t)
        '''
        self.flow_status = flow_status
        self.t_noFlow = t_noFlow
        self.t = t_cuts
        self.thickness = [del_Asp, del_DM]
        self.flux = J_Asp
        self.conc = [C, Cag, Cd]
        self.mass = mass_Asp
        self.pressure = dP
        self.temperature = dT