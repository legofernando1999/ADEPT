# import packages
import numpy as np
import json
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.interpolate import RegularGridInterpolator

# TODO: validate against ADEPT-VBA (file: `ADEPT(2023-01-17).xlsm`)
# TODO: add energy balance model to calculate change in T profile as deposit builds.
# TODO: add multiphase flow simulator to capture velocity and pressure gradients.
# TODO: fix flash calculation failures in TLUT
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

    def __init__(self, T, P, Ceq, yAsp, wtFrac, volFrac, dens, dens_Asp, SP, SP_Asp, visco, velo) -> None:
        self.T = T
        self.P = P
        self.Ceq = Ceq
        self.yAsp = yAsp
        self.wtFrac = wtFrac
        self.volFrac = volFrac
        self.dens = dens
        self.dens_Asp = dens_Asp
        self.SP = SP
        self.SP_Asp = SP_Asp
        self.visco = visco
        self.velo = velo


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
        self.KLUT = KLUT
        self.mix_phase = mix_phase

        file_tuple = self.TLUT_loader(file)
        self.TLUT = file_tuple[0]
        self.prop_label = file_tuple[1]
        self.prop_unit = file_tuple[2]
        self.GOR_range = file_tuple[3]
        self.P_range = file_tuple[4]
        self.T_range = file_tuple[5]

    def TLUT_loader(self, file):
        '''
        TLUT: Thermodynamic Lookup Table, 4D array (Prop, GOR, P, T)
            Ceq: asphaltene solubility (gAsp[L1]/gAsp[T])
            wtFrac[V, L1, L2]: phase frac, mass (g[k]/g)
            volFrac[V, L1, L2]: phase frac, volume (m3[k]/m3)
            dens[V, L1, L2, Asp]: phase density (kg/m3)
            visco[V, L1, L2, Asp]: phase viscosity (cP)
            SP[V, L1, L2, Asp]: solubility parameter (MPa^0.5)
            yAsp[V, L1, L2]: asphaltene composition (gAsp[k] / g[k])
            
        # NOTE: if it's easy to do, I want to change `TLUT` -> `F-LUT` now for `Fluid LUT` because it's more generic. 
        # viscosity and velocity, for example, are transport properties, not thermodynamic properties.
        '''
        prop_dict = json.load(file) if type(file) != dict else file

        TLUT = prop_dict['prop_table']
        prop_label = prop_dict['prop_label']
        prop_unit = prop_dict['prop_unit']
        GOR_range = np.array(prop_dict['GOR'])
        P_range = np.array(prop_dict['P'])
        T_range = np.array(prop_dict['T'])

        return (TLUT, prop_label, prop_unit, GOR_range, P_range, T_range)

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

    def TLUT_interpolator(self, Coord: tuple, Prop: str or tuple) -> np.array:
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
            interp = RegularGridInterpolator((self.GOR_range, self.P_range, self.T_range), self.TLUT[idx])

            # points for interpolation
            pts = np.column_stack(Coord)

            return interp(pts)

        else:
            Prop_list = []
            for p in Prop:
                # get index of property 
                idx = self.prop_label.index(f'{p}')

                # define an interpolating function for the given property
                interp = RegularGridInterpolator((self.GOR_range, self.P_range, self.T_range), self.TLUT[idx])

                # points for interpolation
                pts = np.column_stack(Coord)

                Prop_list.append(interp(pts))

            return np.column_stack(Prop_list)

    def Phase_velo(self, mFlow, wtFrac: np.array, dens: np.array, R_nZ: np.array) -> np.array:
        a = Pi*R_nZ**2      # m2
        A = a.reshape(a.size, 1)
        volFlow = mFlow * wtFrac / dens   # m3/s
        return volFlow / A

    def _mix_aider(self, Frac_V, Frac_L1):
        '''for methods `mix_Dens`, `mix_Visco`, `mix_Velo`'''
        f_sum = Frac_V + Frac_L1
        f_V = Frac_V/f_sum
        f_L1 = Frac_L1/f_sum
        return np.column_stack((f_V, f_L1))

    def mix_Dens(self, df, rho_mix):
        '''
        calculate averaged total density by method: "rho_mix"
        '''      
        if rho_mix == "mass":
            # mass-averaged dens
            wtf = self._mix_aider(df.wtFrac[:, 0], df.wtFrac[:, 1])        # mass fraction
            return 1 / np.sum(wtf/df.dens[:, :2], axis=1)
        elif rho_mix == "volume":
            # vol-averaged dens
            volf = self._mix_aider(df.volFrac[:, 0], df.volFrac[:, 1])     # volume fraction
            return np.sum(df.dens[:, :2]*volf, axis=1)
        else:
            # none (total dens = L1 dens)
            return df.dens[:, 1]  

    def mix_Visco(self, df, rho, mu_mix):
        '''
        calculate averaged total viscosity by method: "mu_mix"
        '''
        if mu_mix == 'mass':
            # mass-averaged visco
            wtf = self._mix_aider(df.wtFrac[:, 0], df.wtFrac[:, 1])        # mass fraction
            return np.sum(wtf/df.visco[:, :2], axis=1)
        elif mu_mix == 'volume':
            # volume-averaged visco
            volf = self._mix_aider(df.volFrac[:, 0], df.volFrac[:, 1])     # volume fraction
            return rho*np.sum(volf*df.visco[:, :2]/df.dens[:, :2], axis=1)
        else:
            # none (total visco = L1 visco)
            return df.visco[:, 1]

    def mix_Velo(self, df, u_mix):
        '''
        calculate averaged total velocity by method: "u_mix"
        '''
        return np.sum(df.velo, axis=1) if u_mix == 'sum' else df.velo[:, 1]

    def ADEPT_Diffusivity(self, T=None, mu=None, Ds=2.e-9):
        '''
        particle diffusivity (m2/s) from Einstein-Stokes equation
        if T, mu not input, then Dm = Ds
        if T, mu are provided, then use Stokes-Einstein equation to calculate Dm
        '''
        return Ds if T is None and mu is None else kB*T/(3*Pi*mu*Ds)

    def pipe_fricFactor(self, Re, d=None):
        '''
        friction factor
        
        # TODO: implement the Colebrook-White function that I sent via discord and allow it to be called by this method.
        # `fric =f(z)` because fluid and pipe properties are not constant across `z` so we either need to solve for `fric` nZ times (or maybe update fric every 10 segments?)
        # or we use averaged fluid props across the length of the pipe. I recommend averaged fluid props and calculate one `fric` for all z.
        '''
        fric = np.empty_like(Re)
        for i in range(Re.size):
            if Re[i] < 1.e-12:
                fric[i] = 0.
            elif Re[i] < 2400:
                fric[i] = 64/Re[i]      # laminar friction factor
            elif d != None:
                fric[i] = (1.14 - 2*np.log(0.0006*0.3048/d + 21.25/Re[i]**0.9)/np.log(10))**(-2)    # Jain friction factor
            else:
                fric[i] = 0.316/Re[i]**0.25         # Blasius (turbulent) friction factor
        return fric
 
    def ADEPT_kP(self, A, T, del_SP, kP_model='default'):
        '''
        Description:  calculates the precipitation kinetics (kP) parameter in the ADEPT model
        Called by:    ADEPT_Solver
        Notes:        kP = f(T, SP)
        
        --------- INPUTS -----------------------
        A: coefficients of correlation (a1, a2, a3, a4)
        T: temperature [K]
        del_SP: diff in solubility parameter between asphaltene and solvent [MPa^0.5]
        kP_model: form of correlation ("nrb" or "T" or "T-SP")
        
        ---------- RETURN -----------------------
        kP: precipitation kinetics (kP) parameter
        '''
        if kP_model in ['nrb', 'narmi']:
            # NRB 2019
            dSP2 = del_SP**2
            lnkP = -(A[0]*np.exp(-A[1]/T) + (A[2]*np.exp(-A[3]/T)/dSP2))
        elif kP_model in ['t-sp', 'il-sp']:
            dSP2 = del_SP**2
            lnkP = np.log(A[0]) - A[1]*1000/(T*dSP2)
        else:   # kP_model = "default"
            dSP2 = 1.
            lnkP = np.log(A[0]) - A[1]*1000/(T*dSP2)

        return np.exp(lnkP)

    def ADEPT_kDiss(self, kP=None):
        '''
        kinetics of redissolution
        '''
        return 0.01 if kP is None else 0.01/kP

    def ADEPT_kAg(self, c0, A, T, del_SP, visco, kAg_model='default'):
        '''
        Description:  calculates the aggregation kinetics (kAg) parameter in the ADEPT model
        Called by:    ADEPT_Solver; on-sheet function
        Notes:        ln(kAg)=f(T,{SP},{mu})
        
        input:
            A: coefficients of correlation (a1, a2, a3, a4)
            T: temperature [K]
            del_SP: diff in solubility parameter [MPa^0.5] between asphaltene and solvent
            visco: viscosity [cP] of solvent
            kAg_model: form of correlation ("nrb" or "IL")
        
        output:
            kAg: aggregation kinetics (kAg) parameter
        '''
        if kAg_model == 'nrb':
            # NRB 2019a
            dSP2 = del_SP**2
            lnkAg = -(A[0]*np.exp(-A[1]/T) + (A[2]*np.exp(-A[3]/T)/dSP2))
        else:   # kAg_model = "default"
            # a[1] represents the collision efficiency
            RT = Rg*T
            lnkAg = np.log((RT/visco)*A[0]/11250)

        return np.exp(lnkAg)*c0

    def ADEPT_rSR(self, A, tau):
        '''
        shear removal rate
        '''
        # extract SR parameters
        tau0, k, n = A
        
        if tau0 < 1.e-12:
            tau0 = 5.       # critical shear stress at wall (Pa), default
   
        return np.where(tau > tau0, k*(tau/tau0 - 1)**n, 0.)

    def Damkohler(self, L, R, Rz, df, c0_Asp, ):
        '''
        Calculate scaling factor (ScF) & Damkohler numbers (Da, reaction parameters) at each spatial point
        Da_P:  precipitation
        Da_Ag: aggregation
        Da_D:  deposition
        df = fluid(T, P, Ceq, yAsp, wtFrac, volFrac, dens, dens_Asp, SP, SP_Asp, visco, velo)
        df: fluid object
        '''
        rho_mix = self.mix_phase[0]
        mu_mix = self.mix_phase[1]
        u_mix = self.mix_phase[2]

        # asphaltene rate parameters
        # Asp_kP = self.KLUT["Asp_kP"]
        # Asp_kAg = self.KLUT["Asp_kAg"]
        Asp_kD = self.KLUT["Asp_kD"]
        kP_param = self.KLUT["kP_param"]
        kP_model = self.KLUT["kP_model"]
        kAg_param = self.KLUT["kAg_param"]
        kAg_model = self.KLUT["kAg_model"]
        # kD_param = self.KLUT["kD_param"]
        # kD_scale = self.KLUT["kD_scale"]
        SR_param = self.KLUT["SR_param"]
        # SR_model = self.KLUT["SR_model"]

        # density and viscosity averaging (="none" for L1 only, "mass", or "volume")
        # velocity averaging (="none" for L1 only, or "sum")
        # TODO: add rho, mu, uz to the `fluid` object
        rho = self.mix_Dens(df, rho_mix)
        mu = self.mix_Visco(df, rho, mu_mix)
        uz = self.mix_Velo(df, u_mix)

        # residence time (s)
        t_res = L/uz

        # axial dispersion, Peclet number
        Dm = self.ADEPT_Diffusivity(df.T, mu)
        Re = 2*Rz*uz*rho/mu                 # Reynolds number
        Dax = np.where(Re < 2500, Dm + (Rz*uz)**2/(48*Dm), 0.4*Rz*uz)
        Pe = L*uz/Dax                       # Peclet number

        # boundary layer calculation
        fricFactor = self.pipe_fricFactor(Re)      # friction factor # TODO: pass in the `fluid` object -- not only `Re` -- so we can use other fricFactor models
        tau = 0.125*rho*fricFactor*uz**2            # shear stress at wall, Pa
        uf = np.sqrt(tau/rho)                       # friction velocity, m/s
        del_wall = mu/(rho*uf)                      # wall layer thickness, m
        del_lam = 5*del_wall                        # laminar boundary layer, m
        del_mom = 125.4*Rz*Re**(-0.875)             # momentum boundary layer, m

        # choose boundary layer
        delta = del_lam

        # ADEPT rate parameters
        del_SP = df.SP_Asp - df.SP[:, 1]

        #-- precipitation
        kP = self.ADEPT_kP(kP_param, df.T, del_SP, kP_model)

        # NOTE: I recommend making these vars {`del_SP`, `kP_param`, `kP_model`, `kAg_param`, `kAg_model`, etc.} attributes of the `fluid` object.
        df.del_SP = df.SP_Asp - df.SP[:,1]
        df.kP = self.ADEPT_kP(df)
        
        #-- redissolution
        kDiss = self.ADEPT_kDiss()

        #-- aggregation
        kAg = self.ADEPT_kAg(c0_Asp, kAg_param, df.T, del_SP, mu, kAg_model)

        #-- deposition
        kD_us = Asp_kD
        phi = Dm/(delta**2*kD_us)
        ScF = (2*delta/R)*(phi/(phi + 1))
        kD = ScF*kD_us
        # if np.sum(kD_param) < 1.e-12:
        #     kD = kD_us
        # else:
        #     kD_param = (mFlow, Rz, rho, mu)
        #     kD = self.ADEPT_kD(kD_us, df.T, kD_param, kD_scale, uz, del_mom)

        #-- shear removal rate, rSR
        rSR = self.ADEPT_rSR(SR_param, tau)

        # Damkohler numbers
        df.Da_P = kP*t_res
        df.Da_Ag = kAg*t_res
        df.Da_D = kD*(1 - rSR)*t_res

        df.rSR = rSR; df.t_res = t_res; df.Pe = Pe
        df.kP = kP; df.kAg = kAg; df.kD = kD; df.kDiss = kDiss
        df.rho = rho; df.mu = mu; df.uz = uz
        return df

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
        t_sim = self.sim.t_sim
        mFlow = self.sim.mFlow
        isTransient = self.sim.isTransient

        ID = 2*R                            # pipe diameter (m)
        V_pipe = self.VolCylinder(L, R)     # total tubing volume
        C_in = 0.                           # concentration of primary particles (C) at inlet (z=0)
        Cf_in = 1.                          # concentration of dissolved asphaltenes (Cf) at inlet (z=0)
        BHP = 1.e6                          # dummy value of bottom-hole P
        kgOil = 0.
        kgAsp = 0.
        C_tol = 1.e-10
        dT = 0.
        dPf = 0.
        dPg = 0.
            
        # discretize pipe into nz points
        z = np.linspace(0, 1, nz)

        # z-axis step size (=1/(nz-1))
        dz = z[1] - z[0] 

        # time-step size [sec]
        dt = t_sim/(nt - 1)

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
        while t < t_sim and flow_status == 'flowing':
            t += dt
            
            # update radius profile due to restriction
            # R_nZ_Asp = R - del_Asp
            R_nZ_DM = R - del_DM
            R_nZ = R_nZ_DM
            
            # extract thermo and transport properties from TLUT at each spatial point
            Coord = (GOR, P, T)
                
            Ceq = self.TLUT_prop(Coord, 'Ceq')
            yAsp = self.TLUT_prop(Coord, ('yAsp_V', 'yAsp_L1', 'yAsp_L2'))
            wtFrac = self.TLUT_prop(Coord, ('wtFrac_V', 'wtFrac_L1', 'wtFrac_L2'))
            volFrac = self.TLUT_prop(Coord, ('volFrac_V', 'volFrac_L1', 'volFrac_L2'))
            dens = self.TLUT_prop(Coord, ('dens_V', 'dens_L1', 'dens_L2'))
            dens_Asp = self.TLUT_prop(Coord, 'dens_Asp')
            SP = self.TLUT_prop(Coord, ('SP_V', 'SP_L1', 'SP_L2'))
            SP_Asp = self.TLUT_prop(Coord, 'SP_Asp')
            visco = self.TLUT_prop(Coord, ('visco_V', 'visco_L1', 'visco_L2'))

            # calculate phase velocity at each point from mass flows
            velo = self.Phase_velo(mFlow, wtFrac, dens, R_nZ)
            
            # instance of fluid object to store data
            depofluid = fluid(T, P, Ceq, yAsp, wtFrac, volFrac, dens, dens_Asp, SP, SP_Asp, visco, velo)

            #zAsp: asphaltene composition (g[Asp]/g[T]) at inlet
            zAsp = np.sum(wtFrac*yAsp, axis=1)[0]

            #c0_Asp: asphaltene concentration (kg[Asp]/m3[T]) at inlet  
            c0_Asp = zAsp*dens[0, 1]

            # mass flows (kg), oil and Asp
            mFlow_Asp = mFlow*zAsp      # kgAsp/s
            kgOil += mFlow*dt           # kg Oil (cumulative; added at each dt)
            kgAsp += mFlow_Asp*dt       # kg Asp (cumulative; added at each dt)

            # calculate scaling factor & Damkohler numbers
            depofluid = self.Damkohler(L, R, R_nZ, depofluid, c0_Asp)

            #----- solve governing PDEs ----------------------------------------------         
            # solve for conc dissolved asphaltenes: Cf(t,z)
            Cf = self.ADEPT_Solver_Cf(Cf0, nz, dt, dz, depofluid, Cf_in, isTransient)

            # solve for conc primary particles: C(t,z)
            C = self.ADEPT_Solver_C(C0, nz, dt, dz, depofluid, Cf0, Cf, C_in, isTransient)
            #-------------------------------------------------------------------------

            # update initial concentrations for next time step
            C_df = Cf - Ceq     # update concentration driving force
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
            rD = depofluid.kD*C*c0_Asp*(1 - depofluid.rSR)

            # deposition flux [kg/m2/s]. might need to revisit to account for V_bl not V_cell.
            J_Asp = 0.5*rD*R_nZ

            # thickness of asphaltenes and DM deposit (m) (Assumes R >> del which is a good approx for small dt)
            del_Asp_z = dt*J_Asp/dens_Asp
            del_DM_z = dt*J_Asp/(dens[:, 2]*yAsp[:, 2])

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
            mass_Asp = V_cyl*dens_Asp

            #----- calculate pressure drop (dP) if at (or near) t_out -----
            # store dT, dP results from prev iteration
            # dT0 = dT
            dPf0, dPg0 = dPf, dPg
            
            # dT @ t; NOTE: to be defined
            # dT = self.energy_balance(depofluid)
            
            # dP @ t; dP=(dPf=friction loss, dPg=gravity loss)
            dPf, dPg = self.pressure_drop(mFlow, L, R, del_DM, depofluid)
            
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
                #     T = self.energy_model(T, TLUT)
                # if P_prof != 'cst':
                #     P = self.well_model(P, TLUT)

        #--/ time loop

        # vol flows (m3), oil and Asp
        m3Oil = kgOil/dens[0, 1]
        m3Asp = kgAsp/dens_Asp[0]

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