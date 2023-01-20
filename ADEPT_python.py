import numpy as np
import json
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.interpolate import RegularGridInterpolator

class pipe(object):
    
    def __init__(self, L: float, R: float, T: float, del_Asp0=None, del_DM0=None) -> None:
        '''
        L: length of pipe
        R: radius of pipe
        T: temperature of pipe
        del_Asp0: initial deposit thickness (m), assuming only asphaltenes deposit
        del_DM0: initial deposit thickness (m), assuming aromatics and resins also deposit
        '''
        self.L = L
        self.R = R
        self.T = T
        self.del_Asp0 = del_Asp0
        self.del_DM0 = del_DM0


class sim(object):
    
    def __init__(self, t_sim: float, mFlow: float, SS: str) -> None:
        '''
        t_sim: simulation time
        mFlow: mass flow rate
        SS: {transient, steady-state}
        '''
        self.t_sim = t_sim
        self.mFlow = mFlow
        self.SS = SS


class depo(object):

    def __init__(self, pipe: pipe, sim: sim, file: json or dict, KLUT: dict, mix_phase: list) -> None:
        '''
        file: file containing thermodynamic information about fluid, such as TLUT:
            TLUT: Thermodynamic Lookup Table, 4D array (Prop, GOR, P, T)
                Ceq: asphaltene solubility (gAsp[L1]/gAsp[T])
                beta[V, L1, L2]: phase amounts (wtf)
                dens[V, L1, L2, Asp]: phase density (kg/m3)
                vol[V, L1, L2]: phase volume (m3/mol)
                visco[V, L1, L2, Asp]: phase viscosity (cP)
                SP[V, L1, L2, Asp]: solubility parameter (MPa^0.5)
                yAsp[V, L1, L2]: asphaltene composition (gAsp[k] / g[k])
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
            dens: density mixing (="volume")
            visco: viscosity mixing (="volume")
            velocity: velocity mixing (="sum")
        '''
        prop_dict = json.load(file) if type(file) != dict else file

        self.TLUT = prop_dict['prop_table']
        self.prop_label = prop_dict['prop_label']
        self.prop_unit = prop_dict['prop_unit']
        self.GOR_range = np.array(prop_dict['GOR'])
        self.P_range = np.array(prop_dict['P'])
        self.T_range = np.array(prop_dict['T'])
        self.pipe = pipe
        self.sim = sim
        self.KLUT = KLUT
        self.mix_phase = mix_phase

    def VolCylinder(self, h, Ro, Ri=0):
        '''
        Computes volume of cylinder
        input:
            h: height
            Ro: outer radius
            Ri: inner radius
        output:
            volume of cylinder
        '''
        return np.pi*(Ro**2 - Ri**2)*h

    def TLUT_prop(self, Coord: tuple, Prop: str) -> np.array:
        # get index of property 
        idx = self.prop_label.index(f'{Prop}')

        # define an interpolating function for the given property
        interp = RegularGridInterpolator((self.GOR_range, self.P_range, self.T_range), self.TLUT[idx])

        # points for interpolation
        pts = np.column_stack((Coord[0], Coord[1], Coord[2]))

        return interp(pts)

    def phase_velo(self, beta: tuple, dens: tuple, R_nZ: np.array) -> tuple:
        mFlow = self.sim.mFlow
        A = np.pi*R_nZ**2  # m2
        volFlow_V = mFlow*beta[0]/dens[0]       # m3/s
        volFlow_L1 = mFlow*beta[1]/dens[1]      # m3/s
        volFlow_L2 = mFlow*beta[2]/dens[2]      # m3/s
        return (volFlow_V/A, volFlow_L1/A, volFlow_L2/A)    # m/s, velo

    def ADEPT_Diffusivity(self, T, mu, Ds=2.e-9):
        '''
        particle diffusivity (m2/s) from Einstein-Stokes equation
        '''
        kB = 1.3806504e-23  # Boltzmann constant
        # particle size
        if Ds <= 0.:
            Ds = 2.e-9    # diffusivity (Narmi ADEPT 2019)

        # diffusivity Dm
        return kB*T / (3*np.pi*mu*Ds) if T > 0. and mu > 0. else Ds 

    def ADEPT_fricFactor(self, Re, d=0):
        '''
        friction factor
        '''
        if Re < 1.e-12:
            return 0.
        elif Re < 2400:
            return 64/Re         # laminar friction factor
        elif d > 0:
            return (1.14 - 2*np.log(0.0006*0.3048/d + 21.25/Re**0.9)/np.log(10))**(-2)      # Jain friction factor
        else:
            return 0.316/Re**0.25       # Blausius (turbulent) friction factor
 
    def ADEPT_kP(self, A, T, del_SP, eqn='default'):
        '''
        Description:  calculates the precipitation kinetics (kP) parameter in the ADEPT model
        Called by:    ADEPT_Solver
        Notes:        kP = f(T, SP)
        
        --------- INPUTS -----------------------
        A:            coefficients of correlation (a1, a2, a3, a4)
        T:            temperature [K]
        del_SP:       diff in solubility parameter between asphaltene and solvent [MPa^0.5]
        eqn:      form of correlation ("nrb" or "T" or "T-SP")
        
        ---------- RETURN -----------------------
        kP:        precipitation kinetics (kP) parameter
        '''
        if eqn in ['nrb', 'narmi']:
            # Narmadha Rajan Babu (thesis, 2019)
            dSP2 = del_SP**2
            lnkP = -(A[0]*np.exp(-A[1]/T) + (A[2]*np.exp(-A[3]/T)/dSP2))
        elif eqn in ['t-sp', 'il-sp']:
            dSP2 = del_SP**2
            lnkP = np.log(A[0]) - A[1]*1000/(T*dSP2)
        else:   # eqn = "default"
            dSP2 = 1.
            lnkP = np.log(A[0]) - A[1]*1000/(T*dSP2)

        return np.exp(lnkP)

    def ADEPT_kAg(self, A, T, del_SP, visco, eqn='default'):
        '''
        Description:  calculates the aggregation kinetics (kAg) parameter in the ADEPT model
        Called by:    ADEPT_Solver; on-sheet function
        Notes:        ln(kAg)=f(T,{SP},{mu})
        
        --------- INPUTS -----------------------
        A:            coefficients of correlation (a1, a2, a3, a4)
        T:            temperature [K]
        del_SP:       diff in solubility parameter [MPa^0.5] between asphaltene and solvent
        visco:        viscosity [cP] of solvent
        eqn:      form of correlation ("nrb" or "IL")
        
        --------- RETURN -----------------------
        kAg:       aggregation kinetics (kAg) parameter
        '''
        if eqn == 'nrb':
            # Narmadha Rajan Babu (thesis, 2019)
            dSP2 = del_SP**2
            lnkAg = -(A[0]*np.exp(-A[1]/T) + (A[2]*np.exp(-A[3]/T)/dSP2))
        else:   # eqn = "default"
            # a[1] represents the collision efficiency
            RT = 8.314*T
            lnkAg = np.log((0.0666667/750)*(RT/visco)*A[0])

        return np.exp(lnkAg)*self.sim.C0

    def ADEPT_kDiss(self, kP):
        '''
        kinetics of redissolution
        '''
        return 0.01/kP if kP > 0 else 0.01

    def ADEPT_rSR(self, param, tau):
        '''
        shear removal rate
        '''
        # extract SR parameters
        tau0 = param[0] 
        k = param[1]
        n = param[2]
        
        if tau0 == 0:
            tau0 = 5.       # critical shear stress at wall (Pa), default
   
        return k*(tau/tau0 - 1)**n if tau > tau0 else 0.

    def mix_Dens(self, wtFrac, volFrac, dens, AvgRule='volume'):
        '''
        calculate averaged total viscosity by method: "AvgRule"
        '''
        pass
    
    def mix_Visco(self):
        pass

    def mix_Velo(self):
        pass

    def Damkohler(self, Rz, T, P, beta, vol, dens, visco, SP, velo, kD_us):
        '''
        Calculate scaling factor (ScF) & Dahmkohler numbers (Da, reaction parameters) at each spatial point
        Da_P:  precipitation
        Da_Ag: aggregation
        Da_D:  deposition
        '''

        del_SP = SP_Asp - SP_L1

        # average density, viscosity, velocity
        beta_sum = beta_V + beta_L1     # total mass
        vol_sum = vol_V + vol_L1        # total volume

        beta_V = beta_V/beta_sum        # mass fraction
        beta_L1 = beta_L1/beta_sum
        volf_V = vol_V/vol_sum          # volume fraction
        volf_L1 = vol_L1/vol_sum

        # density and viscosity averaging (="none" for L1 only, "mass", or "volume")
        # velocity averaging (="none" for L1 only, or "sum")
        rho = self.mix_Dens(beta_V, beta_L1, volf_V, volf_L1, dens_V, dens_L1, rho_mix)
        mu = self.mix_Visco(beta_V, beta_L1, volf_V, volf_L1, dens_V, dens_L1, visco_V, visco_L1, rho, mu_mix)
        uz = self.mix_Velo(beta_V, beta_L1, velo_V, velo_L1, u_mix)

        # residence time (s)
        t_res = L/uz

        # axial dispersion, Peclet number
        Dm = self.ADEPT_Diffusivity(T, mu)
        Re = 2*Rz*uz*rho/mu                 # Reynolds number
        Dax = np.where(Re < 2500, Dm + (Rz*uz)**2/(48*Dm), 0.4*Rz*uz)
        Pe = L*uz/Dax                       # Peclet number

        # boundary layer calculation
        fricFactor = self.ADEPT_fricFactor(Re)      # friction factor
        tau = 0.125*rho*fricFactor*uz**2            # shear stress at wall, Pa
        uf = np.sqrt(tau/rho)                       # friction velocity, m/s
        del_wall = mu/(rho*uf)                      # wall layer thickness, m
        del_lam = 5*del_wall                        # laminar boundary layer, m
        del_mom = 125.4*Rz*Re**(-0.875)             # momentum boundary layer, m

        # choose boundary layer
        delta = del_lam

        # ADEPT rate parameters
        #-- precipitation
        kP = self.ADEPT_kP(kP_param, T, del_SP, kP_model)

        #-- redissolution
        kDiss = self.ADEPT_kDiss(0)

        #-- aggregation
        kAg = self.ADEPT_kAg(kAg_param, T, del_SP, mu*1000, kAg_model)*c0_Asp

        #-- deposition
        if np.sum(kD_param) < 1.e-12:
            kD = kD_us
        else:
            kD_param = (mFlow, R, rho, mu)
            kD = self.ADEPT_kD(kD_us, T, kD_param, kD_scale, uz, del_mom)
            # phi = Dm/(delta**2*kD_us)
            # ScF = (2*delta/R)*(phi/(phi + 1))
            # kD = ScF*kD_us

        #-- shear removal rate, rSR
        rSR = self.ADEPT_rSR(SR_param, tau)

        # Damkohler numbers
        Da_P = kP*t_res
        Da_Ag = kAg*t_res
        Da_D = kD*(1 - rSR)*t_res

        return kP, kAg, kD, kDiss, Da_P, Da_Ag, Da_D, rSR, Pe

    def ADEPT_Solver_Cf(self, u0, m, k, h, V, F, BC, SS='transient'):
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
            SS: transient or steady-state (dCf/dt = 0)
        output
            u: Cf at (j+1)st step in t
        '''
        if SS == 'transient':
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

        elif SS == 'steady-state':
            u = u0

        return u

    def ADEPT_Solver_C(self, u0, m, k, h, A1, A2, A3, A4, A5, V0, V, F, BC, SS='transient'):
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
            SS: transient or steady-state (dCf/dt = 0)
        output
            u: C at (j+1)st step in t
        '''
        if SS == 'transient':
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

        elif SS == 'steady-state':
            u = u0

        return u

    def energy_model(self, T, TLUT):
        pass

    def well_model(self, P, TLUT):
        pass

    def ADEPT_Solver(self, T: np.array, P: np.array, GOR: np.array, nz, nt):
        '''
        Solves asphaltene material balance
        
        Notes:
            1.- Should we choose the time step according to the characteristic time of formation of deposit?
            2.- Currently using a Crank-Nicolson scheme to solve PDEs

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
        SS = self.sim.SS

        rho_mix = self.mix_phase[0]
        mu_mix = self.mix_phase[1]
        u_mix = self.mix_phase[2]

        # asphaltene rate parameters
        Asp_kP = self.KLUT["Asp_kP"]
        Asp_kAg = self.KLUT["Asp_kAg"]
        Asp_kD = self.KLUT["Asp_kD"]
        kP_param = self.KLUT["kP_param"]
        kP_model = self.KLUT["kP_model"]
        kAg_param = self.KLUT["kAg_param"]
        kAg_model = self.KLUT["kAg_model"]
        kD_param = self.KLUT["kD_param"]
        kD_scale = self.KLUT["kD_scale"]
        SR_param = self.KLUT["SR_param"]
        SR_model = self.KLUT["SR_model"]

        ID = 2*R                            # pipe diameter (m)
        V_pipe = self.VolCylinder(L, R)     # total tubing volume
        C_in = 0.                           # concentration of primary particles (C) at inlet (z=0)
        Cf_in = 1.                          # concentration of dissolved asphaltenes (Cf) at inlet (z=0)
        BHP = 1.e6                         # dummy value of bottom-hole P
        kgOil = 0.
        kgAsp = 0.
        C_tol = 1.e-10

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

        # xp = [0, 1]
        # kD_us = np.interp(z, xp, fp=[Asp_kD[0], Asp_kD[-1]])

        #--- time march loop ---
        for _ in range(0, t_sim, dt):

            # extract thermo and transport properties from TLUT at each spatial point
            Coord = (GOR, P, T)

            Ceq = self.TLUT_prop(Coord, 'Ceq')
            yAsp_V = self.TLUT_prop(Coord, 'yAsp_V')
            yAsp_L1 = self.TLUT_prop(Coord, 'yAsp_L1')
            yAsp_L2 = self.TLUT_prop(Coord, 'yAsp_L2')
            beta_V = self.TLUT_prop(Coord, 'wtFrac_V')
            beta_L1 = self.TLUT_prop(Coord, 'wtFrac_L1')
            beta_L2 = self.TLUT_prop(Coord, 'wtFrac_L2')
            vol_V = self.TLUT_prop(Coord, 'volFrac_V')
            vol_L1 = self.TLUT_prop(Coord, 'volFrac_L1')
            vol_L2 = self.TLUT_prop(Coord, 'volFrac_L2')
            dens_V = self.TLUT_prop(Coord, 'dens_V')
            dens_L1 = self.TLUT_prop(Coord, 'dens_L1')
            dens_L2 = self.TLUT_prop(Coord, 'dens_L2')
            dens_Asp = self.TLUT_prop(Coord, 'dens_Asp')
            SP_V = self.TLUT_prop(Coord, 'SP_V')
            SP_L1 = self.TLUT_prop(Coord, 'SP_L1')
            SP_L2 = self.TLUT_prop(Coord, 'SP_L2')
            SP_Asp = self.TLUT_prop(Coord, 'SP_Asp')
            visco_V = self.TLUT_prop(Coord, 'visco_V')
            visco_L1 = self.TLUT_prop(Coord, 'visco_L1')
            visco_L2 = self.TLUT_prop(Coord, 'visco_L2')

            #zAsp: asphaltene composition (g[Asp]/g[T]) at inlet
            zAsp = beta_V*yAsp_V + beta_L1*yAsp_L1 + beta_L2*yAsp_L2

            #c0_Asp: asphaltene concentration (kg[Asp]/m3[T]) at inlet  
            c0_Asp = zAsp[0]*dens_L1[0]

            # mass flows (kg), oil and Asp
            mFlow_Asp = mFlow*zAsp[0]   # kgAsp/s
            kgOil += mFlow*dt           # kg oil (for full t_sim)
            kgAsp += mFlow_Asp*dt       # kgAsp (for full t_sim)

            # update radius profile due to restriction
            R_nZ_Asp = R - del_Asp
            R_nZ_DM = R - del_DM
            R_nZ = R_nZ_DM

            # calculate phase velocity at each point from mass flows
            beta = (beta_V, beta_L1, beta_L2)
            dens = (dens_V, dens_L1, dens_L2)
            velo = self.phase_velo(beta, dens, R_nZ)

            # calculate scaling factor & Dahmkohler numbers
            vol = (vol_V, vol_L1, vol_L2)
            visco = (visco_V, visco_L1, visco_L2)
            SP = (SP_V, SP_L1, SP_L2)
            kP, kAg, kD, kDiss, Da_P, Da_Ag, Da_D, rSR, Pe = self.Damkohler(self, R_nZ, T, P, beta, vol, dens, visco, SP, velo, kD_us)

            #----- solve governing PDEs ----------------------------------------------         
            # solve for conc dissolved asphaltenes: Cf(t,z)
            Cf = self.ADEPT_Solver_Cf(Cf0, nz, dt, dz, Da_P, Ceq, Cf_in, SS)

            # solve for conc primary particles: C(t,z)
            C = self.ADEPT_Solver_C(C0, nz, dt, dz, Pe, Da_Ag, Da_D, Da_P, kDiss, Cf0, Cf, Ceq, C_in, SS)
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
            rD = kD*C*c0_Asp*(1 - rSR)

            # deposition flux [kg/m2/s]. might need to revisit to account for V_bl not V_cell.
            J_Asp = 0.5*rD*R_nZ

            # thickness of asphaltenes and DM deposit (Assumes R >> del which is a good approx for small dt)
            del_Asp_z = dt*J_Asp/dens_Asp
            del_DM_z = dt*J_Asp/(dens_L2*yAsp_L2)

            del_Asp = np.where(np.abs(J_Asp) > 0., del_Asp + del_Asp_z, del_Asp)
            del_DM = np.where(np.abs(J_Asp) > 0., del_DM + del_DM_z, del_DM)

            # set maximum deposit thickness to R_pipe
            del_Asp = np.where(del_Asp < 1.e-8, 0., del_Asp)
            del_Asp = np.where(del_Asp >= R, R, del_Asp)

            del_DM = np.where(del_DM < 1.e-8, 0., del_DM)
            del_DM = np.where(del_DM >= R, R, del_DM)

            #-----/ depo flux (J) and thickness (del) calculation -----

            #----- calculate deposited Asp mass -----
            # element length (m)
            dL_m = L*dz
                
            # volume of deposit (m3)
            V_cyl = self.VolCylinder(dL_m, R, R - del_Asp)
                
            # mass deposit (kg)
            mass_Asp = V_cyl*dens_Asp

            #----- calculate pressure drop (dP) if at (or near) t_out -----
            Set dict_dP = New Scripting.Dictionary
            dict_dP("depth_m") = depth_m: dict_dP("P_bh") = P_bh: dict_dP("Units") = dict_TLUT("Units")
            dict_dP("beta") = beta: dict_dP("vol") = vol: dict_dP("dens") = dens: dict_dP("SP") = SP: dict_dP("visco") = visco: dict_dP("volFlow") = volFlow
            dict_dP("rho") = dens_Z: dict_dP("mu") = visco_Z: dict_dP("u") = velo_Z:
            dict_dP("N_Re") = Re: dict_dP("N_Pe") = Pe
            
            'dP/dt(t=0) (enter only at itr=2 to calculate dp/dt using forward diff)
            If itr = 2 Then
                dP = DepoModel_dP(mFlow, L, R, P_bh, delf_DM, dict_dP)
            End If
            
            'dP(t~=t_out)
            If t_new + dt0 >= T_out And t_new <> T_out Then
                dP = DepoModel_dP(mFlow, L, R, P_bh, delf_DM, dict_dP)
            End If

            'dP(t=t_out) and dP/dt(t=t_out)
            If t_new = T_out Then
                dP = DepoModel_dP(mFlow, L, R, P_bh, delf_DM, dict_dP)
            End If
            
            'extract friction pressure drop, flow status (dP [1]=friction dP; [2]=grav dP; [3]=total (f+g) dP; [4]=flow status)
            dPf = dP(1)             'friction dP
            dP_flowStatus = dP(4)   'flow status
            
            'dP/dt, pressure drop wrt time
            If dP_flowStatus = "flowing" Then
                dPf_arr(1) = dPf_arr(2)
                dPf_arr(2) = dPf
                If dT > 0 Then
                    dPdT = (dPf_arr(2) - dPf_arr(1)) / dT * 86400
                End If
            End If
            
            'exit loop if well not flowing
            If dP_flowStatus <> "flowing" Then
                If t_noFlow = 0 Then t_noFlow = T_out / 86400
                'GoTo ExitLoop
            End If
            '-----/ pressure drop (dP) calculation -----

            # update T and P profiles
            # if T_prof != 'cst':
            #     T = self.energy_model(T, TLUT)
            # if P_prof != 'cst':
            #     P = self.well_model(P, TLUT)

        #--/ time loop

        # vol flows (m3), oil and Asp
        m3Oil = kgOil/dens_L1[0]
        m3Asp = kgAsp/dens_Asp[0]

        return depo_return()


class depo_return(object):
    
    def __init__(self, status, noflow, t_cuts, del_Asp, del_DM, B_Asp, B_DM, J_Asp, C, Cag, Cd, mass_Asp, dP, dT):
        '''
        object containing:
            flow_status: `flowing`, `shutdown`, `plugged`
            t @ noflow:
            t: time cuts
            thickness: del_Asp(t,z), del_DM(t,z); 
            blockage: B_Asp(t,z), B_DM(t,z);
            flux: J_Asp(t,z); 
            conc(t,z): C (primary particle), Cag (aggregates), Cd (deposit))
            mass_Asp(t,z) 
            pressure drop: dP(t)
            temperature profile: dT(t)
        '''
        self.flow_status = status
        self.noflow = noflow
        self.t = t_cuts
        self.thickness = [del_Asp, del_DM]
        self.blockage = [B_Asp, B_DM]
        self.flux = J_Asp
        self.conc = [C, Cag, Cd]
        self.mass = mass_Asp
        self.pressure = dP
        self.temperature = dT