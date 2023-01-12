import numpy as np
from scipy.integrate import solve_ivp
import skfem as fem
from skfem.helpers import dot, grad

class pipe(object):
    
    def __init__(self, length, radius, roughness, temperature, del_Asp0: np.array, del_DM0: np.array):
        '''
        del_Asp0: initial deposit thickness (m), assuming only asphaltenes deposit.
        del_DM0: initial deposit thickness (m), assuming aromatics and resins also deposit.
        T: temperature of pipe.
        '''
        self.L = length
        self.R = radius
        self.rough = roughness
        self.T = temperature
        self.del_Asp0 = del_Asp0
        self.del_DM0 = del_DM0


class sim(object):
    
    def __init__(self, SimTime, mass_flow_rate, C0, kP_param, kAg_param, kD_param, kP_model, kAg_model, SR_param):
        self.t = SimTime    # simulation time (s)
        self.mFlow = mass_flow_rate
        self.C0 = C0
        self.kP_param = kP_param
        self.kAg_param = kAg_param
        self.kD_param = kD_param
        self.SR_param = SR_param
        self.kP_model = kP_model
        self.kAg_model = kAg_model


class depo(object):

    def __init__(self, pipe: pipe, sim: sim, TLUT: np.array, KLUT, mix_phase) -> None:
        '''
        TLUT: thermodynamic lookup table, includes
            Ceq: asphaltene solubility (gAsp[L1]/gAsp[T])
            beta[V, L1, L2]: phase amounts (mol); beta_wt: phase amounts (wtf); beta_vol: phase amounts (volf)
            dens[V, L1, L2, Asp]: phase density (g/cc)
            vol[V, L1, L2]: phase volume (cc/mol)
            visco[V, L1, L2]: phase viscosity (cP)
            SP[V, L1, L2, Asp]: solubility parameter (MPa^0.5)
            yAsp[V, L1, L2]: asphaltene composition (gAsp[k] / g[k])
        '''
        self.pipe = pipe
        self.sim = sim
        self.TLUT = TLUT
        self.KLUT = KLUT
        self.rho_mix = mix_phase[0]
        self.mu_mix = mix_phase[1]
        self.u_mix = mix_phase[2]

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

    def phase_velo(self, beta, dens, R_nZ):
        volFlow = (beta*self.sim.mFlow / dens)   # m3/s
        R = R_nZ        # m
        A = np.pi*R**2  # m2
        A = A.reshape(A.size, 1)    # reshape array to allow numpy to broadcast
        return volFlow/A   # m/s, velo
 
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

    def mix_Dens(self):
        pass
    
    def mix_Visco(self):
        pass

    def mix_Velo(self):
        pass

    def Dahmkohler(self, R_nZ, T, P, beta, vol, dens, visco, SP, velo, kD_us):
        '''
        Calculate scaling factor (ScF) & Dahmkohler numbers (Da, reaction parameters) at each spatial point
        Da_P:  precipitation
        Da_Ag: aggregation
        Da_D:  deposition
        '''
        dens_Z = np.empty_like(R_nZ)
        visco_Z = np.empty_like(R_nZ)
        velo_Z = np.empty_like(R_nZ)
        Re = np.empty_like(R_nZ)
        Pe = np.empty_like(R_nZ)
        tau = np.empty_like(R_nZ)
        kP = np.empty_like(R_nZ)
        kAg = np.empty_like(R_nZ)
        kD = np.empty_like(R_nZ)
        rSR = np.empty_like(R_nZ)
        Da_P = np.empty_like(R_nZ)
        Da_Ag = np.empty_like(R_nZ)
        Da_D = np.empty_like(R_nZ)
        dt_arr = np.empty_like(R_nZ)

        for iZ in range(R_nZ.size):
            Rz = R_nZ[iZ]     # radius at spatial point i
            Tz = T[iZ]
            Pz = P[iZ]
            del_SPi = SP[iZ, -1] - SP(iZ, 1)

            # mean density, viscosity, velocity
            beta_sum = np.sum(beta[iZ, :2])     # total mass
            vol_sum = np.sum(vol[iZ, :2])       # total volume

            betai = beta[iZ, :2]/beta_sum    # mass fraction
            volfi = vol[iZ, :2]/vol_sum      # volume fraction
            densi = dens[iZ, :2]
            SPi = SP[iZ, :2]
            viscoi = visco[iZ, :2]
            veloi = velo[iZ, :2]

            # density and viscosity averaging (="none" for L1 only, "mass", or "volume")
            # velocity averaging (="none" for L1 only, or "sum")
            rho = self.mix_Dens(betai, volfi, densi, self.rho_mix)
            mu = self.mix_Visco(betai, volfi, densi, viscoi, rho, self.mu_mix)
            uz = self.mix_Velo(betai, veloi, self.u_mix)

            # store dens, visco, velocity at current length, z (not necessary for current routine)
            dens_Z[iZ] = rho
            visco_Z[iZ] = mu
            velo_Z[iZ] = uz

            # residence time (s)
            t_res = self.pipe.L/uz

            # axial dispersion, Peclet number
            Dm = self.ADEPT_Diffusivity(Tz, mu)
            Re[iZ] = 2*Rz*uz*rho/mu                 # Reynolds number
            Dax = Dm + ((Rz*uz)**2/48)/Dm if Re[iZ] < 2500 else 2*Rz*uz/5
            Pe[iZ] = uz*self.pipe.L/Dax             # Peclet number

            # boundary layer calculation
            fricFactor = self.ADEPT_fricFactor(Re[iZ])      # friction factor
            tau[iZ] = (1/8)*rho*fricFactor*uz**2            # shear stress at wall, Pa
            uf = (tau[iZ] / rho)**0.5                       # friction velocity, m/s
            del_wall = (mu/rho) / uf                        # wall layer thickness, m
            del_lam = 5*del_wall                            # laminar boundary layer, m
            del_mom = 62.7*(2*Rz)*Re[iZ]**(-7/8)            # momentum boundary layer, m

            # choose boundary layer
            # delta = del_lam

            # ADEPT rate parameters
            #-- precipitation
            kP[iZ] = self.ADEPT_kP(self.sim.kP_param, Tz, del_SPi, self.sim.kP_model)

            #-- redissolution
            kDiss = self.ADEPT_kDiss(0)

            #-- aggregation
            kAg[iZ] = self.ADEPT_kAg(self.sim.kAg_param, Tz, del_SPi, mu*1000, self.sim.kAg_model)

            #-- deposition
            kD0 = kD_us[iZ]         # unscaled depo parameter (=0 to turn off deposition term)
            if kD0 == 0:
                kD[iZ] = 0.
            elif np.sum(self.sim.kD_param) == 0:
                kD[iZ] = kD0

            # 'TEMP (from v2.44)
            # 'scale deposition constant
            # Dm = ADEPT_Diffusivity()
            # delta = del_lam
            # phi = Dm / delta**2 / kD0
            # ScF = 2 * delta / R * (phi / (phi + 1))
            # kD(iZ) = ScF * kD0
            # '/TEMP

            #-- shear removal rate, rSR
            rSR[iZ] = self.ADEPT_rSR(self.sim.SR_param, tau[iZ])

            # Damkohler numbers
            Da_P[iZ] = kP[iZ]*t_res
            Da_Ag[iZ] = kAg[iZ]*t_res
            Da_D[iZ] = kD[iZ]*(1 - rSR[iZ])*t_res

        return kP, kAg, kD, Da_P, Da_Ag, Da_D, rSR

    def ADEPT_Solver_Cf(self, Cf0, nZ, dt, dz, Da_P, Ceq):
        Cf = np.empty(nZ)
        alpha = dt/dz

        A = np.zeros((nZ - 2, nZ - 2))
        B = np.zeros_like(A)
        
        np.fill_diagonal(A, 1 + dt*Da_P[1:nZ-1])
        np.fill_diagonal(B, Da_P[1:nZ-1])

        for i in range(nZ - 3):
            A[i, i+1] = alpha/2
            A[i+1, i] = -alpha/2
        
        v = Ceq[1:nZ-1]
        u0 = Cf0[1:nZ-1]

        C = u0 + dt*np.matmul(B, v)

        u = np.linalg.solve(A, C)

        Cf[0] = 1.  # Boundary condition
        Cf[1:nZ-1] = u
        Cf[nZ] = 0. # What value should be used?
        return Cf

    def ADEPT_Solver_C(self):
        pass

    def energy_model(self, T, TLUT):
        pass

    def well_model(self, P, TLUT):
        pass

    def ADEPT_Solver(self, BHT, WHT, BHP, WHP, GOR, nZ):
        '''
        Solves asphaltene material balance
        
        Notes:
            1.- Should we choose the time step according to the characteristic time of formation of deposit?
            2.- Use scikit-FEM to solve all PDEs

        arguments:

        return:
            depo_return object
        '''
        # z-axis step size
        dz = 1/nZ

        # time-step size [sec]
        dt = 1 

        # Temperature and Pressure initial profiles
        L = self.pipe.L
        z = np.linspace(0, L, nZ)
        xp = [0, L]
        T = np.interp(z, xp, fp=[BHT, WHT])
        P = np.interp(z, xp, fp=[BHP, WHP])

        # initial conditions
        Cf_t0 = np.ones(nZ)     # assumes all asphaltenes are soluble at t=0
        C_t0 = np.zeros(nZ)     # assumes no primary particles (PP) at t=0
                   
        # initial deposition thickness along the pipe
        del_Asp = self.pipe.del_Asp0
        del_DM = self.pipe.del_DM0

        Asp_kD = self.KLUT['Asp_kD']
        kD_us = np.interp(z, xp, fp=[Asp_kD[0], Asp_kD[-1]])

        #--- time march loop ---
        t = 0.
        while t < self.sim.t:

            # extract thermo and transport properties from TLUT
            TLUT_slice = self.read_TLUT((T, P, GOR), self.TLUT)

            # extract beta and dens at each spatial point
            Ceq = TLUT_slice['Ceq']
            beta = TLUT_slice['beta']
            dens = TLUT_slice['dens']
            vol = TLUT_slice['vol']
            visco = TLUT_slice['visco']
            SP = TLUT_slice['SP']

            # update radius profile due to restriction
            R_nZ_Asp = self.R_pipe - (del_Asp/100)
            R_nZ_DM = self.R_pipe - (del_DM/100)
            R_nZ = R_nZ_DM

            # calculate phase velocity at each point from mass flows
            velo = self.phase_velo(beta, dens, R_nZ)
            veloL1 = velo[:, 1]

            # calculate scaling factor & Dahmkohler numbers
            kP, kAg, kD, Da_P, Da_Ag, Da_D, rSR = self.Dahmkohler(self, R_nZ, T, P, beta, vol, dens, visco, SP, velo, kD_us)

            #----- solve governing PDEs ----------------------------------------------         
            # solve for conc dissolved asphaltenes: Cf(t,z)
            Cf = self.ADEPT_Solver_Cf(Cf_t0, nZ, dt, dz, Da_P, Ceq)
            
            # solve for conc primary particles: C(t,z)
            C = self.ADEPT_Solver_C()
            #-------------------------------------------------------------------------

            # calculate depo flux (J) and depo thickness (del) at each spatial point
            # rate of asphaltene deposition [g/cc/s]
            rD = kD*C*self.C0*(1 - rSR)
                
            # deposition flux [g/cm2/s]. might need to revisit to account for V_bl not V_cell.
            R_cm = R_nZ*100
            J_Asp = 0.5*rD*R_cm
                    
            # calculate thickness of asphaltenes and DM (Assumes R >> del which is a good approx for small dt)
            dens_Asp = dens[-1]
            del_Asp += J_Asp*dt / dens_Asp
            del_DM += (J_Asp*dt / dens[:, 2]) / yAsp[:, 2]

            # update initial concentrations for next time step
            Cf_t0 = Cf
            C_t0 = C
            #-----/ depo flux (J) and thickness (del) calculation -----
            
            R_pipe_cm = self.R_pipe*100
            L_cm = self.L*100
            for i in range(nZ):
                # set maximum deposit thickness to R_pipe
                if del_Asp[i] < 1.e-6:
                    del_Asp[i] = 0.
                elif del_Asp[i] >= R_pipe_cm:
                    del_Asp[i] = R_pipe_cm
            
                if del_DM[i] < 1.e-6:
                    del_DM[i] = 0
                elif del_DM[i] >= R_pipe_cm:
                    del_DM[i] = R_pipe_cm
                                
            # calculate deposited Asp mass
            mass_Asp = np.pi*(R_pipe_cm**2 - (R_pipe_cm - del_Asp)**2)*dz*L_cm*dens_Asp
            mass_Asp /= 1000    # kg

            # update T and P profiles
            if T_prof != 'cst':
                T = self.energy_model(T, TLUT)
            if P_prof != 'cst':
                P = self.well_model(P, TLUT)

            # check P profile (if dP too high, then flow stops (mark time at flow stoppage) and exit time loop)

            t += dt
        #--/ time loop
        return depo_return



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