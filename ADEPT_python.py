import numpy as np
from scipy.integrate import solve_ivp

class depo(object):
    '''
    depo: 
    '''
    def __init__(self, OpTime, m_tot, L, R_pipe, C0, TLUT) -> None:
        self.OpTime = OpTime 
        self.m_tot = m_tot
        self.L = L
        self.R_pipe = R_pipe
        self.C0 = C0
        self.TLUT = TLUT

    def func_kP(self, T, P, SP):
        pass

    def func_kAg(self, T, P):
        pass

    def func_kD(self, T, P):
        pass

    def func_SR_tau(self, T, P):
        pass

    def func_SR_k(self, T, P):
        pass

    def func_SR_n(self, T, P):
        pass

    def func_Velo(self, beta, dens, R_nZ):
        volFlow = (beta*self.m_Tot / dens) / 1000   # m3/s
        R = R_nZ        # m
        A = np.pi*R**2  # m2
        A = A.reshape(A.size, 1)    # reshape array to allow numpy to broadcast
        return volFlow/A   # m/s, velo
 
    def mix_Dens(self):
        pass
    
    def mix_Visco(self):
        pass

    def mix_Velo(self):
        pass

    def Scf(self, Dm, kP, kAg, R_nZ, beta, vol, dens, visco, velo, dt, T, P):
        """
        Calculate scaling factor (ScF) & Dahmkohler numbers (Da, reaction parameters) at each spatial point
        """
        Re = np.empty(R_nZ.size)
        Pe = np.empty(R_nZ.size)
        rSR = np.empty(R_nZ.size)
        # get kinetic terms
        kDiss = 0.01 / kP[0]        # Asp redissolution rate (1/s)
        kD_us = self.func_kD(T, P)  # deposition parameter (unscaled)
        for i in range(R_nZ.size):
            R = R_nZ[i]         # radius at segment i

            # calculate avg density, viscosity, velocity
            wt_Tot = np.sum(beta[i, :2])    # total mass
            vol_Tot = np.sum(vol[i, :2])    # total volume

            wtFrac = beta[i, :2] / wt_Tot         # mass fraction
            volFrac = vol[i, :2] / vol_Tot      # volume fraction
            densi = dens[i, :2]
            visci = visco[i, :2]
            veloi = velo[i, :2]
            
            # density and viscosity averaging (="none" for L1 only, "mass", or "volume")
            # velocity averaging (="none" for L1 only, or "sum")
            rho = self.mix_Dens("volume", wtFrac, volFrac, densi)
            mu = self.mix_Visco("volume", wtFrac, volFrac, densi, visci, rho)
            Uz = self.mix_Velo("sum", wtFrac, veloi)
            
            mu *= 10**(-3)      # cP to Pa.s
            rho *= 1000         # density (kg/m3)
            kD0 = kD_us[i]      
            
            Re[i] = 2*R*Uz*rho / mu       # Reynolds number
    
            # axial dispersion corresponding to laminar and turbulent flows, respectively
            Dax = Dm + (Uz**2 * R**2 / 48) / Dm if Re[i] < 2500 else 2*R*Uz / 5

            Pe[i] = Uz*self.L/Dax       # Peclet number

            # calculate non-dimensional time step (for transient simulation)
            # dt_arr[i] = dt*Uz/self.L
                
        # boundary layer calculation
        fricFactor = 0.316 / (Re**0.25)             # Blausius friction factor
        tau = (1/8) * rho * fricFactor * Uz**2      # shear stress at wall, Pa
        uf = (tau / rho)**0.5                       # friction velocity, m/s
        del_wall = (mu / rho) / uf                  # wall layer thickness, m
        del_lam = 5*del_wall                        # laminar boundary layer, m
        del_mom = 62.7*(2*R)*Re**(-0.875)           # momentum boundary layer, m
        
        # choose boundary layer
        Del = del_lam

        # scale deposition constant
        phi = (Dm / Del**2) / kD0
        Scf = (2*Del / R)*(phi / (phi + 1))
        kD = Scf*kD0
            
        # shear removal rate, rSR
        tau_0 = self.fun_SR_tau(T, P)
        kSR = self.fun_SR_k(T, P)
        nSR = self.fun_SR_n(T, P)

        for i in range(R_nZ.size):
            rSR[i] = kSR[i]*(tau[i]/tau_0[i] - 1)**nSR[i] if tau[i] > tau_0[i] else 0.
            
        # Dahmkohler numbers
        Da_P = kP*self.L / Uz
        Da_Ag = kAg*self.L / Uz
        Da_D = kD*(1 - rSR)*self.L / Uz

        return Scf, kD, kDiss, Da_P, Da_Ag, Da_D, rSR
        
    def energy_model(self, T, TLUT):
        pass

    def well_model(self, P, TLUT):
        pass

    def ADEPT_Solver_Cf(self):
        '''
        Program:      ADEPT_Solver_Cf
        Description:  solves PDE for dissolved asphaltene conc (Cf)
        Called by:    ADEPT_Solver
        Notes:        see PDE formulation below
    
    ---------- INPUTS ----------------------
        --- scalars ---
        t0:       initial time
        nT:       number of time nodes
        dt:       array of time steps. made into an array to take into account the varying Uz along tubing
        z0:       initial length
        nZ:       number of steps in length
        dz:       length step
        Cf_in:    dissolved asphaltene conc in the input stream
    
        --- vectors --- (len = num of spatial points)
        Cf_t0:    array of initial (t=0) dissolved asphaltene conc
        Da_p:     precipitation Dahmkohler number
        Ceq:      asphaltene solubility (gAsp_L1 / gAsp_Ovr)
    
        --- solver settings ---
        SS:       {0 = transient, 1 = steady-state}
        SolverType:   {"IE" = implicit euler, "EE" = explicit euler}
    
    ---------- OUTPUTS ---------------------
        Cf:       dissolved asphaltene conc (gAsp_L1 / gAsp_Ovr)
    
    ---------- NOTES -----------------------
        PDE for dissolved asphaltene conc:
        dCf/dt = -dCf/dz - Da_p.(Cf-Ceq)
        IC: Cf(t=0,z) =1    (all Asp dissolved at t=0)
        BC: Cf(t,z=0) =1    (all Asp dissolved at z=0)
        '''
        pass

    def ADEPT_Solver_C(self):
        pass

    def ADEPT_Solver(self, T_b, T_t, T_prof, P_b, P_t, P_prof, del_Asp0: np.array, del_DM0: np.array, GOR0):
        '''
        Solves asphaltene material balance

        TLUT: thermodynamic lookup table
        Ceq: asphaltene solubility (gAsp[L1]/gAsp[T])
        beta[V,L1,L2]: phase amounts (mol); beta_wt: phase amounts (wtf); beta_vol: phase amounts (volf)
        dens[V,L1,L2,Asp]: phase density (g/cc)
        vol[V,L1,L2]: phase volume (cc/mol)
        visco[V,L1,L2]: phase viscosity (cP)
        SP[V,L1,L2,Asp]: solubility parameter (MPa^0.5)

        arguments:

        return:
            depo_return object
        '''
        nZ = 129
        Dm = 2.e-9      # particle diffusivity (m2/s) from Einstein-Stokes equation
        dz = 1/(nZ - 1)

        z = np.linspace(0, self.L, nZ, dtype=np.float64)

        mT = (T_t - T_b)/self.L
        mP = (P_t - P_b)/self.L

        # T and P initial profiles
        T = mT*z + T_b
        P = mP*z + P_b

        # initial GOR
        GOR = GOR0

        # initial conditions
        Cf_t0 = np.ones(nZ)     #assumes all asphaltenes are soluble at t=0
        C_t0 = np.zeros(nZ)     #assumes no primary particles (PP) at t=0
                   
        # initial deposition thickness along the pipe
        del_Asp = del_Asp0
        del_DM = del_DM0
        
        #--- time march loop ---
        for dt in range(self.OpTime*86000):

            # extract thermo and transport properties from TLUT
            TLUT = self.read_TLUT(T, P, GOR)

            # extract beta and dens at each spatial point
            beta = TLUT['beta']
            dens = TLUT['dens']
            vol = TLUT['vol']
            visco = TLUT['visco']
            yAsp = TLUT['yAsp']

            # update radius profile due to restriction
            R_nZ_Asp = self.R_pipe - (del_Asp/100)
            R_nZ_DM = self.R_pipe - (del_DM/100)
            R_nZ = R_nZ_DM

            # calculate phase velocity at each point from mass flows
            velo = self.func_Velo(beta, dens, R_nZ)

            kP = self.func_kP(T, P, TLUT['SP'])
            kAg = self.func_kAg(T, P)

            # calculate scaling factor & Dahmkohler numbers
            Scf, kD, kDiss, Da_P, Da_Ag, Da_D, rSR = self.Scf(Dm, kP, kAg, R_nZ, beta, vol, dens, visco, velo, dt, T, P)

            #----- solve governing PDEs ----------------------------------------------         
            # solve for conc dissolved asphaltenes: Cf(t,z)
            Cf = self.ADEPT_Solver_Cf()
            
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