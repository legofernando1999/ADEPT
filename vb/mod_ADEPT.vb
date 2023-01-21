'------ ABOUT ---------
' DESCRIPTION:  calculations for deposition (ADEPT) and pressure drop (dP) in the wellbore
' CALLED BY:    asphaltene deposition routines
' UI sheet:     "Asphaltenes"
' CALC sheet:   "FPc_Depo"
'----------------------

Option Explicit
Option Base 1

'wrapper to call ADEPT
Public Function Asp_DepoModel(Depo_Solver As Range, Asp_DepoParam As Range, ADEPT_RateParam As Range, Asp_FluidProps As Range, Optional Asp_RateParam As Range, Optional Asp_x0 As Range)
    
    Asp_DepoModel = DepoModel_ADEPT(Depo_Solver, Asp_DepoParam, ADEPT_RateParam, Asp_FluidProps, Asp_RateParam, Asp_x0)
          
End Function

Public Function Asp_DepoRate(rng_T As Range, rng_P As Range, Asp_Fluid As Range, ADEPT_RateParam As Range, Units As Range)

'=========================================================
' Program:      Asp_DepoRate
' Description:  scale ADEPT asp kinetic parameters
' Called by:    FP_Asp_DepoRate
' Notes:
'
'---------- INPUTS ---------------------
' rng_T:        temperature (F)
' rng_P:        pressure (psig)
' Asp_Fluid:    fluid properties along PT-trace
'   beta:       phase fractions (wt%)
'   dens:       density (g/cc)
'   SP:         solubility parameter (MPa^0.5)
'   visco:      viscosity (cP)
' ADEPT_RateParam: scaling parameters for ADEPT asp kinetic parameters
'
'---------- OUTPUTS --------------------
' out:  array(lenPT, 3)
'   kP:         precipitation rate (1/s)
'   KAg:        aggregation rate (1/s)
'   kD:         deposition rate (1/s)
'==========================================================

    On Error GoTo ErrHandle
    
    Dim i As Integer, j As Integer, k As Integer, nPhase As Integer, nPts As Integer
    Dim T As Double, P As Double, rho As Double, mu As Double, vol() As Double, vol_Tot As Double, _
        TK As Double, Pbar As Double, rho_gcc As Double, mu_cP As Double, Tunit As String, Punit As String, _
        del_L1 As Double, del_L2 As Double, del_Asp As Double, del As Double, del2 As Double
    Dim beta As Variant, dens As Variant, visco As Variant, SP As Variant
    Dim a As Double, q As Double, pI As Double, u As Double, d As Double, Re As Double, Scf As Double, mom_bl As Double, phi As Double, _
        Dm As Double, void_frac As Double, U_PB As Double, diam_Sphere As Double, Gu_PB As Double, Gu_WB As Double
    
    Dim ln_kP As Double, ln_kAg As Double, ln_kD As Double, _
        kP As Double, kAg As Double, kD As Double, kp0 As Double, Kag0 As Double, kD0 As Double, _
        flag_kP_Scaling As String, flag_kAg_Scaling As String, flag_kD_Scaling As Integer
    
    Dim out As Variant, coeff As Variant
    
    'read values from Rate_Param
    Dim var As Variant
    var = ADEPT_RateParam.Value2
    
    '-- kP scaling coefficients (a0, a1, b0, b1)
    Dim a0 As Double, a1 As Double, b0 As Double, B1 As Double, str_kP As String, coeff_kP As Variant
    a0 = var(1, 1)
    a1 = var(2, 1)
    b0 = var(3, 1)
    B1 = var(4, 1)
    str_kP = var(5, 1)
    coeff_kP = Array(a0, a1, b0, B1)
    
    '-- kAg scaling coefficients (c0, c1, d0, d1)
    Dim c0 As Double, C1 As Double, d0 As Double, d1 As Double, str_kAg As String, coeff_kAg As Variant
    c0 = var(6, 1)
    C1 = var(7, 1)
    d0 = var(8, 1)
    d1 = var(9, 1)
    str_kAg = var(10, 1)
    coeff_kAg = Array(c0, C1, d0, d1)
    
    Dim C0_Asp As Double
    C0_Asp = var(11, 1)
    flag_kD_Scaling = var(12, 1)   'flag for scaling kD
    
    '-- kinetic parameters
    kP = var(13, 1)         'precipitation kinetic parameter (1/s)
    kAg = var(14, 1)        'aggregation kinetic parameter (1/s)
    kD0 = var(15, 1)        'deposition kinetic parameter (1/s)
    
    '-- shear removal
    'Dim tau As Double, kSR As Double, nSR As Double
    'tau = var(16, 1)        'SR rate critical shear stress (Pa)
    'kSR = var(17, 1)        'SR rate multiplier (-)
    'nSR = var(18, 1)        'SR rate exponent (-)
    
    '-- flow data
    q = var(19, 1)        'flow rate (STB/Day) in wellbore
    d = var(20, 1)        'well diameter (in)
    mu = var(21, 1)       'viscosity (cP), L1
    
    'read phase fraction, density, solubility parameter, viscosity from Asp_Fluid
    beta = Asp_Fluid.Columns("D:F").Value2
    dens = Asp_Fluid.Columns("G:J").Value2
    SP = Asp_Fluid.Columns("K:N").Value2
    visco = Asp_Fluid.Columns("O:R").Value2
    
    '# phases
    nPhase = ArrayLen(beta, 2)
    ReDim vol(1 To nPhase)
    
    'dimensionalize output
    nPts = ArrayLen(beta)
    ReDim out(nPts, 3)
    
    'zTEMP!!
    'read units
    Tunit = "F"     'Units.Cells(1).Value2
    Punit = "psig"  'Units.Cells(2).Value2
    '/zTEMP!!
    
    'constants
    Dm = 0.000000002                            'diffusivity
    pI = 3.14159265358979                       'value of pi
    
    'calculate velocity in wellbore
    q = q * 0.0000018403                        'convert flow rate (STB/D -> m^3/s)
    d = UnitConvert_Length(d, "in", "m")        'convert diameter (in -> m)
    a = pI / 4 * d ^ 2                          'cross-sectional area of wellbore (m^2)
    u = q / a                                   'velocity (m/s)
    
    
    'determine scaling for kP/kAg/kD
    flag_kP_Scaling = "Y"
    If a0 = 0 And a1 = 0 And b0 = 0 And B1 = 0 Then flag_kP_Scaling = "N"
    
    flag_kAg_Scaling = "Y"
    If c0 = 0 And C1 = 0 And d0 = 0 And d1 = 0 Then flag_kAg_Scaling = "N"
            
    If flag_kD_Scaling = 0 Then flag_kD_Scaling = 1     'default is kD from wellbore (no scaling applied)
    
    For i = 1 To nPts
    
        'read T,P
        T = rng_T.Cells(i).Value2
        P = rng_P.Cells(i).Value2
        
        'read density, viscosity, solubility parameters
        rho = dens(i, 2)        'dens L1 (g/cc)
        mu = visco(i, 2)        'visco L1 (cP)
        del_L1 = SP(i, 2)       'SP L1 (MPa^0.5)
        del_L2 = SP(i, 3)       'SP L2
        del_Asp = SP(i, 4)      'SP Asp
        del = del_Asp - del_L1: del2 = del ^ 2
        
        If mu_cP = 0 Then mu_cP = mu
            
        If T > 0 And P > 0 Then
            
            'convert T, P to standard units
            TK = UnitConvert_Temp(T, Tunit, "K")
            Pbar = UnitConvert_Pressure(P, Punit, "bar")
            
            'volume-fraction averaging of density, viscosity (liquids only)
            vol_Tot = 0
            For k = 1 To nPhase
                If dens(i, k) > 0.3 Then vol(k) = beta(i, k) / dens(i, k)
                vol_Tot = vol_Tot + vol(k)
            Next k
            
            rho = 0
            mu = 0
            For k = 1 To nPhase
                rho = rho + vol(k) / vol_Tot * dens(i, k)
                mu = mu + vol(k) / vol_Tot * visco(i, k)
            Next k
            
            'calculate scaling for kP, kAg
            If flag_kP_Scaling = "Y" Then
                ln_kP = ADEPT_kP(coeff_kP, TK, del, str_kP)
                kP = Expt(ln_kP)
            End If
            If flag_kAg_Scaling = "Y" Then
                ln_kAg = ADEPT_kAg(coeff_kAg, TK, del, mu, str_kAg)
                kAg = Expt(ln_kAg)
                kAg = kAg * C0_Asp
            End If
            
            'scaling factor for kd
            If rho_gcc > 0 Then rho = rho_gcc * 1000                'oil density (kg/m3)
            If mu_cP > 0 Then mu = mu_cP * 0.001                    'oil viscosity (Pa.s)
            Re = u * d * rho / mu                                   'Reynolds number
            mom_bl = 62.7 * d * Re ^ (-7 / 8)                       'momentum boundary layer
            Gu_WB = 1 '0.01 * ((RHO / mu) * (U ^ 3 / (2 * D))) ^ 0.5    'gradient of velocity for wellbore
            
            'scale deposition constant depending on EXPERIMENTAL datatype
            If flag_kD_Scaling = 1 Then
                Scf = 1
                kD = Scf * kD0 * Gu_WB                              'deposition constant, wellbore data as input
            ElseIf flag_kD_Scaling = 2 Then
                'values for packed bed column
                void_frac = 0.4                                     'void fraction of packed bed column
                U_PB = 0.00002735                                   'axial velocity of packed bed column (m/s)
                diam_Sphere = 0.00238                               'diameter of spheres in packed bed column (m)
                Gu_PB = 13.4 * ((1 - void_frac) / void_frac) * U_PB / diam_Sphere     'gradient of velocity for packed bed
        
                phi = Dm / (mom_bl ^ 2 * kD0 * Gu_PB)               'dimensionless number for packed bed
                Scf = (4 * mom_bl / d) * (phi / (phi + 1))          'scaling factor for packed bed
                kD = Scf * kD0 * Gu_WB                              'deposition constant, packed bed data as input
            ElseIf flag_kD_Scaling = 3 Then
                'values for capillary tube
                phi = Dm / (mom_bl ^ 2 * kD0)                       'dimensionless number for capillary tube
                Scf = (4 * mom_bl / d) * (phi / (phi + 1))          'scaling factor for capillary tube
                kD = Scf * kD0                                      'deposition constant, capillary data as input
            End If
            
            'store values
            out(i, 1) = kP
            out(i, 2) = kAg
            out(i, 3) = kD
        
        Else
        
            'store values
            out(i, 1) = "" 'kP
            out(i, 2) = "" 'kAg
            out(i, 3) = "" 'kD
        End If
    
    Next i
    
    'return
    Asp_DepoRate = out

ErrHandle:

End Function

'---------------------------------------------------------------------
'--- IL CODES ---
'---------------------------------------------------------------------

Public Function DepoModel_ADEPT(Depo_Solver As Range, Asp_DepoParam As Range, ADEPT_RateParam As Range, Asp_FluidProps As Range, _
        Optional Asp_RateParam As Range, Optional Asp_x0 As Range)
 
'=========================================================
' Program:      DepoModel_ADEPT
' Description:  ADEPT model for asphaltene deposition in wellbore
' Called by:    FP_DepoModel_ADEPT
' Notes:        wrapper for ADEPT_Solver routine
'
'---------- INPUTS ---------------------------
' Depo_Solver
'   (1):        SolverType {IE = implicit eulur, EE = explicit eulur}
'   (2):        Steady State? {0 = transient, 1 = steady state}
'   (3):        nTimeCuts = # of time cuts
' Asp_DepoParam
'   (1):        t = time of operation (days)
'   (2):        q = production rate (STB/day)
'   (3):        L = wellbore length (ft)
'   (4):        D = wellbore ID (in)
'   (5):        mu = oil viscosity (cP)
'   (6):        rho = oil density (g/cc)
'   (7):        rho_dm = deposited material (L2) density (g/cc)
'   (8):        rho_Asp = Asp density (g/cc)
'   (9):        yAsp = fraction Asp in deposited material (gAsp_L2 / gOvr_L2)
'   (10):       Ceq = Asp solubility (gAsp_L1 / gAsp_Ovr)
'   (11):       C0 = Asp concentration (gAsp_Ovr / cc Total)
' ADEPT_RateParam
'   (1-4):      kP scaling constants
'   (5-8):      kAg scaling constants
'   (9):        C0 = total Asp concentration (kgAsp_Ovr / m3 Total)
'   (10):       kD scaling {1 = wellbore, 2 = packed bed, 3 = capillary tube}
'   (11-13):    kP, KAg, kD (1/s) - unscaled
'   (14-16):    shear removal (SR) parameters
'   (17-19):    production data (q, ID, mu)
' Asp_FluidProps (at each point along PT-trace)
'   (1):        Ceq (gAsp_L1 / gAsp_Ovr)
'   (2-3):      Asp compo (wtfrac) in [L1, L2]
'   (4-6):      phase frac (wt%) in [V, L1, L2]
'   (7-10):     density (g/cc) in [V, L1, L2, Asp]
'   (11-14):    solubility parameter (MPa^0.5) in [V, L1, L2, Asp]
'   (15-18):    viscosity (cP) in [V, L1, L2, Asp]
' Asp_RateParam (at each point along PT-trace)
'   (1-3):      kP, KAg, kD (1/s)
' Asp_x0:       initial fracBlock (-)
'
'---------- OUTPUTS ---------------------------
' out:          ADEPT_Solver
'
'==========================================================
     
    Dim i As Integer, j As Integer, k As Integer
    Dim C As Variant, c0 As Double, nZ As Integer, nZ_max As Integer
    
    'extract Solver Settings
    Dim SolverSettings As Variant: SolverSettings = Depo_Solver.Value2
    
    'extract Asp_DepoParam
    Dim T As Double, q As Double, L As Double, R As Double, d As Double, _
        mu As Double, rho As Double, rho_dm As Double, rho_Asp As Double, yAsp_L2 As Double, Ceq_L1 As Double
    T = Asp_DepoParam.Cells(1).Value2
    q = Asp_DepoParam.Cells(2).Value2
    L = Asp_DepoParam.Cells(3).Value2
    d = Asp_DepoParam.Cells(4).Value2: R = d / 2
    mu = Asp_DepoParam.Cells(5).Value2
    rho = Asp_DepoParam.Cells(6).Value2
    rho_dm = Asp_DepoParam.Cells(7).Value2
    rho_Asp = Asp_DepoParam.Cells(8).Value2
    yAsp_L2 = Asp_DepoParam.Cells(9).Value2
    Ceq_L1 = Asp_DepoParam.Cells(10).Value2
    c0 = Asp_DepoParam.Cells(11).Value2
    
    'extract ADEPT rate parameters
    Dim kP As Double, kAg As Double, kD As Double, _
        SR_tau As Double, SR_k As Double, SR_n As Double
    kP = ADEPT_RateParam.Cells(11).Value2       'kP: Asp precip rate
    kAg = ADEPT_RateParam.Cells(12).Value2      'kAg: Asp aggregation rate
    kD = ADEPT_RateParam.Cells(13).Value2       'kD: Asp deposition rate
    SR_tau = ADEPT_RateParam.Cells(14).Value2   'tau_SR: SR rate critical shear stress at the wall (Pa)
    SR_k = ADEPT_RateParam.Cells(15).Value2     'kSR: SR rate multiplier (-)
    SR_n = ADEPT_RateParam.Cells(16).Value2     'nSR: SR rate exponent (-)
    
    'extract fluid properties
    Dim Asp_Ceq As Variant, Asp_beta As Variant, Asp_dens As Variant, Asp_SP As Variant, Asp_visco As Variant, Asp_yAsp As Variant
    Asp_Ceq = Asp_FluidProps.Columns("A").Value2
    Asp_yAsp = Asp_FluidProps.Columns("B:C").Value2
    Asp_beta = Asp_FluidProps.Columns("D:F").Value2
    Asp_dens = Asp_FluidProps.Columns("G:J").Value2
    Asp_SP = Asp_FluidProps.Columns("K:N").Value2
    Asp_visco = Asp_FluidProps.Columns("O:R").Value2
    
    'extract Asp rate parameters
    Dim Asp_kP As Variant, Asp_kAg As Variant, Asp_kD As Variant
    Asp_kP = Asp_RateParam.Columns("A").Value2
    Asp_kAg = Asp_RateParam.Columns("B").Value2
    Asp_kD = Asp_RateParam.Columns("C").Value2
    
    'extract initial fracBlock
    Dim Asp_fracBlock As Variant
    Asp_fracBlock = Asp_x0.Value2
    
    'max # length cuts (nZ_Max) and # non-zero length cuts (nZ)
    nZ_max = Asp_FluidProps.Rows.Count
    For i = 1 To nZ_max
        If IsEmpty(Asp_Ceq(i, 1)) = False Then
            nZ = nZ + 1
        End If
    Next i
    
    Dim Ceq() As Double, beta() As Double, dens() As Double, visco() As Double, yAsp() As Double, _
        kPv() As Double, kAgv() As Double, kDv() As Double
    ReDim Ceq(1 To nZ), beta(1 To nZ, 1 To 3), dens(1 To nZ, 1 To 4), visco(1 To nZ, 1 To 4), yAsp(1 To nZ, 1 To 3), _
        kPv(1 To nZ), kAgv(1 To nZ), kDv(1 To nZ)
    For i = 1 To nZ
    
        'thermo properties
        Ceq(i) = Asp_Ceq(i, 1)
        For j = 1 To 3
            beta(i, j) = Asp_beta(i, j)
            visco(i, j) = Asp_visco(i, j)
            dens(i, j) = Asp_dens(i, j)
        Next j
        visco(i, 4) = Asp_visco(i, 4)
        dens(i, 4) = Asp_dens(i, 4)
        For j = 1 To 2
            yAsp(i, j + 1) = Asp_yAsp(i, j)
        Next j
        
        'ADEPT asp kinetic parameters (kP, kAg, kD)
        kPv(i) = Asp_kP(i, 1)
        kAgv(i) = Asp_kAg(i, 1)
        kDv(i) = Asp_kD(i, 1)
    Next i
    
    'populate input dict
    Dim dict_ADEPT As New Scripting.Dictionary
    '---solver settings
    dict_ADEPT("SolverSettings") = SolverSettings
    '---fluid properties
    dict_ADEPT("Ceq") = Ceq
    dict_ADEPT("beta") = beta
    dict_ADEPT("dens") = dens
    dict_ADEPT("visco") = visco
    dict_ADEPT("yAsp") = yAsp
    '---shear removal (SR) rate parameters
    dict_ADEPT("SR_tau") = SR_tau: dict_ADEPT("SR_k") = SR_k: dict_ADEPT("SR_n") = SR_n:
    '---asphaltene (Asp) rate parameters
    dict_ADEPT("Asp_kP") = kPv: dict_ADEPT("Asp_kAg") = kAgv: dict_ADEPT("Asp_kD") = kDv:
    '---initial frac blocked
    dict_ADEPT("Asp_fracBlock") = Asp_fracBlock
    
    'run ADEPT to solve deposition model
    DepoModel_ADEPT = ADEPT_Solver(T, q, L, R, c0, dict_ADEPT, C)
    
End Function

Private Function ADEPT_Solver(SimTime As Double, m_Tot As Double, L As Double, R_pipe As Double, c0_gcc As Double, _
        dict_ADEPT As Scripting.Dictionary, _
        Optional Conc_PP)
    
'=========================================================
' Program:      ADEPT_Solver
' Description:  ADEPT model for asphaltene deposition in wellbore
' Called by:    DepoModel_ADEPT
' Notes:        see below
'
'---------- INPUTS ---------------------------
' --- scalars ---
' SimTime:  time of operation (s)
' m_Tot:    production rate (kg/s)
' L:        wellbore length (m)
' R_pipe:   wellbore radius (m)
' c0:       asphaltene concentration (g/cc) in L1 at inlet
'
' --- arrays --- (nRows = number of spatial points)
' dict_ADEPT:
'   Ceq:        asphaltene solubility (gAsp in L1 / gAsp in Ovr)
'   beta:       phase fractions (wt%) in [V,L1,L2]
'   visco:      viscosity (cP) in [V,L1,L2,Asp]
'   dens:       density (g/cc) in [V,L1,L2,Asp]
'   yAsp:       asphaltene composition (g/gOvr) in [V,L1,L2]
'   Asp_kP:     precipitation rate (1/s)
'   Asp_KAg:    aggregation rate (1/s)
'   Asp_kD:     deposition rate, unscaled (1/s)
'   Asp_fracBlock: initial pipe blockage
'   SR_tau:     shear removal (SR) rate critical shear stress (Pa)
'   SR_k:       SR rate multiplier (-)
'   SR_n:       SR rate exponent (-)
'
' --- solver settings ---
' SS:       {0 = transient, 1 = steady-state}
' SolverType: {"IE" = implicit euler, "EE" = explicit euler}
'
'---------- OUTPUTS ---------------------------
' out: array along length of pipe
'   C:          primary particle concentration
'   Cf-Ceq:     driving force for precipitation
'   del_Asp:    deposit thickness (in), Asp
'   del_DM:     deposit thickness (in), L2
'   B_Asp:      fraction blockage (-), Asp
'   B_DM:       fraction blockage (-), L2
'   flux_Asp:   deposition flux (g/m2/d), Asp
'   --- time cuts ---
'   {B_Asp, B_DM, flux_Asp}
'   --- dP(t) at each time cut ---
'   t:          time at cut (day)
'   dP(t):      pressure drop (psi)
'   dP/dt(t):   pressure drop wrt time (psi/d)
'   flux(t):    deposition flux (g/m2/d)
'
'---------- NOTES ----------------------------
' Author: Mohammed IL Abutaqiya
' Objective: perform time march, solve PDEs for dissolved
'   Asp conc (Cf) and primary particle conc (C) at each iter
'
' 1. routine includes terms for asphaltene kinetics [kP,kAg,kD]
'   and shear removal [tau0, kSR, nSR]
' 2. pressure drop (dP) calculation is performed after each
'   time march, dP=f(L, del=depo thickness)
' 3. velocity updates at each time marching step to include
'   pipe restriction due to deposition, u=f(del)
' 4. thermo property values [Ceq, dens, visco, etc.] should update
'   to reflect changes in dP after each time march (not implemented in current routine!!)
'==========================================================
        
    Dim i As Integer, j As Integer, k As Integer, nP As Integer, nPh As Integer, nRow As Integer, nCol As Integer, nT_solve As Integer, iT_out As Integer, nT_out As Integer, dt_out As Double, T_out As Double, _
        sumY As Double, sumR As Double, pI As Double, val As Double, val1 As Double, val2 As Double
    Dim Z As Double, Z0 As Double, zf As Double, dz As Double, iZ As Integer, nZ As Integer, z_nP() As Double, nZ_max As Integer, _
        m As Double, Y As Double, y1 As Double, y2 As Double, x As Double, x1 As Double, x2 As Double, found As Integer, _
        T As Double, dti As Double, T0 As Double, tf As Double, dt As Double, dt_arr() As Double, iT As Integer, nT As Integer, _
        dt1 As Double, dt2 As Double, dt3 As Double, _
        R0 As Double, L0 As Double, q As Double, _
        id As Double, R_cm As Double, L_cm As Double, a As Double, Re() As Double, Pe() As Double, Dax As Double
    Dim kDiss As Double, kD0 As Double, Dm As Double, phi As Double, Scf As Double
    Dim c0 As Double, Cf() As Double, C() As Double, Cf_solve() As Double, C_solve() As Double, Ceq() As Double, beta() As Double, vol() As Double, dens() As Double, dens_Asp() As Double, visco() As Double, yAsp() As Double, _
        velo_L1() As Double, Q_VLL() As Double, velo() As Double, volFlow() As Double
    Dim kP() As Double, kAg() As Double, kD() As Double, kD_us() As Double, Da_P() As Double, Da_Ag() As Double, Da_D() As Double, tau_arr() As Double, _
        del_Asp() As Double, del_DM() As Double, del_Asp0() As Double, del_DM0() As Double, _
        J_Asp() As Double, mass_Asp() As Double, y_Asp As Double, vol_Tot As Double, wt_Tot As Double, mass_Tot As Double
    Dim Uz As Double, rho As Double, mu As Double, rho_dm As Double, rho_Asp As Double, _
        V_Tube As Double, V_TubeAsp As Double, V_TubeDM As Double, Vfrac_Asp As Double, Vfrac_DM As Double, dz_m As Double, dr_m As Double, _
        mass_Asp_Tot As Double, mass_Asp_Max As Double, mass_Asp_Frac As Double, max_flux As Double
    Dim det_Asp As Double, det_DM As Double, rD As Double
    Dim del_Asp_norm() As Double, del_DM_norm() As Double
    Dim del_lam As Double, del_wall As Double, del_mom As Double, del As Double, _
        fricFactor As Double, tau As Double, uf As Double, kinVisco As Double
    Dim Cf_in As Double, Cf_t0() As Double, C_in As Double, C_t0() As Double
    Dim dPdT As Double, dP() As Variant, dPf As Double, dPf_arr(1 To 2) As Double, dP_flowStatus As String, P_bh As Double, dPf_old As Double
    Dim SimPercent As Integer, ti_Day As Double, t_noFlow As Double, str_msg As String
            
    'extract fluid properties
    Dim Ceq_wtf() As Double, beta_wt() As Double, dens_gcc() As Double, visco_cP() As Double, yAsp_wtf() As Double, _
        Asp_kP() As Double, Asp_kAg() As Double, Asp_kD() As Double, Asp_fracBlock() As Variant, _
        tau_0 As Double, kSR As Double, nSR As Double, _
        SolverSettings() As Variant, SS As Integer, SolverType As String, nTimeCuts As Integer, nTC_Vars As Integer
    Ceq_wtf = dict_ADEPT("Ceq")
    beta_wt = dict_ADEPT("beta")
    dens_gcc = dict_ADEPT("dens")
    visco_cP = dict_ADEPT("visco")
    yAsp_wtf = dict_ADEPT("yAsp")
    'extract SR rate parameters
    tau_0 = dict_ADEPT("SR_tau"): kSR = dict_ADEPT("SR_k"): nSR = dict_ADEPT("SR_n")
    'extract Asp rate parameters
    Asp_kP = dict_ADEPT("Asp_kP"): Asp_kAg = dict_ADEPT("Asp_kAg"): Asp_kD = dict_ADEPT("Asp_kD")
    'extract initial fraction blocked
    Asp_fracBlock = dict_ADEPT("Asp_fracBlock")
    'extract sim settings
    SolverSettings = dict_ADEPT("SolverSettings")
    SolverType = SolverSettings(1, 1): SS = SolverSettings(2, 1): nTimeCuts = SolverSettings(3, 1)
    
    'length of input Ceq array
    nP = ArrayLen(Ceq_wtf)
    ReDim z_nP(1 To nP)
    
    pI = 3.14159265358979:
    c0 = c0_gcc         'conc Asp (gAsp/cc oil)
    C_in = 0            'conc primary particles (C) at inlet (z=0)
    Cf_in = 1           'conc dissolved asphaltenes (Cf) at inlet (z=0)
    id = R_pipe * 2     'pipe diameter (m)
    P_bh = 10 ^ 6       'dummy value of bottom-hole P
    
    'handle time = 0
    'If SimTime = 0 Then SimTime = 0.01
    
    'discretize pipe into nP points
    For i = 1 To nP
        z_nP(i) = ((i - 1) / (nP - 1))
    Next i
    
    'spatial steps
    nZ_max = 500
    nZ = nP
    Z0 = 0
    zf = 1
    dz = (zf - Z0) / (nZ - 1)
    
    'extract fluid properties and rate parameters (& interpolate) along spatial steps
    nPh = ArrayLen(beta_wt, 2)
    ReDim Ceq(1 To nZ), beta(1 To nZ, 1 To nPh), dens(1 To nZ, 1 To nPh), dens_Asp(1 To nZ), vol(1 To nZ, 1 To nPh), _
        visco(1 To nZ, 1 To nPh), yAsp(1 To nZ, 1 To nPh)
    ReDim kP(1 To nZ), kAg(1 To nZ), kD(1 To nZ), kD_us(1 To nZ)
    ReDim velo_L1(1 To nZ), Q_VLL(1 To nZ, 1 To nPh), velo(1 To nZ, 1 To nPh), volFlow(1 To nZ, 1 To nPh), _
        Da_P(1 To nZ), Da_Ag(1 To nZ), Da_D(1 To nZ), Re(1 To nZ), Pe(1 To nZ), tau_arr(1 To nZ), _
        del_Asp(1 To nZ), del_DM(1 To nZ), del_Asp0(1 To nZ), del_DM0(1 To nZ), del_Asp_norm(1 To nZ), del_DM_norm(1 To nZ)
    
    Dim volFrac() As Double, wtFrac() As Double, densi() As Double, visci() As Double, veloi() As Double
    ReDim densi(1 To 3), visci(1 To 3), veloi(1 To 3), volFrac(1 To 3), wtFrac(1 To 3)

    Dim R_nZ() As Double, R_nZ_Asp() As Double, R_nZ_DM() As Double, depth_m() As Double, depth_ft() As Double
    ReDim R_nZ(1 To nZ), R_nZ_Asp(1 To nZ), R_nZ_DM(1 To nZ), depth_m(1 To nZ), depth_ft(1 To nZ), _
        Cf_t0(1 To nZ), C_t0(1 To nZ), dt_arr(1 To nZ), Cf(1 To nZ), C(1 To nZ)
    
    'check if initial deposit is input
    Dim flag_fracBlock As Integer
    If IsEmpty(Asp_fracBlock) = False Then
        flag_fracBlock = 1
    End If
    
    'shear removal (SR) rate parameters
    Dim rSR() As Double: ReDim rSR(1 To nZ)
    
    i = 1
    x = 0
    For iZ = 1 To nZ
        
        'populate array of depth (ft). required for dP model
        depth_m(iZ) = L * (zf - dz * (iZ - 1))
        depth_ft(iZ) = depth_m(iZ) / 0.3048
        
        'find the upper and lower bound of current z
        i = 1
        x2 = z_nP(i)
        Do While i < nP And x2 <= x
            i = i + 1
            x2 = z_nP(i)
        Loop
        j = i - 1
        x1 = z_nP(j)
            
        'do linear interpolation
        m = (x - x1) / (x2 - x1)
        
        y1 = Ceq_wtf(j): y2 = Ceq_wtf(i): Y = y1 + m * (y2 - y1)
        Ceq(iZ) = Y                     'asphaltene solubility (gL1 / gOvr)
        
        wt_Tot = 0: vol_Tot = 0
        For k = 1 To nPh
            
            y1 = beta_wt(j, k): y2 = beta_wt(i, k): Y = y1 + m * (y2 - y1)
            beta(iZ, k) = Y             'phase fraction in VLL (wtfrac)
            
            y1 = dens_gcc(j, k): y2 = dens_gcc(i, k): Y = y1 + m * (y2 - y1)
            dens(iZ, k) = Y             'density in VLL (g/cc)
            
            If dens(iZ, k) > 0 Then vol(iZ, k) = beta(iZ, k) / dens(iZ, k)
            
            y1 = visco_cP(j, k): y2 = visco_cP(i, k): Y = y1 + m * (y2 - y1)
            visco(iZ, k) = Y            'viscosity in VLL (cP)
                
            y1 = yAsp_wtf(j, k): y2 = yAsp_wtf(i, k): Y = y1 + m * (y2 - y1)
            yAsp(iZ, k) = Y             'Asp compo in VLL (g/g)
            
            'sum total mass, volume
            wt_Tot = wt_Tot + beta(iZ, k)
            vol_Tot = vol_Tot + vol(iZ, k)
            
        Next k
        
        'normalize mass, volume
        For k = 1 To nPh
            beta(iZ, k) = beta(iZ, k) / wt_Tot
            vol(iZ, k) = vol(iZ, k) / vol_Tot
        Next k
        
        y1 = dens_gcc(j, 4): y2 = dens_gcc(i, 4): Y = y1 + m * (y2 - y1)
        dens_Asp(iZ) = Y                'Asp density (g/cc)
        
        y1 = Asp_kP(j): y2 = Asp_kP(i): Y = y1 + m * (y2 - y1)
        kP(iZ) = Y                      'Asp precipitation rate (1/s)
    
        y1 = Asp_kAg(j): y2 = Asp_kAg(i): Y = y1 + m * (y2 - y1)
        kAg(iZ) = Y                     'Asp aggregation rate (1/s)
    
        y1 = Asp_kD(j): y2 = Asp_kD(i): Y = y1 + m * (y2 - y1)
        kD_us(iZ) = Y                   'Asp deposition rate, unscaled (1/s)
        
        If flag_fracBlock = 1 Then
            y1 = Asp_fracBlock(j, 1): y2 = Asp_fracBlock(i, 1): Y = y1 + m * (y2 - y1)
            del_Asp0(iZ) = R_pipe * Y   'pipe fraction blocked, initial (m)
            
            y1 = Asp_fracBlock(j, 2): y2 = Asp_fracBlock(i, 2): Y = y1 + m * (y2 - y1)
            del_DM0(iZ) = R_pipe * Y    'pipe fraction blocked, initial (m)
        End If
        
        'next model z
        x = x + dz
            
        'populate radius profile (will update during time march loop due to deposition)
        del_Asp(iZ) = del_Asp0(iZ) * 100        'Asp thickness, initial (cm)
        del_DM(iZ) = del_DM0(iZ) * 100          'DM thickness, initial (cm)
        del_DM_norm(iZ) = del_DM0(iZ) / R_pipe
        R_nZ_Asp(iZ) = R_pipe - (del_Asp(i) / 100)  'initial pipe radius, Asp (m)
        R_nZ_DM(iZ) = R_pipe - (del_DM(i) / 100)    'initial pipe radius, DM (m)
        R_nZ(iZ) = R_nZ_DM(iZ)
        
        'populate initial conditions for Cf and C
        Cf_t0(iZ) = Cf_in   'conc dissolved Asp (Cf), initial
        C_t0(iZ) = C_in     'conc primary particles (C), initial
                
    Next iZ
    
    '=========================================================
    '--- time march loop (solve SS ADEPT at each step) ---
    '=========================================================
    Dim R As Double, R_pipe_cm As Double, dt0 As Double, flag_timecut As Boolean, tnew As Double, t_out_days As Double
    Dim iter As Integer, iter_Max As Integer, num_TC As Integer
    
    Dim dens_Z() As Double, visco_Z() As Double, velo_Z() As Double
    ReDim dens_Z(1 To nZ), visco_Z(1 To nZ), velo_Z(1 To nZ)
    
    'handle SimTime=0 (for output_TimeCuts var)
    If SimTime <= 0 Then
        SimTime = 0
        num_TC = 0
    End If
    
    'handle nTimeCuts=0 (for output_TimeCuts var)
    If nTimeCuts = 0 Then
        nT_out = 1
    Else
        nT_out = nTimeCuts + 1
        nCol = 0
    End If
    dt_out = 0
    If SimTime > 0 Then
        dt_out = SimTime / nTimeCuts    '?? how to handle if nTimeCuts = 0
    End If
    T_out = 0 'dt_out ' first time step to report output
    iT_out = 0
    
    'dimensionalize output_TimeCuts variable
    Dim output_TimeCuts() As Variant
    If nTimeCuts > 0 Then
        nTC_Vars = 10
        ReDim output_TimeCuts(1 To nZ_max, 1 To (3 * nT_out + nTC_Vars))   '3 properties at each time cut (nT_out) + 10 time-dependent vars
    End If
    
    'time step (s)
    dt0 = (1) * 86400 'day to sec
    
    'ensure step size corresponds to a number of steps at least = nTimeCuts
    nT_solve = SimTime / dt0
    If nT_solve < nT_out Then dt0 = dt_out
    
        
    '--- time march loop ---
    T = 0
    dti = dt0
    iter = 0
    iter_Max = 1000
    If SimTime = 0 Then iter_Max = 1    'handle empty pipe (SimTime=0) simulation
    Do While T <= SimTime And iter < iter_Max
        iter = iter + 1
             
        If iter = 1 Then
            dti = 0 'first iteration is empty pipe
        ElseIf iter = 8 Then
            iter = iter 'break point for testing
        End If
        
        'try a full time step
        tnew = T + dti
        
        'restrict time step if tnew exceeds t_out
        If tnew > T_out Then
            dti = tnew - T_out  'will change only for the next time step
            tnew = T_out        'restrict step so values are exactly equal
        Else
            dti = dt0           'return to original time step
        End If
        
        'actual time step taken
        dt = tnew - T
        
        'update radius profile due to restriction
        For i = 1 To nZ
            R_nZ_Asp(i) = R_pipe - (del_Asp(i) / 100)
            R_nZ_DM(i) = R_pipe - (del_DM(i) / 100)
            R_nZ(i) = R_nZ_DM(i)
        Next i
        
        'calculate phase velocity at each point from mass flows
        For i = 1 To nZ
            R = R_nZ(i)     'm
            a = pI * R ^ 2  'm2
            
            For k = 1 To nPh
                If beta(i, k) > 0 Then
                    volFlow(i, k) = beta(i, k) * m_Tot / dens(i, k) / 1000  'm3/s
                    velo(i, k) = volFlow(i, k) / a 'm/s
                End If
            Next k
            velo_L1(i) = velo(i, 2)
        Next i
        
        '===============================================================================================
        '--- calculate scaling factor (ScF) & Dahmkohler numbers (Da, reaction parameters) at each spatial point
        '===============================================================================================
        Dm = 0.000000002        'particle diffusivity (m2/s) from Einstein-Stokes equation
        kDiss = 0.01 / kP(1)    'Asp redissolution rate (1/s)
        For i = 1 To nZ
            R = R_nZ(i)     'radius at spatial point i
   
            'calculate avg density, viscosity, velocity
            wt_Tot = 0
            vol_Tot = 0
            For k = 1 To 2
                wt_Tot = wt_Tot + beta(i, k)        'total mass
                vol_Tot = vol_Tot + vol(i, k)       'total volume
            Next k
            For k = 1 To 2
                wtFrac(k) = beta(i, k) / wt_Tot     'mass fraction
                volFrac(k) = vol(i, k) / vol_Tot    'volume fraction
                densi(k) = dens(i, k)
                visci(k) = visco(i, k)
                veloi(k) = velo(i, k)
            Next k
            
            'density and viscosity averaging (="none" for L1 only, "mass", or "volume")
            'velocity averaging (="none" for L1 only, or "sum")
            rho = mix_Dens("volume", wtFrac, volFrac, densi)
            mu = mix_Visco("volume", wtFrac, volFrac, densi, visci, rho) * 10 ^ -3 'cP to Pa.s
            Uz = mix_Velo("sum", wtFrac, veloi)
            
            'store dens, visco, velocity at current length, z (not necessary for current routine)
            dens_Z(i) = rho
            visco_Z(i) = mu
            velo_Z(i) = Uz
            
            rho = rho * 1000           'density (kg/m3)
            kD0 = kD_us(i)             'deposition parameter (unscaled)
            
            Re(i) = 2 * R * Uz * rho / mu               'Reynolds number
            If Re(i) < 2500 Then                        'axial dispersion
                Dax = Dm + Uz ^ 2 * R ^ 2 / 48 / Dm     'laminar, e.g. capillary tube
            Else
                Dax = 2 * R * Uz / 5                    'turbulent flow, e.g. wellbore
            End If
            Pe(i) = Uz * L / Dax                        'Peclet number
                
            'boundary layer calculation
            fricFactor = 0.316 / (Re(i) ^ 0.25)         'Blausius friction factor
            tau = 1 / 8 * rho * fricFactor * Uz ^ 2     'shear stress at wall, Pa
            uf = (tau / rho) ^ 0.5                      'friction velocity, m/s
            del_wall = mu / rho / uf                    'wall layer thickness, m
            del_lam = 5 * del_wall                      'laminar boundary layer, m
            del_mom = 62.7 * (2 * R) * Re(i) ^ (-7 / 8) 'momentum boundary layer, m
            
            'choose boundary layer
            del = del_lam
    
            'scale deposition constant
            phi = Dm / del ^ 2 / kD0
            Scf = 2 * del / R * (phi / (phi + 1))
            kD(i) = Scf * kD0
            
            'shear removal rate, rSR
            If tau > tau_0 Then
                rSR(i) = kSR * (tau / tau_0 - 1) ^ nSR
            Else
                rSR(i) = 0
            End If
            
            'Dahmkohler numbers
            Da_P(i) = kP(i) * L / Uz
            Da_Ag(i) = kAg(i) * L / Uz
            Da_D(i) = kD(i) * (1 - rSR(i)) * L / Uz
            
            'store wall shear stress
            tau_arr(i) = tau
            
            'calculate non-dimensional time step (for transient simulation)
            dt_arr(i) = dt * Uz / L
            
        Next i
        '-----/ ScF and Da number calculation -----
        
        
        '----- solve governing PDEs ----------------------------------------------
        'take only one time step
        nT = 2
        
        'solve for conc dissolved asphaltenes: Cf(t,z)
        Cf_solve = ADEPT_Solver_Cf(T0, nT, dt_arr, Z0, nZ, dz, Cf_in, Cf_t0, Da_P, Ceq, SS, SolverType)
        
        'solve for conc primary particles: C(t,z)
        C_solve = ADEPT_Solver_C(T0, nT, dt_arr, Z0, nZ, dz, C_in, C_t0, kDiss, Pe, Da_P, Da_Ag, Da_D, Cf_solve, Ceq, SS, SolverType)
        '-------------------------------------------------------------------------
        
        'time
'        ti_Day = tnew / 86000
'        SimPercent = ti_Day / nT_solve * 100
'        Call ADEPT_Solver_StatusBar(SimPercent)
        
        '=========================================================
        '--- calculate deposition profile, flux, and other important outputs
        '=========================================================
        ReDim J_Asp(1 To nZ), mass_Asp(1 To nZ)
        
        Conc_PP = C_solve
        
        'calculate average y_Asp, rho_DM
        k = 0
        sumY = 0
        sumR = 0
        For i = 1 To nZ
            If yAsp(i, 3) > 0 Then
                k = k + 1
                sumY = sumY + yAsp(i, 3)
                sumR = sumR + dens(i, 3)
            End If
        Next i
        y_Asp = sumY / k
        rho_dm = sumR / k
             
        'calculate depo flux (J) and depo thickness (del) at each spatial point
        For i = 1 To nZ
            'rho_DM = dens_VLL(i,3)
            rho_Asp = dens_Asp(i)
            'y_Asp = yAsp_VLL(i,3)
            
            'extract C and Cf
            If SS = 1 Then
                C(i) = C_solve(i)
                Cf(i) = Cf_solve(i)
            Else
                C(i) = C_solve(nT, i)
                Cf(i) = Cf_solve(nT, i)
            End If
                
            'rate of asphaltene deposition [g/cc/s]
            rD = kD(i) * C(i) * c0 * (1 - rSR(i))
            
            'deposition flux [g/cm2/s]. might need to revisit to account for V_bl not V_cell.
            R_cm = R_nZ(i) * 100
            If rD < 0.00000000001 Then  'zNOTE!!: this should be removed!
                J_Asp(i) = rD * (2 * R_cm) / 4 '=0 in the old version
            Else
                J_Asp(i) = rD * (2 * R_cm) / 4
            End If
                
            'calculate thickness of asphaltenes and DM (Assumes R >> del which is a good approx for small dt)
            del_Asp(i) = del_Asp(i) + J_Asp(i) * dt / rho_Asp
            del_DM(i) = del_DM(i) + J_Asp(i) * dt / rho_dm / y_Asp

            'update initial concentrations for next time step
            Cf_t0(i) = Cf(i)
            C_t0(i) = C(i)
        Next i
        '-----/ depo flux (J) and thickness (del) calculation -----
        
        'post processing
        R_pipe_cm = R_pipe * 100
        L_cm = L * 100
        For i = 1 To nZ
            
            'set maximum deposit thickness to R_pipe
            If del_Asp(i) < 10 ^ -6 Then
                del_Asp(i) = 0
            ElseIf del_Asp(i) >= R_pipe_cm Then
                del_Asp(i) = R_pipe_cm
            End If
        
            If del_DM(i) < 10 ^ -6 Then
                del_DM(i) = 0
            ElseIf del_DM(i) >= R_pipe_cm Then
                del_DM(i) = R_pipe_cm
            End If
                             
            'calculate deposited Asp mass
            mass_Asp(i) = pI * (R_pipe_cm ^ 2 - (R_pipe_cm - del_Asp(i)) ^ 2) * dz * L_cm * dens_Asp(i)
            mass_Asp(i) = mass_Asp(i) / 1000    'kg
             
            'normalize deposit thickness (0 to 1)
            del_Asp_norm(i) = del_Asp(i) / R_pipe_cm
            del_DM_norm(i) = del_DM(i) / R_pipe_cm
        Next i
        
        '----- calculate pressure drop (dP) if at (or near) t_out -----
        'calc dP/dt(t=0) (enter only at iter=2 to calculate dp/dt using forward diff)
        If iter = 2 Then
            dP = DepoModel_dP(m_Tot, L, id, depth_ft, P_bh, del_DM_norm, beta, vol, dens, visco, volFlow)
            dP_flowStatus = dP(4, 1)    'extract flow status
            'If dP_flowStatus = "flowing" Then
                dPf = dP(1, 1)
                dPf_arr(2) = dPf
                If dt > 0 Then
                    dPdT = (dPf_arr(2) - dPf_arr(1)) / dt * 86400
                End If
            'End If
            output_TimeCuts(iT_out, 3 * nT_out + 3) = dPdT ' output results
        End If
        
        'calc dP(t~=t_out)
        If tnew + dt0 >= T_out And tnew <> T_out Then
            dP = DepoModel_dP(m_Tot, L, id, depth_ft, P_bh, del_DM_norm, beta, vol, dens, visco, volFlow)
            dP_flowStatus = dP(4, 1)    'extract flow status
            'If dP_flowStatus = "flowing" Then
                dPf = dP(1, 1)
                dPf_arr(1) = dPf
            'End If
        End If

        'calc dP(t=t_out) and dP/dt(t=t_out)
        If tnew = T_out Then
            dP = DepoModel_dP(m_Tot, L, id, depth_ft, P_bh, del_DM_norm, beta, vol, dens, visco, volFlow)
            dP_flowStatus = dP(4, 1)    'extract flow status
            'If dP_flowStatus = "flowing" Then
                dPf = dP(1, 1)
                dPf_arr(2) = dPf
                If dt > 0 Then
                    dPdT = (dPf_arr(2) - dPf_arr(1)) / dt * 86400
                End If
            'End If
            
            'handles cases of t=0 and when dt0=dt_out
            If dt0 = dt_out Or iter = 1 Then dPf_arr(1) = dPf_arr(2)

        End If
        
        'exit loop if well not flowing
        If dP_flowStatus <> "flowing" Then
            If t_noFlow = 0 Then t_noFlow = T_out / 86400
            'GoTo ExitLoop
        End If
        '-----/ pressure drop (dP) calculation -----
                
        '----- populate output_TimeCuts -----
        If nTimeCuts > 0 Then
            If tnew = T_out Then
                iT_out = iT_out + 1
                t_out_days = T_out / 86400
    
                For i = 1 To nZ
                    output_TimeCuts(i, nCol + 1) = del_Asp_norm(i)
                    output_TimeCuts(i, nCol + 2) = del_DM_norm(i)
                    
                    'val = Cf(i) - Ceq(i)
                    val = J_Asp(i) * 10000 * 86400 'deposition flux [g/m2/day]

                    If Abs(val) < 10 ^ -10 Then val = 0
                    output_TimeCuts(i, nCol + 3) = val
                Next i
                
                'total tubing volume, m3
                V_Tube = (pI * R_pipe ^ 2 * L)
                
                'frac volume deposited
                Dim V_Tube0 As Double, Lfrac As Double, dz_cm As Double
                L0 = 0: V_Tube0 = 0: V_TubeAsp = 0: V_TubeDM = 0
                For i = 1 To nZ
                    dz_m = L * dz
                    dr_m = R_pipe - R_nZ_DM(i)
                    If Abs(dr_m) > 10 ^ -10 Then
                        L0 = L0 + dz_m
                        V_Tube0 = V_Tube0 + pI * R_pipe ^ 2 * dz_m
                        V_TubeAsp = V_TubeAsp + pI * (R_pipe ^ 2 - R_nZ_Asp(i) ^ 2) * dz_m
                        V_TubeDM = V_TubeDM + pI * (R_pipe ^ 2 - R_nZ_DM(i) ^ 2) * dz_m
                    End If
                Next i
                Lfrac = 0: Vfrac_Asp = 0: Vfrac_DM = 0
                If V_Tube0 > 0 Then
                    Lfrac = L0 / L                      'length frac where deposit occurs
                    Vfrac_Asp = V_TubeAsp / V_Tube0     'volFrac of deposit, Asp (wrt clean tubing)
                    Vfrac_DM = V_TubeDM / V_Tube0       'volFrac of deposit, DM (wrt clean tubing)
                End If
                
                'total amount of Asp deposited
                mass_Asp_Tot = Sum(mass_Asp)                'total Asp mass deposited (kg)
                mass_Asp_Max = V_Tube * (rho_Asp) * 1000    'total Asp mass present (kg)
                mass_Asp_Frac = mass_Asp_Tot / mass_Asp_Max 'fraction Asp mass deposited
                max_flux = max(J_Asp) * 10000 * 86400       'max flux (g/m2/day)
                
                'populate time-dependent variables at the end of range
                output_TimeCuts(iT_out, 3 * nT_out + 1) = t_out_days        'time (days)
                output_TimeCuts(iT_out, 3 * nT_out + 2) = dPf               'pressure drop (psi)
                output_TimeCuts(iT_out, 3 * nT_out + 3) = dPdT              'dPdt (psi/day)
                output_TimeCuts(iT_out, 3 * nT_out + 4) = max(del_Asp_norm) 'max blockage, DM
                output_TimeCuts(iT_out, 3 * nT_out + 5) = max(del_DM_norm)  'max blockage, Asp
                output_TimeCuts(iT_out, 3 * nT_out + 6) = max_flux          'max flux (g/m2/day)
                output_TimeCuts(iT_out, 3 * nT_out + 7) = mass_Asp_Frac     'frac Asp mass deposited
                output_TimeCuts(iT_out, 3 * nT_out + 8) = Vfrac_Asp * 100   'vol% of deposit, Asp (wrt clean tubing)
                output_TimeCuts(iT_out, 3 * nT_out + 9) = Vfrac_DM * 100    'vol% of deposit, DM (wrt clean tubing)
                
                If dPdT = 0 Then output_TimeCuts(iT_out, 3 * nT_out + 3) = ""
                
                'update time_out and nCol
                T_out = T_out + dt_out
                nCol = nCol + 3
            End If
        End If
        '-----/ populate output_TimeCuts -----
        
        'update time
        T = tnew
        
        'exit loop when T >= SimTime
        If T >= SimTime Then
            iter = iter_Max
        End If
    Loop
    '-----/ time march loop -----

ExitLoop:

'--- OUTPUT ---
    
    'show day when flow stopped
    If dP_flowStatus <> "flowing" Then
        output_TimeCuts(1, 3 * nT_out + 10) = t_noFlow
    End If
    
    '''''''''''
    output_TimeCuts(1, 3 * nT_out + 10) = V_Tube * 6.29     'tubing volume (bbl oil)
    output_TimeCuts(2, 3 * nT_out + 10) = V_Tube0 * 6.29    'tubing volume (bbl oil), after deposit starts
    output_TimeCuts(3, 3 * nT_out + 10) = V_TubeDM * 6.29   'tubing volume (bbl oil),
    output_TimeCuts(4, 3 * nT_out + 10) = V_TubeAsp * 6.29  'tubing volume (bbl oil)
    
    'number of output columns (before timecuts)
    nCol = 8
    ReDim output_R(1 To nZ_max, 1 To nCol)
    For i = 1 To nZ
        val1 = C(i): If Abs(val1) < 10 ^ -12 Then val1 = 0
        val2 = Cf(i) - Ceq(i): If Abs(val2) < 10 ^ -10 Then val2 = 0
        
        output_R(i, 1) = val1                       'conc primary particles (C)
        output_R(i, 2) = val2                       'conc driving force
        output_R(i, 3) = del_Asp(i) / 2.54          'deposit thickness, Asp (in)
        output_R(i, 4) = del_DM(i) / 2.54           'deposit thickness, DM (in)
        output_R(i, 5) = del_Asp_norm(i)            'fraction blocked, Asp
        output_R(i, 6) = del_DM_norm(i)             'fraction blocked, DM
        output_R(i, 7) = J_Asp(i) * 10000 * 86400   'flux Asp (g/m2/day)
        output_R(i, 8) = mass_Asp(i)                'mass deposited, Asp (kg)
    Next i
    
    If nTimeCuts >= 0 Then
        ReDim output_Combined(1 To nZ_max, 1 To nCol + 3 * nT_out + nTC_Vars)
        For i = 1 To nZ
            For j = 1 To nCol
                output_Combined(i, j) = output_R(i, j)
            Next j
            For j = 1 To (3 * nT_out + nTC_Vars)
                output_Combined(i, nCol + j) = output_TimeCuts(i, j)
            Next j
            'output_Combined(i, nCol + 1) = dens_Z(i)            'rNOTE: COMMENT for release, UNCOMMENT to test property calc at prop(z,t)
            'output_Combined(i, nCol + 2) = visco_Z(i) * 1000    'rNOTE: COMMENT for release
            'output_Combined(i, nCol + 3) = velo_Z(i)            'rNOTE: COMMENT for release
        Next i
    End If
    
    'fill remaining outputs with empty
    If nTimeCuts > 0 Then
        For i = (nZ + 1) To nZ_max
            For j = 1 To 3 * nT_out
                output_TimeCuts(i, j) = ""
            Next j
        Next i
    End If
    If nTimeCuts > 0 Then
        For i = (nZ + 1) To nZ_max
            For j = 1 To nCol + 3 * nT_out
                output_Combined(i, j) = ""
            Next j
        Next i
    Else
        For i = (nZ + 1) To nZ_max
            For j = 1 To nCol
                output_R(i, j) = ""
            Next j
        Next i
    End If
    
    'return
    If nTimeCuts > 0 Then
        ADEPT_Solver = output_Combined
    Else
        ADEPT_Solver = output_R
    End If
    
End Function

Private Sub ADEPT_Solver_StatusBar(SimPercent As Integer)
    Dim str_msg As String: str_msg = "Deposition Simulation: " & SimPercent & "%"
    Application.ScreenUpdating = True
    Application.StatusBar = str_msg
    Application.ScreenUpdating = False
End Sub

Public Function ADEPT_Solver_C(T0 As Double, nT As Integer, dt() As Double, Z0 As Double, nZ As Integer, dz As Double, _
    C_in As Double, C_t0() As Double, _
    kDiss As Double, Pe() As Double, Da_P() As Double, Da_Ag() As Double, Da_D() As Double, Cf() As Double, Ceq() As Double, _
    Optional SS As Integer = 0, Optional SolverType As String = "IE")

'==========================================================
' Program:      ADEPT_Solver_C
' Description:  solves PDE for asphaltene primary particles conc (C)
' Called by:    ADEPT_Solver
' Notes:        see PDE formulation below
'
'---------- INPUTS ----------------------
' --- scalars ---
' t0:       initial time
' nT:       number of time nodes
' dt:       array of time steps. made into an array to take into account the varying Uz along tubing
' z0:       initial length
' nZ:       number of steps in length
' dz:       length step
' C_0:      primary particle conc at inlet (z=0)
' kDiss:    asphaltene redissolution rate (1/s)
'
' --- vectors --- (len = num of spatial points)
' Pe:       Peclet number
' Da_p:     precipitation Dahmkohler number
' Da_Ag:    aggregation Dahmkohler number
' Da_D:     deposition Dahmkohler number
' Cf:       dissolved asphaltene conc (gAsp_L1 / gAsp_Ovr)
' Ceq:      asphaltene solubility (gAsp_L1 / g_Ovr)
'
' --- solver settings ---
' SS:       {0 = transient, 1 = steady-state}
' SolverType:   {"IE" = implicit euler, "EE" = explicit euler}
'
'---------- OUTPUTS ---------------------
' C:        primary particle conc (gAsp_L1 / gAsp_Ovr)
'
'---------- NOTES -----------------------
' PDE for primary particle conc:
'   dC/dt = 1/Pe.d2C/dz2 - dC/dz + rp - Da_ag.C^2 - Da_d.C
'       rp = Da_p.(Cf-Ceq)          for Cf > Ceq (precipitation)
'       rp = rdiss = -kdiss.Da_p.C  for Cf < Ceq (redissolution)
'   IC: C(t=0,z) =0     (no PP at t=0)
'   BC1: C(t,z=0) =0    (no PP at z=0)
'   BC2: dC/dz(z=1) =0  (no gradient at end of pipe)??
'==========================================================

    Dim i As Integer, j As Integer, iZ As Integer, jZ As Integer, iT As Integer
    Dim Z As Double
    Dim C_z0 As Double, Cf_z As Double, Ceq_z As Double, Cz As Double, Czp As Double, Czf As Double, _
        C() As Double, c_T() As Double, c0() As Double, dC() As Double
    Dim f() As Double, jA() As Double, _
        tol As Double, obj As Double, conv As Integer, iter As Integer, iterMax As Integer
    Dim a1 As Double, a2 As Double, a3 As Double, a4 As Double, S As Double
    Dim rP As Double, rAg As Double, rD As Double
    
    'initial and boundary conditions
    'C_z0 = 0    ' no primary particles at z = 0
    'C_t0 = 0    ' no primary particles at t = 0
    
    'calculate number of nodes
    'nZ = Round((zf - z0) / dz) + 1
    'nT = Round((tf - t0) / dt) + 1
    
    'preallocate solution matrix (tSteps,zSteps) & populate IC and BC
    ReDim f(1 To nZ), jA(1 To nZ, 1 To nZ)
    
    If SS = 1 Then
        ReDim C(1 To nZ)
        C(1) = C_in     'BC
    Else
        ReDim C(1 To nT, 1 To nZ), c_T(1 To nZ)
        
        For i = 1 To nZ
            C(1, i) = C_t0(i)   'IC
        Next i
        For i = 1 To nT
            C(i, 1) = C_in      'BC
        Next i
    End If
    
    '----- steady-state, implicit Euler (IE-SS) -----
    '(a1.C(z-1) + a2.C(z) + a3.C(z)^2 + a4.C(z+1) + S = 0)
    If SS = 1 And InStr(1, "IE", SolverType) > 0 Then
        'initial guess
        For iZ = 1 To nZ
            Z = Z0 + (iZ - 1) * dz
            Cf_z = Cf(iZ)
            C(iZ) = 1 - Cf_z
        Next iZ
        C(nZ) = C(nZ - 1)
        
        tol = 0.0000001
        conv = 0
        iter = 0
        iterMax = 20
        Do While conv = 0 And iter <= iterMax
            iter = iter + 1
            
            For iZ = 1 To nZ
                Z = Z0 + (iZ - 1) * dz
                Cf_z = Cf(iZ)
                Ceq_z = Ceq(iZ)
                
                'define coefficients of node equations
                a1 = -1 / dz - 1 / Pe(iZ) / dz ^ 2
                a3 = Da_Ag(iZ)
                a4 = -1 / Pe(iZ) / dz ^ 2
                If Cf_z >= Ceq_z Then
                    a2 = 1 / dz + 2 / Pe(iZ) / dz ^ 2 + Da_D(iZ)
                    S = -Da_P(iZ) * (Cf_z - Ceq_z)
                Else
                    a2 = kDiss * Da_P(iZ) + 1 / dz + 2 / Pe(iZ) / dz ^ 2 + Da_D(iZ)
                    S = 0
                End If
                    
                'populate vector of objective functions and Jacobian
                If iZ = 1 Then
                    f(iZ) = C(iZ) - C_in
                    jA(iZ, 1) = 1
                ElseIf iZ = nZ Then
                    f(iZ) = C(iZ) - C(iZ - 1)
                    jA(iZ, nZ - 1) = -1
                    jA(iZ, nZ) = 1
                Else
                    Czp = C(iZ - 1)
                    Cz = C(iZ)
                    Czf = C(iZ + 1)
                    f(iZ) = a1 * Czp + a2 * Cz + a3 * Cz ^ 2 + a4 * Czf + S
                    
                    'Jacobian
                    For jZ = 1 To nZ
                        If jZ = iZ - 1 Then
                            jA(iZ, jZ) = a1
                        ElseIf jZ = iZ Then
                            jA(iZ, jZ) = a2 + 2 * a3 * Cz
                        ElseIf jZ = iZ + 1 Then
                            jA(iZ, jZ) = a4
                        End If
                    Next jZ
                End If
            Next iZ
            
            'calculate objective & check convergence
            obj = Norm(f)
            If obj < tol Then
                conv = 1
            Else
                'if not converged, take update step
                For iZ = 1 To nZ
                    f(iZ) = -f(iZ)
                Next iZ
                dC = SolveLinSys(jA, f)
                For iZ = 1 To nZ
                    C(iZ) = C(iZ) + dC(iZ)
                Next iZ
            End If
        Loop
    End If
    '----- /IE-SS -----
    
    '----- transient, explicit Euler (EE-t) -----
    If SS = 0 And InStr(1, "EE", SolverType) > 0 Then
        For iT = 2 To nT
            For iZ = 2 To (nZ - 1)
                Z = Z0 + (iZ - 1) * dz
                Cf_z = Cf(nT - 1, iZ)
                Ceq_z = Ceq(iZ)
                
                'need some work ??
                If Cf_z - Ceq_z >= 0 Then
                    rP = Da_P(iZ) * (Cf(nT - 1, iZ) - Ceq(iZ))
                Else
                    rP = -kDiss * Da_P(iZ) * C(nT - 1, iZ)
                End If
                
                rAg = Da_Ag(iZ) * C(nT - 1, iZ) ^ 2
                rD = Da_D(iZ) * C(nT - 1, iZ)
                
                C(iT, iZ) = C(iT - 1, iZ) + dt(iZ) / dz ^ 2 / Pe(iZ) * (C(iT - 1, iZ + 1) - 2 * C(iT - 1, iZ) + C(iT - 1, iZ - 1)) - dt(iZ) / (dz) * (C(iT - 1, iZ) - C(iT - 1, iZ - 1)) + dt(iZ) * (rP - rAg - rD)
            Next iZ
            
            'populate boundary condition (z=1)
            C(iT, nZ) = C(iT, nZ - 1)
            
        Next iT
    End If
    '----- /EE-t -----
    
    '----- transient, implicit Euler (IE-t) -----
    If SS = 0 And InStr(1, "IE", SolverType) > 0 Then
        'initial guess
        ReDim c0(1 To nZ)
        
        For iZ = 1 To nZ
            c0(iZ) = Cf(nT, iZ)
        Next iZ
        
        For iZ = 1 To nZ
            Z = Z0 + (iZ - 1) * dz
            Cf_z = c0(iZ)
            c_T(iZ) = 1 - Cf_z
        Next iZ
        
        c_T(nZ) = c_T(nZ - 1)
        
        'time-stepping
        For iT = 2 To nT
             
            'Newton loop
            tol = 0.0000001
            conv = 0
            iter = 0
            iterMax = 20
            Do While conv = 0 And iter <= iterMax
                iter = iter + 1
    
                For iZ = 1 To nZ
                    Z = Z0 + (iZ - 1) * dz
    
                    Cf_z = c0(iZ)
                    Ceq_z = Ceq(iZ)
    
                    ' define coefficients of node equations
                    a1 = dt(iZ) * (1 / dz + 1 / Pe(iZ) / dz ^ 2)
                    a3 = dt(iZ) * (-Da_Ag(iZ))
                    a4 = dt(iZ) * (1 / Pe(iZ) / dz ^ 2)
                    If Cf_z >= Ceq_z Then
                        a2 = dt(iZ) * (-1 / dz - 2 / Pe(iZ) / dz ^ 2 - Da_D(iZ))
                        S = dt(iZ) * (Da_P(iZ) * (Cf_z - Ceq_z))
                    Else
                        a2 = dt(iZ) * (-kDiss * Da_P(iZ) - 1 / dz - 2 / Pe(iZ) / dz ^ 2 - Da_D(iZ))
                        S = 0
                    End If
    
                    ' populate vector of objective functions and Jacobian
                    If iZ = 1 Then
                        f(iZ) = c_T(iZ) - C_in
                        jA(iZ, 1) = 1
                    ElseIf iZ = nZ Then
                        f(iZ) = c_T(iZ) - c_T(iZ - 1)
                        jA(iZ, nZ - 1) = -1
                        jA(iZ, nZ) = 1
                    Else
                        Czp = c_T(iZ - 1)
                        Cz = c_T(iZ)
                        Czf = c_T(iZ + 1)
                        f(iZ) = a1 * Czp + a2 * Cz + a3 * Cz ^ 2 + a4 * Czf + S
    
                        ' Jacobian
                        For jZ = 1 To nZ
                            If jZ = iZ - 1 Then
                                jA(iZ, jZ) = a1
                            ElseIf jZ = iZ Then
                                jA(iZ, jZ) = a2 + 2 * a3 * Cz
                            ElseIf jZ = iZ + 1 Then
                                jA(iZ, jZ) = a4
                            End If
                        Next jZ
                    End If
                Next iZ
                
                'calculate objective & check convergence
                obj = Norm(f)
                If obj < tol Then
                    conv = 1
                Else
                    'if not converged, take update step
                    For iZ = 1 To nZ
                        f(iZ) = -f(iZ)
                    Next iZ
                    dC = SolveLinSys(jA, f)
                    For iZ = 1 To nZ
                        c_T(iZ) = c_T(iZ) + dC(iZ)
                    Next iZ
                End If
            Loop
            
            'store solution for current time cut
            For iZ = 1 To nZ
                C(iT, iZ) = c_T(iZ)
            Next iZ
        Next iT
    End If
    '----- /IE-t -----
    
    'return
    ADEPT_Solver_C = C
    
End Function

Public Function ADEPT_Solver_Cf(T0 As Double, nT As Integer, dt() As Double, Z0 As Double, nZ As Integer, dz As Double, _
    Cf_in As Double, Cf_t0() As Double, Da_P() As Double, Ceq() As Double, _
    Optional SS As Integer = 0, Optional SolverType As String = "IE")

'==========================================================
' Program:      ADEPT_Solver_Cf
' Description:  solves PDE for dissolved asphaltene conc (Cf)
' Called by:    ADEPT_Solver
' Notes:        see PDE formulation below
'
'---------- INPUTS ----------------------
' --- scalars ---
' t0:       initial time
' nT:       number of time nodes
' dt:       array of time steps. made into an array to take into account the varying Uz along tubing
' z0:       initial length
' nZ:       number of steps in length
' dz:       length step
' Cf_in:    dissolved asphaltene conc in the input stream
'
' --- vectors --- (len = num of spatial points)
' Cf_t0:    array of initial (t=0) dissolved asphaltene conc
' Da_p:     precipitation Dahmkohler number
' Ceq:      asphaltene solubility (gAsp_L1 / gAsp_Ovr)
'
' --- solver settings ---
' SS:       {0 = transient, 1 = steady-state}
' SolverType:   {"IE" = implicit euler, "EE" = explicit euler}
'
'---------- OUTPUTS ---------------------
' Cf:       dissolved asphaltene conc (gAsp_L1 / gAsp_Ovr)
'
'---------- NOTES -----------------------
' PDE for dissolved asphaltene conc:
'   dCf/dt = -dCf/dz - Da_p.(Cf-Ceq)
'   IC: Cf(t=0,z) =1    (all Asp dissolved at t=0)
'   BC: Cf(t,z=0) =1    (all Asp dissolved at z=0)
'==========================================================

    Dim i As Integer, j As Integer, iZ As Integer, iT As Integer
    Dim Cf() As Double, Cf_t() As Double, Cf_z As Double, Ceq_z As Double, _
        rP As Double, rp_v() As Double, _
        a() As Double, B() As Double, a1 As Double, a2 As Double, B1 As Double, B2 As Double, m As Double, Z As Double
    
    'initial and boundary conditions
    'Cf_in = 1 ' everything in solution
    'Cf_t0 = 1 ' everything in solution

    'calculate number of nodes
    'nZ = Round((zf - z0) / dz) + 1
    'nT = Round((tf - t0) / dt) + 1

    'preallocate solution matrix (tSteps,zSteps) & populate IC and BC
    ReDim a(1 To nZ, 1 To nZ), B(1 To nZ)
        
    If SS = 1 Then
        ReDim Cf(1, 1 To nZ), rp_v(1, 1 To nZ)
        Cf(1, 1) = Cf_in        ' BC
        
    Else
        ReDim Cf(1 To nT, 1 To nZ), rp_v(1 To nT, 1 To nZ), Cf_t(1 To nZ)
        
        For i = 1 To nZ
            Cf(1, i) = Cf_t0(i)     'IC
        Next i
        For i = 1 To nT
            Cf(i, 1) = Cf_in        'BC
        Next i
    End If

    'steady-state, explicit Euler
    If SS = 1 And InStr(1, "EE", SolverType) > 0 Then
    
        For iZ = 2 To nZ
            Z = Z0 + (iZ - 1) * dz
            Cf_z = Cf(1, iZ - 1)
            Ceq_z = Ceq(iZ - 1)     'extract Ceq at previous node
            rP = Da_P(iZ - 1) * (Cf_z - Ceq_z)

            Cf(1, iZ) = Cf_z - dz * rP

            'store rate of precipitation
            rp_v(1, iZ) = rP
        Next iZ
        
    End If
    
    'steady-state, implicit Euler: a1*C(i) + a2*C(i+1) + b
    If SS = 1 And InStr(1, "IE", SolverType) > 0 Then

        For iZ = 1 To nZ
            Z = Z0 + (iZ - 1) * dz
            Ceq_z = Ceq(iZ)         'extract Ceq at current node
            
            'coefficients of node equation
            a1 = -1
            a2 = 1 + dz * Da_P(iZ)
            B1 = -dz * Da_P(iZ) * Ceq_z
    
            'construct linear system A*C+B = 0
            If iZ = 1 Then
                a(iZ, 1) = 1
                B(iZ) = -Cf_in
            Else
                a(iZ, iZ - 1) = a1
                a(iZ, iZ) = a2
                B(iZ) = B1
            End If
            
        Next iZ
    
        'solve linear system
        For iZ = 1 To nZ
            B(iZ) = -B(iZ)
        Next iZ
        Cf = SolveLinSys(a, B)

    End If
    
    'transient, explicit Euler
    If SS = 0 And InStr(1, "EE", SolverType) > 0 Then
        
        For iT = 2 To nT
            For iZ = 2 To nZ
                
                Z = Z0 + (iZ - 1) * dz
                Ceq_z = Ceq(iZ)     'extract Ceq at current node
                rP = Da_P(iZ) * (Cf(iT - 1, iZ) - Ceq_z)
                
                Cf(iT, iZ) = Cf(iT - 1, iZ) - dt(iZ) / dz * (Cf(iT - 1, iZ) - Cf(iT - 1, iZ - 1)) - dt(iZ) * rP
            Next iZ
        Next iT
        
    End If
    
    'transient, implicit Euler: a1*C(i-1) + a2*C(i) + b
    If SS = 0 And InStr(1, "IE", SolverType) > 0 Then
        
        For iT = 2 To nT
            For iZ = 1 To nZ
                Z = Z0 + (iZ - 1) * dz
                Ceq_z = Ceq(iZ)     'extract Ceq at current node

                'coefficients of node equation
                a1 = -dt(iZ) / dz
                a2 = 1 + dt(iZ) / dz + dt(iZ) * Da_P(iZ)
                B1 = -Cf(iT - 1, iZ) - dt(iZ) * Da_P(iZ) * Ceq_z

                'construct linear system A*C+B = 0
                If iZ = 1 Then
                    a(iZ, 1) = 1
                    B(iZ) = -Cf_in
                Else
                    a(iZ, iZ - 1) = a1
                    a(iZ, iZ) = a2
                    B(iZ) = B1
                End If
            Next iZ

            'solve linear system
            For iZ = 1 To nZ
                B(iZ) = -B(iZ)
            Next iZ
            Cf_t = SolveLinSys(a, B)
            
            'populate solution at current time cut
            For iZ = 1 To nZ
                Cf(iT, iZ) = Cf_t(iZ)
            Next iZ
            
        Next iT
        
    End If
    
    'return
    ADEPT_Solver_Cf = Cf
   
End Function

Private Function mix_Dens(str_Avg As String, wtFrac() As Double, volFrac() As Double, dens() As Double)
'calculate averaged total viscosity by method: "str_Avg"

    Dim k As Integer, nPhase As Integer, rho As Double, mu As Double
    nPhase = ArrayLen(wtFrac)
    
    If str_Avg = "mass" Then
        For k = 1 To nPhase
            If wtFrac(k) > 0 Then rho = rho + wtFrac(k) / dens(k)   'mass-averaged dens
        Next k
        rho = 1 / rho
    ElseIf str_Avg = "volume" Then
        For k = 1 To nPhase
            If volFrac(k) > 0 Then rho = rho + dens(k) * volFrac(k) 'vol-averaged dens
        Next k
    Else 'none
        rho = dens(2)                                               'none (total dens = L1 dens)
    End If
    
    'output
    mix_Dens = rho
    
End Function

Private Function mix_Visco(str_Avg As String, wtFrac() As Double, volFrac() As Double, dens() As Double, visco() As Double, rho)
'calculate averaged total viscosity by method: "str_Avg"

    Dim k As Integer, nPhase As Integer, mu As Double
    nPhase = ArrayLen(wtFrac)
    
    If str_Avg = "mass" Then
        For k = 1 To nPhase
            If wtFrac(k) > 0 Then mu = mu + wtFrac(k) / visco(k)            'mass-averaged visco
        Next k
        mu = 1 / mu
    ElseIf str_Avg = "volume" Then
        For k = 1 To nPhase
            If wtFrac(k) > 0 Then mu = mu + wtFrac(k) * visco(k) / dens(k)  'vol-averaged visco
        Next k
        mu = rho * mu
    Else 'none
        mu = visco(2)                                                       'none (total visco = L1 visco)
    End If
    
    'return
    mix_Visco = mu
    
End Function

Private Function mix_Velo(str_Avg As String, wtFrac() As Double, velo() As Double)
'calculate averaged total velocity by method: "str_Avg"

    Dim k As Integer, nPhase As Integer, Uz As Double
    nPhase = ArrayLen(wtFrac)
    
    If str_Avg = "sum" Then
        For k = 1 To nPhase
            Uz = Uz + velo(k)   'sum of all velocities
        Next k
    Else
        Uz = velo(2)            'none (total velo = L1 velo)
    End If
    
    'return
    mix_Velo = Uz
    
End Function

Public Function Asp_DepoModel_dP(m_Tot As Double, L As Double, id As Double, Asp_zProf As Range, P_bh As Double, Asp_xProf As Range, Asp_FluidProps As Range)

'=========================================================
' Program:      Asp_DepoModel_dP
' Description:  calculates pressure drop in a pipe with deposits
' Called by:    on-sheet function
' Notes:        calculates dP along length (L) of pipe
'
'---------- INPUTS -----------------------
' --- scalars ---
' m_Tot:        total flow rate (kg/s)
' L:            pipe length (m)
' ID:           pipe diameter (m)
' P_bh:         bottomhole pressure (psi)
'
' --- ranges ---
' Asp_zProf:    depth (ft)
' Asp_xProf:    deposit thickness (-)
' Asp_FluidProps: oil properties along depth profile
'   beta:       phase amounts (wt%)
'   dens:       phase density (g/cc)
'   visco:      phase viscosity (cP)
'
'---------- OUTPUTS ----------------------
' dP:           pressure drop
'==========================================================

    Dim i As Integer, k As Integer, nZ As Integer, nPh As Integer
    Dim Asp_depth As Variant, Asp_P As Variant, Asp_del As Variant, Asp_beta As Variant, Asp_dens As Variant, Asp_visco As Variant
    Dim depth_ft() As Double, P_psig() As Double, depo_dm() As Double, wtFrac() As Double, volFrac() As Double, dens() As Double, visco() As Double, volFlow() As Double, _
        R As Double, wt_Tot As Double, vol_Tot As Double
    
    Asp_depth = Asp_zProf.Columns("A").Value2
    'Asp_P = Asp_pProf.Columns("A").Value2
    Asp_del = Asp_xProf.Columns("A").Value2
    'Asp_Ceq = Asp_FluidProps.Columns("A").Value2
    'Asp_yAsp = Asp_FluidProps.Columns("B:C").Value2
    Asp_beta = Asp_FluidProps.Columns("D:F").Value2
    Asp_dens = Asp_FluidProps.Columns("G:J").Value2
    'Asp_SP = Asp_FluidProps.Columns("K:N").Value2
    Asp_visco = Asp_FluidProps.Columns("O:R").Value2
    
    'dimensionalize inputs
    nZ = ArrayLen(Asp_beta, , , nPh)
    ReDim depth_ft(1 To nZ), P_psig(1 To nZ), depo_dm(1 To nZ), _
        wtFrac(1 To nZ, 1 To nPh), volFrac(1 To nZ, 1 To nPh), dens(1 To nZ, 1 To nPh), visco(1 To nZ, 1 To nPh), volFlow(1 To nZ, 1 To nPh)
    
    'extract inputs from ranges
    For i = 1 To nZ
        depth_ft(i) = Asp_depth(i, 1)   'depth (ft)
        'P_psig(i) = Asp_P(i, 1)         'pressure (psig)
        depo_dm(i) = Asp_del(i, 1)      'deposit thickness (in)
        
        If depth_ft(i) > 0 Then
            wt_Tot = 0: vol_Tot = 0
            For k = 1 To nPh
                wtFrac(i, k) = Asp_beta(i, k)
                dens(i, k) = Asp_dens(i, k)
                visco(i, k) = Asp_visco(i, k)
                
                'calculate volume
                If wtFrac(i, k) > 0 Then volFrac(i, k) = wtFrac(i, k) / dens(i, k)
                
                'total mass, volume
                wt_Tot = wt_Tot + wtFrac(i, k)
                vol_Tot = vol_Tot + volFrac(i, k)
                
            Next k
            
            'normalize mass, volume
            For k = 1 To nPh
                wtFrac(i, k) = wtFrac(i, k) / wt_Tot
                volFrac(i, k) = volFrac(i, k) / vol_Tot
                
                If wtFrac(i, k) > 0 Then
                    volFlow(i, k) = wtFrac(i, k) * m_Tot / dens(i, k) / 1000  'm3/s
                End If
            Next k
            
        End If
    Next i
    
    'call routine
    Asp_DepoModel_dP = DepoModel_dP(m_Tot, L, id, depth_ft, P_bh, depo_dm, wtFrac, volFrac, dens, visco, volFlow)
    
End Function

Private Function DepoModel_dP(m_Tot As Double, L As Double, id As Double, depth_ft() As Double, P_bh As Double, depo_dm() As Double, _
        beta() As Double, vol() As Double, dens() As Double, visco() As Double, volFlow() As Double)

'=========================================================
' Program:      DepoModel_dP
' Description:  calculates pressure drop in a pipe with deposits
' Called by:    Asp_DepoModel_dP
' Notes:        scans from bottomhole (BH) to wellhead (WH)
'               calculating pressure drop in each segment
'
'---------- INPUTS -----------------------
' --- scalars ---
' m_Tot:        total flow rate (kg/s)
' L:            pipe length (m)
' ID:           pipe diameter (m)
' P_bh:         bottomhole pressure (psi)
'
' --- ranges ---
' rng_depth:    depth (ft)
' rng_deposit:  deposit thickness (-)
' rng_dens:     oil (L1) density (g/cc)
' rng_visco:    oil (L1) viscosity (cP)
'
'---------- OUTPUTS ----------------------
' dP:           pressure drop
'==========================================================
    
    Dim i As Integer, k As Integer, iZ As Integer, nZ As Integer, nPh As Integer, flag As String, flag_Err As String
    Dim depth() As Double, deposit() As Double, velo() As Double, _
        densi() As Double, visci() As Double, veloi() As Double, wtFrac() As Double, volFrac() As Double, _
        depo As Double, rho As Double, mu As Double, wt_Tot As Double, vol_Tot As Double
    Dim dPf_i As Double, dPg_i As Double, dP_i As Double, dP As Double, dPf As Double, dPg As Double, _
        Re As Double, fricFactor As Double, Uz As Double, a As Double, pI As Double, R As Double, R_true As Double, L_seg As Double
    Dim P_wh As Double, delP As Double
    
    nZ = ArrayLen(depth_ft)
    nPh = ArrayLen(beta, 2)
    ReDim depth(1 To nZ), deposit(1 To nZ), _
        densi(1 To nPh), visci(1 To nPh), veloi(1 To nPh), wtFrac(1 To nPh), volFrac(1 To nPh)
    
    'convert
    pI = 3.14159265358979:
    R = id / 2
    'Q = Q * 0.1589873 / 86400       'STB/d to m3/s
    'ID = ID * 0.0254                'in to m
    
    'get bottomhole pressure
    'P_bh = P_psig(1)    'max(P_psig)
    
    'scan from bottomhole (BH) to wellhead (WH)
    'calculate pressure drop for each segment
    dP = 0      'total pressure loss
    dPf = 0     'friction pressure loss
    dPg = 0     'gravity pressure loss
    flag = "flowing"
    flag_Err = "err"
    For i = 1 To nZ - 1
        
        If depth_ft(i) > 0 Then
            iZ = iZ + 1
            
            'extract properties
            L_seg = (depth_ft(i) - depth_ft(i + 1))     'ft
            L_seg = L_seg * 0.3048                      'ft to m
            depo = depo_dm(i) * R                       'm
            
            'calculate diameter available for flow
            R_true = R - depo               'm
            a = pI * R_true ^ 2             'm2
            
            If R_true <= 0 Then
                flag = "plugged"
                GoTo ErrHandle
            End If
            
            'calculate phase velocities
            For k = 1 To nPh
                'volFlow(i, k) = beta(i, k) * m_Tot / dens(i, k) / 1000  'm3/s
                veloi(k) = volFlow(i, k) / a      'm/s
            Next k
            
            'calculate average density, viscosity, velocity
            wt_Tot = 0
            vol_Tot = 0
            For k = 1 To 2
                wt_Tot = wt_Tot + beta(i, k)        'total mass
                vol_Tot = vol_Tot + vol(i, k)       'total volume
            Next k
            For k = 1 To 2
                wtFrac(k) = beta(i, k) / wt_Tot     'mass fraction
                volFrac(k) = vol(i, k) / vol_Tot    'volume fraction
                densi(k) = dens(i, k)
                visci(k) = visco(i, k)
                'veloi(k) = velo(i, k)
            Next k
            
            'density and viscosity averaging (="none" for L1 only, "mass", or "volume")
            'velocity averaging (="none" for L1 only, or "sum")
            rho = mix_Dens("volume", wtFrac, volFrac, densi)
            mu = mix_Visco("volume", wtFrac, volFrac, densi, visci, rho)
            Uz = mix_Velo("sum", wtFrac, veloi)
            
            'convert to standard units
            rho = rho * 1000                        'g/cc to kg/m3
            mu = mu * 10 ^ -3                       'cP to Pa.s
                    
            'Re and friction factor
            Re = rho * Uz * (2 * R_true) / mu
            If Re = 0 Then
                fricFactor = 0
            ElseIf Re < 2400 Then
                fricFactor = 64 / Re
            Else
                fricFactor = 0.316 / (Re ^ 0.25)    'Blausius friction factor
                'fricFactor = (1 / (1.14 - 2 * Log((0.0006 * 0.3048) / (2 * R_true) + 21.25 / Re ^ 0.9) / Log(10))) ^ 2 ' Jain friction factor
                'fricFactor = 0.018
            End If
            
            'friction pressure drop
            dPf_i = fricFactor * L_seg * Uz ^ 2 * rho / R_true / 4   'Darcy-Weisbach formula (Pa)
            dPf = dPf + dPf_i
            
            'gravity pressure drop
            dPg_i = L_seg * 9.81 * rho      'Pa
            dPg = dPg + dPg_i
            
            'sum pressure drop in segment
            dP_i = dPf_i + dPg_i
            
            'cumulative dp
            dP = dP + dP_i
            
        End If
        
    Next i
    
    'get wellhead pressure
    P_wh = 0 'P_psig(iZ)
    
    'maximum allowable pressure drop
    delP = P_bh - P_wh      'psi
    
    'convert pressure
    dPf = dPf / 6894.76     'Pa to psi
    dPg = dPg / 6894.76
    dP = dP / 6894.76
    
    'determine if shutdown
    If dP > delP Then
        flag = "shutdown"
    End If
    flag_Err = ""       'lower flag if calc reaches this point
    
ErrHandle:
    
    'output
    Dim out(4, 1) As Variant
    If flag_Err <> "" Then
        out(1, 1) = "": out(2, 1) = "": out(3, 1) = ""
        out(4, 1) = flag_Err
    ElseIf flag = "plugged" Then
        out(1, 1) = "": out(2, 1) = "": out(3, 1) = ""
        out(4, 1) = flag
    Else
        out(1, 1) = dPf: out(2, 1) = dPg: out(3, 1) = dP
        out(4, 1) = flag
    End If
    
    If out(1, 1) > 10 ^ 6 Then
        'out(1, 1) = "> 10^6"
        'out(2, 1) = ""
        'out(3, 1) = ""
    End If
    
    'return
    DepoModel_dP = out

End Function

Private Function ADEPT_kP(a As Variant, T As Double, Optional del_SP As Double, Optional str_eqn As String = "default")

'=========================================================
' Program:      ADEPT_kP
' Description:  calculates the precipitation kinetics (kP) parameter in the ADEPT model
' Called by:    Asp_DepoRate; on-sheet function
' Notes:        ln(kP) =f(T,{SP})
'
'---------- INPUTS -----------------------
' a:            coefficients of correlation (a1, a2, a3, a4)
' T:            temperature [K]
' del_SP:       diff in solubility parameter between asphaltene and solvent [MPa^0.5]
' str_eqn:      form of correlation ("narmi" or "IL" or "IL-SP")
'
'---------- RETURN -----------------------
' ln_kP:        precipitation kinetics (kP) parameter
'==========================================================

    'log(kP)=f(T,{SP})
    Dim str As String, x As Double
    str = LCase(str_eqn)
    If str = "narmi" Then
        x = -(a(1) * Exp(-a(2) / T) + (a(3) * Exp(-a(4) / T)) / del_SP ^ 2)
    ElseIf str = "il-sp" Then
        x = Log(a(1)) - a(2) * 1000 / (T * del_SP ^ 2)
    Else 'str = "default" Then
        x = Log(a(1)) - a(2) * 1000 / T
    End If
    
    'return
    ADEPT_kP = x
    
End Function

Private Function ADEPT_kAg(a As Variant, T As Double, Optional del_SP As Double, Optional visco As Double, _
    Optional str_eqn As String = "default")

'=========================================================
' Program:      ADEPT_kAg
' Description:  calculates the aggregation kinetics (kAg) parameter in the ADEPT model
' Called by:    Asp_DepoRate; on-sheet function
' Notes:        ln(kAg)=f(T,{SP},{mu})
'
'---------- INPUTS -----------------------
' a:            coefficients of correlation (a1, a2, a3, a4)
' T:            temperature [K]
' del_SP:       diff in solubility parameter [MPa^0.5] between asphaltene and solvent
' visco:        viscosity [cP] of solvent
' str_eqn:      form of correlation ("narmi" or "IL")
'
'---------- RETURN -----------------------
' ln_kAg:       aggregation kinetics (kAg) parameter
'==========================================================

    Dim str As String, x As Double, RT As Double
    str = LCase(str_eqn)
    If str = "narmi" Then
        x = -(a(1) * Exp(-a(2) / T) + (a(3) * Exp(-a(4) / T)) / del_SP ^ 2)
    Else 'str = "default" Then
        'a[1] represents the collision efficiency
        RT = 8.314 * T
        x = Log(1 / 750 * (2 / 3) * RT / visco * a(1) * 0.1)
    End If
    
    'return
    ADEPT_kAg = x
    
End Function

