'------ ABOUT ---------
' DESCRIPTION:  calculations for deposition (ADEPT) and pressure drop (dP) in the wellbore
' CALLED BY:    asphaltene deposition routines
' UI sheet:     "Asphaltenes"
' CALC sheet:   "FPc_Depo"
'----------------------

Option Explicit
Option Base 1

'wrapper to call ADEPT
Public Function Asp_DepoModel(ADEPT_Input As Range, Asp_DepoInputs As Range, Optional sht_P As String = "FPp_main", Optional out_type = "sheet")
        
    'run calculation
    Dim dict As New Scripting.Dictionary
    Set dict = DepoModel_ADEPT(ADEPT_Input, Asp_DepoInputs, sht_P)
    
    'return
    Dim str As String
    str = LCase(out_type)
    If str = "sheet" Then
        Asp_DepoModel = dict("sheet")
    ElseIf str = "tc" Or str = "time cut" Then
        Asp_DepoModel = dict("TC_vars")
    Else
        Set Asp_DepoModel = dict
    End If
    
End Function

'---------------------------------------------------------------------
'--- IL CODES ---
'---------------------------------------------------------------------

Private Function DepoModel_ADEPT(ADEPT_Input As Range, Asp_DepoInputs As Range, Optional sht_P As String = "FPp_main")
 
'=========================================================
' Program:      DepoModel_ADEPT
' Description:  ADEPT model for asphaltene deposition in wellbore
' Called by:    FP_DepoModel_ADEPT
' Notes:        wrapper for ADEPT_Solver routine
'
'---------- INPUTS ---------------------------
' ADEPT_Input:      input range
'   production data
'   (1)     t_sim:      simulation time (s)
'   (2)     mFlow:      massflow rate (kg/s)
'   (3)     L:          pipe length (m)
'   (4)     R:          pipe radius (m)
'   fluid properties
'   (5)     Ca0:        asph concentration (kgAsp / m3 oil)
'   (6)     Ceq:        asph solubility (gAsp[L1] / gAsp)
'   (7-9)   dens:       density (kg/m3) of [L1,L2,Asp]
'   (10-12) SP:         solubility parameter (MPa^0.5) of [L1,L2,Asp]
'   (13-15) visco:      viscosity (Pa.s) of [L1,L2,Asp]
'   (16-17) yAsp:       asph composition (gAsp[k]/g[k])
'   (18)    props:      {'table'=thermo props from LUT, 'constant'=thermo props from this range)
'   ADEPT parameters
'   (19)                kP/kAg scaling {constant or scaled}
'   (20)                kD scaling {1=wellbore, 2=packed bed, 3=capillary tube}
'   (21-23)             kP, KAg, kD (1/s) - unscaled
'   (24-25) kP_param    kP scaling constants, kP_model
'   (26-27) kAg_param   kAg scaling constants, kAg_model
'   (28-29) SR_param    shear removal (SR) parameters, SR_model
'   (30-32)             mixing rules (rho=density, mu=viscosity, u=velocity)
'   depo_solver
'   (33)    SolverType  {IE=implicit eulur, EE=explicit eulur}
'   (34)    SS          {0=transient, 1=steady state}
'   (35)    nTimeCuts   # time cuts
' Asp_DepoInputs:   lookup table
'   Asp_zPTx: PT-profile
'   (1)     z           depth (-)
'   (2)     z_ft        depth (ft)
'   (3)     P           pressure
'   (4)     T           temperature
'   Asp_x0: depo-profile
'   (1)     x0_Asp      initial fracBlock (-), Asp
'   (2)     x0_DM       initial fracBlock (-), DM
'   Asp_FluidProps: thermo and transport props at each TP
'   (1)     Ceq         Asp solubility (gAsp[L1]/gAsp)
'   (2-3)   yAsp:       Asp compo (wtf) in [L1,L2]
'   (4-6)   beta:       phase amt (wt%) in [V,L1,L2]
'   (7-10)  dens:       density (g/cc) in [V,L1,L2,Asp]
'   (11-14) SP:         solubility parameter (MPa^0.5) in [V,L1,L2,Asp]
'   (15-18) visco:      viscosity (cP) in [V,L1,L2,Asp]
'   Asp_RateParam: ADEPT parameters at each TP
'   (1-3):      kP, KAg, kD (1/s)
'
'---------- RETURN ---------------------------
' out:          ADEPT_Solver
'
'==========================================================
     
    Dim i As Integer, j As Integer, k As Integer, nP As Integer, nPh As Integer
    Dim nZ As Integer, nZ_max As Integer
    Dim var As Variant, str As String
    
    'extract `ADEPT_Input`
    var = Math_Transpose(ADEPT_Input.Value2)
    
    '-- production data
    Dim t_sim As Double, mFlow As Double, q As Double, L As Double, R As Double, d As Double, _
        rho As Double, mu As Double
    t_sim = var(1)      'simulation time (s)
    mFlow = var(2)      'mass flow rate (kg/s)
    L = var(3)          'pipe length (m)
    R = var(4): d = 2 * R 'pipe ID (m)
    
    '-- other settings
    Dim mix_phase As Variant, Solver_Vars As Variant
    k = 30: mix_phase = Array(var(k + 0), var(k + 1), var(k + 2))
    k = 33: Solver_Vars = Array(var(k + 0), var(k + 1), var(k + 2))
    
    If t_sim = 0 Or mFlow = 0 Or L = 0 Or R = 0 Then
        GoTo ExitFunc
    End If
    
    'mass flow, vol flow
    'mFlow = q * rho     'm[kg/s]=q[m3/s]*rho[kg/m3]
    q = mFlow / rho
    q = UnitConverter("volflow", q, "m3/s", "stb/d")
    
    'get units
    Dim Units As Variant: Units = Read_Units(sht_P)
    
    'extract Asp_DepoInputs
    Dim Asp_DepoVal As Range
    Set Asp_DepoVal = subRange(Asp_DepoInputs, 3, , 1)
    
    Dim Asp_zPTx As Range, Asp_x0 As Range, Asp_FluidProps As Range, Asp_RateParam As Range
    Set Asp_zPTx = Asp_DepoVal.Columns("a:d")
    Set Asp_x0 = Asp_DepoVal.Columns("g:h")
    Set Asp_FluidProps = Asp_DepoVal.Columns("i:z")
    Set Asp_RateParam = Asp_DepoVal.Columns("aa:ac")
    
    'thermodynamics lookup table (TLUT)
    Dim dict_TLUT As New Scripting.Dictionary
    Set dict_TLUT = ADEPT_LUT_Thermo(ADEPT_Input, Asp_zPTx, Asp_FluidProps, Units)
    
    nZ = dict_TLUT("nZ")
    If nZ = 0 Then
        GoTo ExitFunc
    End If
    
    'kinetics lookup table (KLUT)
    Dim dict_KLUT As New Scripting.Dictionary
    Set dict_KLUT = ADEPT_LUT_Kinetics(ADEPT_Input, Asp_x0, Asp_RateParam, nZ)
    
    'populate input dict
    Dim dict_ADEPT As New Scripting.Dictionary
    '---solver settings
    dict_ADEPT("Solver_Vars") = Solver_Vars
    dict_ADEPT("mix_phase") = mix_phase
    '---thermo and kinetics lookup table
    Set dict_ADEPT("dict_TLUT") = dict_TLUT
    Set dict_ADEPT("dict_KLUT") = dict_KLUT
    
    'solve ADEPT deposition model and return
    Set DepoModel_ADEPT = ADEPT_Solver(t_sim, mFlow, L, R, dict_ADEPT)
    
ExitFunc:

End Function

Private Function ADEPT_LUT_Thermo(ADEPT_Input As Range, Asp_zPTx As Range, Asp_FluidProps As Range, Optional Units As Variant, Optional rho_mix As String = "volume")

'=========================================================
' Program:      ADEPT_LUT_Thermo
' Description:  construct thermo lookup table (TLUT)
' Called by:    DepoModel_ADEPT
' Notes:
'
'---------- INPUTS ---------------------------
' ADEPT_Input:  refer to DepoModel_ADEPT function
' Asp_zPTx:     PT-profile
'   (1):        depth (-)
'   (2):        depth (ft)
'   (3):        pressure (psig)
'   (4):        temperature (F)
' Asp_FluidProps: refer to DepoModel_ADEPT function
' Units:        input/output units
'
'---------- RETURN ---------------------------
' dict_TLUT:    thermo lookup table (TLUT)
'
'==========================================================

    Dim i As Integer, j As Integer, k As Integer, _
        nZ As Integer, nZ_max As Integer
    Dim var As Variant
    
    'extract units
    If IsEmpty(Units) = True Then
        Units = Read_Units()
    End If
    Dim unit_T As String, unit_P As String, unit_V As String, unit_D As String, unit_N As String, unit_Visco As String
    Call Get_Units(Units, "input", unit_T, unit_P, unit_V, unit_D, unit_N, , , unit_Visco)
    
    'extract data from ranges
    
    'extract `ADEPT_input`
    var = Math_Transpose(ADEPT_Input.Value2)
    
    '-- fluid properties
    Dim c0 As Double, c_Ceq As Double, c_dens As Variant, c_SP As Variant, c_visco As Variant, c_yAsp As Variant, str_thermo As String
    c0 = var(5)                                 'Asp conc (kgAsp/m3 Oil)
    c_Ceq = var(6)                              'Asp solubility (gAsp[L1]/gAsp)
    c_dens = Array(var(7), var(8), var(9))      'density (kg/m3) in [L1,L2,Asp]
    c_SP = Array(var(10), var(11), var(12))     'solubility parameter (MPa^0.5) in [L1,L2,Asp]
    c_visco = Array(var(13), var(14), var(15))  'visco (Pa.s) in [L1,L2,Asp]
    c_yAsp = Array(var(16), var(17))            'Asp compo (wtf) in [L1,L2]
    str_thermo = var(18)                        'lookup table input type (table or constant)
    
    '-- Asp_zPTx: PT-profile
    Dim z_ As Variant, z_ft As Variant, T As Variant, P As Variant
    z_ = Math_Transpose(Asp_zPTx.Columns("a").Value2)
    z_ft = Math_Transpose(Asp_zPTx.Columns("b").Value2)
    P = Math_Transpose(Asp_zPTx.Columns("c").Value2)
    T = Math_Transpose(Asp_zPTx.Columns("d").Value2)
    
    '-- Asp_FluidProps: fluid thermo and transport properties
    Dim Asp_Ceq As Variant, Asp_beta As Variant, Asp_dens As Variant, Asp_SP As Variant, Asp_visco As Variant, Asp_yAsp As Variant
    Asp_Ceq = Math_Transpose(Asp_FluidProps.Columns("a").Value2)
    Asp_yAsp = Asp_FluidProps.Columns("b:c").Value2
    Asp_beta = Asp_FluidProps.Columns("d:f").Value2
    Asp_dens = Asp_FluidProps.Columns("g:j").Value2
    Asp_SP = Asp_FluidProps.Columns("k:n").Value2
    Asp_visco = Asp_FluidProps.Columns("o:r").Value2
    
    'extract properties to arrays
    Dim nPh As Integer
    Dim depth() As Double, depth_ft() As Double, depth_m() As Double
    Dim TK() As Double, Pbar() As Double, zAsp As Double, c0_Asp As Double, beta_sum As Double, vol_sum As Double
    Dim Ceq() As Double, beta() As Double, yAsp() As Double, volFrac() As Double, dens() As Double, SP() As Double, visco() As Double
    
    'max # length cuts (nZ_Max) and # non-zero length cuts (nZ)
    nZ_max = Asp_FluidProps.Rows.Count
    For i = 1 To nZ_max
        If IsEmpty(Asp_Ceq(i)) = False Then
            nZ = nZ + 1
        End If
    Next i
    
    'depth
    ReDim depth(1 To nZ), depth_ft(1 To nZ), depth_m(1 To nZ)
    For i = 1 To nZ
        depth(i) = z_(i)
        depth_ft(i) = z_ft(i)
        depth_m(i) = UnitConverter("length", z_ft(i), "ft", "m")
    Next i
    
    'TP properties
    ReDim TK(1 To nZ), Pbar(1 To nZ)
    For i = 1 To nZ
        TK(i) = UnitConverter("temp", T(i), unit_T, "K")
        Pbar(i) = UnitConverter("pressure", P(i), unit_P, "bar")
    Next i
    
    nPh = ArrayLen(Asp_beta, 2)
    ReDim Ceq(1 To nZ), beta(1 To nZ, 1 To nPh), yAsp(1 To nZ, 1 To nPh), _
        dens(1 To nZ, 1 To nPh + 1), SP(1 To nZ, 1 To nPh + 1), visco(1 To nZ, 1 To nPh + 1)
    
    If str_thermo = "constant" Then
        For i = 1 To nZ
            Ceq(i) = c_Ceq
            beta(i, 2) = 1
            For j = 1 To nPh
                dens(i, j + 1) = c_dens(j)
                SP(i, j + 1) = c_SP(j)
                visco(i, j + 1) = c_visco(j)
            Next j
            For j = 1 To 2
                yAsp(i, j + 1) = c_yAsp(j)
            Next j
        Next i
    Else
        For i = 1 To nZ
            'asph solubility
            Ceq(i) = Asp_Ceq(i)
            
            'phase fractions
            beta_sum = 0
            For j = 1 To nPh
                beta(i, j) = Asp_beta(i, j)
                beta_sum = beta_sum + beta(i, j)
            Next j
            For j = 1 To nPh
                beta(i, j) = beta(i, j) / beta_sum
            Next j
            'asph composition
            For j = 1 To (nPh - 1)
                yAsp(i, j + 1) = Asp_yAsp(i, j)
            Next j
            'dens, SP, visco
            For j = 1 To (nPh + 1)
                dens(i, j) = Asp_dens(i, j) * 1000  'dens (g/cc -> kg/m3)
                SP(i, j) = Asp_SP(i, j)
                visco(i, j) = Asp_visco(i, j) / 1000 'visco (cP -> Pa.s)
                
                'convert density, viscosity
                If Asp_dens(i, j) > 0 Then
                    'dens(i, j) = UnitConverter("density", Asp_dens(i, j), unit_D, "kg/m3")
                    'visco(i, j) = UnitConverter("visco", Asp_visco(i, j), unit_Visco, "Pa.s")
                End If
            Next j
        Next i
    End If
    
    'volume fraction
    ReDim volFrac(1 To nZ, 1 To nPh)
    For i = 1 To nZ
        vol_sum = 0
        For j = 1 To nPh
            If beta(i, j) > 0 Then
                volFrac(i, j) = beta(i, j) / dens(i, j)
                vol_sum = vol_sum + volFrac(i, j)
            End If
        Next j
        For j = 1 To nPh
            volFrac(i, j) = volFrac(i, j) / vol_sum
        Next j
    Next i
    
    'asph amount
    Dim betai() As Double, densi() As Double, volfi() As Double, dens_mix As Double, str As String
    ReDim betai(nPh), densi(nPh), volfi(nPh)
    i = 1
    For j = 1 To 2
        betai(j) = beta(i, j)
        densi(j) = dens(i, j)
        volfi(j) = volFrac(i, j)
    Next j
    str = rho_mix
    dens_mix = mix_Dens(betai, volfi, densi, str)
    
    zAsp = 0
    For j = 1 To nPh
        zAsp = zAsp + yAsp(i, j) * beta(i, j)   'gAsp/g
    Next j
    
    If str_thermo = "constant" Then
        c0_Asp = c0
    Else
        c0_Asp = zAsp * dens_mix     'kgAsp/cc oil
    End If
    
    'populate dict
    Dim dict As New Scripting.Dictionary
    dict("T") = TK
    dict("P") = Pbar
    dict("zAsp") = zAsp
    dict("c0_Asp") = c0_Asp
    dict("Ceq") = Ceq
    dict("yAsp") = yAsp
    dict("beta") = beta
    dict("dens") = dens
    dict("volFrac") = volFrac
    dict("SP") = SP
    dict("visco") = visco
    
    dict("depth") = depth
    dict("depth_ft") = depth_ft
    dict("depth_m") = depth_m
    dict("P_bh") = P(1)
    dict("nZ") = nZ
    dict("Units") = Units
    
    'return
    Set ADEPT_LUT_Thermo = dict
    
End Function

Private Function ADEPT_LUT_Kinetics(ADEPT_Input As Range, Asp_x0 As Range, Asp_RateParam As Range, nZ As Integer)

'=========================================================
' Program:      ADEPT_LUT_Kinetics
' Description:  construct kinetics lookup table (KLUT)
' Called by:    DepoModel_ADEPT
' Notes:
'
'---------- INPUTS ---------------------------
' ADEPT_Input: refer to DepoModel_ADEPT
' Asp_x0
'   (1):        initial fracBlock (-), Asp
'   (2):        initial fracBlock (-), DM
' Asp_RateParam (at each point along PT-trace)
'   (1-3):      kP, KAg, kD (1/s)
'
'---------- RETURN ---------------------------
' dict_KLUT:    kinetics lookup table (KLUT)
'
'==========================================================

    Dim i As Integer, j As Integer, k As Integer
    Dim var As Variant
    
    'extract `Asp_RareParam`: ADEPT rate parameters
    Dim Asp_kP As Variant, Asp_kAg As Variant, Asp_kD As Variant
    Asp_kP = Math_Transpose(Asp_RateParam.Columns("a").Value2)
    Asp_kAg = Math_Transpose(Asp_RateParam.Columns("b").Value2)
    Asp_kD = Math_Transpose(Asp_RateParam.Columns("c").Value2)
    
    'extract `Asp_x0`: initial fracBlock
    Dim Asp_fracBlock As Variant
    Asp_fracBlock = Asp_x0.Value2
    
    'extract `ADEPT_Input`
    var = Math_Transpose(ADEPT_Input.Value2)
    
    '-- ADEPT parameters
    Dim kP As Double, kAg As Double, kD As Double
    Dim kP_param As Variant, kP_model As String, kAg_param As Variant, kAg_model As String, _
        kD_param As Variant, kD_scale As String, SR_param As Variant, SR_model As String
    Dim mFlow As Double, R As Double, rho As Double, mu As Double
    
    k = 21: kP = var(k): kAg = var(k + 1): kD = var(k + 2)
    k = 24: kP_param = parse_varStr(var(k)): kP_model = var(k + 1)
    k = 26: kAg_param = parse_varStr(var(k)): kAg_model = var(k + 1)
    k = 28: SR_param = parse_varStr(var(k)): SR_model = var(k + 1)
    mFlow = var(2): R = var(4): rho = var(7): mu = var(13)
    kD_param = Array(mFlow, R, rho, mu): kD_scale = var(20)
    
    '-- mixing rules
    Dim mix_phase As Variant:
    k = 30: mix_phase = Array(var(k + 0), var(k + 1), var(k + 2))
    
    Dim kPv() As Double, kAgv() As Double, kDv() As Double
    ReDim kPv(1 To nZ), kAgv(1 To nZ), kDv(1 To nZ)
    
    'ADEPT asp kinetic parameters (kP, kAg, kD)
    If oSum(Asp_kP) > 0 Then
        'kPv = Asp_kP    'test
        For i = 1 To nZ
            kPv(i) = Asp_kP(i)
        Next i
    Else
        For i = 1 To nZ
            kPv(i) = kP
        Next i
    End If
    If oSum(Asp_kAg) > 0 Then
        For i = 1 To nZ
            kAgv(i) = Asp_kAg(i)
        Next i
    Else
        For i = 1 To nZ
            kAgv(i) = kAg
        Next i
    End If
    If oSum(Asp_kD) > 0 Then
        For i = 1 To nZ
            kDv(i) = Asp_kD(i)
        Next i
    Else
        For i = 1 To nZ
            kDv(i) = kD
        Next i
    End If
    
    'populate dict
    Dim dict As New Scripting.Dictionary
    
    '--kinetic properties (shear removal and asphaltene kinetics)
    dict("mix_phase") = mix_phase
    '--asphaltene (Asp) rate parameters
    dict("kP_param") = kP_param: dict("kP_model") = kP_model:       'precip
    dict("kAg_param") = kAg_param: dict("kAg_model") = kAg_model:   'aggreg
    dict("kD_param") = kD_param: dict("kD_scale") = kD_scale:       'depo
    dict("SR_param") = SR_param: dict("SR_model") = SR_model:       'shear removal
    dict("Asp_kP") = kPv: dict("Asp_kAg") = kAgv: dict("Asp_kD") = kDv:
    '--initial frac blocked
    dict("Asp_fracBlock") = Asp_fracBlock
    
    'return
    Set ADEPT_LUT_Kinetics = dict
    
End Function

Private Function ADEPT_Solver(t_sim As Double, mFlow As Double, L As Double, R As Double, _
        dict_ADEPT As Scripting.Dictionary)
    
'=========================================================
' Program:      ADEPT_Solver
' Description:  ADEPT model for asphaltene deposition in wellbore
' Called by:    DepoModel_ADEPT
' Notes:        see below
'
'---------- INPUTS ---------------------------
' t_sim:    simulation time (s)
' mFlow:    production rate (kg/s)
' L:        pipe length (m)
' R:        pipe radius (m)
'
' --- arrays --- (nRows = number of spatial points)
' dict_ADEPT:       dict of ADEPT inputs
'   Solver_Vars
'       SS:         {(0 = transient), 1 = steady-state}
'       SolverType: {("IE" = implicit euler), "EE" = explicit euler}
'       nTimeCuts:  # time cuts (=10)
'   mix_phase
'       dens:       density mixing (="volume")
'       visco:      viscosity mixing (="volume")
'       velocity:   velocity mixing (="sum")
'   dict_TLUT:  thermo lookup table (TLUT)
'       T:          temperature [K]
'       P:          pressure [bar]
'       zAsp:       asphaltene composition (g[Asp]/g[T]) at inlet
'       c0_Asp:     asphaltene concentration (kg[Asp]/m3[T]) at inlet
'       Ceq:        asphaltene solubility (gAsp[L1] / gAsp)
'       yAsp:       asphaltene composition (gAsp[k] / g[k]) in [V,L1,L2]
'       beta:       phase fractions (wtf) in [V,L1,L2]
'       dens:       density (kg/m3) in [V,L1,L2,Asp]
'       SP:         solubility parameter (MPa^0.5) in [V,L1,L2,Asp]
'       visco:      viscosity (cP) in [V,L1,L2,Asp]
'   dict_KLUT:  kinetics lookup table (KLUT)
'       Asp_kP:     precipitation rate (1/s)
'       Asp_KAg:    aggregation rate (1/s)
'       Asp_kD:     deposition rate, unscaled (1/s)
'       SR_param:   shear removal rate
'       Asp_fracBlock: initial pipe blockage
'
'---------- OUTPUTS ---------------------------
' out: array along length of pipe
'   C:          primary particle concentration
'   C_df:       driving force (Cf-Ceq) for precipitation
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
' Authors: Mohammed IL Abutaqiya, Caleb J Sisco
' Objective: perform time march, solve PDEs for dissolved
'   Asp conc (Cf) and primary particle conc (C) at each iter
'
' 1. routine includes terms for asphaltene kinetics [kP,kAg,kD] and shear removal [tau0, k, n]
' 2. pressure drop (dP) calculation is performed after each time march, dP=f(L, del=depo thickness)
' 3. velocity updates at each time marching step to include pipe restriction due to deposition, u=f(del)
' 4. thermo property values [Ceq, dens, visco, etc.] should update to reflect changes in dP after each time march (not implemented in current routine!!)
'==========================================================

    Dim val As Double, val1 As Double, val2 As Double, Pi As Double
    Dim i As Integer, j As Integer, k As Integer, nP As Integer, nPh As Integer, nRo As Integer, nCo As Integer, nT_solve As Integer, iT_out As Integer, nT_out As Integer, dt_out As Double, T_out As Double
    Dim Z As Double, Z0 As Double, zf As Double, dz As Double, iZ As Integer, nZ As Integer, z_nP() As Double, nZ_max As Integer, _
        m As Double, Y As Double, y1 As Double, y2 As Double, X As Double, x1 As Double, x2 As Double, found As Integer, _
        T As Double, dti As Double, T0 As Double, tf As Double, dT As Double, dt_arr() As Double, iT As Integer, nT As Integer, _
        dt1 As Double, dt2 As Double, dt3 As Double, _
        R0 As Double, L0 As Double, q As Double, _
        id As Double, d As Double, Ri As Double, L_cm As Double, A As Double
    Dim kDiss As Double, kD0 As Double, Dm As Double, phi As Double, ScF As Double
    Dim c0 As Double, Cf() As Double, C() As Double, Cag() As Double, Cd() As Double, C_df() As Double, _
        Cf_solve() As Double, C_solve() As Double, Cag_solve() As Double, Cd_solve() As Double
    
    
    Dim uz As Double, rho As Double, mu As Double, rho_dm As Double, rho_Asp As Double, _
        V_pipe As Double, V_pipeAsp As Double, V_pipeDM As Double, Vfrac_Asp As Double, Vfrac_DM As Double, dz_m As Double, dr_m As Double, _
        mass_Asp_Sum As Double, mass_Asp_Max As Double, mass_Asp_Frac As Double, J_Asp_max As Double
    Dim det_Asp As Double, det_DM As Double, rD As Double
    Dim delf_Asp() As Double, delf_DM() As Double
    Dim delf_Asp_max As Double, delf_DM_max As Double, idx_max_Asp As Long, idx_max_DM As Long
    
    Dim Cf_in As Double, Cf_t0() As Double, C_in As Double, C_t0() As Double
    Dim dPdT As Double, dP() As Variant, dPf As Double, dPf_arr(1 To 2) As Double, dP_flowStatus As String, P_bh As Double, dPf_old As Double
    Dim SimPercent As Integer, ti_Day As Double, t_noFlow As Double, str_msg As String
    
    Dim dict_Cfsolve As New Scripting.Dictionary, dict_Csolve As New Scripting.Dictionary
    
    'extract from `dict_ADEPT`
    
    '-- sim settings
    Dim Solver_Vars() As Variant, SS As Integer, SolverType As String, nTC As Integer, nTC_Vars As Integer
    Dim mix_phase As Variant, rho_mix As String, mu_mix As String, u_mix As String
    
    '--     solver settings
    Solver_Vars = dict_ADEPT("Solver_Vars")
    SolverType = Solver_Vars(1): SS = Solver_Vars(2): nTC = Solver_Vars(3)
    '--     mixing rules
    mix_phase = dict_ADEPT("mix_phase")
    rho_mix = mix_phase(1): mu_mix = mix_phase(2): u_mix = mix_phase(3)
    
    '-- fluid (thermo/transport) properties
    Dim dict_TLUT As New Scripting.Dictionary
    Dim TLUT_T() As Double, TLUT_P() As Double, TLUT_zAsp As Double, TLUT_c0 As Double, _
        TLUT_Ceq() As Double, TLUT_yAsp() As Double, TLUT_beta() As Double, TLUT_dens() As Double, TLUT_visco() As Double, TLUT_SP() As Double
    
    Set dict_TLUT = dict_ADEPT("dict_TLUT")
    TLUT_T = dict_TLUT("T")
    TLUT_P = dict_TLUT("P")
    TLUT_zAsp = dict_TLUT("zAsp")
    TLUT_c0 = dict_TLUT("c0_Asp")
    TLUT_Ceq = dict_TLUT("Ceq")
    TLUT_yAsp = dict_TLUT("yAsp")
    TLUT_beta = dict_TLUT("beta")
    TLUT_dens = dict_TLUT("dens")
    TLUT_SP = dict_TLUT("SP")
    TLUT_visco = dict_TLUT("visco")
    
    '-- kinetic (ADEPT model) properties
    Dim dict_KLUT As New Scripting.Dictionary, dict_dP As New Scripting.Dictionary
    Dim kP_param As Variant, kP_model As String, kAg_param As Variant, kAg_model As String, _
        kD_param As Variant, kD_scale As String, SR_param As Variant, SR_model As String, _
        Asp_kP() As Double, Asp_kAg() As Double, Asp_kD() As Double, Asp_fracBlock() As Variant
        
    Set dict_KLUT = dict_ADEPT("dict_KLUT")
    '-- Asp rate parameters
    Asp_kP = dict_KLUT("Asp_kP"): Asp_kAg = dict_KLUT("Asp_kAg"): Asp_kD = dict_KLUT("Asp_kD")
    kP_param = dict_KLUT("kP_param"): kP_model = dict_KLUT("kP_model")
    kAg_param = dict_KLUT("kAg_param"): kAg_model = dict_KLUT("kAg_model")
    kD_param = dict_KLUT("kD_param"): kD_scale = dict_KLUT("kD_scale")
    SR_param = dict_KLUT("SR_param"): SR_model = dict_KLUT("SR_model")
    '-- initial fraction blocked
    Asp_fracBlock = dict_KLUT("Asp_fracBlock")
    
    '--/ fluid properties
    
    'length of input Ceq array
    nP = ArrayLen(TLUT_Ceq)
    
    c0 = TLUT_c0 * 1    'conc Asp (kgAsp/m3 oil)
    C_in = 0            'conc primary particles (C) at inlet (z=0)
    Cf_in = 1           'conc dissolved asphaltenes (Cf) at inlet (z=0)
    id = 2 * R          'pipe diameter (m)
    P_bh = 10 ^ 6       'dummy value of bottom-hole P
    
    'handle time = 0
    'If t_sim = 0 Then t_sim = 0.01
    
    'discretize pipe into nP points
    ReDim z_nP(1 To nP)
    For iZ = 1 To nP
        z_nP(iZ) = ((iZ - 1) / (nP - 1))
    Next iZ
    
    'spatial steps
    nZ_max = 500
    nZ = nP
    Z0 = 0
    zf = 1
    dz = (zf - Z0) / (nZ - 1)
    
    'extract fluid properties and rate parameters (& interpolate) along spatial steps
    nPh = ArrayLen(TLUT_beta, 2)
    
    Dim Ts() As Double, Ps() As Double, Ceq() As Double, beta() As Double, dens() As Double, dens_Asp() As Double, vol() As Double, _
        SP() As Double, SP_Asp() As Double, visco() As Double, yAsp() As Double, _
        velo_L1() As Double, velo() As Double, volFlow() As Double
    Dim kP() As Double, kAg() As Double, kD() As Double, kD_us() As Double, _
        Da_P() As Double, Da_Ag() As Double, Da_D() As Double, rSR() As Double, Re() As Double, Pe() As Double, tau() As Double, _
        Dax As Double, fricFactor As Double, uf As Double, del_lam As Double, del_wall As Double, del_mom As Double, del As Double
    Dim del_Asp() As Double, del_DM() As Double, del_Asp0() As Double, del_DM0() As Double, _
        J_Asp() As Double, mass_Asp() As Double, y_Asp As Double, vol_sum As Double, beta_sum As Double, mass_Tot As Double
    
    ReDim Ts(1 To nZ), Ps(1 To nZ), Ceq(1 To nZ), beta(1 To nZ, 1 To nPh), dens(1 To nZ, 1 To nPh), dens_Asp(1 To nZ), vol(1 To nZ, 1 To nPh), _
        SP(1 To nZ, 1 To nPh), SP_Asp(1 To nZ), visco(1 To nZ, 1 To nPh), yAsp(1 To nZ, 1 To nPh)
    ReDim kP(1 To nZ), kAg(1 To nZ), kD(1 To nZ), kD_us(1 To nZ)
    ReDim velo_L1(1 To nZ), Q_VLL(1 To nZ, 1 To nPh), velo(1 To nZ, 1 To nPh), volFlow(1 To nZ, 1 To nPh), _
        Da_P(1 To nZ), Da_Ag(1 To nZ), Da_D(1 To nZ), rSR(1 To nZ), Re(1 To nZ), Pe(1 To nZ), tau(1 To nZ), _
        del_Asp(1 To nZ), del_DM(1 To nZ), del_Asp0(1 To nZ), del_DM0(1 To nZ), delf_Asp(1 To nZ), delf_DM(1 To nZ)
    
    Dim Tz As Double, Pz As Double, del_SPi As Double, _
        volfi() As Double, betai() As Double, densi() As Double, SPi() As Double, visci() As Double, veloi() As Double
    ReDim densi(1 To 3), SPi(1 To 3), visci(1 To 3), veloi(1 To 3), volfi(1 To 3), betai(1 To 3)

    Dim T_nZ() As Double, P_nZ() As Double, _
        R_nZ() As Double, R_nZ_Asp() As Double, R_nZ_DM() As Double, depth_m() As Double, depth_ft() As Double
    ReDim T_nZ(1 To nZ), P_nZ(1 To nZ), R_nZ(1 To nZ), R_nZ_Asp(1 To nZ), R_nZ_DM(1 To nZ), depth_m(1 To nZ), depth_ft(1 To nZ), _
        Cf_t0(1 To nZ), C_t0(1 To nZ), dt_arr(1 To nZ), Cf(1 To nZ), C(1 To nZ), Cag(1 To nZ), C_df(1 To nZ)
    
    'check if initial deposit is input
    Dim flag_fracBlock As Integer
    If IsEmpty(Asp_fracBlock) = False Then
        If oSum(Asp_fracBlock) > 0 Then
            flag_fracBlock = 1
        Else:
            flag_fracBlock = 0
        End If
    End If
    
    'construct lookup table (this will change in Python code)
    i = 1
    X = 0
    For iZ = 1 To nZ
        
        'populate array of depth (ft). required for dP model
        depth_m(iZ) = L * (zf - dz * (iZ - 1))
        depth_ft(iZ) = depth_m(iZ) / 0.3048
        
        'find the upper and lower bound of current z
        i = 1
        x2 = z_nP(i)
        Do While i < nP And x2 <= X
            i = i + 1
            x2 = z_nP(i)
        Loop
        j = i - 1
        x1 = z_nP(j)
        
        'interpolate thermo properties and kinetics (rate) parameters
        '-- thermo properties
        T_nZ(iZ) = Interpolate(X, x1, x2, TLUT_T(j), TLUT_T(i))     'temp (K)
        P_nZ(iZ) = Interpolate(X, x1, x2, TLUT_P(j), TLUT_P(i))     'pressure (bar)
        Ceq(iZ) = Interpolate(X, x1, x2, TLUT_Ceq(j), TLUT_Ceq(i))  'asph solubility (gL1/gOvr)
        
        beta_sum = 0: vol_sum = 0
        For k = 1 To nPh
            yAsp(iZ, k) = Interpolate(X, x1, x2, TLUT_yAsp(j, k), TLUT_yAsp(i, k))  'asph composition (wtfrac)
            beta(iZ, k) = Interpolate(X, x1, x2, TLUT_beta(j, k), TLUT_beta(i, k))  'phase fraction (wtfrac)
            dens(iZ, k) = Interpolate(X, x1, x2, TLUT_dens(j, k), TLUT_dens(i, k))  'density (kg/m3)
            SP(iZ, k) = Interpolate(X, x1, x2, TLUT_SP(j, k), TLUT_SP(i, k))        'solubility parameter (MPa^0.5)
            visco(iZ, k) = Interpolate(X, x1, x2, TLUT_visco(j, k), TLUT_visco(i, k)) 'viscosity (Pa.s) (1 cP = 1 mPa.s = 0.001 Pa.s)
            
            'sum total mass, volume
            If dens(iZ, k) > 0 Then
                vol(iZ, k) = beta(iZ, k) / dens(iZ, k)
                beta_sum = beta_sum + beta(iZ, k)
                vol_sum = vol_sum + vol(iZ, k)
            End If
        Next k
        
        '-- normalize mass, volume
        For k = 1 To nPh
            beta(iZ, k) = beta(iZ, k) / beta_sum
            vol(iZ, k) = vol(iZ, k) / vol_sum
        Next k
        
        '-- rate parameters
        k = 4
        dens_Asp(iZ) = Interpolate(X, x1, x2, TLUT_dens(j, 4), TLUT_dens(i, 4)) 'Asp density (kg/m3)
        SP_Asp(iZ) = Interpolate(X, x1, x2, TLUT_SP(j, 4), TLUT_SP(i, 4))       'Asp SP (MPa^0.5)
        kP(iZ) = Interpolate(X, x1, x2, Asp_kP(j), Asp_kP(i))           'Asp precip rate (1/s)
        kAg(iZ) = Interpolate(X, x1, x2, Asp_kAg(j), Asp_kAg(i))        'Asp aggreg rate (1/s)
        kD_us(iZ) = Interpolate(X, x1, x2, Asp_kD(j), Asp_kD(i))        'Asp depo rate (1/s)
        
        If flag_fracBlock = 1 Then
            del_Asp0(iZ) = R * Interpolate(X, x1, x2, Asp_fracBlock(j, 1), Asp_fracBlock(i, 1))     'initial pipe frac blocked, Asp (m)
            del_DM0(iZ) = R * Interpolate(X, x1, x2, Asp_fracBlock(j, 2), Asp_fracBlock(i, 2))      'initial pipe frac blocked, DM (m)
        End If
        
        'next model z
        X = X + dz
                
    Next iZ
    
    'populate radius profile (will update during time march loop due to deposition)
    For iZ = 1 To nZ
        del_Asp(iZ) = del_Asp0(iZ) * 1      'depo thickness, initial (m) Asp
        del_DM(iZ) = del_DM0(iZ) * 1        'depo thickness, initial (m) DM
        
        delf_Asp(iZ) = del_Asp(iZ) / R      'frac blocked, Asp
        delf_DM(iZ) = del_DM(iZ) / R        'frac blocked, DM
        
        R_nZ_Asp(iZ) = R - del_Asp(iZ) / 1  'initial pipe radius, Asp (m)
        R_nZ_DM(iZ) = R - del_DM(iZ) / 1    'initial pipe radius, DM (m)
        R_nZ(iZ) = R_nZ_DM(iZ)
    Next iZ
    
    'populate initial conditions for Cf and C
    For iZ = 1 To nZ
        Cf_t0(iZ) = Cf_in   'conc dissolved Asp (Cf), initial
        C_t0(iZ) = C_in     'conc primary particles (C), initial
    Next iZ
    
    '=========================================================
    '--- time march loop (solve SS ADEPT at each step) ---
    '=========================================================
    Dim Rz As Double, R_pipe_cm As Double, d_in As Double, dt0 As Double, flag_timecut As Boolean, t_new As Double, t_res As Double, t_out_days As Double
    Dim itr As Integer, itr_max As Integer, num_TC As Integer
    
    Dim dens_Z() As Double, visco_Z() As Double, velo_Z() As Double
    ReDim dens_Z(1 To nZ), visco_Z(1 To nZ), velo_Z(1 To nZ)
    
    'handle t_sim=0 (for output_TC var)
    If t_sim <= 0 Then
        t_sim = 0
        num_TC = 0
    End If
    
    'handle nTimeCuts=0 (for output_TC var)
    If nTC = 0 Then
        nT_out = 1
    Else
        nT_out = nTC + 1
        nCo = 0
    End If
    dt_out = 0
    If t_sim > 0 Then
        dt_out = t_sim / nTC    '?? how to handle if nTC = 0
    End If
    T_out = 0 'dt_out ' first time step to report output
    iT_out = 0
    
    'time cut vars
    'dimensionalize output_TC variable
    Dim output_TC() As Variant
    Dim TC_t_out(), TC_C_sum(), TC_Cag_sum(), TC_Cd_sum(), _
        TC_kgAsp_Tot(), TC_kgAsp_Frac(), _
        TC_del_Asp(), TC_del_DM(), TC_Vfrac_Asp(), TC_Vfrac_DM(), _
        TC_J_Asp_max(), TC_dPf(), TC_dPdt(), TC_delf_max_zLoc(), _
        TC_Depo_Asp_kg(), TC_Depo_Asp_kgf()
    ReDim TC_t_out(nT_out), TC_C_sum(nT_out), TC_Cag_sum(nT_out), TC_Cd_sum(nT_out), _
        TC_kgAsp_Tot(nT_out), TC_kgAsp_Frac(nT_out), _
        TC_del_Asp(nT_out), TC_del_DM(nT_out), TC_Vfrac_Asp(nT_out), TC_Vfrac_DM(nT_out), _
        TC_J_Asp_max(nT_out), TC_dPf(nT_out), TC_dPdt(nT_out), TC_delf_max_zLoc(nT_out), _
        TC_Depo_Asp_kg(nT_out), TC_Depo_Asp_kgf(nT_out)
    
    If nTC > 0 Then
        nTC_Vars = 10
        k = 3 * nT_out
        ReDim output_TC(1 To nZ_max, 1 To (k + nTC_Vars))   '3 properties at each time cut (nT_out) + 10 time-dependent vars
    End If
    
    'time step (s)
    Dim nDays As Double
    nDays = t_sim / 86400
    If nDays > 10 Then
        dt0 = (1) * 86400 'day to sec
    ElseIf nDays < 1.5 Then
        dt0 = (0.01) * 86400
    Else
        dt0 = (0.1) * 86400
    End If
    'dt0 = (1) * 86400 'day to sec
    
    'ensure step size corresponds to a number of steps at least = nTC
    nT_solve = t_sim / dt0
    If nT_solve < nT_out Then dt0 = dt_out
    
    'total properties
    'total tubing volume, m3
    Dim V_pipe_eq As Double
    V_pipe = shp_VolCylinder(L, R)
    
    'total flows
    ' t_sim:    time of operation (s)
    ' mFlow:    production rate (kg/s)

    'mass flows (kg), oil and Asp
    Dim kgOil_Tot As Double, mFlow_Asp As Double, kgAsp_Tot As Double
    mFlow_Asp = mFlow * TLUT_zAsp   'kgAsp/s
    kgOil_Tot = mFlow * t_sim       'kg oil (for full t_sim)
    kgAsp_Tot = mFlow_Asp * t_sim   'kgAsp (for full t_sim)
    
    'vol flows (m3), oil and Asp
    Dim m3Oil_Tot As Double, m3Asp_Tot As Double, dens_Oil As Double
    m3Oil_Tot = kgOil_Tot / dens(1, 2)
    m3Asp_Tot = kgAsp_Tot / dens_Asp(1)
    
    
    '--- time march loop ---
    T = 0
    dti = dt0
    itr = 0
    itr_max = 1000
    If t_sim = 0 Then itr_max = 1    'handle empty pipe (t_sim=0) simulation
    Do While T <= t_sim And itr < itr_max
        itr = itr + 1
             
        'break point for testing
        If itr = 8 Then
            'itr = itr
        End If
        
        If itr = 1 Then
            dti = 0 'first iteration is empty pipe
        End If
        
        'try a full time step
        t_new = T + dti
        
        'restrict time step if t_new exceeds t_out
        If t_new > T_out Then
            dti = t_new - T_out 'will change only for the next time step
            t_new = T_out       'restrict step so values are exactly equal
        Else
            dti = dt0           'return to original time step
        End If
        
        'actual time step taken
        dT = t_new - T
        
        'update radius profile (m) due to restriction
        For iZ = 1 To nZ
            R_nZ_Asp(iZ) = R - del_Asp(iZ)
            R_nZ_DM(iZ) = R - del_DM(iZ)
            R_nZ(iZ) = R_nZ_DM(iZ)
        Next iZ
        
        'calculate phase velocity (m/s) at each point from mass flows
        For iZ = 1 To nZ
            A = shp_AreaCircle(R_nZ(iZ)) 'm2
            
            For k = 1 To nPh
                If beta(iZ, k) > 0 Then
                    volFlow(iZ, k) = mFlow * beta(iZ, k) / dens(iZ, k) '[m3/s] .=[kg/s]*[kg[k]/kg]*[m3/kg]
                    velo(iZ, k) = volFlow(iZ, k) / A 'm/s
                End If
            Next k
            velo_L1(iZ) = velo(iZ, 2)
        Next iZ
        
        '===============================================================================================
        '--- scaling factor (ScF) & Dahmkohler numbers (Da, reaction parameters) at each spatial point
        'Da_P:  precip
        'Da_Ag: aggreg
        'Da_D:  depo
        '===============================================================================================
        
        For iZ = 1 To nZ
            Rz = R_nZ(iZ)     'radius at spatial point i
            Tz = T_nZ(iZ)
            Pz = P_nZ(iZ)
            del_SPi = SP_Asp(iZ) - SP(iZ, 2)
            
            'avg density, viscosity, velocity
            beta_sum = 0: vol_sum = 0
            For k = 1 To 2
                beta_sum = beta_sum + beta(iZ, k)   'total mass
                vol_sum = vol_sum + vol(iZ, k)      'total volume
            Next k
            For k = 1 To 2
                betai(k) = beta(iZ, k) / beta_sum   'mass fraction
                volfi(k) = vol(iZ, k) / vol_sum     'volume fraction
                densi(k) = dens(iZ, k)
                SPi(k) = SP(iZ, k)
                visci(k) = visco(iZ, k)
                veloi(k) = velo(iZ, k)
            Next k
            
            'density and viscosity averaging (="none" for L1 only, "mass", or "volume")
            'velocity averaging (="none" for L1 only, or "sum")
            rho = mix_Dens(betai, volfi, densi, rho_mix)
            mu = mix_Visco(betai, volfi, densi, visci, rho, mu_mix)
            uz = mix_Velo(betai, veloi, u_mix)
            
            'convert rho, mu
            'rho = UnitConverter("dens", rho, "g/cc", "kg/m3")   'density (g/cc -> kg/m3)
            'mu = UnitConverter("visco", mu, "cP", "Pa.s")       'visco (cP -> Pa.s)
            
            'store dens, visco, velocity at current length, z (not necessary for current routine)
            dens_Z(iZ) = rho
            visco_Z(iZ) = mu
            velo_Z(iZ) = uz
            
            'residence time (s)
            t_res = L / uz
            
            'axial dispersion, Peclet number
            Dm = ADEPT_Diffusivity(Tz, mu)
            Re(iZ) = 2 * Rz * uz * rho / mu          'Reynolds number
            If Re(iZ) < 2500 Then
                Dax = Dm + (Rz * uz) ^ 2 / 48 / Dm   'laminar, e.g. capillary tube
            Else
                Dax = 2 * Rz * uz / 5                'turbulent flow, e.g. wellbore
            End If
            Pe(iZ) = uz * L / Dax                   'Peclet number
                
            'boundary layer calculation
            fricFactor = ADEPT_fricFactor(Re(iZ))           'friction factor
            tau(iZ) = 1 / 8 * rho * fricFactor * uz ^ 2     'shear stress at wall, Pa
            uf = (tau(iZ) / rho) ^ 0.5                      'friction velocity, m/s
            del_wall = mu / rho / uf                        'wall layer thickness, m
            del_lam = 5 * del_wall                          'laminar boundary layer, m
            del_mom = 62.7 * (2 * Rz) * Re(iZ) ^ (-7 / 8)   'momentum boundary layer, m
            
            'choose boundary layer
            del = del_lam
            
            'ADEPT rate parameters
            '-- precipitation
            If oSum(kP_param) = 0 Then
                'kP(iZ) = Asp_kP(iZ)
            Else
                kP(iZ) = Exp(ADEPT_kP(kP_param, Tz, del_SPi, kP_model))
            End If
            
            '-- redissolution
            kDiss = ADEPT_kDiss(0) 'kP(iZ))
            
            '-- aggregation
            If oSum(kAg_param) = 0 Then
                'kAg(iZ) = Asp_kAg(i)
            Else
                kAg(iZ) = Exp(ADEPT_kAg(kAg_param, Tz, del_SPi, mu * 1000, kAg_model)) * c0
            End If
            
            '-- deposition
            kD0 = kD_us(iZ)     'unscaled depo parameter (=0 to turn off deposition term)
            If kD0 = 0 Then
                kD(iZ) = 0
            ElseIf oSum(kD_param) = 0 Then
                kD(iZ) = kD0
            Else
                'kD_param(1) = mFlow: kD_param(2) = R: kD_param(3) = rho: kD_param(4) = mu
                'kD(iZ) = ADEPT_kD(kD0, Tz, kD_param, kD_scale, uz, del_mom)
            End If
            
            'TEMP (from v2.44)
            'scale deposition constant
            Dm = ADEPT_Diffusivity()
            del = del_lam
            phi = Dm / del ^ 2 / kD0
            ScF = 2 * del / R * (phi / (phi + 1))
            kD(iZ) = ScF * kD0
            '/TEMP
            
            '-- shear removal rate, rSR
            rSR(iZ) = ADEPT_rSR(SR_param, tau(iZ))
            
            'Damkohler numbers
            Da_P(iZ) = kP(iZ) * t_res
            Da_Ag(iZ) = kAg(iZ) * t_res
            Da_D(iZ) = kD(iZ) * (1 - rSR(iZ)) * t_res
            
            'non-dimensional time step (for transient simulation)
            dt_arr(iZ) = dT / t_res
            
        Next iZ
        '-----/ ScF and Da number calculation -----
        
        
        '----- solve governing PDEs ----------------------------------------------
        'take only one time step per iteration
        nT = 2
        
'        Dim dict_C As New Scripting.Dictionary
'        dict_C("Ceq") = Ceq: dict_C("Cf_in") = Cf_in: dict_C("Cf_t0") = Cf_t0:
'        dict_C("C_in") = C_in: dict_C("C_t0") = C_t0:
'        dict_C("Pe") = Pe: dict_C("kDiss") = kDiss: dict_C("Da_P") = Da_P: dict_C("Da_Ag") = Da_Ag: dict_C("Da_D") = Da_D
'        dict_C("SolverSettings") = Solver_Vars
'        Set dict_Csolve = ADEPT_Solver_Conc(T0, nT, dt_arr, Z0, nZ, dz, dict_C)
        
        'solve Cf(t,z), dissolved asphaltene conc
        Cf_solve = ADEPT_Solver_Cf(T0, nT, dt_arr, Z0, nZ, dz, Cf_in, Cf_t0, Da_P, Ceq, SS, SolverType)
        
        'solve C(t,z), primary particle conc
        Set dict_Csolve = ADEPT_Solver_C(T0, nT, dt_arr, Z0, nZ, dz, C_in, C_t0, kDiss, Pe, Da_P, Da_Ag, Da_D, Cf_solve, Ceq, SS, SolverType)
        C_solve = dict_Csolve("C")
        Cag_solve = dict_Csolve("Cag")
        Cd_solve = dict_Csolve("Cd")
        
        'extract solved Cf, C, Cag
        ReDim Cf(1 To nZ), C(1 To nZ), Cag(1 To nZ), Cd(1 To nZ)
        If SS = 1 Then
            'steady-state
            For iZ = 1 To nZ
                Cf(iZ) = Cf_solve(iZ)
                C(iZ) = C_solve(iZ)
                Cag(iZ) = Cag_solve(iZ)
                Cd(iZ) = Cd_solve(iZ)
            Next iZ
        Else
            'transient (nT is the last time element of array)
            For iZ = 1 To nZ
                Cf(iZ) = Cf_solve(nT, iZ)
                C(iZ) = C_solve(nT, iZ)
                Cag(iZ) = Cag_solve(nT, iZ)
                Cd(iZ) = Cd_solve(nT, iZ)
            Next iZ
        End If
                
        'update initial concs for next time step
        Const C_tol As Double = 10 ^ -10
        For iZ = 1 To nZ
            
            'update conc driving force
            C_df(iZ) = Cf(iZ) - Ceq(iZ)
            
            If Abs(Cf(iZ)) < C_tol Then Cf(iZ) = 0
            If Abs(C(iZ)) < C_tol Then C(iZ) = 0
            If Abs(C_df(iZ)) < C_tol Then C(iZ) = 0
            
            Cf_t0(iZ) = Cf(iZ)  'soluble asphaltenes
            C_t0(iZ) = C(iZ)    'primary particles
        Next iZ
        
        
        'material balance on asphaltenes
        
        'convert conc (kg/m3) to mass amount (kg) by multiplying by the volume (m3) of the element
        
        '-------------------------------------------------------------------------
        
'        ti_Day = t_new / 86000
'        SimPercent = ti_Day / nT_solve * 100
'        Call ADEPT_Solver_StatusBar(SimPercent)
        
        '=========================================================
        '--- calculate deposition profile, flux, and other important outputs
        ' post-processing step after PDEs are solved
        '=========================================================
        
        'average y_Asp, rho_DM
        Dim Y_sum As Double, R_sum As Double, rhoA_dm As Double
        k = 0
        Y_sum = 0
        R_sum = 0
        For iZ = 1 To nZ
            If yAsp(iZ, 3) > 0 Then
                k = k + 1
                Y_sum = Y_sum + yAsp(iZ, 3)
                R_sum = R_sum + dens(iZ, 3)
            End If
        Next iZ
        y_Asp = Y_sum / k
        rho_dm = R_sum / k
        rhoA_dm = y_Asp * rho_dm
        
        'depo flux (J) and depo thickness (del) at each spatial point
        Dim del_Asp_z As Double, del_DM_z As Double
        ReDim J_Asp(1 To nZ), mass_Asp(1 To nZ)
        Dim V_cyl As Double, A_cyl As Double, dL_m As Double
        For iZ = 1 To nZ
            rhoA_dm = dens(iZ, 3) * yAsp(iZ, 3)
            rho_Asp = dens_Asp(iZ)
            
            'volume and surface area of cylindrical element
            dL_m = L * dz
            V_cyl = shp_VolCylinder(dL_m, R, R - del_Asp(iZ))
            A_cyl = shp_AreaCylinder(dL_m, R - del_Asp(iZ))
            
            'rate of asph depo [kg/m3/s]
            rD = kD(iZ) * C(iZ) * c0 * (1 - rSR(iZ))
            
            'deposition flux [kg/m2/s]. might need to revisit to account for V_bl not V_cell.
            If rD < 0.00000000001 Then  'zNOTE!!: this should be removed!
                J_Asp(iZ) = rD * (2 * R_nZ(iZ)) / 4 '=0 in the old version
            Else
                J_Asp(iZ) = rD * (2 * R_nZ(iZ)) / 4
            End If
                
            'thickness of Asp and DM deposit (assumes R >> del which is a good approx for small dt)
            If J_Asp(iZ) <> 0 Then
                del_Asp_z = dT * J_Asp(iZ) / rho_Asp
                del_DM_z = dT * J_Asp(iZ) / rhoA_dm
                del_Asp(iZ) = del_Asp(iZ) + del_Asp_z
                del_DM(iZ) = del_DM(iZ) + del_DM_z
            End If
            
            'set maximum deposit thickness to R_pipe
            If del_Asp(iZ) < 10 ^ -8 Then
                del_Asp(iZ) = 0
            ElseIf del_Asp(iZ) >= R Then
                del_Asp(iZ) = R
            End If
        
            If del_DM(iZ) < 10 ^ -8 Then
                del_DM(iZ) = 0
            ElseIf del_DM(iZ) >= R Then
                del_DM(iZ) = R
            End If
                
            'frac blocked (0=no deposit, 1=fully plugged)
            delf_Asp(iZ) = del_Asp(iZ) / R
            delf_DM(iZ) = del_DM(iZ) / R
            
        Next iZ
        '---/ depo flux (J) and thickness (del) calculation ---
        
        'mass deposited
        For iZ = 1 To nZ
            
            'element length (m)
            dL_m = L * dz
            
            'volume of deposit (m3)
            V_cyl = shp_VolCylinder(dL_m, R, R - del_Asp(iZ))
            
            'mass deposit (kg)
            mass_Asp(iZ) = V_cyl * dens_Asp(iZ)
        Next iZ
        
        '----- calculate pressure drop (dP) if at (or near) t_out -----
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
                
        '----- populate output_TC -----
        If nTC > 0 Then
            If t_new = T_out Then
                iT_out = iT_out + 1
                t_out_days = T_out / 86400
                
                'frac volume deposited =f(r)
                Dim V_pipe0 As Double, Lfrac As Double, dz_cm As Double
                L0 = 0: V_pipe0 = 0: V_pipeAsp = 0: V_pipeDM = 0
                For iZ = 1 To nZ
                    dL_m = L * dz
                    dr_m = R - R_nZ_DM(iZ)
                    V_pipe0 = V_pipe0 + shp_VolCylinder(dL_m, R)
                    If Abs(dr_m) > 10 ^ -10 Then
                        L0 = L0 + dz_m
                        V_pipeAsp = V_pipeAsp + shp_VolCylinder(dL_m, R, R_nZ_Asp(iZ))
                        V_pipeDM = V_pipeDM + shp_VolCylinder(dL_m, R, R_nZ_DM(iZ))
                    End If
                Next iZ
                Lfrac = 0: Vfrac_Asp = 0: Vfrac_DM = 0
                If V_pipe0 > 0 Then
                    Lfrac = L0 / L                      'length frac where deposit occurs
                    Vfrac_Asp = V_pipeAsp / V_pipe0     'volFrac of deposit, Asp (wrt clean tubing)
                    Vfrac_DM = V_pipeDM / V_pipe0       'volFrac of deposit, DM (wrt clean tubing)
                End If
                
                'convert deposition flux
                For iZ = 1 To nZ
                    J_Asp(iZ) = J_Asp(iZ) * 0.1 * 86400 'deposition flux [kg/m2/s -> g/m2/day]
                    If Abs(J_Asp(iZ)) < 10 ^ -10 Then J_Asp(iZ) = 0
                Next iZ
                
                'max blockage (frac) and max flux at current time step
                delf_Asp_max = oMax(delf_Asp, idx_max_Asp)
                delf_DM_max = oMax(delf_DM, idx_max_DM)
                J_Asp_max = oMax(J_Asp)                     'max flux (g/m2/day)
                
                'total amount of Asp deposited
                Dim mass_Asp0 As Double
                If T > 0 Then
                    mass_Asp_Sum = oSum(mass_Asp)               'total Asp mass deposited (kg)
                    mass_Asp0 = mFlow_Asp * T                   'total Asp mass present (kg)
                    mass_Asp_Frac = mass_Asp_Sum / mass_Asp0 'fraction Asp mass deposited
                End If
                
                'mFlow_Asp * dt_arr(iz)
                
                'conc of PP, Ag
                Dim C_tz As Double, Cag_tz As Double, Cd_tz As Double
                C_tz = 0: Cag_tz = 0: Cd_tz = 0
                For iZ = 1 To nZ
                    C_tz = C_tz + Cf(iZ)
                    Cag_tz = Cag_tz + Cag(iZ)
                    Cd_tz = Cd_tz + Cd(iZ)
                Next iZ
                
                'store properties at each time cut
                TC_t_out(iT_out) = t_out_days           'current time (day)
                TC_Depo_Asp_kg(iT_out) = mass_Asp_Sum   'sum Asp mass deposited
                TC_Depo_Asp_kgf(iT_out) = mass_Asp_Frac 'frac Asp mass deposited
                TC_del_Asp(iT_out) = delf_Asp_max       'max blockage (frac of cross-section blocked), Asp
                TC_del_DM(iT_out) = delf_DM_max         'max blockage (frac of cross-section blocked), DM
                TC_Vfrac_Asp(iT_out) = Vfrac_Asp        'volume of deposit, Asp
                TC_Vfrac_DM(iT_out) = Vfrac_DM          'volume of deposit, DM
                TC_J_Asp_max(iT_out) = J_Asp_max        'asp flux, max
                TC_dPf(iT_out) = dPf                    'friction pressure drop (psi)
                TC_dPdt(iT_out) = dPdT                  'dPdt (psi/day)
                TC_C_sum(iT_out) = C_tz                 'sum of C at t
                TC_Cag_sum(iT_out) = Cag_tz             'sum of Cag at t
                TC_Cd_sum(iT_out) = Cd_tz               'sum of Cd at t
                If idx_max_Asp > 0 Then TC_delf_max_zLoc(iT_out) = z_nP(idx_max_Asp)    'location of max deposit
                
                'time cut vars (for time cut plots)
                For iZ = 1 To nZ
                    output_TC(iZ, nCo + 1) = delf_Asp(iZ)
                    output_TC(iZ, nCo + 2) = delf_DM(iZ)
                    output_TC(iZ, nCo + 3) = J_Asp(iZ)
                Next iZ
                
                'update time_out and nCol
                T_out = T_out + dt_out
                nCo = nCo + 3
            End If
        End If
        '-----/ populate output_TC -----
        
        'update time
        T = t_new
        
        'exit loop when T >= t_sim
        If T >= t_sim Then
            itr = itr_max
        End If
    Loop
    '-----/ time march loop -----
    
ExitLoop:

    If nTC > 0 Then
        For j = 1 To nT_out
            iT_out = j
            
            'populate time-dependent variables at the end of output range
            k = 3 * nT_out
            output_TC(iT_out, k + 1) = TC_t_out(j)      'time (days)
            output_TC(iT_out, k + 2) = TC_dPf(j)        'pressure drop (psi)
            output_TC(iT_out, k + 3) = TC_dPdt(j)       'dPdt (psi/day)
            output_TC(iT_out, k + 4) = TC_del_Asp(j)    'max blockage (frac), Asp
            output_TC(iT_out, k + 5) = TC_del_DM(j)     'max blockage (frac), DM
            output_TC(iT_out, k + 6) = TC_J_Asp_max(j)  'max flux (g/m2/day)
            output_TC(iT_out, k + 7) = TC_kgAsp_Frac(j) 'Asp mass (frac) deposited
            output_TC(iT_out, k + 8) = TC_Vfrac_Asp(j)  'volf of deposit, Asp (wrt clean tubing)
            output_TC(iT_out, k + 9) = TC_Vfrac_DM(j)   'volf of deposit, DM (wrt clean tubing)
            If TC_del_Asp(j) > 0 Then
                'output_TC(iT_out, k + 10) = depth_ft(idx_max_Asp)    'location of max blockage (depth, ft)
                output_TC(iT_out, k + 10) = z_nP(idx_max_Asp)    'location of max blockage (dimensionless depth)
            End If
            
            'zTEMP (for oxy project)
            'outputs: pressure drop
            k = 3 * nT_out
            output_TC(iT_out, k + 5) = TC_C_sum(j)      'conc PP
            output_TC(iT_out, k + 6) = TC_Cag_sum(j)    'conc Ag
            output_TC(iT_out, k + 7) = TC_Depo_Asp_kgf(j) 'frac Asp mass deposited
            '/zTEMP
            
        Next j
        
        Dim TC_vars As Variant, TC_vars_label As Variant
        ReDim TC_vars(1 To nT_out, 1 To 12)
        TC_vars_label = Array("t(day)", "dPf(psi)", "dP/dt(psi/d)", "del_Asp(in)", "del_DM(in)", "Jasp(g/m2/d)", _
            "kgAsp frac deposit", "Vfrac_Asp deposit", "Vfrac_DM deposit", "z(-) max deposit", _
            "sum(C)", "sum(Cag)", "")
        For j = 1 To nT_out
            TC_vars(j, 1) = TC_t_out(j)
            TC_vars(j, 2) = TC_dPf(j)
            TC_vars(j, 3) = TC_dPdt(j)
            TC_vars(j, 4) = TC_del_Asp(j)
            TC_vars(j, 5) = TC_del_DM(j)
            TC_vars(j, 6) = TC_J_Asp_max(j)
            TC_vars(j, 7) = TC_kgAsp_Frac(j)
            TC_vars(j, 8) = TC_Vfrac_Asp(j)
            TC_vars(j, 9) = TC_Vfrac_DM(j)
            TC_vars(j, 10) = TC_delf_max_zLoc(j)
            TC_vars(j, 11) = TC_C_sum(j)
            TC_vars(j, 12) = TC_Cag_sum(j)
            'TC_vars(j, 13) = TC_kgAsp_Frac(j)
        Next j
        
    End If

'--- OUTPUT ---
    
    'show day when flow stopped
    k = 3 * nT_out
    If dP_flowStatus <> "flowing" Then
        output_TC(1, k + 10) = t_noFlow
    End If
    'output_TC(1, k + 10) = V_Tube * 6.29     'tubing volume (bbl oil)
    'output_TC(2, k + 10) = V_Tube0 * 6.29    'tubing volume (bbl oil), after deposit starts
    'output_TC(3, k + 10) = V_TubeDM * 6.29   'tubing volume (bbl oil),
    'output_TC(4, k + 10) = V_TubeAsp * 6.29  'tubing volume (bbl oil)
    
    'number of output columns (before timecuts)
    nCo = 8
    ReDim output_R(1 To nZ_max, 1 To nCo)
    For iZ = 1 To nZ
        output_R(iZ, 1) = C(iZ)                 'conc primary particles (C)
        output_R(iZ, 2) = C_df(iZ)              'conc driving force
        output_R(iZ, 3) = del_Asp(iZ) / 2.54    'deposit thickness, Asp (in)
        output_R(iZ, 4) = del_DM(iZ) / 2.54     'deposit thickness, DM (in)
        output_R(iZ, 5) = delf_Asp(iZ)          'radial fraction blocked, Asp
        output_R(iZ, 6) = delf_DM(iZ)           'radial fraction blocked, DM
        output_R(iZ, 7) = J_Asp(iZ)             'flux Asp (g/m2/day)
        output_R(iZ, 8) = mass_Asp(iZ)          'mass deposited, Asp (kg)
    Next iZ
    
    If nTC >= 0 Then
        k = 3 * nT_out
        ReDim output_Combined(1 To nZ_max, 1 To nCo + k + nTC_Vars)
        For iZ = 1 To nZ
            For j = 1 To nCo
                output_Combined(iZ, j) = output_R(iZ, j)
            Next j
            For j = 1 To (3 * nT_out + nTC_Vars)
                output_Combined(iZ, nCo + j) = output_TC(iZ, j)
            Next j
        Next iZ
    End If
    
    'fill remaining outputs with empty
    k = 3 * nT_out
    If nTC > 0 Then
        For i = (nZ + 1) To nZ_max
            For j = 1 To k
                output_TC(i, j) = ""
            Next j
        Next i
    End If
    If nTC > 0 Then
        For i = (nZ + 1) To nZ_max
            For j = 1 To (nCo + k)
                output_Combined(i, j) = ""
            Next j
        Next i
    Else
        For i = (nZ + 1) To nZ_max
            For j = 1 To nCo
                output_R(i, j) = ""
            Next j
        Next i
    End If
    
    'populate dictionary
    Dim dict As New Scripting.Dictionary
    dict("sheet") = output_Combined
    dict("output_Combined") = output_Combined
    dict("output_R") = output_R
    dict("TC_vars") = TC_vars
    dict("TC_vars_label") = TC_vars_label
    'time cut vars
    dict("TC_t_days") = TC_t_out
    dict("TC_kgAsp_Tot") = TC_kgAsp_Tot
    dict("TC_del_Asp") = TC_del_Asp         'max blockage (frac of cross-section blocked), Asp
    dict("TC_del_DM") = TC_del_DM           'max blockage (frac of cross-section blocked), DM
    dict("TC_Vfrac_Asp") = TC_Vfrac_Asp     'volume of deposit, Asp
    dict("TC_Vfrac_DM") = TC_Vfrac_DM       'volume of deposit, DM
    dict("TC_J_Asp_max") = TC_J_Asp_max     'asp flux, max
    dict("TC_dPf") = TC_dPf                 'friction pressure drop (psi)
    dict("TC_dPdt") = TC_dPdt               'dPdt (psi/day)
    dict("TC_C_sum") = TC_C_sum             'sum of C at t
    dict("TC_Cag_sum") = TC_Cag_sum         'sum of Cag at t
    
    'return
    Set ADEPT_Solver = dict
    
End Function

Private Sub ADEPT_Solver_StatusBar(SimPercent As Integer)
    Dim str_msg As String: str_msg = "Deposition Simulation: " & SimPercent & "%"
    Application.ScreenUpdating = True
    Application.StatusBar = str_msg
    Application.ScreenUpdating = False
End Sub

Private Function ADEPT_Solver_Conc(T0 As Double, nT As Integer, dt_arr() As Double, Z0 As Double, nZ As Integer, dz As Double, _
    dict_C As Scripting.Dictionary)
    
    Dim dict_Csolve As New Scripting.Dictionary
'    'solve Cf(t,z), dissolved asphaltene conc
'    Cf_solve = ADEPT_Solver_Cf(T0, nT, dt_arr, Z0, nZ, dz, Cf_in, Cf_t0, Da_P, Ceq, SS, SolverType)
'
'    'solve C(t,z), primary particle conc
'    Set dict_Csolve = ADEPT_Solver_C(T0, nT, dt_arr, Z0, nZ, dz, C_in, C_t0, kDiss, Pe, Da_P, Da_Ag, Da_D, Cf_solve, Ceq, SS, SolverType)
'
'    dict_Csolve("Cf_solve") = Cf_solve
    
    'return
    Set ADEPT_Solver_Conc = dict_Csolve
    
End Function

Public Function ADEPT_Solver_C(T0 As Double, nT As Integer, dT() As Double, Z0 As Double, nZ As Integer, dz As Double, _
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
    Dim Z As Double, dz2 As Double
    Dim C_z0 As Double, Cf_z As Double, Ceq_z As Double, Cz As Double, Czp As Double, Czf As Double, _
        C() As Double, c_T() As Double, c0() As Double, dC() As Double, alfa As Double
    Dim f() As Double, Ja() As Double, _
        tol As Double, obj As Double, conv As Integer, iter As Integer, iterMax As Integer
    Dim a1 As Double, a2 As Double, a3 As Double, a4 As Double, S As Double
    Dim rP As Double, rAg As Double, rD As Double
    Dim Ctr() As Double, Cp() As Double, Cag() As Double, Cd() As Double
    
    'initial and boundary conditions
    'C_z0 = 0    ' no primary particles at z = 0
    'C_t0 = 0    ' no primary particles at t = 0
    
    'calculate number of nodes
    'nZ = Round((zf - z0) / dz) + 1
    'nT = Round((tf - t0) / dt) + 1
    
    'preallocate solution matrix (tSteps,zSteps) & populate IC and BC
    ReDim f(1 To nZ), Ja(1 To nZ, 1 To nZ)
    
    If SS = 1 Then
        ReDim C(1 To nZ), Ctr(1 To nZ), Cp(1 To nZ), Cag(1 To nZ), Cd(1 To nZ)
        C(1) = C_in     'BC
    Else
        ReDim C(1 To nT, 1 To nZ), Ctr(1 To nT, 1 To nZ), Cp(1 To nT, 1 To nZ), Cag(1 To nT, 1 To nZ), Cd(1 To nT, 1 To nZ)
        ReDim c_T(1 To nZ)
        
        For iZ = 1 To nZ
            C(1, iZ) = C_t0(iZ)   'IC
        Next iZ
        For iT = 1 To nT
            C(iT, 1) = C_in      'BC
        Next iT
    End If
    
    '----- steady-state, implicit Euler (IE-SS) -----
    'solve f(C)=0: f(C) =a1.C(z-1) + a2.C(z) + a3.C(z)^2 + a4.C(z+1) + S
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
                Ceq_z = Ceq(iZ) 'solubility
                Cf_z = Cf(iZ)   'dissolved asp
                Cz = C(iZ)      'PP
                
                'define coefficients of node equations
                a1 = -1 / dz - 1 / Pe(iZ) / dz ^ 2  'advection/diffusion
                a4 = -1 / Pe(iZ) / dz ^ 2           'advection/diffusion
                a3 = Da_Ag(iZ)                      'aggregation
                If (Cf_z - Ceq_z) >= 0 Then
                    S = -Da_P(iZ) * (Cf_z - Ceq_z)  'precipitation
                    a2 = 1 / dz + 2 / Pe(iZ) / dz ^ 2 + Da_D(iZ)    'deposition
                Else
                    S = 0   'precipitation
                    a2 = kDiss * Da_P(iZ) + 1 / dz + 2 / Pe(iZ) / dz ^ 2 + Da_D(iZ) 'deposition
                End If
                    
                'populate vector of objective functions and Jacobian
                If iZ = 1 Then
                    f(iZ) = Cz - C_in
                    Ja(iZ, 1) = 1
                ElseIf iZ = nZ Then
                    f(iZ) = Cz - C(iZ - 1)
                    Ja(iZ, nZ - 1) = -1
                    Ja(iZ, nZ) = 1
                Else
                    Czp = C(iZ - 1) 'PP, prev iZ
                    Czf = C(iZ + 1) 'PP, next iZ
                    f(iZ) = (a1 * Czp) + (a2 * Cz + a3 * Cz ^ 2) + (a4 * Czf) + S
                    
                    'conc calculation (?? are these expressions correct?)
                    Ctr(iZ) = (a4 * Czf) - (a1 * Czp)
                    Cp(iZ) = S
                    Cag(iZ) = a3 * Cz ^ 2
                    Cd(iZ) = a2 * Cz
                              
                    'Jacobian
                    For jZ = 1 To nZ
                        If jZ = (iZ - 1) Then
                            Ja(iZ, jZ) = a1
                        ElseIf jZ = iZ Then
                            Ja(iZ, jZ) = a2 + 2 * a3 * Cz
                        ElseIf jZ = (iZ + 1) Then
                            Ja(iZ, jZ) = a4
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
                alfa = 1
                dC = NewtonStep(f, Ja)
                For iZ = 1 To nZ
                    C(iZ) = C(iZ) + alfa * dC(iZ)
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
                Ceq_z = Ceq(iZ)         'solubility
                Cf_z = Cf(nT - 1, iZ)   'dissolved asp (prev time)
                Cz = C(nT - 1, iZ)      'PP (prev time)
                
                'rate of precipitation/redissolution
                If (Cf_z - Ceq_z) >= 0 Then
                    rP = Da_P(iZ) * (Cf_z - Ceq_z)
                Else
                    rP = -kDiss * Da_P(iZ) * Cz
                End If
                
                'rate of aggregation and deposition
                rAg = Da_Ag(iZ) * Cz ^ 2
                rD = Da_D(iZ) * Cz
                
                'conc
                Ctr(iT, iZ) = 1
                Cp(iT, iZ) = rP * dT(iZ)
                Cag(iT, iZ) = rAg * dT(iZ)
                Cd(iT, iZ) = rD * dT(iZ)
                                
                C(iT, iZ) = C(iT - 1, iZ) + dT(iZ) / dz ^ 2 / Pe(iZ) * (C(iT - 1, iZ + 1) - 2 * C(iT - 1, iZ) + C(iT - 1, iZ - 1)) - dT(iZ) / (dz) * (C(iT - 1, iZ) - C(iT - 1, iZ - 1)) + dT(iZ) * (rP - rAg - rD)
            Next iZ
            
            'populate boundary condition (z=1)
            C(iT, nZ) = C(iT, nZ - 1)
            
        Next iT
    End If
    '----- /EE-t -----
    
    '----- transient, implicit Euler (IE-t) -----
    'solve f(C)=0: f(C)=a1.C(z-1) + a2.C(z) + a3.C(z)^2 + a4.C(z+1) + S
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
             
            'NR loop
            tol = 0.0000001
            conv = 0
            iter = 0
            iterMax = 20
            Do While conv = 0 And iter <= iterMax
                iter = iter + 1
    
                For iZ = 1 To nZ
                    Z = Z0 + (iZ - 1) * dz
                    Ceq_z = Ceq(iZ)     'solubility
                    Cf_z = c0(iZ)       'dissolved asp
                    Cz = c_T(iZ)        'PP
                        
                    ' define coefficients of node equations
                    a1 = dT(iZ) * (1 / dz + 1 / Pe(iZ) / dz ^ 2)    'advection/diffusion
                    a4 = dT(iZ) * (1 / Pe(iZ) / dz ^ 2)             'advection/diffusion
                    a3 = dT(iZ) * (-Da_Ag(iZ))                      'aggregation
                    If (Cf_z - Ceq_z) >= 0 Then
                        S = dT(iZ) * (Da_P(iZ) * (Cf_z - Ceq_z))    'precipitation
                        a2 = dT(iZ) * (-1 / dz - 2 / Pe(iZ) / dz ^ 2 - Da_D(iZ))    'deposition
                    Else
                        S = 0   'precipitaiton
                        a2 = dT(iZ) * (-kDiss * Da_P(iZ) - 1 / dz - 2 / Pe(iZ) / dz ^ 2 - Da_D(iZ)) 'deposition
                    End If
    
                    ' populate vector of objective functions (f) and Jacobian (Ja)
                    If iZ = 1 Then
                        f(iZ) = Cz - C_in
                        Ja(iZ, 1) = 1
                    ElseIf iZ = nZ Then
                        f(iZ) = Cz - c_T(iZ - 1)
                        Ja(iZ, nZ - 1) = -1
                        Ja(iZ, nZ) = 1
                    Else
                        Czp = c_T(iZ - 1) 'PP, prev iZ
                        Czf = c_T(iZ + 1) 'PP, next iZ
                        f(iZ) = (a1 * Czp) + (a2 * Cz + a3 * Cz ^ 2) + (a4 * Czf) + S
                        
                        ' Jacobian
                        For jZ = 1 To nZ
                            If jZ = iZ - 1 Then
                                Ja(iZ, jZ) = a1
                            ElseIf jZ = iZ Then
                                Ja(iZ, jZ) = a2 + 2 * a3 * Cz
                            ElseIf jZ = iZ + 1 Then
                                Ja(iZ, jZ) = a4
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
                    alfa = 1
                    dC = NewtonStep(f, Ja)
                    For iZ = 1 To nZ
                        c_T(iZ) = c_T(iZ) + alfa * dC(iZ)
                    Next iZ
                End If
            Loop
            
            'store solution for current time cut
            For iZ = 1 To nZ
                C(iT, iZ) = c_T(iZ)
            Next iZ
        Next iT
    End If
    '-----/ IE-t -----
    
    'post-processing step
    
    '-- Cag(t,z), conc of aggregated particles
    '-- Cd(t,z), conc of deposited particles
    If SS = 0 Then
        'transient
        'ReDim Ctr(1 To nT, 1 To nZ), Cag(1 To nT, 1 To nZ), Cd(1 To nT, 1 To nZ)
        For iT = 1 To nT
            For iZ = 1 To nZ
                Ctr(iT, iZ) = 1 'temp
                Cag(iT, iZ) = dT(iZ) * Da_Ag(iZ) * C(iT, iZ) ^ 2
                Cd(iT, iZ) = dT(iZ) * Da_D(iZ) * C(iT, iZ)
            Next iZ
        Next iT
    Else
        'steady-state
        'ReDim Ctr(1 to nZ), Cag(1 To nZ), Cd(1 To nZ)
        For iZ = 1 To nZ
            Ctr(iZ) = 1 'temp
            Cag(iZ) = Da_Ag(iZ) * C(iZ) ^ 2
            Cd(iZ) = Da_D(iZ) * C(iZ)
        Next iZ
    End If
    '/ post-processing
    
    'populate output dictionary
    Dim dict As New Scripting.Dictionary
    dict("C") = C
    dict("Cf") = Cf
    dict("Cag") = Cag
    dict("Cd") = Cd
    
    'return
    Set ADEPT_Solver_C = dict
    
End Function

Public Function ADEPT_Solver_Cf(T0 As Double, nT As Integer, dT() As Double, Z0 As Double, nZ As Integer, dz As Double, _
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
    Dim a1 As Double, a2 As Double, B1 As Double, B2 As Double, m As Double, Z As Double
    
    
    'initial and boundary conditions
    'Cf_in = 1 ' everything in solution
    'Cf_t0 = 1 ' everything in solution

    'calculate number of nodes
    'nZ = Round((zf - z0) / dz) + 1
    'nT = Round((tf - t0) / dt) + 1

    'preallocate solution matrix (tSteps,zSteps) & populate IC and BC
    Dim A() As Double, B() As Double
    ReDim A(1 To nZ, 1 To nZ), B(1 To nZ)
        
    Dim Cf() As Double, Cf_t() As Double, rP() As Double
    If SS = 1 Then
        ReDim Cf(1, 1 To nZ), rP(1, 1 To nZ)
        Cf(1, 1) = Cf_in        ' BC
        
    Else
        ReDim Cf(1 To nT, 1 To nZ), rP(1 To nT, 1 To nZ), Cf_t(1 To nZ)
        
        For i = 1 To nZ
            Cf(1, i) = Cf_t0(i)     'IC
        Next i
        For i = 1 To nT
            Cf(i, 1) = Cf_in        'BC
        Next i
    End If

    'steady-state, explicit Euler
    Dim Cf_z As Double, Ceq_z As Double
    If SS = 1 And InStr(1, "EE", SolverType) > 0 Then
    
        For iZ = 2 To nZ
            Z = Z0 + (iZ - 1) * dz
            Cf_z = Cf(1, iZ - 1)
            Ceq_z = Ceq(iZ - 1)     'extract Ceq at previous node
            rP(1, iZ) = Da_P(iZ - 1) * (Cf_z - Ceq_z)

            Cf(1, iZ) = Cf_z - dz * rP(1, iZ)
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
            B1 = -dz * Da_P(iZ) * Ceq(iZ)
    
            'construct linear system A*C+B = 0
            If iZ = 1 Then
                A(iZ, 1) = 1
                B(iZ) = -Cf_in
            Else
                A(iZ, iZ - 1) = a1
                A(iZ, iZ) = a2
                B(iZ) = B1
            End If
            
        Next iZ
    
        'solve linear system
        Cf = NewtonStep(B, A)

    End If
    
    'transient, explicit Euler
    If SS = 0 And InStr(1, "EE", SolverType) > 0 Then
        
        For iT = 2 To nT
            For iZ = 2 To nZ
                Z = Z0 + (iZ - 1) * dz
                rP(iT, iZ) = Da_P(iZ) * (Cf(iT - 1, iZ) - Ceq(iZ))
                
                Cf(iT, iZ) = Cf(iT - 1, iZ) - dT(iZ) / dz * (Cf(iT - 1, iZ) - Cf(iT - 1, iZ - 1)) - dT(iZ) * rP(iT, iZ)
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
                a1 = -dT(iZ) / dz
                a2 = 1 + dT(iZ) / dz + dT(iZ) * Da_P(iZ)
                B1 = -Cf(iT - 1, iZ) - dT(iZ) * Da_P(iZ) * Ceq_z

                'construct linear system A*C+B = 0
                If iZ = 1 Then
                    A(iZ, 1) = 1
                    B(iZ) = -Cf_in
                Else
                    A(iZ, iZ - 1) = a1
                    A(iZ, iZ) = a2
                    B(iZ) = B1
                End If
            Next iZ

            'solve linear system
            Cf_t = NewtonStep(B, A)
            
            'populate solution at current time cut
            For iZ = 1 To nZ
                Cf(iT, iZ) = Cf_t(iZ)
            Next iZ
            
        Next iT
        
    End If
    
    'return
    ADEPT_Solver_Cf = Cf
   
End Function

Public Function mix_Dens(wtFrac() As Double, volFrac() As Double, dens() As Double, Optional str_Avg As String = "volume")
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

Public Function mix_Visco(wtFrac() As Double, volFrac() As Double, dens() As Double, visco() As Double, rho, Optional str_Avg As String = "volume")
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

Public Function mix_Velo(wtFrac() As Double, velo() As Double, Optional str_Avg As String = "sum")
'calculate averaged total velocity by method: "str_Avg"

    Dim k As Integer, nPhase As Integer, uz As Double
    nPhase = ArrayLen(wtFrac)
    
    If str_Avg = "sum" Then
        For k = 1 To nPhase
            uz = uz + velo(k)   'sum of all velocities
        Next k
    Else
        uz = velo(2)            'none (total velo = L1 velo)
    End If
    
    'return
    mix_Velo = uz
    
End Function

'------------------------
' auxiliary methods for ADEPT routine
'------------------------

Public Function ADEPT_kP(A As Variant, T As Double, Optional del_SP As Double, Optional str_eqn As String = "default")

'=========================================================
' Program:      ADEPT_kP
' Description:  calculates the precipitation kinetics (kP) parameter in the ADEPT model
' Called by:    ADEPT_Solver; on-sheet function
' Notes:        ln(kP) =f(T,{SP})
'
'---------- INPUTS -----------------------
' a:            coefficients of correlation (a1, a2, a3, a4)
' T:            temperature [K]
' del_SP:       diff in solubility parameter between asphaltene and solvent [MPa^0.5]
' str_eqn:      form of correlation ("nrb" or "T" or "T-SP")
'
'---------- RETURN -----------------------
' ln_kP:        precipitation kinetics (kP) parameter
'==========================================================

    'log(kP)=f(T,{SP})
    Dim str As String, X As Double, dSP2 As Double
    str = LCase(str_eqn)
    If str = "nrb" Or str = "narmi" Then
        'Narmadha Rajan Babu (thesis, 2019)
        dSP2 = del_SP ^ 2
        X = -(A(1) * Exp(-A(2) / T) + (A(3) * Exp(-A(4) / T)) / dSP2)
    ElseIf str = "t-sp" Or str = "il-sp" Then
        dSP2 = del_SP ^ 2
        X = Log(A(1)) - A(2) * 1000 / (T * dSP2)
        'X = Log(A(1)) - A(2) * 1000 / (T ^ A(3) * dSP2 ^ A(4))
    Else 'str = "default" Then
        dSP2 = 1
        X = Log(A(1)) - A(2) * 1000 / (T * dSP2)
    End If
    
    'return
    ADEPT_kP = X
    
End Function

Public Function ADEPT_kAg(A As Variant, T As Double, Optional del_SP As Double, Optional visco As Double, _
    Optional str_eqn As String = "default")

'=========================================================
' Program:      ADEPT_kAg
' Description:  calculates the aggregation kinetics (kAg) parameter in the ADEPT model
' Called by:    ADEPT_Solver; on-sheet function
' Notes:        ln(kAg)=f(T,{SP},{mu})
'
'---------- INPUTS -----------------------
' a:            coefficients of correlation (a1, a2, a3, a4)
' T:            temperature [K]
' del_SP:       diff in solubility parameter [MPa^0.5] between asphaltene and solvent
' visco:        viscosity [cP] of solvent
' str_eqn:      form of correlation ("nrb" or "IL")
'
'---------- RETURN -----------------------
' ln_kAg:       aggregation kinetics (kAg) parameter
'==========================================================

    Dim str As String, X As Double, RT As Double, dSP2 As Double
    str = LCase(str_eqn)
    If str = "nrb" Then
        'Narmadha Rajan Babu (thesis, 2019)
        dSP2 = del_SP ^ 2
        X = -(A(1) * Exp(-A(2) / T) + (A(3) * Exp(-A(4) / T)) / dSP2)
    Else 'str = "default" Then
        'a[1] represents the collision efficiency
        RT = 8.314 * T
        X = Log(0.0666667 / 750 * RT / visco * A(1))
    End If
    
    'return
    ADEPT_kAg = X
    
End Function

Public Function ADEPT_kD(kD0 As Double, T As Double, param As Variant, Optional str_scale As String = "wb", _
    Optional ByVal uz As Double, Optional ByVal mom_bl As Double)
    
'=========================================================
' Program:      ADEPT_kD
' Description:  scales the deposition kinetics (kD0) parameter to wellbore (WB) conditions
' Called by:    ADEPT_Solver; on-sheet function
' Notes:        kD =f({flow_props},{fluid_props})
'
'---------- INPUTS -----------------------
' kD0:          deposition (kD) parameter [unscaled]
' T:            temperature [K]
' kD_param:
'   mFlow:      massflow rate [kg/s]
'   r:          pipe radius [m]
'   rho:        density [kg/m3] of solvent
'   mu:         viscosity [Pa.s] of solvent
' str_scale:    form of unscaled kD ("wb", "pb", "cap")
' uz:           velocity (m/s) -> if provided, overwrites u = q/A
' mom_bl:       momentum boundary layer (m) -> if provided, overwrites mom_bl calculation
'
'---------- RETURN -----------------------
' kD:           aggregation kinetics (kAg) parameter [scaled to WB]
'==========================================================
    
    If str_scale = "" Then
        str_scale = "wb"
    End If
    
    'extract inputs
    Dim q As Double, mFlow As Double, R As Double, rho As Double, mu As Double
    mFlow = param(1): R = param(2): rho = param(3): mu = param(4)
    q = mFlow / rho
    'mFlow = q * rho
    
    'convert inputs
    'q = q * 0.0000018403                                'oil flow rate (STB/d -> m^3/s)
    'd = d * 0.0254
    'rho = rho * 1000
    'mu = mu / 1000
    
    'q = UnitConverter("volflow", q, "stb/d", "m3/s")
    'd = UnitConvert_Length(d, "in", "m")                'pipe diameter (in -> m)
    'rho = UnitConverter("dens", rho, "g/cc", "kg/m3")   'oil density (g/cc -> kg/m3)
    'mu = UnitConverter("visco", mu, "cP", "Pa.s")       'oil viscosity (cP -> Pa.s)

    'velocity in wellbore
    Dim d As Double, A As Double, u As Double
    If uz = 0 Then
        d = 2 * R
        A = shp_AreaCircle(R)   'cross-sectional area of wellbore (m^2)
        u = q / A               'velocity (m/s)
    Else
        u = uz
    End If
    
    'scaling factor for kd
    Dim Re As Double, Gu_WB As Double
    If mom_bl = 0 Then
        Re = u * d * rho / mu           'Reynolds number (-)
        mom_bl = 62.7 * d * Re ^ (-7 / 8)   'momentum boundary layer, m
    End If

    'scale deposition constant from EXPERIMENTAL setup to wellbore conditions
    Dim kD As Double, ScF As Double, phi As Double, Dm As Double
    Dim void_frac As Double, diam_Sphere As Double, Gu_PB As Double, Gu_Cap As Double, uz_PB As Double
    
    'scaling factor (ScF) and velocity gradient (Gu)
    str_scale = LCase(str_scale)
    If str_scale = "wb" Or str_scale = "1" Or str_scale = "no-scale" Then
        'wellbore (WB) setup
        Gu_WB = 1 '0.01 * ((RHO / mu) * (U ^ 3 / (2 * D))) ^ 0.5 'gradient of velocity for wellbore
        ScF = 1
        
    ElseIf str_scale = "pb" Or str_scale = "2" Then
        'packed bed (PB) setup
        void_frac = 0.4                                 'void fraction of packed bed column
        uz_PB = 0.00002735                              'axial velocity of packed bed column (m/s)
        diam_Sphere = 0.00238                           'diameter of spheres in packed bed column (m)
        Gu_PB = 13.4 * ((1 - void_frac) / void_frac) * uz_PB / diam_Sphere     'gradient of velocity for packed bed
        Gu_WB = 1
        
        Dm = ADEPT_Diffusivity(T, mu)    'diffusivity
        phi = Dm / (mom_bl ^ 2 * kD0 * Gu_PB)           '
        ScF = (4 / d * mom_bl) * (phi / (phi + 1))      'scaling factor (PB)
        
    ElseIf str_scale = "cap" Or str_scale = "3" Then
        'capillary tube (Cap) setup
        Gu_Cap = 1
        Gu_WB = 1
        Dm = ADEPT_Diffusivity(T, mu)    'diffusivity
        phi = Dm / (mom_bl ^ 2 * kD0 * Gu_Cap)          '
        ScF = (4 / d * mom_bl) * (phi / (phi + 1))      'scaling factor (Cap)
        
    End If
    
    'scaled deposition constant
    kD = kD0 * ScF * Gu_WB

    'return
    ADEPT_kD = kD
    
End Function

Public Function ADEPT_rSR(param As Variant, tau As Double)
'shear removal rate
    
    'extract SR parameters
    Dim tau0 As Double, k As Double, n As Double, eta As Double
    tau0 = param(1): k = param(2): n = param(3)
    
    If tau0 = 0 Then
        tau0 = 5    'critical shear stress at wall (Pa), default
    End If
    
    'return
    If tau > tau0 Then
        ADEPT_rSR = k * (tau / tau0 - 1) ^ n
    Else
        ADEPT_rSR = 0
    End If
    
End Function

Public Function ADEPT_kDiss(Optional ByVal kP As Double)
'kinetics of redissolution
    If kP > 0 Then
        ADEPT_kDiss = 0.01 / kP
    Else
        ADEPT_kDiss = 0.01
    End If
End Function

Public Function ADEPT_Diffusivity(Optional T As Double, Optional mu As Double, Optional Ds As Double = 0.000000002)
'particle diffusivity (m2/s) from Einstein-Stokes equation
    
    Const Pi As Double = 3.14159265358979   'value of pi
    Const kB As Double = 1.3806504 * 10 ^ -23 'Boltzmann constant

    'particle size
    'Dim Ds As Double
    If Ds <= 0 Then
        Ds = 0.000000035    'asphaltenes in toluene (Andrews et al. 2006; doi: 10.1021/jp062099n)
        Ds = 0.000000002    'diffusivity (Narmi ADEPT 2019)
    End If
    
    'diffusivity
    Dim Dm As Double
    If T > 0 And mu > 0 Then
        Dm = kB * T / (3 * Pi * mu * Ds)  'Brownian particle diffusivity (Stokes-Einstein), m2/s
    Else
        Dm = Ds
    End If
    
    'return
    ADEPT_Diffusivity = Dm
    
End Function

Public Function ADEPT_fricFactor(Re As Double, Optional d As Double = 0)
'friction factor
    
    Dim f As Double
    If Re = 0 Then
        f = 0
    ElseIf Re < 2400 Then
        f = 64 / Re             'laminar friction factor
    ElseIf d > 0 Then
        f = (1 / (1.14 - 2 * Log((0.0006 * 0.3048) / d + 21.25 / Re ^ 0.9) / Log(10))) ^ 2 ' Jain friction factor
    Else
        f = 0.316 / (Re ^ 0.25) 'Blausius (turbulent) friction factor
    End If
            
    'return
    ADEPT_fricFactor = f
    
End Function

'------------------------
' pressure drop calculation
'------------------------

Public Function Asp_DepoModel_dP(mFlow As Double, L As Double, R As Double, P_bh As Double, _
    ADEPT_Input As Range, Asp_DepoInputs As Range, Asp_xProf As Range, Optional sht_P As String = "FPp_main")

'=========================================================
' Program:      Asp_DepoModel_dP
' Description:  calculates pressure drop in a pipe with deposits
' Called by:    on-sheet function
' Notes:        calculates dP along length (L) of pipe
'
'---------- INPUTS -----------------------
' --- scalars ---
' mFlow:        total flow rate (kg/s)
' L:            pipe length (m)
' R:            pipe radius (m)
' P_bh:         bottomhole pressure (psi)
'
' --- ranges ---
' ADEPT_Input:  full input range for ADEPT model
' Asp_DepoInputs:   full outputproperty table from zPTx and Ceq calculation
'   Asp_zProf:  depth (ft)
'   Asp_FluidProps: oil properties along depth profile
'       yAsp:       asph compo (wtf)
'       beta:       phase amounts (wt%)
'       dens:       phase density (g/cc)
'       visco:      phase viscosity (cP)
' Asp_xProf:    deposit thickness (-)
'
'---------- OUTPUTS ----------------------
' dP:           pressure drop
'==========================================================

    Dim i As Integer, k As Integer, nZ As Integer, nPh As Integer
    
    If mFlow = 0 Or L = 0 Or R = 0 Then
        GoTo ExitFunc
    End If
        
    'get units
    Dim Units As Variant, unit_T As String, unit_P As String, unit_D As String, unit_N As String, unit_X As String, unit_GOR As String, unit_Visco As String
    Units = Read_Units(sht_P)
    Call Get_Units(Units, "input", unit_T, unit_P, , unit_D, unit_N, unit_X, unit_GOR, unit_Visco)
    
    'extract Asp_DepoInputs
    Dim Asp_DepoVal As Range, nRo As Integer, nCo As Integer
    nRo = Asp_DepoInputs.Rows.Count
    nCo = Asp_DepoInputs.Columns.Count
    Set Asp_DepoVal = subRange(Asp_DepoInputs, 3, nRo, 1, nCo)
    
    Dim Asp_zPTx As Range, Asp_x0 As Range, Asp_FluidProps As Range, Asp_RateParam As Range
    Set Asp_zPTx = Asp_DepoVal.Columns("a:d")
    'Set Asp_x0 = Asp_DepoVal.Columns("g:h")
    Set Asp_FluidProps = Asp_DepoVal.Columns("i:z")
    'Set Asp_RateParam = Asp_DepoVal.Columns("aa:ac")
    
    'thermodynamics lookup table (TLUT)
    Dim dict_TLUT As New Scripting.Dictionary ', ADEPT_Input As Range
    Set dict_TLUT = ADEPT_LUT_Thermo(ADEPT_Input, Asp_zPTx, Asp_FluidProps, Units)
    
    'extract deposit thickness (del)
    Dim Asp_depth As Variant, Asp_P As Variant, Asp_del As Variant, Asp_beta As Variant, Asp_dens As Variant, Asp_SP As Variant, Asp_visco As Variant
    Dim depth_ft() As Double, del() As Double
    Asp_del = Math_Transpose(Asp_xProf.Value2)
    If IsError(Asp_del(1)) = True Then
        GoTo ExitFunc
    End If
    
    nZ = dict_TLUT("nZ")
    ReDim del(1 To nZ)
    For i = 1 To nZ
        del(i) = Asp_del(i)
    Next i
    
    'dP =f(del)
    Asp_DepoModel_dP = Math_Transpose(DepoModel_dP(mFlow, L, R, P_bh, del, dict_TLUT))
 
ExitFunc:

End Function

Public Function DepoModel_dP(mFlow As Double, L As Double, R As Double, P_bh As Double, del_frac() As Double, _
        dict As Scripting.Dictionary)

'=========================================================
' Program:      DepoModel_dP
' Description:  calculates pressure drop in a pipe with deposits
' Called by:    Asp_DepoModel_dP
' Notes:        scans from bottomhole (BH) to wellhead (WH)
'               calculating pressure drop in each segment
'
'---------- INPUTS -----------------------
' --- scalars ---
' mFlow:        total flow rate (kg/s)
' L:            pipe length (m)
' R:            pipe radius (m)
' P_bh:         bottomhole pressure (psi)
'
' --- ranges ---
' depth:        depth (ft)
' del_frac:     deposit thickness (frac blocked)
' dict_TLUT:    thermo lookup table (TLUT)
'   beta:         phase amounts (wtf)
'   vol:          volume (cc/mol)
'   dens:         oil (L1) density (g/cc)
'   SP:           oil (L1) SP (MPa^0.5)
'   visco:        oil (L1) viscosity (cP)
'
'---------- OUTPUTS ----------------------
' dP:           pressure drop
'   [1] friction dP
'   [2] gravity dP
'   [3] total (f+g) dP
'   [4] flow status {"flowing", "shutdown", "plugged", "err"}
'==========================================================
    
    Dim i As Integer, k As Integer, iZ As Integer, nZ As Integer, nPh As Integer, flag As String, flag_Err As String
    Dim depth() As Double, depth_ft() As Double, deposit() As Double, velo() As Double, _
        densi() As Double, visci() As Double, veloi() As Double, betai() As Double, volfi() As Double, _
        del() As Double, beta_sum As Double, vol_sum As Double
    Dim dPf_i As Double, dPg_i As Double, dP_i As Double, dP As Double, dPf As Double, dPg As Double, _
        fricFactor As Double, A As Double, id As Double, R0 As Double, d0 As Double, dL_m As Double
    Dim P_wh As Double, dP_max As Double
    
    'extract units
    Dim Units As Variant, unit_T As String, unit_P As String, unit_Pa As String
    Units = dict("Units")
    Call Get_Units(Units, "input", unit_T, unit_P)
    unit_Pa = Replace(unit_P, "g", "a")
    
    'extract thermo properties
    Dim P0 As Double
    Dim beta() As Double, vol() As Double, dens() As Double, SP() As Double, visco() As Double, volFlow() As Double
    Dim Re() As Double, Pe() As Double, rho() As Double, mu() As Double, u() As Double, flag_Re As Integer
    P0 = dict("P_bh")
    depth = dict("depth_m")
    beta = dict("beta")
    dens = dict("dens")
    SP = dict("SP")
    visco = dict("visco")
    If P0 > 0 Then
        P_bh = P0
    End If
    nZ = ArrayLen(depth)
    If dict.Exists("N_Re") Then
        flag_Re = 1
        Re = dict("N_Re")
        Pe = dict("N_Pe")
        rho = dict("rho")
        mu = dict("mu")
        u = dict("u")
    Else
        flag_Re = 0
        ReDim Re(1 To nZ), rho(1 To nZ), mu(1 To nZ), u(1 To nZ)
    End If
    
    nPh = ArrayLen(beta, 2)
    ReDim deposit(1 To nZ), del(1 To nZ), _
        vol(1 To nZ, 1 To nPh), volFlow(1 To nZ, 1 To nPh), velo(1 To nZ, 1 To nPh), _
        densi(1 To nPh), visci(1 To nPh), veloi(1 To nPh), betai(1 To nPh), volfi(1 To nPh)
    
    'convert
    id = 2 * R
    'Q = UnitConverter("volflow", Q, "stb/d", "m3/s") 'volumetric flow (STB/d -> m3/s) q = q * 0.1589873 / 86400
    'ID = ID * 0.0254                'in to m
    
    'scan from bottomhole (BH) to wellhead (WH)
    'calculate pressure drop for each segment
    dP = 0      'total pressure loss
    dPf = 0     'friction pressure loss
    dPg = 0     'gravity pressure loss
    flag = "flowing"
    flag_Err = "err"
    For i = 1 To nZ - 1
        
        If depth(i) > 0 Then
            iZ = iZ + 1
            
            'extract properties
            dL_m = (depth(i) - depth(i + 1))    'segment length (m)
            del(i) = del_frac(i) * R               'deposit thickness (m)
            
            'R0, radius of flow area
            R0 = R - del(i)             'radius of flow area (m) (subtract off deposit)
            d0 = 2 * R0                 'diameter (m)
            A = shp_AreaCircle(R0)      'cross-sectional area (m2)
            
            If d0 <= 0 Then
                flag = "plugged"
                GoTo ErrHandle
            End If
            
            If flag_Re = 0 Then
            
                'volume
                For k = 1 To nPh
                    If beta(i, k) > 0 Then
                        vol(i, k) = beta(i, k) / dens(i, k) 'm3/kg .= (g[k]/g)*(m3/kg)
                    End If
                Next k
                
                'phase velocities
                For k = 1 To nPh
                    If beta(i, k) > 0 Then
                        volFlow(i, k) = mFlow * vol(i, k)   'm3/s .= (kg/s)*(m3/kg)
                        velo(i, k) = volFlow(i, k) / A      'm/s .= (m3/s)*(1/m2)
                    End If
                Next k
                
                'average density, viscosity, velocity
                beta_sum = 0: vol_sum = 0
                For k = 1 To 2
                    beta_sum = beta_sum + beta(i, k)    'total mass
                    vol_sum = vol_sum + vol(i, k)       'total volume
                Next k
                For k = 1 To 2
                    betai(k) = beta(i, k) / beta_sum    'mass fraction
                    volfi(k) = vol(i, k) / vol_sum      'volume fraction
                    densi(k) = dens(i, k)
                    visci(k) = visco(i, k)
                    veloi(k) = velo(i, k)
                Next k
                
                'density and viscosity averaging (="none" for L1 only, "mass", or "volume")
                'velocity averaging (="none" for L1 only, or "sum")
                rho(i) = mix_Dens(betai, volfi, densi, "volume")
                mu(i) = mix_Visco(betai, volfi, densi, visci, rho(i), "volume")
                u(i) = mix_Velo(betai, veloi, "sum")
                
                'convert to standard units
                'rho = UnitConverter("dens", rho, "g/cc", "kg/m3")   'oil density (g/cc -> kg/m3)
                'mu(i) = UnitConverter("visco", mu(i), "cP", "Pa.s") 'oil visco (cP -> Pa.s)
                   
                'Reynolds number
                Re(i) = u(i) * d0 * rho(i) / mu(i)
            
            End If
            
            'friction factor
            fricFactor = ADEPT_fricFactor(Re(i))
            
            'segment pressure drop: friction (f) and gravity (g)
            dPf_i = fricFactor * dL_m * rho(i) * u(i) ^ 2 / d0 / 2   'Darcy-Weisbach formula (Pa)
            dPg_i = dL_m * rho(i) * 9.80665    'P[Pa]=rho[kg/m3]*g[m/s2]*h[m]
            dP_i = dPf_i + dPg_i
            
            'cumulative dP
            dPf = dPf + dPf_i
            dPg = dPg + dPg_i
            dP = dP + dP_i
            
        End If
        
    Next i
    
    'get atm pressure
    P_wh = UnitConverter("pressure", 1, "atm", unit_P)
    
    'maximum allowable dP
    dP_max = P_bh - P_wh      'psi
    
    'convert pressure (Pa -> unit_P)
    dPf = UnitConverter("dP", dPf, "Pa", unit_P)
    dPg = UnitConverter("dP", dPg, "Pa", unit_P)
    dP = dPf + dPg
    
    'determine if shutdown
    If dP > dP_max Then
        flag = "shutdown"
    End If
    flag_Err = ""       'lower flag if calc reaches this point
    
ErrHandle:
    
    'output
    Dim out(4) As Variant
    out(1) = "": out(2) = "": out(3) = ""
    If flag_Err <> "" Then
        out(4) = flag_Err
    ElseIf flag = "plugged" Then
        out(4) = flag
    Else
        out(1) = dPf: out(2) = dPg: out(3) = dP
        out(4) = flag
    End If
    
    If out(1) > 10 ^ 6 Then
        'out(1, 1) = "> 10^6"
        'out(2, 1) = ""
        'out(3, 1) = ""
    End If
    
    Dim dict_Out As New Scripting.Dictionary
    dict_Out("out") = out
    dict_Out("dPf") = dPf
    dict_Out("dPg") = dPg
    dict_Out("dP") = dP

ExitFunc:

    'return
    'Set DepoModel_dP = dict_out
    DepoModel_dP = out
    
End Function

