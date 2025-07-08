import casadi as ca # type: ignore
import numpy as np
import pandas as pd


# Growth dynamics function
def DAE_system(t, x, z, params, constants):
    """
    Function to define the DAE system for the growth dynamics.
    """
    # State variables
    X, C, N, CO2, O = x[0], x[1], x[2], x[3], x[4]
    # Algebraic variable
    pH = z[0] 

    # Explicit algebraic equations
    # pH inhibition factor
    Iph = ca.exp((params['I_val'] * ((pH - params['pH_UL']) / (params['pH_UL'] - params['pH_LL']))) ** 2)

    # Specific growth rate
    mu = (params['mu_max'] 
          * (1 - ca.exp(-t / constants['t_lag'])) 
          * (C / (C + params['k_C'])) 
          * (N / (N + params['k_N'])) 
          * (O / (O + params['k_O']))
          * (1 - (X / (params['Xmax']))) 
          * Iph)
    
    ka7 = 10 ** (-constants['pka7'])

    # Differential equations
    dXdt   = (mu- params['k_d']) * X                                                     # Biomass
    dCdt   = - (mu / params['YX_C']) * X                                             # Glycerol
    dNdt   = - (mu / params['YX_N']) * X                                             # Ammonia
    dCO2dt = ((mu / params['YX_CO2']) * X) - ka7 * (CO2 / (((10 ** -pH) / ka7) + 1))  # CO2
    dOdt   = constants['k_La'] * (params['O2_sat'] - O) - (mu / params['YX_O2']) * X    # O2

    return ca.vertcat(dXdt, dCdt, dNdt, dCO2dt, dOdt)


def simulate_model(x0, parameters, constants, time):
    """
    Function to simulate the model with the given parameters and initial conditions.
    """
    # Symbolic variables
    t = ca.MX.sym('t')
    x = ca.MX.sym('x', 5)  # [X, C, N, CO2, O2]
    z = ca.MX.sym('z')     # [pH]

    # Systems's differential equations
    dxdt = DAE_system(t, x, z, parameters, constants)

    # Algebraic equation
    # Parameters
    ka1 = 10 ** (-constants['pka1'])  # KH2PO4
    ka2 = 10 ** (-constants['pka2'])  # C6H8O7
    ka3 = 10 ** (-constants['pka3'])  # (C6H7O7)-
    ka4 = 10 ** (-constants['pka4'])  # (C6H6O7)2-
    ka7 = 10 ** (-constants['pka7'])  # CO2
    ka9 = 10 ** (-constants['pka9'])  # H2O

    H = 10 ** (-z) 

    # Concentration of charges according to H+ ions
    KHPO4 = constants['KH2PO4'] / ((H / ka1) + 1)
    C6H5O7 = constants['C6H8O7'] / ((H ** 3 / (ka2 * ka3 * ka4)) + (H ** 2 / (ka3 * ka4)) + (H / ka4) + 1)
    C6H6O7 = (H / ka4) * C6H5O7
    C6H7O7 = (H / ka3) * C6H6O7
    HCO3 = x[3] / ((H / ka7) + 1)
    OH = ka9 / H

    f_z = OH + HCO3 + KHPO4 + (3 * C6H5O7) + (2 * C6H6O7) + C6H7O7 - constants['pH_alk'] - H

    # CasADi function
    f = ca.Function('f', [t, x, z], [dxdt])
    
    time_stamps = time
    # ODE system

    dae = {'t': t, 'x': x, 'z': z, 'ode': f(t, x, z), 'alg': f_z}

    integrator = ca.integrator('integrator', 'idas', dae, {'grid': time_stamps, 'output_t0': True})

    # Solve

    sol = integrator(x0=x0[:-1], z0=x0[-1])

    # Extract results
    t = time_stamps
    x = sol['xf'].full().T
    z = sol['zf'].full().T

    X = x[:, 0]
    C = x[:, 1]
    N = x[:, 2]
    CO2 = x[:, 3]
    O = x[:, 4]
    pH = z[:, 0]
    H = 10 ** (-pH)
    # Compute specific growth rate over time
    mu_values = np.zeros_like(C)
    for i in range(len(C)):
        mu_values[i] = (parameters['mu_max'] *
            (1 - np.exp(-t[i] / constants['t_lag'])) *
            (C[i] / (C[i] + parameters['k_C'])) *
            (N[i] / (N[i] + parameters['k_N'])) *
            np.exp((parameters['I_val'] *
                    (pH[i] - parameters['pH_UL']) /
                (parameters['pH_UL'] - parameters['pH_LL'])) ** 2) *
            (1 - X[i] / parameters['Xmax']))

    df_results = pd.DataFrame({
        't': t,
        'X': X,
        'C': C,
        'N': N,
        'CO2': CO2,
        'O': O,
        'pH': pH,
        'H': H,
        'mu_values': mu_values
    })

    return df_results

    

def DAE_system_calibrating(t, x, z, p, parameters, constants, param_list):
    """
    Function to define the DAE system for calibration.
    """
    # for key, value in parameters.items():
    #     print(f"{key}: {value}")

    for key, value in parameters.items():
        globals()[key] = value
    
    for i, param in enumerate(param_list):
        globals()[param] = p[i]
    # Unpack variables
    X, C, N, CO2, O = x[0], x[1], x[2], x[3], x[4]
    pH = z[0] 

    # for key, value in parameters.items():
    #     print(f"{key}: {globals()[key]}")
    
    Iph = ca.exp((globals()['I_val'] * ((pH - globals()['pH_UL']) / (globals()['pH_UL'] - globals()['pH_LL']))) ** 2)  

    # Specific growth rate
    mu = (globals()['mu_max'] 
        * (1 - ca.exp(-t / constants['t_lag']))   
        * (C / (C + globals()['k_C']))   
        * (N / (N + globals()['k_N']))   
        * (O / (O + globals()['k_O']))  
        * (1 - (X / (globals()['Xmax']))) 
        * Iph)

    ka7 = 10 ** (-constants['pka7'])

    # Differential equations
    dXdt   = (mu - globals()['k_d']) * X                                                                     # Biomass
    dCdt   = - (mu / globals()['YX_C']) * X                                                                                 # Glycerol
    dNdt   = - (mu / globals()['YX_N']) * X                                                                               # Ammonia
    dCO2dt = ((mu / globals()['YX_CO2']) * X) - ka7 * (CO2 / (((10 ** -pH) / ka7) + 1))                 # CO2
    dOdt   = constants['k_La'] * (globals()['O2_sat'] - O) - (mu / globals()['YX_O2']) * X    # O2  

    return ca.vertcat(dXdt, dCdt, dNdt, dCO2dt, dOdt)


def simulate_model_calibrating(p_vars, x0, parameters, constants, param_list, t_exp):
    """
    Function to simulate the model with the given parameters and initial conditions.
    """

    # Symbolic variables
    t = ca.MX.sym('t')
    x = ca.MX.sym('x', 5)  # [X, C, N, CO2, O2]
    z = ca.MX.sym('z')     # [pH]

    # Systems's differential equations
    dxdt = DAE_system_calibrating(t, x, z, p_vars, parameters, constants, param_list)


    # Algebraic equation
    # Parameters
    ka1 = 10 ** (-constants['pka1'])  # KH2PO4
    ka2 = 10 ** (-constants['pka2'])  # C6H8O7
    ka3 = 10 ** (-constants['pka3'])  # (C6H7O7)-
    ka4 = 10 ** (-constants['pka4'])  # (C6H6O7)2-
    ka7 = 10 ** (-constants['pka7'])  # CO2
    ka9 = 10 ** (-constants['pka9'])  # H2O

    H = 10 ** (-z) 

    # Concentration of charges according to H+ ions
    KHPO4 = constants['KH2PO4'] / ((H / ka1) + 1)
    C6H5O7 = constants['C6H8O7'] / ((H ** 3 / (ka2 * ka3 * ka4)) + (H ** 2 / (ka3 * ka4)) + (H / ka4) + 1)
    C6H6O7 = (H / ka4) * C6H5O7
    C6H7O7 = (H / ka3) * C6H6O7
    HCO3 = x[3] / ((H / ka7) + 1)
    OH = ka9 / H

    f_z = OH + HCO3 + KHPO4 + (3 * C6H5O7) + (2 * C6H6O7) + C6H7O7 - constants['pH_alk'] - H

    # CasADi function
    f = ca.Function('f', [t, x, z], [dxdt])



    # Simulation time vector
    
    t_values = t_exp 
    # ODE system
    dae = {'t': t, 'x': x, 'z': z, 'ode': f(t,x,z), 'alg': f_z}
    integrator = ca.integrator('integrator', 'idas', dae, {'grid': t_values, 'output_t0': True})

    
    # Solve
    sol = integrator(x0=x0[:-1], z0=x0[-1])

    # Extract results
    t = t_exp
    x = sol['xf'].full().T
    z = sol['zf'].full().T

    X = x[:, 0]
    C = x[:, 1]
    N = x[:, 2]
    CO2 = x[:, 3]
    O = x[:, 4]
    pH = z[:, 0]
    H = 10 ** (-pH)
    # Compute specific growth rate over time
    mu_values = np.zeros_like(C)
    for i in range(len(C)):
        mu_values[i] = (parameters['mu_max'] *
            (1 - np.exp(-t[i] / constants['t_lag'])) *
            (C[i] / (C[i] + parameters['k_C'])) *
            (N[i] / (N[i] + parameters['k_N'])) *
            np.exp((parameters['I_val'] *
                    (pH[i] - parameters['pH_UL']) /
                (parameters['pH_UL'] - parameters['pH_LL'])) ** 2) *
            (1 - X[i] / parameters['Xmax']))

    df_results = pd.DataFrame({
        't': t,
        'X': X,
        'C': C,
        'N': N,
        'CO2': CO2,
        'O': O,
        'pH': pH,
        'H': H,
        'mu_values': mu_values
    })

    return df_results
