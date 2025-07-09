import casadi as ca # type: ignore
import numpy as np
import pandas as pd


# Growth dynamics function
def DAE_system(t, x, z, params, constants):
    """
    Function to define the DAE system for the growth dynamics.
    Parameters:
        - t: time variable.
        - x: state variables.
        - z: algebraic variables.
        - params: dictionary of parameters.
        - constants: dictionary of constants.
    Returns:
        - dXdt: vector of differential equations.
    
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




    

def DAE_system_calibrating(t, x, z, p, parameters, constants, param_list):
    """
    Function to define the DAE system for calibration.
    Parameters:
        - t: time variable.
        - x: state variables.
        - z: algebraic variables.
        - p: parameters to calibrate.
        - parameters: dictionary of fixed parameters.
        - constants: dictionary of constants.
        - param_list: list of parameter names to calibrate.
    Returns:
        - dXdt: vector of differential equations.

    This function performs the following steps:
    1. Updates the global parameters with the provided values.
    2. Extracts the state variables and algebraic variable from the input.
    3. Computes the algebraic variable based on the provided parameters.
    4. Defines the differential equations for the system.
    5. Returns the vector of differential equations.



    """

    for key, value in parameters.items():
        globals()[key] = value
    
    for i, param in enumerate(param_list):
        globals()[param] = p[i]

    X, C, N, CO2, O = x[0], x[1], x[2], x[3], x[4]
    pH = z[0] 

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
    dXdt   = (mu - globals()['k_d']) * X                                                    # Biomass
    dCdt   = - (mu / globals()['YX_C']) * X                                                 # Glycerol
    dNdt   = - (mu / globals()['YX_N']) * X                                                 # Ammonia
    dCO2dt = ((mu / globals()['YX_CO2']) * X) - ka7 * (CO2 / (((10 ** -pH) / ka7) + 1))     # CO2
    dOdt   = constants['k_La'] * (globals()['O2_sat'] - O) - (mu / globals()['YX_O2']) * X  # O2  

    return ca.vertcat(dXdt, dCdt, dNdt, dCO2dt, dOdt)


def simulate_model(simulation_type, x0, parameters, constants, time, p_vars=None, param_list=None):
    """
    Function to simulate the DAE system.
    Parameters:
        - simulation_type: 'calibrating' or 'normal'.
        - x0: initial conditions for the state variables.
        - parameters: dictionary of fixed parameters.
        - constants: dictionary of constants.
        - time: time vector for the simulation.
        - p_vars: parameters to calibrate (only for 'calibrating' type).
        - param_list: list of parameter names to calibrate (only for 'calibrating' type).
    Returns:
        - df_results: DataFrame with the simulation results.

    """
    # Symbolic variables
    t = ca.MX.sym('t')
    x = ca.MX.sym('x', 5)  # [X, C, N, CO2, O2]
    z = ca.MX.sym('z')     # [pH]

    # Systems's differential equations
    if simulation_type == 'calibrating':
        dxdt = DAE_system_calibrating(t, x, z, p_vars, parameters, constants, param_list)
    elif simulation_type == 'normal':
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
    

    # ODE system
    dae = {'t': t, 'x': x, 'z': z, 'ode': f(t, x, z), 'alg': f_z}
    integrator = ca.integrator('integrator', 'idas', dae, {'grid': time, 'output_t0': True})

    # Solve
    simulation_run = True
    dict_results = {}
    try:
        sol = integrator(x0=x0[:-1], z0=x0[-1])
    except RuntimeError as e:
        print("Integration was not performed:", e)
        simulation_run = False
        # You can decide what to return when it fails:
        return None

    # Extract results
    t = time
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
        
    dict_results['t'] = t
    dict_results['X'] = X
    dict_results['C'] = C
    dict_results['N'] = N
    dict_results['CO2'] = CO2
    dict_results['O'] = O
    dict_results['pH'] = pH
    dict_results['H'] = H
    dict_results['mu_values'] = mu_values

    # Create DataFrame with results
    df_results = pd.DataFrame(dict_results)
    return df_results