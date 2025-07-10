import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.stattools import durbin_watson # type: ignore

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from System_info import system_info
from Aux_Functions import compute_sensitivity, compute_FIM, residuals_equations
from DAE_Systems_Simulations import simulate_model




def validation_analysis(system_data, var_names, parameters, params_list, type):
    """

    """
    print(" ")
    print("                              --------------- Validation - Residual Analysis ---------------                              ")
    

    if type in ['initial', 'Initial']:
        str_name = "_og"
        params_list = []
    else:
        str_name = "_new"
    x0_sim_v = system_data['x0_sim_v']
    constants = system_data['constants']
    t_exp_v = system_data['t_exp_v']
    df_val = system_data['df_val']


    y_val = []
    y_sim = []

    # Simulation with experimental validation data time points, to compare with experimental data (same size)
    sol = simulate_model(simulation_type='normal', 
                            x0=x0_sim_v, 
                            parameters=parameters, 
                            constants=constants, 
                            time=t_exp_v)
    
    if sol is None:
        print("!!!!!!!!!!!!!               Simulation for validation failed. Please check the parameters and initial conditions.")
        return None

    for var in var_names:
        var_exp_v = var
        y_val.append(df_val[var_exp_v].values)
        y_sim.append(sol[var].values)
    
    df_validation_states = pd.DataFrame({})

    y_val_c = np.concatenate(y_val)
    y_sim_c = np.concatenate(y_sim)

    validacion_estados = {}

    for i in range(len(var_names)):
        estado = var_names[i]
        y_v = y_val[i]
        y_s = y_sim[i]
        res_results_var = residuals_equations(y_v, y_s, params_list)
        
        df_validation_states[estado + str_name]= [res_results_var['rmse'], res_results_var['nmrse'], res_results_var['mape']]
        
        validacion_estados['RMSE_'+var_names[i]+str_name] = res_results_var['rmse']
        validacion_estados['NMRSE_'+var_names[i]+str_name] = res_results_var['nmrse']
        validacion_estados['MAPE_'+var_names[i]+str_name] = res_results_var['mape']

    validation_final_results = pd.DataFrame([validacion_estados])

    res_results_model = residuals_equations(y_val_c, y_sim_c, params_list)

    # AIC, BIC, RMSE, MAPE y NRMSE por modelo

    validation_final_results['AIC'+ str_name] = res_results_model['aic']
    validation_final_results['BIC'+ str_name] = res_results_model['bic']
    validation_final_results['RMSE'+ str_name] = res_results_model['rmse']
    validation_final_results['NMRSE'+ str_name] =  res_results_model['nmrse']
    validation_final_results['MAPE'+ str_name] = res_results_model['mape']

    print("\nValidation Results:")
    print('AIC: ', validation_final_results['AIC'+ str_name].values)
    
    
    residuals = res_results_model['res']

    result = stats.anderson(residuals)


    # Anderson-Darling test
    print(f"\nAnderson-Darling test statistic: {result.statistic}")
    print("Critical values and significance levels:")
    for i in range(len(result.critical_values)):
        level = result.significance_level[i]
        critical_value = result.critical_values[i]
        print(f"  Significance level {level}%: Critical value {critical_value}")

    # Durbin-Watson test
    dw_statistic = durbin_watson(residuals)
    print(f"\nDurbin-Watson statistic: {dw_statistic}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Histogram of residuals and normal distribution fit
    ax0 = axes[0]
    n, bins, patches = ax0.hist(residuals, bins='auto', density=True, alpha=0.6, color='blue')
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = ax0.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax0.plot(x, p, 'k', linewidth=2)
    ax0.set_xlabel('Residuals')
    ax0.set_ylabel('Density distribution')
    ax0.set_title('Histogram of Residuals\nand Normal Distribution')
    ax0.grid(True)

    # ---  Normal Probability Plot 
    ax1 = axes[1]
    stats.probplot(residuals, dist="norm", plot=ax1)
    ax1.set_title('Normal Probability Plot')
    ax1.set_xlabel('Theoretical Quantiles')
    ax1.set_ylabel('Ordered Values')
    ax1.grid(True)

    plt.tight_layout()
    plt.show()


    # Plotting comparison of original and new parameters with validation data

    plotting_comparison(params_list=params_list,
                        parameters_updated=parameters,
                        type=type)

    return {'Validation results': validation_final_results,
            'Residuals': residuals}


def parameter_analysis(perturbation, correlation_threshold, params_list, parameters, type , delta):
    """

    """
    
    
    system_data = system_info
    var_names = system_data['var_names']
    constants = system_data['constants']
    x0_sim = system_data['x0_sim']
    x0_exp = system_data['x0_exp']
    x0_sim_v = system_data['x0_sim_v']
    time_stamps_sim = system_data['time_stamps_sim']
    t_exp = system_data['t_exp']
    weights_exp_stack = system_data['weights_exp_stack']

    if type in ['initial', 'Initial']:
        params_list = []


    print(" ")
    print("                                  ----------------- Parameter Analysis -----------------                                  ") # WITH PE DATA

    sensitivity = compute_sensitivity(x0_sim, parameters, constants, time_stamps_sim, perturbation, var_names) #SENSITIVITY WITH NEW PARAMS AND SIMULATION
    FIM = compute_FIM(x0_exp, parameters, constants, t_exp, weights_exp_stack, correlation_threshold, var_names, params_list, delta, type) #FIM WITH WITH NEW PARAMS AND PE DATA
    

    if FIM is None:
        print("!!!!!!!!!!!!!               FIM Analysis failed. Please check the parameters and initial conditions.")
        return None
    
    corr_matrix = FIM['correlation_matrix']
    t_values = FIM['t_values']
    FIM = FIM['FIM']  

    
    return {'correlation_matrix':corr_matrix,
            't_values': t_values,
            'sensitivity': sensitivity} # Return the DataFrame with t-values


def plotting_comparison(params_list, parameters_updated, type):
    """

    """

    print(" ")
    print("--------------------------------------------------------------------------------------------------------------------------")
    print("------------------------------------------ MODEL COMPARISON TO VALIDATION DATA -------------------------------------------")
    print("--------------------------------------------------------------------------------------------------------------------------")
    print(" ")

        
    def darken_color(color, factor=0.6):
        rgb = mcolors.to_rgb(color)
        return tuple(factor * c for c in rgb)

    system_data = system_info

    var_names = system_data['var_names']
    colors = system_data['colors']
    constants = system_data['constants']
    x0_sim_v = system_data['x0_sim_v']
    time_stamps_sim = system_data['time_stamps_sim']
    df_val = system_data['df_val']
    parameters_og = system_data['parameters']

    t_exp_v = system_data['t_exp_v']

    original_sol_v = simulate_model(simulation_type='normal', 
                                    x0=x0_sim_v, 
                                    parameters=parameters_og, 
                                    constants=constants, 
                                    time=time_stamps_sim)

    if original_sol_v is None:
        print("!!!!!!!!!!!!!               Simulation with original parameters and validation data failed. Please check the parameters and initial conditions.")
        return None

    if type not in ['initial', 'Initial']:
        new_sol_v = simulate_model(simulation_type='normal', 
                                    x0=x0_sim_v, 
                                    parameters=parameters_updated, 
                                    constants=constants, 
                                    time=time_stamps_sim)
        if new_sol_v is None:
            print("!!!!!!!!!!!!!               Simulation with updated parameters and validation data failed. Please check the parameters and initial conditions.")
            return None
    

    fig, axes = plt.subplots(len(var_names), 1, figsize=(2*len(var_names), 12))

    for i in range(len(var_names)):
        var = var_names[i]
    
        axes[i].scatter(t_exp_v, df_val[var], marker='o', label=f"{var} exp validation", color=colors[var])
        axes[i].plot(time_stamps_sim, original_sol_v[var], '-', label=f"{var} original", color=colors[var])
        if type not in ['initial', 'Initial']:
            axes[i].plot(time_stamps_sim, new_sol_v[var], '--', label=f"{var} original",color=darken_color(colors[var]))
        axes[i].set_xlabel("Time (h)")
        axes[i].set_ylabel(var)
        axes[i].legend()
        axes[i].grid(True)
    
    if type in ['initial', 'Initial']:
        fig.suptitle(f'Initial Model vs. Validation Data', fontsize=18)
    else:
        fig.suptitle(f' New Model fitting {params_list} vs. Validation Data', fontsize=18)
    plt.tight_layout()
    plt.show()

    return True


