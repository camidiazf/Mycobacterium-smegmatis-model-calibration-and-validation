import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.stattools import durbin_watson # type: ignore

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from System_info import system_info
from Aux_Functions import compute_sensitivity, compute_FIM, residuals_equations
from DAE_Systems_Simulations import simulate_model


def validation_analysis(y_val, y_sim_og, var_names, parameters, y_sim_new = None, type = None):
    """
    Function to compute the validation metrics for the model.
    Parameters:
        - y_exp: list, experimental data for comparison.
        - y_val: list, validation data.
        - y_sim_og: list, original simulated data.
        - var_names: list, names of the variables in the model.
        - params_list_og: list, list of original parameters.
        - y_sim_new: list, new simulated data (optional).
        -type: str, type of validation ('initial' for initial validation, 'new' for new validation).
    Returns:
        - validation_final_results: DataFrame, containing the validation metrics for each variable and the model.
    This function performs the following steps:
        1. Initializes a DataFrame to store validation results.
        2. Computes residuals for each variable and the model.
        3. Calculates RMSE, MAPE, and NRMSE for each variable.
        4. Computes AIC and BIC for the model.
        5. Performs residual analysis using the Anderson-Darling test and Durbin-Watson test.
        6. Plots histograms and normal probability plots of the residuals.
        7. Returns a DataFrame with the final validation results.
    """
    print(" ")
    print("                              --------------- Validation - Residual Analysis ---------------                              ")

    df_validation_states = pd.DataFrame({})

    y_val_c = np.concatenate(y_val)
    y_sim_og_c = np.concatenate(y_sim_og)
    if y_sim_new is not None:
        y_sim_new_c = np.concatenate(y_sim_new)

    validacion_estados = {}

    for i in range(len(var_names)):
        estado = var_names[i]
        y_v = y_val[i]
        y_o = y_sim_og[i]
        res_results_var_og = residuals_equations(y_v, y_o, parameters)
        
        if type not in ['initial', 'Initial']:
            y_n = y_sim_new[i]
            res_results_var_new = residuals_equations(y_v, y_n, parameters)
            str_type = '_new'
            df_validation_states['New ' + estado]= [res_results_var_new['rmse'], res_results_var_new['nmrse'], res_results_var_new['mape']]
            res_results_var = res_results_var_new
        else:
            str_type = '_og'
            res_results_var = res_results_var_og

        df_validation_states['Original ' + estado]= [res_results_var_og['rmse'], res_results_var_og['nmrse'], res_results_var_og['mape']]
        
        validacion_estados['RMSE_'+var_names[i]+str_type] = res_results_var['rmse']
        validacion_estados['NMRSE_'+var_names[i]+str_type] = res_results_var['nmrse']
        validacion_estados['MAPE_'+var_names[i]+str_type] = res_results_var['mape']

    validation_final_results = pd.DataFrame([validacion_estados])

    if type not in ['initial', 'Initial']:
        res_results_model = residuals_equations(y_val_c, y_sim_new_c, parameters)
    else:
        res_results_model = residuals_equations(y_val_c, y_sim_og_c, parameters)

    # AIC, BIC, RMSE, MAPE y NRMSE por modelo

    validation_final_results['AIC'+ str_type] = res_results_model['aic']
    validation_final_results['BIC'+ str_type] = res_results_model['bic']
    validation_final_results['RMSE'+ str_type] = res_results_model['rmse']
    validation_final_results['NMRSE'+ str_type] =  res_results_model['nmrse']
    validation_final_results['MAPE'+ str_type] = res_results_model['mape']

    print("\nValidation Results:")
    print('AIC: ', validation_final_results['AIC'+ str_type].values)
    
    
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

    # --- Histograma con ajuste normal 
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

    # --- Gráfico de probabilidad normal
    ax1 = axes[1]
    stats.probplot(residuals, dist="norm", plot=ax1)
    ax1.set_title('Normal Probability Plot')
    ax1.set_xlabel('Theoretical Quantiles')
    ax1.set_ylabel('Ordered Values')
    ax1.grid(True)

    plt.tight_layout()
    plt.show()

    return validation_final_results


def parameter_analysis(perturbation, correlation_threshold, params_list = None, parameters_updated = None, type = None, delta=1e-4):
    """
    Function to perform parameter analysis for the system.
    Parameters:
        - condition: str, the experimental condition (e.g., 'Normal').
        - perturbation: float, the perturbation value for sensitivity analysis.
        - correlation_threshold: float, threshold for correlation in FIM computation.
        - original_sol: dict, original solution from the simulation.
        - params_list: list, list of parameters to be analyzed (optional).
        - parameters_updated: dict, updated parameters for the system (optional).
        - new_sol: dict, new solution from the simulation (optional).
        - type: str, type of analysis ('initial' for initial analysis, 'new' for new analysis).
        - delta: float, small perturbation value for sensitivity analysis.
    Returns:
        - t_values: DataFrame, containing the t-values for the parameters.
    This function performs the following steps:
        1. Retrieves system data based on the specified condition.
        2. Computes sensitivity analysis using the original parameters and simulation data.
        3. Computes the Fisher Information Matrix (FIM) using the original parameters and experimental data.
        4. Computes t-values for the parameters based on the FIM.
        5. If type is 'initial', plots comparison of original solution with experimental data.
        6. If type is 'new', plots comparison of new solution with experimental data.
        7. Returns a DataFrame with the t-values for the parameters.
    
    """
    
    
    system_data = system_info
    parameters_og = system_data['parameters']
    var_names = system_data['var_names']
    constants = system_data['constants']
    x0_sim = system_data['x0_sim']
    x0_exp = system_data['x0_exp']
    x0_sim_v = system_data['x0_sim_v']
    time_stamps_sim = system_data['time_stamps_sim']
    t_exp = system_data['t_exp']
    weights_exp_stack = system_data['weights_exp_stack']

    if type in ['initial', 'Initial']:
        parameters = parameters_og
        params_list = []
    
    else:
        parameters = parameters_updated
    

    print(" ")
    print("                                  ----------------- Parameter Analysis -----------------                                  ") # WITH PE DATA

    sensitivity = compute_sensitivity(x0_sim, parameters, constants, time_stamps_sim, perturbation, var_names) #SENSITIVITY WITH NEW PARAMS AND SIMULATION
    FIM = compute_FIM(x0_exp, parameters, constants, t_exp, weights_exp_stack, correlation_threshold, var_names, params_list, delta, type) #FIM WITH WITH NEW PARAMS AND PE DATA
    
    if FIM is None:
        print("!!!!!!!!!!!!!               FIM Analysis failed. Please check the parameters and initial conditions.")
        return None
    
    return FIM[1] # Return the DataFrame with t-values


def plotting_comparison(original_sol_v, params_list, parameters_updated, type = None):
    """
    Function to plot the comparison of model predictions with experimental validation data.
    Parameters:
        - condition: str, the experimental condition (e.g., 'Normal').
        - original_sol: DataFrame, the original simulation results.
        - params_list: list, list of parameters to calibrate (optional).
        - parameters_updated: dict, updated parameters after calibration (optional).
        - new_sol: DataFrame, the new simulation results with updated parameters (optional).
        - type: str, type of comparison ('initial' or 'updated').
    Returns:
        - None, displays the plots directly.
    
    """

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
    t_exp_v = system_data['t_exp_v']

    
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
