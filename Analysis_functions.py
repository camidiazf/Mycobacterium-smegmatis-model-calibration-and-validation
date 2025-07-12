import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.stattools import durbin_watson # type: ignore

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from System_info import system_info as system_data
from Aux_Functions import compute_sensitivity, compute_FIM, residuals_equations
from DAE_Systems_Simulations import simulate_model




def validation_analysis(iteration, parameters, params_list):
    """

    """
    # print(" ")
    # print("                              --------------- Validation - Residual Analysis ---------------                              ")
    

    x0_sim_v = system_data['x0_sim_v']
    var_names = system_data['var_names']
    x0_exp_v = system_data['x0_exp_v']
    t_exp_v = system_data['t_exp_v']
    df_val = system_data['df_val']


    y_val = []
    y_sim = []

    # Simulation with experimental validation data time points, to compare with experimental data (same size)
    sol = simulate_model(simulation_type='normal', 
                            x0=x0_exp_v, 
                            parameters=parameters, 
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


    val_results_list = []

    for i in range(len(var_names)):
        estado = var_names[i]
        y_v = y_val[i]
        y_s = y_sim[i]
        res_results_var = residuals_equations(y_v, y_s)
        val_results_list.extend(res_results_var)


    res_results_model = residuals_equations(y_val_c, y_sim_c, params_list)

    # AIC, BIC, RMSE, MAPE y NRMSE por modelo
    
    val_results_list.extend(res_results_model[0])

    print("\nValidation Results:")
    print('AIC: ', res_results_model[0][0])
    
    
    residuals = res_results_model[1]

    result = stats.anderson(residuals)


    # # Anderson-Darling test
    # print(f"\nAnderson-Darling test statistic: {result.statistic}")
    # print("Critical values and significance levels:")
    # for i in range(len(result.critical_values)):
    #     level = result.significance_level[i]
    #     critical_value = result.critical_values[i]
    #     print(f"  Significance level {level}%: Critical value {critical_value}")

    # # Durbin-Watson test
    # dw_statistic = durbin_watson(residuals)
    # print(f"\nDurbin-Watson statistic: {dw_statistic}")

    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # # Histogram of residuals and normal distribution fit
    # ax0 = axes[0]
    # n, bins, patches = ax0.hist(residuals, bins='auto', density=True, alpha=0.6, color='blue')
    # mu, std = stats.norm.fit(residuals)
    # xmin, xmax = ax0.get_xlim()
    # x = np.linspace(xmin, xmax, 100)
    # p = stats.norm.pdf(x, mu, std)
    # ax0.plot(x, p, 'k', linewidth=2)
    # ax0.set_xlabel('Residuals')
    # ax0.set_ylabel('Density distribution')
    # ax0.set_title('Histogram of Residuals\nand Normal Distribution')
    # ax0.grid(True)

    # # ---  Normal Probability Plot 
    # ax1 = axes[1]
    # stats.probplot(residuals, dist="norm", plot=ax1)
    # ax1.set_title('Normal Probability Plot')
    # ax1.set_xlabel('Theoretical Quantiles')
    # ax1.set_ylabel('Ordered Values')
    # ax1.grid(True)

    # plt.tight_layout()
    # plt.show()


    # Plotting comparison of original and new parameters with validation data

    # plotting_comparison(iteration = iteration, 
    #                     params_list=params_list,
    #                     parameters_updated=parameters)

    return {'Validation results': val_results_list,
            'Residuals': residuals}


def parameter_analysis(iteration, params_list, parameters ):
    """

    """
    
    x0_sim = system_data['x0_sim']
    x0_exp = system_data['x0_exp']
    time_stamps_sim = system_data['time_stamps_sim']
    t_exp = system_data['t_exp']

    # print(" ")
    # print("                                  ----------------- Parameter Analysis -----------------                                  ") # WITH PE DATA

    sensitivity = compute_sensitivity(x0 = x0_sim,
                                        parameters = parameters,
                                        time_stamps = time_stamps_sim) # SENSITIVITY WITH ORIGINAL PARAMS AND SIMULATION
    FIM = compute_FIM(iteration = iteration,
                        x0= x0_exp, 
                        parameters = parameters, 
                        time_stamps = t_exp,
                        params_list = params_list) # FIM WITH ORIGINAL PARAMS AND PE DATA
    

    if FIM is None:
        print("!!!!!!!!!!!!!               FIM Analysis failed. Please check the parameters and initial conditions.")
        return None
    
    corr_matrix = FIM['correlation_matrix']
    t_values = FIM['t_values']
    FIM = FIM['FIM']  

    
    return {'correlation_matrix':corr_matrix,
            't_values': t_values,
            'sensitivity': sensitivity} # Return the DataFrame with t-values


def plotting_comparison(iteration, params_list, parameters_updated):
    """

    """

    # print(" ")
    # print("--------------------------------------------------------------------------------------------------------------------------")
    # print("------------------------------------------ MODEL COMPARISON TO VALIDATION DATA -------------------------------------------")
    # print("--------------------------------------------------------------------------------------------------------------------------")
    # print(" ")

        
    def darken_color(color, factor=0.6):
        rgb = mcolors.to_rgb(color)
        return tuple(factor * c for c in rgb)


    var_names = system_data['var_names']
    colors = system_data['colors']
    x0_sim_v = system_data['x0_sim_v']
    time_stamps_sim = system_data['time_stamps_sim']
    df_val = system_data['df_val']
    parameters_og = system_data['parameters']

    t_exp_v = system_data['t_exp_v']

    original_sol_v = simulate_model(simulation_type='normal', 
                                    x0=x0_sim_v, 
                                    parameters=parameters_og, 
                                    time=time_stamps_sim)

    if original_sol_v is None:
        print("!!!!!!!!!!!!!               Simulation with original parameters and validation data failed. Please check the parameters and initial conditions.")
        return None

    if iteration is not None:
        new_sol_v = simulate_model(simulation_type='normal', 
                                    x0=x0_sim_v, 
                                    parameters=parameters_updated, 
                                    time=time_stamps_sim)
        if new_sol_v is None:
            print("!!!!!!!!!!!!!               Simulation with updated parameters and validation data failed. Please check the parameters and initial conditions.")
            return None
    

    # fig, axes = plt.subplots(len(var_names), 1, figsize=(2*len(var_names), 12))

    # for i in range(len(var_names)):
    #     var = var_names[i]
    
    #     axes[i].scatter(t_exp_v, df_val[var], marker='o', label=f"{var} exp validation", color=colors[var])
    #     axes[i].plot(time_stamps_sim, original_sol_v[var], '-', label=f"{var} original", color=colors[var])
    #     if iteration is not None:
    #         axes[i].plot(time_stamps_sim, new_sol_v[var], '--', label=f"{var} new",color=darken_color(colors[var]))
    #     axes[i].set_xlabel("Time (h)")
    #     axes[i].set_ylabel(var)
    #     axes[i].legend()
    #     axes[i].grid(True)
    
    # if iteration == None:
    #     fig.suptitle(f'Initial Model vs. Validation Data', fontsize=18)
    # else:
    #     fig.suptitle(f' New Model fitting {params_list} vs. Validation Data', fontsize=18)
    # plt.tight_layout()
    # plt.show()




