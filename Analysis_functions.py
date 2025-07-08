import numpy as np
import pandas as pd
import copy
from scipy import stats
from numpy.linalg import inv
from statsmodels.stats.stattools import durbin_watson # type: ignore

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns # type: ignore

from DAE_Systems_Simulations import simulate_model
from System_info import system_info
from Aux_Functions import compute_sensitivity, compute_FIM, compute_t_values, residuals, rmse, nrmse, mape, aic_bic, plotting_comparison


def compute_validation(y_exp, y_val, y_sim_og, var_names, params_list_og, y_sim_new = None, type = None):

    print("                                ----------------------------------------------")
    print("                                ------- Validation - Residual Analysis -------")
    print("                                ----------------------------------------------")
    print(" ")

    df_validation_states = pd.DataFrame({})
    y_val_c = np.concatenate(y_val)
    y_sim_og_c = np.concatenate(y_sim_og)
    y_val_range_c = np.max(y_val_c) - np.min(y_val_c)
    y_exp_c = np.concatenate(y_exp)

    if type in ['initial', 'Initial']:
        validacion_estados = {}
        #RMSE, MAPE y NRMSE por variable
        for i in range(len(var_names)):
            estado = var_names[i]
            y_v = y_val[i]
            y_o = y_sim_og[i]
            y_v_range = np.max(y_v) - np.min(y_v)
            res_state = residuals(y_v, y_o)
            df_validation_states['Original ' + estado]= [rmse(res_state), nrmse(y_v_range, res_state), mape(y_v, res_state)]
            
            validacion_estados['RMSE_'+var_names[i]+'_new'] = rmse(res_state)
            validacion_estados['NMRSE_'+var_names[i]+'_new'] = nrmse(y_v_range, res_state)
            validacion_estados['MAPE_'+var_names[i]+'_new'] = mape(y_v, res_state)
        validation_final_results = pd.DataFrame([validacion_estados])
        # print(df_validation_states)
        res_model = residuals(y_val_c, y_sim_og_c)
        res_model_aicbic = residuals(y_exp_c, y_sim_og_c)
        # AIC, BIC, RMSE, MAPE y NRMSE por modelo
        df_validation_model = pd.DataFrame({
            'Metric': ['AIC', 'BIC', 'RMSE', 'NMRSE', 'MAPE'],
            'Original Model': [aic_bic(res_model_aicbic, y_exp_c, params_list_og)[0], aic_bic(res_model_aicbic, y_exp_c, params_list_og)[1], rmse(res_model), nrmse(y_val_range_c, res_model), mape(y_val_c, res_model)]
        })

        validation_final_results['AIC_new'] = aic_bic(res_model_aicbic, y_exp_c, params_list_og)[0]
        validation_final_results['BIC_new'] = aic_bic(res_model_aicbic, y_exp_c, params_list_og)[1]
        validation_final_results['RMSE_new'] = rmse(res_model_aicbic)
        validation_final_results['NMRSE_new'] = nrmse(y_val_range_c, res_model_aicbic)
        validation_final_results['MAPE_new'] = mape(y_val_c, res_model_aicbic)

        # print(' ')
        # print(df_validation_model)

        
        res = res_model



    else:
        validacion_estados = {}
        y_sim_new_c = np.concatenate(y_sim_new)
        #RMSE, MAPE y NRMSE por variable
        for i in range(len(var_names)):
            estado = var_names[i]
            y_v = y_val[i]
            y_o = y_sim_og[i]
            y_n = y_sim_new[i]
            y_v_range = np.max(y_v) - np.min(y_v)
            res_new_state = residuals(y_v, y_n)
            res_og_state = residuals(y_v, y_o)
            df_validation_states['New '+ estado] =  [rmse(res_new_state), nrmse(y_v_range, res_new_state), mape(y_v, res_new_state)]
            df_validation_states['Original ' + estado]= [rmse(res_og_state), nrmse(y_v_range, res_og_state), mape(y_v, res_og_state)]
            
            validacion_estados['RMSE_'+var_names[i]+'_new'] = rmse(res_new_state)
            validacion_estados['NMRSE_'+var_names[i]+'_new'] = nrmse(y_v_range, res_new_state)
            validacion_estados['MAPE_'+var_names[i]+'_new'] = mape(y_v, res_new_state)
        
        validation_final_results = pd.DataFrame([validacion_estados])
        # print(df_validation_states)
        # AIC, BIC, RMSE, MAPE y NRMSE por modelo
        res_model_new = residuals(y_val_c, y_sim_new_c)
        res_model_og = residuals(y_val_c, y_sim_og_c)
        res_model_aicbic_new = residuals(y_exp_c, y_sim_new_c)
        res_model_aicbic_og = residuals(y_exp_c, y_sim_og_c)
        df_validation_model = pd.DataFrame({
            'Metric': ['AIC', 'BIC', 'RMSE', 'NMRSE', 'MAPE'],
            'New Model':      [aic_bic(res_model_aicbic_new, y_exp_c, params_list_og)[0], aic_bic(res_model_aicbic_new, y_exp_c, params_list_og)[1], rmse(res_model_new), nrmse(y_val_range_c, res_model_new), mape(y_val_c, res_model_new)],
            'Original Model': [aic_bic(res_model_aicbic_og, y_exp_c, params_list_og)[0], aic_bic(res_model_aicbic_og, y_exp_c, params_list_og)[1], rmse(res_model_og), nrmse(y_val_range_c, res_model_og), mape(y_val_c, res_model_og)]
        })

        validation_final_results['AIC_new'] = aic_bic(res_model_aicbic_new, y_exp_c, params_list_og)[0]
        validation_final_results['BIC_new'] = aic_bic(res_model_aicbic_new, y_exp_c, params_list_og)[1]
        validation_final_results['RMSE_new'] = rmse(res_model_new)
        validation_final_results['NMRSE_new'] = nrmse(y_val_range_c, res_model_new)
        validation_final_results['MAPE_new'] = mape(y_val_c, res_model_new)

        # print(' ')
        # print(df_validation_model)

        res = res_model_new
        
    # Residual analysis  model
    
    result = stats.anderson(res)


    # Anderson-Darling test
    print(f"\nAnderson-Darling test statistic: {result.statistic}")
    print("Critical values and significance levels:")
    for i in range(len(result.critical_values)):
        level = result.significance_level[i]
        critical_value = result.critical_values[i]
        print(f"  Significance level {level}%: Critical value {critical_value}")

    # Durbin-Watson test
    dw_statistic = durbin_watson(res)
    print(f"\nDurbin-Watson statistic: {dw_statistic}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- Histograma con ajuste normal 
    ax0 = axes[0]
    n, bins, patches = ax0.hist(res, bins='auto', density=True, alpha=0.6, color='blue')
    mu, std = stats.norm.fit(res)
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
    stats.probplot(res, dist="norm", plot=ax1)
    ax1.set_title('Normal Probability Plot')
    ax1.set_xlabel('Theoretical Quantiles')
    ax1.set_ylabel('Ordered Values')
    ax1.grid(True)

    plt.tight_layout()
    plt.show()

    return validation_final_results


def parameter_analysis(condition, perturbation, correlation_threshold, original_sol, params_list = None, parameters_updated = None, new_sol = None, type = None, delta=1e-4):
    
    print(" ")
    print(" ")
    print("                                ----------------------------------")
    print("                                ------- Parameter Analysis -------") # WITH PE DATA
    print("                                ----------------------------------")
    print(" ")
    system_data = system_info(condition)
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
        parameters_og = system_data['parameters']
        params_list = parameters_og.keys()
        plotting_comparison(condition = condition, 
                            original_sol = original_sol, 
                            type = type)
        sensitivity = compute_sensitivity(x0_sim, parameters_og, constants, time_stamps_sim, perturbation, var_names) #SENSITIVITY WITH ORIGINAL PARAMS AND SIMULATION
        FIM = compute_FIM(x0_exp, parameters_og, constants, t_exp, weights_exp_stack, correlation_threshold, delta) #FIM WITH ORIGINAL PARAMS AND PE DATA
        t_values = compute_t_values(parameters_og, params_list, FIM)
    else:
        plotting_comparison(condition = condition, 
                            original_sol = original_sol, 
                            params_list = params_list, 
                            new_sol = new_sol, 
                            parameters_updated = parameters_updated, 
                            type = type)   

        sensitivity = compute_sensitivity(x0_sim, parameters_updated, constants, time_stamps_sim, perturbation, var_names) #SENSITIVITY WITH NEW PARAMS AND SIMULATION
        FIM = compute_FIM(x0_exp, parameters_updated, constants, t_exp, weights_exp_stack, correlation_threshold, delta) #FIM WITH WITH NEW PARAMS AND PE DATA
        t_values = compute_t_values(parameters_updated, params_list, FIM)

    return t_values