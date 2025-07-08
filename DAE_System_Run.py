import numpy as np
import pandas as pd
import copy
import time


### Agregar mas que solo PSO, q admitan restricciones
## Avisar si se llega a limites

from mealpy import FloatVar, PSO # type: ignore
from mealpy.utils.problem import FloatVar # type: ignore
from mealpy.swarm_based import PSO # type: ignore

from System_info import system_info
from DAE_Systems_Simulations import simulate_model, simulate_model_calibrating
from Analysis_functions import compute_validation, parameter_analysis




def RUN_MAIN(iteration, condition, perturbation, correlation_threshold, params_list, lb, ub):

    system_data = system_info(condition)

    var_names = system_data['var_names']
    parameters_og = system_data['parameters']
    constants = system_data['constants']
    x0_exp = system_data['x0_exp']
    x0_sim_v = system_data['x0_sim_v']
    x0_exp_v = system_data['x0_exp_v']
    df_exp = system_data['df_exp']
    df_val = system_data['df_val']
    t_exp = system_data['t_exp']
    t_exp_v = system_data['t_exp_v']
    
    X_exp = df_exp['Biomass (g/L)']
    C_exp = df_exp['Glycerol (g/L)']
    N_exp = df_exp['Ammonia (g/L)']
    ph_exp = df_exp['pH']

    X_exp_v = df_val['Biomass (g/L)']
    C_exp_v = df_val['Glycerol (g/L)']
    N_exp_v = df_val['Ammonia (g/L)']
    ph_exp_v = df_val['pH']
    

    # ORIGINAL SIMULATION USING VALIDATION DATA
    y_exp = [X_exp, C_exp, N_exp, ph_exp]
    y_val = [X_exp_v, C_exp_v, N_exp_v, ph_exp_v]
    original_sol = simulate_model(x0_sim_v, parameters_og, constants, t_exp_v)
    y_sim_og = [original_sol['X'], original_sol['C'], original_sol['N'], original_sol['pH']]
    params_list_og = list(parameters_og.keys())



    if iteration == 0:
        print(" ")
        print(" ")
        print("                                ------------------------------")
        print("                                ------- ORIGINAL MODEL -------")
        print("                                ------------------------------")

        print(" ")
        df_results_validation_initial = compute_validation(y_exp = y_exp, 
                                                    y_val = y_val, 
                                                    y_sim_og = y_sim_og,
                                                    var_names = var_names,
                                                    params_list_og = params_list_og,
                                                    type = 'initial')
        df_t_values_initial = parameter_analysis(condition = condition, 
                                            perturbation = perturbation, 
                                            correlation_threshold = correlation_threshold,
                                            original_sol = original_sol, 
                                            type = 'initial')
        parameters_initial_dict = {}
        for key, value in parameters_og.items():
            if key in params_list:
                parameters_initial_dict[key + '_value'] = value
        df_parameters_initial = pd.DataFrame([parameters_initial_dict])


        final_results = pd.concat([df_parameters_initial, df_results_validation_initial, df_t_values_initial], axis=1)

    else:
        print(" ")
        print(" ")
        print("                                --------------------------------")
        print("                                ------- PSO optimization -------")
        print("                                --------------------------------")

        print(" ")

        def cost_function(p_vars): # COST FUNCTION USING PE DATA
            try:
                df_results = simulate_model_calibrating(p_vars, x0_exp, parameters_og, constants, params_list, t_exp)

                X_sim, C_sim, N_sim, pH_sim = df_results['X'], df_results['C'], df_results['N'], df_results['pH']

                err = np.sum((X_sim - X_exp)**2) + np.sum((C_sim - C_exp)**2) + np.sum((N_sim - N_exp)**2) + np.sum((pH_sim - ph_exp)**2)
                return [err]
            except:
                return [1e6]  
            
        problem = {
        "obj_func": cost_function,
        "bounds": FloatVar(lb=lb, ub=ub),
        "minmax": "min"
        }
        pso = PSO.OriginalPSO(epoch=15, pop_size=50, c1=1.5, c2=1.5, w=0.5)
        start = time.perf_counter()

        g_best = pso.solve(problem)

        end = time.perf_counter()
        
        print("Mejores parámetros encontrados:", g_best.solution)
        print("Error mínimo:", g_best.target.fitness)
        print(f"Tiempo de optimización: {end - start:.2f} s")
        

        print(" ")
        print("                                ------------------------------")
        print("                                ------- New Parameters -------")
        print("                                ------------------------------")

        print(" ")

        new_params = g_best.solution
        new_params_dict = dict(zip(params_list, new_params))
        parameters_updated = copy.deepcopy(parameters_og)
        
        param_dict = {}
        
        i = 0
        for key, value in parameters_og.items():
            if key in params_list:
                new_value = new_params_dict[key]
                parameters_updated[key] = new_value
                param_dict[key + '_value'] = new_value
                if new_value == ub[i]:
                    print(f"Warning: Parameter '{key}' reached its upper limit ({ub[i]}).")
                elif new_value == lb[i]:
                    print(f"Warning: Parameter '{key}' reached its lower limit ({lb[i]}).")                
                i += 1
            




        df_new_params = pd.DataFrame({
            "Parámetro": params_list,
            "original": [parameters_og[key] for key in params_list],
            "new": [parameters_updated[key] for key in params_list]})
        print(df_new_params)

        df_parameters_updated = pd.DataFrame([param_dict])
        new_sol = simulate_model(x0_sim_v, parameters_updated, constants, t_exp_v) 

    # NEW SIMULATION USING VALIDATION DATA
        y_sim_new = [new_sol['X'], new_sol['C'], new_sol['N'], new_sol['pH']]
        df_results_validation = compute_validation(y_exp = y_exp,
                                                    y_val = y_val, 
                                                    y_sim_og = y_sim_og,
                                                    var_names = var_names,
                                                    params_list_og = params_list_og,
                                                    y_sim_new = y_sim_new)



        df_t_values = parameter_analysis(condition = condition, 
                                            perturbation = perturbation, 
                                            correlation_threshold = correlation_threshold, 
                                            original_sol = original_sol,
                                            params_list = params_list,
                                            parameters_updated = parameters_updated, 
                                            new_sol = new_sol)
        
        final_results = pd.concat([df_parameters_updated, df_results_validation, df_t_values], axis=1)


    return final_results


