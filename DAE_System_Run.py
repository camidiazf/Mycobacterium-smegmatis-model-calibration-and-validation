import numpy as np
import pandas as pd
import copy
import time
import os


from mealpy import FloatVar, PSO # type: ignore
from mealpy.utils.problem import FloatVar # type: ignore
from mealpy.swarm_based import PSO # type: ignore

from System_info import system_info
from DAE_Systems_Simulations import simulate_model
from Analysis_functions import validation_analysis, parameter_analysis, plotting_comparison

def RUN_MAIN(iterations, path, condition, perturbation, delta, correlation_threshold, params_list, lb, ub):
    if os.path.exists(path):
        os.remove(path)
        print(f">>> Existing file {path!r} removed – starting fresh")

    df = pd.DataFrame()
    print(">>> Starting with empty DataFrame")

    df.to_excel(path, index=False)
    print(f">>> Written new file {path!r}")
        
    for i in range(iterations):
        print("")
        print("")
        print(f"                                 ...... Running iteration {i+1} of {iterations} ......                                   ")
        print("")
        Results = RUN_DAE_CALIBRATION(iteration = i, 
                                condition = condition,
                                perturbation = perturbation, 
                                delta = delta,
                                correlation_threshold = correlation_threshold, 
                                params_list = params_list,
                                lb = lb,
                                ub = ub)
        
        if Results is None:
            if i == 0:
                print("!!!!!!!!!!!!!               Initial run failed. Please check the parameters and initial conditions.")
                return None
            else:
                print(f"!!!!!!!!!!!!!               Iteration {i+1} failed. Skipping to next iteration.")
                continue
            

        if i == 0:
            run_name = "Original Model"
        else:
            run_name = f"Run_{i+1}"
        Results.insert(0, 'Run', run_name)

        before = df.shape[0]
        df = pd.concat([df, Results], ignore_index=True)
        after = df.shape[0]
        print(f"Appended row, df rows went {before} → {after}")

        df.to_excel(path, index=False)
        print(f"Saved to {path}")

    print(" ALL ITERATIONS DONE — final df shape:", df.shape)
    return df


def RUN_DAE_CALIBRATION(iteration, condition, perturbation, delta, correlation_threshold, params_list, lb, ub):
    """
    Main function to run the DAE system simulation and calibration/validation process.
    Parameters:
        - iteration: int, the current iteration number (0 for initial run, >0 for optimization).
        - condition: str, the experimental condition (e.g., 'Normal').
        - perturbation: float/int, the type of perturbation applied to the system.
        - correlation_threshold: float, threshold for correlation analysis.
        - params_list: list, list of parameters to calibrate.
        - lb: list, lower bounds for the parameters to calibrate.
        - ub: list, upper bounds for the parameters to calibrate.
    Returns:
        - final_results: DataFrame, containing the results of the calibration/validation process.
    
    This function performs the following steps:
    1. Retrieves system data based on the specified condition.
    2. Simulates the model using the original parameters and validation data.
    3. If iteration is 0, computes initial validation results and performs parameter analysis.
    4. If iteration > 0, performs PSO optimization to calibrate the model parameters.
    5. Simulates the model with the new parameters and computes validation results.
    6. Returns a DataFrame with the final results, including updated parameters, validation metrics, and t-values.

    Note: The function assumes that the necessary data files and system information are available in the specified format.
    """

    system_data = system_info(condition)

    str_info = system_data['str_info']
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
    time_stamps_sim = system_data['time_stamps_sim']
    


    # ORIGINAL SIMULATION USING VALIDATION DATA
    y_val = []
    y_sim_og_v_tv = []

    original_sol = simulate_model(simulation_type='normal', 
                                            x0=x0_sim_v, 
                                            parameters=parameters_og, 
                                            constants=constants, 
                                            time=t_exp_v)
    
    if original_sol is None:
        print("!!!!!!!!!!!!!               Initial simulation failed. Please check the parameters and initial conditions.")
        return None

    for var in var_names:
        if str_info[var][1] == "":
            var_exp_v = str_info[var][0]
        else:
            var_exp_v = str_info[var][0]+" "+str_info[var][1]
        y_val.append(df_val[var_exp_v].values)
        y_sim_og_v_tv.append(original_sol[var].values)



    if iteration == 0:
        print(" ")
        print("--------------------------------------------------------------------------------------------------------------------------")
        print("----------------------------------------------------- ORIGINAL MODEL -----------------------------------------------------")
        print("--------------------------------------------------------------------------------------------------------------------------")
        print(" ")
        df_results_validation_initial = validation_analysis(y_val = y_val, 
                                                    y_sim_og = y_sim_og_v_tv,
                                                    var_names = var_names,
                                                    parameters = parameters_og,
                                                    type = 'initial')
        df_t_values_initial = parameter_analysis(condition = condition, 
                                            perturbation = perturbation, 
                                            correlation_threshold = correlation_threshold,
                                            type = 'initial')
        parameters_initial_dict = {}
        for key, value in parameters_og.items():
            if key in params_list:
                parameters_initial_dict[key + '_value'] = value
        df_parameters_initial = pd.DataFrame([parameters_initial_dict])


        final_results = pd.concat([df_parameters_initial, df_results_validation_initial, df_t_values_initial], axis=1)

    else:
        print(" ")
        print("--------------------------------------------------------------------------------------------------------------------------")
        print("--------------------------------------- PSO OPTIMIZATION FOR PARAMETER CALIBRATION ---------------------------------------")
        print("--------------------------------------------------------------------------------------------------------------------------")
        print(" ")

        def cost_function(p_vars): # COST FUNCTION USING PE DATA
            try:
                df_results = simulate_model(simulation_type='calibrating', 
                                            x0=x0_exp, 
                                            parameters=parameters_og, 
                                            constants=constants, 
                                            time=t_exp,
                                            p_vars=p_vars,
                                            param_list=params_list
                                        )
                
                err = None
                for var in var_names:
                    err += np.sum((df_results[var].values - df_exp[var].values)**2)
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
        
        print("Optimization Results:")
        print("     Best Solutions: ", g_best.solution)
        print("     Minimum Error:", g_best.target.fitness)
        print(f"     Optimization Time: {end - start:.2f} s")
        

        print(" ")
        print("--------------------------------------------------------------------------------------------------------------------------")
        print(f"--------------------------------------------- NEW PARAMETERS MODEL {iteration} -------------------------------------------------")
        print("--------------------------------------------------------------------------------------------------------------------------")
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
                    print(f"!!!       Warning: Parameter '{key}' reached its upper limit ({ub[i]}).")
                if new_value == lb[i]:
                    print(f"!!!       Warning: Parameter '{key}' reached its lower limit ({lb[i]}).")                
                if new_value != ub[i] and new_value != lb[i]:
                    print(f"Parameter '{key}' is between the limits ({lb[i]}, {ub[i]}).")
                i += 1
            

        df_new_params = pd.DataFrame({
            "Parameter": params_list,
            "Original": [parameters_og[key] for key in params_list],
            "New": [parameters_updated[key] for key in params_list]})
        print(df_new_params)

        df_parameters_updated = pd.DataFrame([param_dict])

        new_sol = simulate_model(simulation_type='normal', 
                                x0=x0_sim_v, 
                                parameters=parameters_updated, 
                                constants=constants, 
                                time=t_exp_v)
        
        if new_sol is None:
            print(f"!!!!!!!!!!!!!               Simulation with new parameters failed at iteration {iteration}. Please check the parameters and initial conditions.")
            return None
        
        if iteration == 1:
            parameters = parameters_og
            type = 'initial'
            params_list = parameters_og.keys()
        else:
            parameters = parameters_updated
            type = 'new'
        original_sol_v = simulate_model(simulation_type='normal', 
                                    x0=x0_sim_v, 
                                    parameters=parameters_og, 
                                    constants=constants, 
                                    time=time_stamps_sim)
            

        print(" ")
        print("--------------------------------------------------------------------------------------------------------------------------")
        print("------------------------------------------ MODEL COMPARISON TO VALIDATION DATA -------------------------------------------")
        print("--------------------------------------------------------------------------------------------------------------------------")
        print(" ")
        plotting_comparison(condition = condition, 
                            original_sol_v = original_sol_v,
                            params_list = params_list,
                            parameters_updated = parameters,
                            type = type)   

        y_sim_new_v_tv = []
        for var in var_names:
            y_sim_new_v_tv.append(new_sol[var].values)

        df_results_validation = validation_analysis(y_val = y_val, 
                                                    y_sim_og = y_sim_og_v_tv,
                                                    var_names = var_names,
                                                    parameters = df_parameters_updated,
                                                    y_sim_new = y_sim_new_v_tv)
        if df_results_validation is None:
            print(f"!!!!!!!!!!!!!               Validation analysis failed at sensibility, on iteration {iteration}. Please check the simulation results.")
            df_results_validation = pd.DataFrame()
            


        df_t_values = parameter_analysis(condition = condition, 
                                        perturbation = perturbation, 
                                        correlation_threshold = correlation_threshold, 
                                        params_list = params_list,
                                        parameters_updated = parameters_updated, 
                                        delta=delta)
        if df_t_values is None:
            print(f"!!!!!!!!!!!!!               Parameter analysis failed at t-values, on iteration {iteration}. Please check the parameters and simulation results.")
            df_results_validation = pd.DataFrame()
        
        final_results = pd.concat([df_parameters_updated, df_results_validation, df_t_values], axis=1)


    return final_results


