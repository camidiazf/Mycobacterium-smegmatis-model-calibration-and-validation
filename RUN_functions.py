import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time
import os
import itertools

from scipy import stats

from mealpy import FloatVar, PSO # type: ignore
from mealpy.utils.problem import FloatVar # type: ignore
from mealpy.swarm_based import PSO # type: ignore

from System_info import system_info as system_data
from Analysis_functions import validation_analysis, parameter_analysis
from Aux_Functions import define_cost_function, format_number

import sys
import contextlib


# Suppress all output from PSO Optimization to avoid cluttering the console during optimization
# Comment this out if you want to see the output from PSO
# Note: This will suppress all output, including errors in the optimization, so use with caution.
@contextlib.contextmanager
def suppress_all_output():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr



def RUN_PARAMETERS_ITERATIONS(iterations, path, params_list, lb, ub, new_og = None):
    """
    Parameters:
        - n: int, number of iterations for each parameter combination
        - path: str, path to save the results Excel file
        - params_list: list of str, names of the parameters to be calibrated
        - param_ranges: dict, dictionary with parameter names as keys and their ranges as values (list of tuples with lower and upper bounds)
    Returns:
        - df: pandas DataFrame, containing the results of the parameter calibration and validation
    Steps:
        1. Check if the file at `path` exists, if so, remove it to start fresh.
        2. Create an empty DataFrame with the necessary columns.
        3. Run the initial model with original parameters and store results.
        4. Iterate through all combinations of parameters and their ranges.
        5. For each combination, run the scenario and store results in the DataFrame.
        6. Save the DataFrame to an Excel file at `path`.
        7. Return the DataFrame with all results.


    """
    
    if os.path.exists(path):
        os.remove(path)
        print(f">>> Existing file {path!r} removed | starting fresh")

    df = pd.DataFrame()
    print(">>> Starting with empty DataFrame")


    parameters_og = system_data['parameters']
    parameters_og_new = {}
    if new_og is not None:
        for key, value in parameters_og.items():
            if key in list(new_og.keys()):
                parameters_og_new[key] = new_og[key]
            else:
                parameters_og_new[key] = value
    else:
        parameters_og_new = parameters_og.copy()
    print(parameters_og_new)
            

    column_names = []

    columns_values = []
    columns_t_values = []
    for key, value in parameters_og_new.items():
        columns_values.append(key)
        columns_t_values.append('t_value_'+ key)

    columns_var_validation = []
    var_names = system_data['var_names']
    for var in var_names:
        columns_var_validation.append('RMSE_'+ var)
        columns_var_validation.append('NMRSE_'+ var)
        columns_var_validation.append('MAPE_'+ var)
    
    column_names.extend(['Model', 'Param Combo', 'lb', 'ub'])
    column_names.extend(columns_values)
    column_names.extend(columns_t_values)
    column_names.extend(columns_var_validation)
    column_names.extend(['AIC', 'BIC', 'RMSE', 'NMRSE', 'MAPE'])

    df = pd.DataFrame(columns=column_names)

    initial_results = RUN_INITIAL(new_og=new_og)
    row_data = ['Original', '', '','']
    row_data.extend(initial_results)
    df.loc[len(df)] = row_data
    df.to_excel(path, index=False)
    print(f">>> Row {len(df)} written to file")

    
    print(f"\n>>> RUN_SCENARIO with parameters: {params_list}")
    print(f"    lb = {format_number(lb)}")
    print(f"    ub = {format_number(ub)}")

        # print(' ')
    final_results_escenario = RUN_SCENARIO(iterations=iterations,
                            params_list=list(params_list),
                            lb=lb,
                            ub=ub,
                            new_og=new_og
                            )
    scenario_rows = []
    for g in range(len(final_results_escenario)):
        row_data = [f"Model_iteration_{g+1}", str(params_list), str(format_number(lb)), str(format_number(ub))]
        row_data.extend(final_results_escenario[g])
        scenario_rows.append(row_data)

    df = pd.concat([df, pd.DataFrame(scenario_rows, columns=df.columns)], ignore_index=True)
    df.to_excel(path, index=False)
    print(f">>> Row {len(df)} written to file")

    return df



    
def RUN_INITIAL(new_og=None):
    """ 
    Runs the initial model with original parameters and returns the results.
    This function retrieves the original parameters from the system data, runs the analysis,
    and formats the results for output.
    Returns:
        - final_results_initial: list, containing the original parameters, t-values, and validation results.
    Steps:
        1. Retrieve the original parameters and their list from the system data.
        2. Run the analysis using the original parameters.
        3. Format the t-values and validation results for output.
        4. Combine the original parameters, t-values, and validation results into a final results list.
        5. Return the final results list.

    """
    parameters_og = system_data['parameters']
    parameters_og_new = {}
    if new_og is not None:
        for key, value in parameters_og.items():
            if key in list(new_og.keys()):
                parameters_og_new[key] = new_og[key]
            else:
                parameters_og_new[key] = value
    else:
        parameters_og_new = parameters_og.copy()
    parameters_og_new_list = list(parameters_og_new.keys())


    print(" ")
    print("-----------------------------------------------------------------------------------------------------")
    print("------------------------------------------ ORIGINAL MODEL -------------------------------------------")
    print("-----------------------------------------------------------------------------------------------------")
    print(" ")

    parameters_og_values = [parameters_og_new[key] for key in parameters_og_new_list]

    ANALYSIS_RESULTS = RUN_ANALYSIS(iteration = None,
                                    parameters=parameters_og_new,
                                    params_list=parameters_og_new_list,
                                    new_og=new_og)
    
    validation_results = ANALYSIS_RESULTS['validation_results']
    t_values = ANALYSIS_RESULTS['t_values']
    t_values_formatted = [format_number(x) for x in t_values]
    validation_results_formatted = [format_number(x) for x in validation_results]
    corr_matrix = ANALYSIS_RESULTS['correlation_matrix']
    sensitivity_df = ANALYSIS_RESULTS['sensitivity']
    residuals = ANALYSIS_RESULTS['residuals']

    final_results_initial = parameters_og_values + t_values_formatted + validation_results_formatted

    return final_results_initial

def RUN_SCENARIO(iterations, params_list, lb, ub, new_og = None):
    """ 
    Runs the scenario for parameter calibration and validation using PSO optimization.
    Parameters:
        - iterations: int, number of iterations to run the PSO optimization.
        - params_list: list of str, names of the parameters to be calibrated.
        - lb: list of float, lower bounds for the parameters.
        - ub: list of float, upper bounds for the parameters.
    Returns:
        - final_results_escenario: list of lists, containing the results for each iteration.
    Steps:
        1. Initialize empty lists to store results from all iterations.
        2. Loop through the number of iterations:
        3. Print the current iteration number.
        4. Call the `RUN_PSO_CALIBRATION` function with the current iteration, parameters, and bounds.
        5. Append the results from each iteration to the respective lists
        6. Print a summary of the calibration results, including parameter names and bounds.
        7. Return the final results for all iterations as a list of lists.
    """

    sensitivity_df_all = []
    corr_matrix_all = []
    residuals_all = []
    final_results_escenario = []

    for i in range(iterations):
        print("")
        print("")
        print(f"                                 ...... Running iteration {i+1} ......                                   ")
        print("")
        
        Results = RUN_PSO_CALIBRATION(iteration = i, 
                                params_list = params_list,
                                lb = lb,
                                ub = ub,
                                new_og=new_og)
                                
        final_result_iteration = Results['FINAL RESULTS']
        final_results_escenario.append(final_result_iteration)
        sensitivity_df = Results['SENSITIVITY']
        corr_matrix = Results['CORRELATION MATRIX']
        residuals = Results['RESIDUALS']

        if sensitivity_df is not None:
            sensitivity_df_all.append(sensitivity_df)
        else:
            print('Not adding sensitivity data for this iteration, it is None')
        if corr_matrix is not None:
            corr_matrix_all.append(corr_matrix)
        else:
            print('Not adding correlation matrix for this iteration, it is None')
        if residuals is not None:
            residuals_all.append(residuals)
        else:
            print('Not adding residuals for this iteration, it is None')

    print(" ")
    print(f"                                 ...... ALL ITERATIONS FOR COMBO DONE ......                                   ")    
    print(" ")


    print("--------------------------------------------------------------------------------------------------------")
    print(f"---------------------- SUMMARY OF {params_list} CALIBRATION -----------------------------------")
    print(f"---------------------- Lower Bounds: {lb} -------------------------------------------------")
    print(f"---------------------- Upper Bounds: {ub} -------------------------------------------------")
    print("--------------------------------------------------------------------------------------------------------")
    print(" ")

    RUN_SUMMARY_ANALYSIS(sensitivity_df_all, corr_matrix_all, residuals_all)

    return final_results_escenario




def RUN_PSO_CALIBRATION(iteration, params_list, lb, ub, new_og=None):
    """
    Runs the PSO optimization for parameter calibration and validation.
    Parameters:
        - iteration: int, current iteration number for the calibration.
        - params_list: list of str, names of the parameters to be calibrated.
        - lb: list of float, lower bounds for the parameters.
        - ub: list of float, upper bounds for the parameters.
    Returns:
        - final_results_escenario: dict, containing the final results of the calibration and validation.
    Steps:
        1. Retrieve the original parameters and their list from the system data.
        2. Print the current iteration number and the parameters being calibrated.
        3. Define the PSO optimization problem with the objective function, bounds, and minimization goal.
        4. Initialize the PSO optimizer with specified parameters.
        5. Suppress all output from the PSO optimization to avoid cluttering the console.
        6. Solve the PSO optimization problem and get the best solution and minimum error.
        7. Print the optimization results, including best solutions, minimum error, and optimization time.
        8. Update the original parameters with the new values obtained from PSO optimization.
        9. Run analysis on the updated parameters and store results in a dictionary.
        10. Return a dictionary containing final results, sensitivity analysis, correlation matrix, and residuals.
    """

    parameters_og = system_data['parameters']
    parameters_og_new = {}
    if new_og is not None:
        for key, value in parameters_og.items():
            if key in list(new_og.keys()):
                parameters_og_new[key] = new_og[key]
            else:
                parameters_og_new[key] = value
    else:
        parameters_og_new = parameters_og.copy()
    parameters_og_new_list = list(parameters_og_new.keys())

    print(" ")
    print("------------------------------------------------------------------------------------------------------------")
    print(f"----------------- PSO OPTIMIZATION FOR PARAMETER CALIBRATION | Iteration {iteration + 1} ------------------")
    print("------------------------------------------------------------------------------------------------------------")
    print(" ")

    problem = {
    "obj_func": define_cost_function(params_list=params_list, new_og=new_og),
    "bounds": FloatVar(lb=lb, ub=ub),
    "minmax": "min"
    }
    pso = PSO.OriginalPSO(epoch=100, pop_size=50, c1=1.5, c2=1.5, w=0.5)
    
    start = time.perf_counter()
    with suppress_all_output():

        g_best = pso.solve(problem)

    end = time.perf_counter()
    
    print("Optimization Results:")
    print("     Best Solutions: ", g_best.solution)
    print("     Minimum Error:", g_best.target.fitness)
    print(f"     Optimization Time: {end - start:.2f} s")

    print(" ")
    print("------------------------------------------------------------------------------------------------------------")
    print(f"-------------------------------------- NEW PARAMETERS MODEL {iteration +1} ------------------------------------------")
    print("------------------------------------------------------------------------------------------------------------")
    print(" ")
    
    new_params = g_best.solution
    new_params_dict = dict(zip(params_list, new_params))
    
    parameters_updated = copy.deepcopy(parameters_og_new)
    
    parameters_values = []
    i_param = 0
    for param in parameters_og_new_list:
        if param not in params_list:
            parameters_updated[param] = parameters_og_new[param]
            parameters_values.append(None)
        else:
            new_value = new_params_dict[param]
            parameters_updated[param] = new_value
            parameters_values.append(new_value)
            upper_limit = True
            lower_limit = True
            if abs((abs(new_value) - abs(ub[i_param]))) < 1e-3:
                print(f"!!!       Warning: Parameter '{param}' reached its upper limit ({format_number(ub[i_param])}).")
                upper_limit = False
            if abs((abs(new_value) - abs(lb[i_param]))) < 1e-3:
                print(f"!!!       Warning: Parameter '{param}' reached its lower limit ({format_number(lb[i_param])}).")                
                lower_limit = False
            if upper_limit and lower_limit:
                print(f"Parameter '{param}' is between the limits ({format_number(lb[i_param])}, {format_number(ub[i_param])}).")
            i_param += 1

        parameters_values_formatted = [format_number(x) for x in parameters_values]

    df_new_params = pd.DataFrame({
        "Parameter": params_list,
        "Original": [parameters_og_new[key] for key in params_list],
        "New": [parameters_updated[key] for key in params_list]})
    print(df_new_params)

    ANALYSIS_RESULTS = RUN_ANALYSIS(iteration = iteration,
                                    parameters=parameters_updated,
                                    params_list=params_list,
                                    new_og=new_og)
    
    validation_results = ANALYSIS_RESULTS['validation_results']
    validation_results_formatted = [format_number(x) for x in validation_results]
    t_values = ANALYSIS_RESULTS['t_values']
    t_values_formatted = [format_number(x) for x in t_values]

    corr_matrix = ANALYSIS_RESULTS['correlation_matrix']
    sensitivity_df = ANALYSIS_RESULTS['sensitivity']
    residuals = ANALYSIS_RESULTS['residuals']

    final_results_escenario = parameters_values_formatted + t_values_formatted + validation_results_formatted
    return {'FINAL RESULTS' : final_results_escenario,
                'SENSITIVITY' : sensitivity_df,
                'CORRELATION MATRIX' : corr_matrix,
                'RESIDUALS' : residuals
                }

def RUN_ANALYSIS(iteration, parameters, params_list, new_og=None):
    """ 
    Runs the validation and parameter analysis for the given parameters.
    Parameters:
        - iteration: int, current iteration number for the analysis.
        - parameters: dict, dictionary containing the parameters to be analyzed.
        - params_list: list of str, names of the parameters to be analyzed.
    Returns:
        - analysis_results: dict, containing validation results, residuals, t-values, correlation matrix, and sensitivity analysis.
    Steps:
        1. Retrieve the original parameters list from the system data.
        2. If iteration is None, initialize an empty params_list.
        3. Run validation analysis using the provided parameters and iteration.
        4. Extract validation results and residuals from the validation analysis.
        5. Run parameter analysis using the provided parameters and iteration.
        6. If parameter analysis fails, set t-values, correlation matrix, and sensitivity DataFrame to None.
        7. Otherwise, extract t-values, correlation matrix, and sensitivity DataFrame from the parameter analysis.
        8. Return a dictionary containing validation results, residuals, t-values, correlation matrix, and sensitivity DataFrame.
    """
    parmeters_og_list = system_data['parameters_og_list']
    if iteration is None:
        params_list = []
    
    # Run validation analysis, if initial, use original parameters
    val_analysis = validation_analysis(iteration = iteration,
                                        parameters = parameters,
                                        params_list= params_list,
                                        new_og=new_og)
        
    validation_results = val_analysis['Validation results']
    residuals = val_analysis['Residuals']

    param_analysis = parameter_analysis(iteration = iteration,
                                        params_list = params_list,
                                        parameters = parameters)
    if param_analysis is None:
        print("!!!!!!!!!!!!!               Parameter Analysis failed. Please check the parameters and initial conditions.")
        t_values = [None] * len(parmeters_og_list)
        corr_matrix = None
        sensitivity_df = None
    else:
        t_values = param_analysis['t_values']
        corr_matrix = param_analysis['correlation_matrix']
        sensitivity_df = param_analysis['sensitivity']

    return {'validation_results': validation_results,
            'residuals': residuals,
            't_values': t_values,
            'correlation_matrix': corr_matrix,
            'sensitivity': sensitivity_df}



def RUN_SUMMARY_ANALYSIS(sensitivity_df_all, corr_matrix_all, residuals_all):
    """ 
    Generates summary plots for sensitivity analysis, residuals, and correlation matrix.
    Parameters:
        - sensitivity_df_all: list of pandas DataFrame, containing sensitivity data from all iterations.
        - corr_matrix_all: list of numpy.ndarray, containing correlation matrices from all iterations.
        - residuals_all: list of numpy.ndarray, containing residuals from all iterations.
    Returns:
        - None, displays plots for sensitivity analysis, residuals, and correlation matrix.
    """
    
    # Sensitivity Analysis Plot
    if sensitivity_df_all is None or len(sensitivity_df_all) == 0 or sensitivity_df_all == []:
        print("No sensitivity data available for plotting.")
    else:
        
        stacked_sens = np.stack([df.values for df in sensitivity_df_all], axis=0)  # (n_iter, n_params, n_states)
        mean_sens = stacked_sens.mean(axis=0)
        std_sens  = stacked_sens.std(axis=0)

        params = sensitivity_df_all[0].index.tolist()
        states = sensitivity_df_all[0].columns.tolist()

        fig, axes = plt.subplots(len(states), 1, figsize=(8, 2*len(states)), sharex=True)
        for j, state in enumerate(states):
            axes[j].bar(params,
                        mean_sens[:, j],
                        yerr=std_sens[:, j],
                        capsize=4,
                        color='red',
                        linewidth=1.5) 
            axes[j].set_title(f"Sensitivity of “{state}” (mean ± σ)")
            axes[j].set_ylabel("Sensitivity")
            axes[j].tick_params(axis="x", rotation=90)
            axes[j].grid(axis="y", alpha=0.5)
        axes[-1].set_xlabel("Parameter")
        plt.tight_layout()
        plt.show()

    # Residuals Histogram and Q-Q Plot
    if residuals_all is None or len(residuals_all) == 0 or residuals_all == []:
        print("No residuals data available for plotting.")
    else:
        all_res    = np.concatenate(residuals_all)
        mu_all, std_all = stats.norm.fit(all_res)

        bins       = np.histogram_bin_edges(all_res, bins="auto")
        bin_centers = (bins[:-1] + bins[1:]) / 2
        densities  = np.vstack([np.histogram(r, bins=bins, density=True)[0] for r in residuals_all])

        mean_den = densities.mean(axis=0)
        std_den  = densities.std(axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax0 = axes[0]
        ax0.bar(bin_centers,
                mean_den,
                width=np.diff(bins),
                alpha=0.7,
                yerr=std_den,
                capsize=4,
                color='blue',
                linewidth=1.5)
        x_norm = np.linspace(bin_centers.min(), bin_centers.max(), 200)
        ax0.plot(x_norm,
                stats.norm.pdf(x_norm, mu_all, std_all),
                'k-',
                lw=2,
                label="Normal fit")
        ax0.set_xlabel("Residual value")
        ax0.set_ylabel("Density")
        ax0.set_title("Residuals Histogram\n(mean ± σ) with Normal Fit")
        ax0.legend()
        ax0.grid(alpha=0.4)

        ax1 = axes[1]
        m     = residuals_all[0].size
        probs = (np.arange(1, m+1) - 0.5) / m
        theo  = stats.norm.ppf(probs, loc=mu_all, scale=std_all)
        sorted_all = np.vstack([np.sort(r) for r in residuals_all])
        mean_q     = sorted_all.mean(axis=0)
        std_q      = sorted_all.std(axis=0)

        ax1.errorbar(theo,
                    mean_q,
                    yerr=std_q,
                    fmt='o',
                    ecolor='gray',
                    elinewidth=1,
                    capsize=3)
        lims = [min(theo.min(), mean_q.min()), max(theo.max(), mean_q.max())]
        ax1.plot(lims, lims, 'r--')
        ax1.set_xlabel("Theoretical Quantiles")
        ax1.set_ylabel("Mean Ordered Residuals")
        ax1.set_title("Aggregated Q–Q Plot\n(mean ± σ)")
        ax1.grid(alpha=0.4)

        plt.tight_layout()
        plt.show()
    
    # Correlation Matrix Heatmap
    if corr_matrix_all is None or len(corr_matrix_all) == 0 or corr_matrix_all == []:
        print("No correlation matrix data available for plotting.")
    else:
        parameters_og_list = system_data['parameters_og_list']
        # 2.4 Correlation-matrix heatmap with mean±σ annotations
        corr_stack = np.stack(corr_matrix_all, axis=0)
        mean_corr  = corr_stack.mean(axis=0)
        std_corr   = corr_stack.std(axis=0)
        
        n      = mean_corr.shape[0]
        annot  = np.empty((n, n), dtype=object)

        for i in range(n):
            for j in range(n):
                # two-line annotation: mean on top, ±σ below
                annot[i, j] = f"{mean_corr[i,j]:.2f}\n±{std_corr[i,j]:.2f}"

        plt.figure(figsize=(12, 9))
        sns.heatmap(mean_corr,
                    annot=annot,
                    fmt="",
                    xticklabels=parameters_og_list,
                    yticklabels=parameters_og_list,
                    cmap="coolwarm",
                    center=0,
                    annot_kws={"fontsize":10, 'fontweight':'bold'})
        plt.title("Parameter Correlation Matrix (mean ± σ)")

        plt.tight_layout()
        plt.show()

        #most correltaed parameters
        print("Most correlated parameters:")
        corr_flat = mean_corr.flatten()
        indices = np.argsort(np.abs(corr_flat))[-10:]  # Get indices of the 10 largest absolute correlations
        for idx in indices:
            i, j = np.unravel_index(idx, mean_corr.shape)
            print(f"{parameters_og_list[i]} ↔ {parameters_og_list[j]}: {mean_corr[i,j]:.2f} ± {std_corr[i,j]:.2f}")
            

