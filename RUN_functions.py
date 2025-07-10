import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time
import os


from scipy import stats

from mealpy import FloatVar, PSO # type: ignore
from mealpy.utils.problem import FloatVar # type: ignore
from mealpy.swarm_based import PSO # type: ignore

from System_info import system_info
from Analysis_functions import validation_analysis, parameter_analysis
from Aux_Functions import define_cost_function



def RUN_MAIN(iterations, path, perturbation, delta, correlation_threshold, params_list, lb, ub):
    if os.path.exists(path):
        os.remove(path)
        print(f">>> Existing file {path!r} removed | starting fresh")

    df = pd.DataFrame()
    print(">>> Starting with empty DataFrame")

    df.to_excel(path, index=False)
    print(f">>> Written new file {path!r}")
        
    sensitivity_df_all = []
    corr_matrix_all = []
    residuals_all = []

    for i in range(iterations):
        print("")
        print("")
        print(f"                                 ...... Running iteration {i+1} of {iterations} ......                                   ")
        print("")
    
        Results = RUN_DAE_CALIBRATION(iteration = i, 
                                perturbation = perturbation, 
                                delta = delta,
                                correlation_threshold = correlation_threshold, 
                                params_list = params_list,
                                lb = lb,
                                ub = ub)
        
        FINAL_RESULTS = Results['FINAL RESULTS']
        sensitivity_df = Results['SENSITIVITY']
        corr_matrix = Results['CORRELATION MATRIX']
        residuals = Results['RESIDUALS']

        if FINAL_RESULTS is None:
            if i == 0:
                print("!!!!!!!!!!!!!               Initial run failed. Please check the parameters and initial conditions.")
                return None
            else:
                print(f"!!!!!!!!!!!!!               Iteration {i+1} failed. Skipping to next iteration.")
                continue
            

        if i == 0:
            run_name = "Original Model"
        else:
            run_name = f"Model{i+1}"
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
                

            
        FINAL_RESULTS.insert(0, 'Model_', run_name)

        before = df.shape[0]
        df = pd.concat([df, FINAL_RESULTS], ignore_index=True)
        after = df.shape[0]
        print(" ")
        print(f"Appended row, df rows went {before} → {after}")

        df.to_excel(path, index=False)
        print(f"Saved to {path}")
    
    print(" ")
    print(" ")
    print(" ")

    print(" ALL ITERATIONS DONE — final df shape:", df.shape)

    
    print(" ")
    print("--------------------------------------------------------------------------------------------------------------------------")
    print(f"------------------------------- SUMMARY OF {params_list} CALIBRATION --------------------------------------------")
    print("--------------------------------------------------------------------------------------------------------------------------")
    print(" ")

    RUN_SUMMARY_ANALYSIS(sensitivity_df_all, corr_matrix_all, residuals_all)


    return FINAL_RESULTS


def RUN_DAE_CALIBRATION(iteration, perturbation, delta, correlation_threshold, params_list, lb, ub):
    """

    """

    system_data = system_info

    var_names = system_data['var_names']

    parameters_og = system_data['parameters']
    
    

    if iteration == 0:
        print(" ")
        print("--------------------------------------------------------------------------------------------------------------------------")
        print("----------------------------------------------------- ORIGINAL MODEL -----------------------------------------------------")
        print("--------------------------------------------------------------------------------------------------------------------------")
        print(" ")


        parameters_initial_dict = {}
        for key, value in parameters_og.items():
            if key in params_list:
                parameters_initial_dict[key + '_value'] = value
        df_parameters_value= pd.DataFrame([parameters_initial_dict])


        ANALYSIS_RESULTS = RUN_ANALYSIS(system_data=system_data,
                                        perturbation=perturbation,
                                        correlation_threshold=correlation_threshold,
                                        delta = delta,
                                        parameters=parameters_og,
                                        var_names=var_names,
                                        params_list=params_list,
                                        type="initial")




    else:
        print(" ")
        print("--------------------------------------------------------------------------------------------------------------------------")
        print(f"------------------------ PSO OPTIMIZATION FOR PARAMETER CALIBRATION | Iteration {iteration + 1} -------------------------")
        print("--------------------------------------------------------------------------------------------------------------------------")
        print(" ")


        
        problem = {
        "obj_func": define_cost_function(system_data=system_data,
                                            var_names=var_names,
                                            params_list=params_list),
        "bounds": FloatVar(lb=lb, ub=ub),
        "minmax": "min"
        }
        pso = PSO.OriginalPSO(epoch=100, pop_size=50, c1=1.5, c2=1.5, w=0.5)
        start = time.perf_counter()

        g_best = pso.solve(problem)

        end = time.perf_counter()
        
        print("Optimization Results:")
        print("     Best Solutions: ", g_best.solution)
        print("     Minimum Error:", g_best.target.fitness)
        print(f"     Optimization Time: {end - start:.2f} s")
        

        print(" ")
        print("--------------------------------------------------------------------------------------------------------------------------")
        print(f"--------------------------------------------- NEW PARAMETERS MODEL {iteration +1} -------------------------------------------------")
        print("--------------------------------------------------------------------------------------------------------------------------")
        print(" ")

        new_params = g_best.solution
        new_params_dict = dict(zip(params_list, new_params))
        parameters_updated = copy.deepcopy(parameters_og)
        
        param_dict = {}
        
        i_param = 0
        for i_param in range(len(params_list)):
            key = params_list[i_param]
            new_value = new_params_dict[key]
            parameters_updated[key] = new_value
            param_dict[key + '_value'] = new_value
            upper_limit = True
            lower_limit = True

            if abs((abs(new_value) - abs(ub[i_param]))) < 1e-3:
                print(f"!!!       Warning: Parameter '{key}' reached its upper limit ({ub[i_param]}).")
                upper_limit = False
            if abs((abs(new_value) - abs(lb[i_param]))) < 1e-3:
                print(f"!!!       Warning: Parameter '{key}' reached its lower limit ({lb[i_param]}).")                
                lower_limit = False
            if upper_limit and lower_limit:
                print(f"Parameter '{key}' is between the limits ({lb[i_param]}, {ub[i_param]}).")
        
        df_new_params = pd.DataFrame({
            "Parameter": params_list,
            "Original": [parameters_og[key] for key in params_list],
            "New": [parameters_updated[key] for key in params_list]})
        print(df_new_params)

        df_parameters_value = pd.DataFrame([param_dict])
        # df_parameters_value or paramsupdates ?????????????????????????????????/ for val in RUN

        ANALYSIS_RESULTS = RUN_ANALYSIS(system_data=system_data,
                                        perturbation=perturbation,
                                        correlation_threshold=correlation_threshold,
                                        delta = delta,
                                        parameters=parameters_updated,
                                        var_names=var_names,
                                        params_list=params_list,
                                        type="new")

    


    validation_results = ANALYSIS_RESULTS['validation_results']
    t_values_FIM = ANALYSIS_RESULTS['t_values_FIM']
    corr_matrix = ANALYSIS_RESULTS['correlation_matrix']
    sensitivity_df = ANALYSIS_RESULTS['sensitivity']
    residuals = ANALYSIS_RESULTS['residuals']

    final_results = pd.concat([df_parameters_value, validation_results, t_values_FIM], axis=1)
    return {'FINAL RESULTS' : final_results,
            'SENSITIVITY' : sensitivity_df,
            'CORRELATION MATRIX' : corr_matrix,
            'RESIDUALS' : residuals
            }





def RUN_ANALYSIS(system_data, perturbation, correlation_threshold, delta, parameters, var_names, params_list, type):
    
    # Run validation analysis, if initial, use original parameters
    val_analysis = validation_analysis(system_data = system_data,
                                                            var_names = var_names,
                                                            parameters = parameters,
                                                            params_list= params_list,
                                                            type = type)
        
    validation_results = val_analysis['Validation results']
    residuals = val_analysis['Residuals']

    param_analysis = parameter_analysis(perturbation = perturbation, 
                                        correlation_threshold = correlation_threshold,
                                        params_list = params_list,
                                        parameters = parameters,
                                        delta = delta,
                                        type = type)
    if param_analysis is None:
        print("!!!!!!!!!!!!!               Parameter Analysis failed. Please check the parameters and initial conditions.")
        t_values_FIM = None
        corr_matrix = None
        sensitivity_df = None
    else:
        t_values_FIM = param_analysis['t_values']
        corr_matrix = param_analysis['correlation_matrix']
        sensitivity_df = param_analysis['sensitivity']

    return {'validation_results': validation_results,
            'residuals': residuals,
            't_values_FIM': t_values_FIM,
            'correlation_matrix': corr_matrix,
            'sensitivity': sensitivity_df}



def RUN_SUMMARY_ANALYSIS(sensitivity_df_all, corr_matrix_all, residuals_all):
    # — Stack data from all iterations —
    # sensitivity_df_all: list of pandas.DataFrame (shape: n_params × n_states)
    # corr_matrix_all:   list of numpy.ndarray (shape: n_params × n_params)
    # residuals_all:     list of numpy.ndarray
    if sensitivity_df_all is None or len(sensitivity_df_all) == 0 or sensitivity_df_all == []:
        print("No sensitivity data available for plotting.")
    else:
        
        # 1) Sensitivity Bar‐Plots with ±1σ Error Bars
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


    if residuals_all is None or len(residuals_all) == 0 or residuals_all == []:
        print("No residuals data available for plotting.")
    else:
        # 2.2 Residuals histogram with ±1σ shading and normal fit curve
        all_res    = np.concatenate(residuals_all)
        mu_all, std_all = stats.norm.fit(all_res)

        bins       = np.histogram_bin_edges(all_res, bins="auto")
        bin_centers = (bins[:-1] + bins[1:]) / 2
        densities  = np.vstack([np.histogram(r, bins=bins, density=True)[0] for r in residuals_all])

        mean_den = densities.mean(axis=0)
        std_den  = densities.std(axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: histogram with ±1σ error bars and normal‐fit curve
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

        # Right: aggregated Q–Q plot with ±1σ error bars
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
    
    if corr_matrix_all is None or len(corr_matrix_all) == 0 or corr_matrix_all == []:
        print("No correlation matrix data available for plotting.")
    else:
        system_data = system_info
        parameters_og = system_data['parameters']
        params = list(parameters_og.keys())
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
                    xticklabels=params,
                    yticklabels=params,
                    cmap="coolwarm",
                    center=0,
                    annot_kws={"fontsize":10, 'fontweight':'bold'})
        plt.title("Parameter Correlation Matrix (mean ± σ)")

        plt.tight_layout()
        plt.show()