import numpy as np
import pandas as pd
import copy
from numpy.linalg import inv

import matplotlib.pyplot as plt
import seaborn as sns # type: ignore

from DAE_Systems_Simulations import simulate_model
from System_info import system_info as system_data


def define_cost_function(params_list):
    """
    Function to define the cost function for calibration using experimental data.
    Parameters:
        - params_list: list of parameter names to be calibrated.
    Returns:
        - cost_function: function that computes the cost based on the difference between simulated and experimental data.
    """
    # Load system information
    var_names = system_data['var_names']
    x0_exp = system_data['x0_exp']
    parameters_og = system_data['parameters']
    t_exp = system_data['t_exp']
    df_exp = system_data['df_exp']

    def cost_function(p_vars): # COST FUNCTION USING PE DATA
        """
        Computes the cost function based on the difference between simulated and experimental data.
        Parameters:
            - p_vars: array of parameter values to be calibrated
        Returns:
            - err: total error between simulated and experimental data
        """
        try:
            df_results = simulate_model(simulation_type='calibrating', 
                                        x0=x0_exp, 
                                        parameters=parameters_og,
                                        time=t_exp,
                                        p_vars=p_vars,
                                        param_list=params_list
                                    )
            
            err = 0
            for var in var_names:
                var_new = df_results[var]
                var_exp = df_exp[var]

                err += np.sum((var_new - var_exp)**2)
        
            return err
        except:
            err = 1e6
            return err
    return cost_function

def sim_plus_minus(key, x0, parameters, time_stamps, base_val = None):
    """
    Function to simulate the model with perturbed parameters for sensitivity analysis or FIM analysis.
    Parameters:
        - key: parameter name to be perturbed.
        - x0: initial conditions for the simulation.
        - parameters: dictionary of model parameters.
        - time_stamps: time points for the simulation.
        - base_val: base value of the parameter for sensitivity analysis (optional).
    Returns:
        - [Y_plus, Y_minus]: list containing the results of the simulation with perturbed parameters.
        - None: if the simulation fails.

    Steps:
        1. Load system information.
        2. Create copies of the parameters for perturbation.
        3. Perturb the parameter based on whether it's a sensitivity analysis or FIM analysis.
        4. Simulate the model with perturbed parameters.
        5. Check if the simulation was successful.
        6. Extract the results for the perturbed parameters.
        7. Return the results as a list of two arrays: Y_plus and Y_minus.
    """
    var_names = system_data['var_names']
    

    params_plus = copy.deepcopy(parameters)
    params_minus = copy.deepcopy(parameters)

    if base_val is not None:                             #Sensitivity analysis
        perturbation = system_data['perturbation']
        params_plus[key] = base_val * (1 + perturbation)
        params_minus[key] = base_val * (1 - perturbation)

    else:                                                 # FIM analysis
        delta = system_data['delta']
        params_plus[key] += delta
        params_minus[key] -= delta

    sim_plus = simulate_model(simulation_type='normal', 
                                x0=x0, 
                                parameters=params_plus, 
                                time=time_stamps)
    if sim_plus is None:
        print(f"!!!!!!!!!!!!!               Simulation with parameter {key} perturbed up failed. Please check the parameters and initial conditions.")
    
    sim_minus = simulate_model(simulation_type='normal', 
                                x0=x0, 
                                parameters=params_minus, 
                                time=time_stamps)
    if sim_minus is None:
        print(f"!!!!!!!!!!!!!               Simulation with parameter {key} perturbed down failed. Please check the parameters and initial conditions.")
    
    if sim_plus is None or sim_minus is None:
        return None
    Y_plus = []
    Y_minus = []
    for var in var_names:
        Y_plus.append(sim_plus[var])
        Y_minus.append(sim_minus[var])

    return [Y_plus, Y_minus]


def residuals_equations(y_val, y_sim, params_list = None):
    """
    Function to compute residuals, RMSE, NRMSE, and MAPE between experimental and simulated data.
    Parameters:
        - y_val: experimental data.
        - y_sim: simulated data.
        - params_list: list of parameter names (optional, used for AIC/BIC calculation).
    Returns:
        - [rmse, nmrse, mape]: list containing RMSE, NRMSE, and MAPE (if params_list is None).
        - [aic, bic, rmse, nmrse, mape]: list containing AIC, BIC, RMSE, NRMSE, and MAPE (if params_list is provided).
        - res: residuals between experimental and simulated data (if params_list is provided).
    """
    y_val_range = np.max(y_val) - np.min(y_val)

    res = y_val - y_sim

    rmse = np.sqrt(np.mean(res**2))

    nmrse = np.sqrt(np.mean(res**2)) / y_val_range
    
    mape = np.mean(np.abs(res/ y_val)) * 100

    if params_list is None:
        return [rmse, nmrse, mape]
    else:
        n = len(y_val)
        k = len(params_list) 

        rss = np.sum(res**2)

        aic = 2 * k + n * np.log(rss / n)

        bic = k * np.log(n) + n * np.log(rss / n)

        return [[aic, bic, rmse, nmrse, mape], res]




def compute_FIM(iteration, x0, parameters, time_stamps, params_list):
    """
    Function to compute the Fisher Information Matrix (FIM) for parameter sensitivity analysis.
    Parameters:
        - iteration: current iteration number (used for t-values).
        - x0: initial conditions for the state variables.
        - parameters: dictionary of model parameters.
        - time_stamps: time points for the simulation.
        - params_list: list of parameter names to be calibrated.
    returns:
        - FIM: Fisher Information Matrix.
        - correlation_matrix: correlation matrix of the parameters.
        - t_values_complete: t-values for the parameters (if iteration is not None).
        - None: if the simulation fails.

    Steps:
        1. Load system information.
        2. Initialize the Fisher Information Matrix (FIM) and Jacobian matrix (J).
        3. Loop through each parameter in the parameters list.
        4. Simulate the model with perturbed parameters using `sim_plus_minus`.
        5. Compute the Jacobian matrix (dY_dp) for the perturbed parameters.
        6. Compute the FIM by multiplying the Jacobian matrix with its transpose.
        7. Compute eigenvalues, condition number, and rank of the FIM.
        8. Compute the correlation matrix from the FIM.
        9. Compute t-values for the parameters if iteration is not None.
        10. Return the FIM, correlation matrix, and t-values.

    """

    print(" ")
    print("                                            >>>>>>>>>> FIM Analysis <<<<<<<<<<                                            ")
    print(" ")

    weights_exp = system_data['weights_exp_stack']
    parameters_og_list = system_data['parameters_og_list']
    n_params = len(parameters_og_list)
    n_outputs = weights_exp.shape[0] * weights_exp.shape[1]
    delta = system_data['delta']
    J = np.zeros((n_outputs, n_params))

    i = 0
    for key in parameters_og_list:
        sim_plus_minus_results = sim_plus_minus(key, x0, parameters, time_stamps)
        if sim_plus_minus_results is None:
            print(f"!!!!!!!!!!!!!               FIM Analysis for parameter {key} failed. Please check the parameters and initial conditions.")
            FIM = None
            return None
        sim_plus = np.vstack(sim_plus_minus_results[0]).T
        sim_minus = np.vstack(sim_plus_minus_results[1]).T
        dY_dp = (sim_plus - sim_minus) / (2 * delta)
        dY_dp_weighted = dY_dp * weights_exp

        J[:, i] = dY_dp_weighted.flatten()
        i += 1
    
    FIM = J.T @ J


    eigvals = np.linalg.eigvalsh(FIM)
    cond_number = np.linalg.cond(FIM)
    rank = np.linalg.matrix_rank(FIM)

    print(f"FIM condition number: {cond_number:.2e}")
    print(f"FIM rank: {rank}")
    print(f"FIM eigenvalues: {eigvals}")

    corr_matrix = compute_correlation_matrix(FIM)
    t_values_complete = compute_t_values(iteration, parameters, params_list, FIM)

    if t_values_complete is None:
        print(f"!!!!!!!!!!!!!               Parameter analysis, t-values, failed on iteration. Please check the parameters and simulation results.")
        t_values_complete = [None] * len(parameters_og_list)
        
    return {'FIM': FIM,
            'correlation_matrix': corr_matrix,
            't_values': t_values_complete,}

def compute_correlation_matrix(FIM):
    """
    Function to compute the correlation matrix from the Fisher Information Matrix (FIM).
    Parameters:
        - FIM: Fisher Information Matrix.
    Returns:
        - corr_matrix: correlation matrix of the parameters.
    """
    parameters_og_list = system_data['parameters_og_list']
    correlation_threshold = system_data['correlation_threshold']
    FIM_inv = inv(FIM)
    corr_matrix = np.zeros_like(FIM)

    for i in range(len(parameters_og_list)):
        for j in range(len(parameters_og_list)):
            try:
                corr_matrix[i, j] = FIM_inv[i, j] / np.sqrt(FIM_inv[i, i] * FIM_inv[j, j])
                #to catch invalid srqt operations
            except ValueError:
                print(f"Invalid sqrt operation for indices ({i}, {j})")
                corr_matrix[i, j] = None

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        corr_matrix,
        xticklabels=parameters_og_list,
        yticklabels=parameters_og_list,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        center=0
    )
    plt.title("Parameter correlation matrix")
    plt.tight_layout()
    plt.show()

    highly_correlated_pairs = []

    for i in range(len(parameters_og_list)):
        for j in range(i + 1, len(parameters_og_list)):
            corr_val = corr_matrix[i, j]
            if abs(corr_val) > correlation_threshold:
                highly_correlated_pairs.append((parameters_og_list[i], parameters_og_list[j], corr_val))

    print(f"Highly correlated parameter pairs (|ρ| > {correlation_threshold}):\n")
    for p1, p2, corr in highly_correlated_pairs:
        print(f"{p1:10s} <--> {p2:10s} | correlation: {corr:.4f}")

    return corr_matrix
    

def compute_t_values(iteration, parameters, params_list, FIM):
    """
    Function to compute t-values for the parameters based on the Fisher Information Matrix (FIM).
    Parameters:
        - iteration: current iteration number (used for t-values).
        - parameters: dictionary of model parameters.
        - params_list: list of parameter names to be calibrated.
        - FIM: Fisher Information Matrix.
    Returns:
        - t_values_complete: list of t-values for the parameters (if iteration is not None).
        - None: if the simulation fails.

    Steps:
        1. Load system information.
        2. Check if iteration is None, if so, initialize t_values_complete with None.
        3. If iteration is not None, compute the adjusted parameters and their indices.
        4. Compute the adjusted FIM and its inverse.
        5. Extract the adjusted parameters based on params_list.
        6. Compute the standard errors from the diagonal of the covariance matrix.
        7. Compute the t-values for each parameter.
        8. Create a complete list of t-values, including None for parameters not in params_list.
        9. Print the t-values for each parameter.
        10. Return the t_values_complete list.
    """

    print(" ")
    print("                                              >>>>>>>>>> t-values <<<<<<<<<<                                              ")
    print(" ")
    parameters_og_list = system_data['parameters_og_list']
    if iteration == None:
        t_values_complete = [None] * len(parameters_og_list)
    else:
        params_adjusted = copy.deepcopy(parameters)
    
        adjusted_indices = [params_list.index(k) for k in params_list]

        FIM_adj = FIM[np.ix_(adjusted_indices, adjusted_indices)]
        Cov_adj = inv(FIM_adj)

        theta_adj = np.array([params_adjusted[k] for k in params_list])
        try:
            std_errors = np.sqrt(np.diag(Cov_adj))
        except RuntimeError as e:
            print("Error in computing standard errors:", e)
            return None
        t_values = theta_adj / std_errors
        t_values_complete = []
        for key in parameters_og_list:
            if key in params_list:
                t_values_complete.append(t_values[params_list.index(key)])
            else:
                t_values_complete.append(None)
        for k, theta, se, t in zip(params_list, theta_adj, std_errors, t_values):
            
            print(f"{k:<10}: θ = {theta:.6f}, SE = {se:.6f}, t-value = {t:.2f}")
    return t_values_complete


def compute_sensitivity(x0, parameters, time_stamps):
    """
    Function to compute the sensitivity of the model parameters using perturbation analysis.
    Parameters:
        - x0: initial conditions for the state variables.
        - parameters: dictionary of model parameters.
        - time_stamps: time points for the simulation.
    Returns:
        - sensitivity_df: DataFrame containing the sensitivity values for each parameter and state variable.
        - None: if the simulation fails.
    Steps:
        1. Load system information.
        2. Initialize a DataFrame to store sensitivity values.
        3. Simulate the model with the original parameters.
        4. Check if the simulation was successful.
        5. Extract the base values for each state variable.
        6. Loop through each parameter in the parameters list.
        7. Perturb the parameter using `sim_plus_minus`.
        8. Check if the simulation with perturbed parameters was successful.
        9. Compute the relative sensitivity for each state variable.
        10. Sort the sensitivity DataFrame by the mean sensitivity value.
        11. Plot the sensitivity values for each state variable.
        12. Plot the top 5 most sensitive parameters.
        13. Return the sensitivity DataFrame.
    """

    print(" ")
    print("                                        >>>>>>>>>> Sensitivity Analysis <<<<<<<<<<                                        ")
    print(" ")

    perturbation = system_data['perturbation']
    parameters_og_list = system_data['parameters_og_list']
    var_names = system_data['var_names']

    sensitivity_df = pd.DataFrame(index = parameters_og_list, 
                                columns=var_names)
    model_sim_sensitivity = simulate_model(simulation_type='normal', 
                                            x0=x0, 
                                            parameters=parameters,
                                            time=time_stamps)
    if model_sim_sensitivity is None:
        print("!!!!!!!!!!!!!               Simulation for model sensitivity failed. Please check the parameters and initial conditions.")
        return None
    Y_base = []
    for var in var_names:
        Y_base.append(model_sim_sensitivity[var])

    for key in parameters_og_list:
        base_val = parameters[key]
        if base_val == 0 or np.isnan(base_val):
            continue

        sim_plus_minus_results = sim_plus_minus(key=key,
                                                x0=x0,
                                                parameters=parameters,
                                                time_stamps=time_stamps,
                                                base_val=base_val)
        
        if sim_plus_minus_results is None:
            print(f"!!!!!!!!!!!!!               Validation Analysis for parameter {key} failed. Please check the parameters and initial conditions.")
            return None
        sim_plus = sim_plus_minus_results[0]
        sim_minus = sim_plus_minus_results[1]    

        for i, var in enumerate(var_names):
            delta_Y = sim_plus[i] - sim_minus[i]
            rel_Y = delta_Y / (2 * perturbation * Y_base[i])
            mean_S = np.mean(np.abs(rel_Y))
            sensitivity_df.loc[key, var] = mean_S

    sensitivity_df = sensitivity_df.astype(float)
    sensitivity_df['Mean'] = sensitivity_df.mean(axis=1)
    sensitivity_sorted = sensitivity_df.sort_values('Mean', ascending=False)
    top5_df = sensitivity_sorted.head(5)
    sensitivity_df.drop(columns='Mean', inplace=True)

    fig, axes = plt.subplots(len(var_names), 1, figsize=(10, 12), sharey=False)

    for i, state in enumerate(sensitivity_df.columns):
        axes[i].bar(sensitivity_df.index, sensitivity_df[state], color='red')
        axes[i].set_title(f'State {state}')
        axes[i].set_ylabel('Sensitivity')
        axes[i].set_xticks(range(len(sensitivity_df.index)))
        axes[i].set_xticklabels(sensitivity_df.index, rotation=90, ha='right')
        axes[i].grid(True, axis='y', linestyle='-', alpha=0.6)

    axes[-1].set_xlabel('Parameter')
    plt.tight_layout()
    plt.show()

    top5_df = sensitivity_df.sum(axis=1).nlargest(5)
    top5_keys = top5_df.index
    top5_plot_df = sensitivity_df.loc[top5_keys]

    palette = sns.color_palette("Set2", n_colors=sensitivity_df.shape[1])
    top5_plot_df.plot(kind='barh', stacked=True, color=palette)
    plt.gca().invert_yaxis()
    plt.xlabel('Mean relative sensitivity')
    plt.title('5 most sensitive parameters')
    plt.legend(title='Output variable')
    plt.tight_layout()
    plt.show()

    return sensitivity_df


def format_number(x, decimals=3):
    """
    Function to format a number to a specified number of decimal places.
    Parameters:
        - x: number to be formatted.
        - decimals: number of decimal places to format to (default is 3).
    Returns:
        - formatted number as a string with the specified number of decimal places.
        - x: if x is None or cannot be converted to float.
    """
    if x is None:
        return None
    if isinstance(x, list) and len(x) == 1:
        x = x[0]
    try:
        return f"{float(x):.{decimals}f}"
    except (ValueError, TypeError):
        return x