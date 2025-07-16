import numpy as np
import pandas as pd
import copy
from numpy.linalg import cond, matrix_rank, pinv
import logging
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore

from DAE_Systems_Simulations import simulate_model
from System_info import system_info as system_data

logger = logging.getLogger(__name__)


def define_cost_function(params_list, new_og=None):
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

    t_exp = system_data['t_exp']
    df_exp = system_data['df_exp']

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
                                        parameters=parameters_og_new,
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




def compute_FIM(iteration, x0, parameters, time_stamps, params_list, new_og=None):
    """
    Compute the Fisher Information Matrix (FIM), parameter correlation matrix, and t-values for sensitivity analysis.

    Parameters:
        x0: Initial state vector.
        parameters: Dict of all model parameters.
        time_stamps: Array of time points for simulation.
        system_data: Dict containing:
            - 'weights_exp_stack': weights array (n_outputs x 1).
            - 'parameters_og_list': list of parameter names.
            - 'delta': perturbation magnitude for finite differences.
            - 'correlation_threshold': threshold for plotting.
        sim_plus_minus: Function(key, x0, parameters, time_stamps) -> (sim_plus, sim_minus) or None.
        compute_t_values: Function(iteration, parameters, params_list, FIM) -> t-values array.

    Returns:
        Dict with keys:
            'FIM': FIM matrix or None on failure.
            'correlation_matrix': Correlation matrix or None.
            't_values': Array of t-values or None.
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

    parameters_og_list = system_data['parameters_og_list']
    weights_exp = system_data['weights_exp_stack']
    delta = system_data['delta']
    n_params = len(parameters_og_list)
    n_outputs = weights_exp.size
    J = np.zeros((n_outputs, n_params))


    # Build Jacobian via finite differences
    for i, key in enumerate(parameters_og_list):
        result = sim_plus_minus(key, x0, parameters, time_stamps)
        if result is None:
            logger.error("Simulation failed for parameter %s", key)
            return {'FIM': None, 'correlation_matrix': None, 't_values': None}
        sim_plus, sim_minus = result
        # Flatten and weight
        dY_dp = (np.vstack(sim_plus).T - np.vstack(sim_minus).T) / (2 * delta)
        J[:, i] = (dY_dp * weights_exp).flatten()

    # Compute FIM
    FIM = J.T @ J
    cond_num = cond(FIM)
    rank = matrix_rank(FIM)
    logger.info("FIM condition number: %.2e | rank: %d", cond_num, rank)


    # Correlation matrix
    corr_matrix = compute_correlation_matrix(FIM)

    # Compute t-values
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
    # Use pseudo-inverse to handle singular or ill-conditioned FIM
    FIM_inv = pinv(FIM)
    # Vectorized computation of correlation matrix
    var = np.diag(FIM_inv)
    std = np.sqrt(var)
    # Avoid division by zero
    std[std == 0] = np.nan
    corr_matrix = FIM_inv / (std[:, None] * std[None, :])
    # Clamp to [-1, 1]
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    # Mask invalid entries
    mask = np.isnan(corr_matrix)
    # Plot heatmap
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        xticklabels=parameters_og_list,
        yticklabels=parameters_og_list,
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        cmap='coolwarm',
        center=0
    )
    plt.title("Parameter Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # Identify and return highly correlated pairs
    n = corr_matrix.shape[0]
    high_pairs: List[Tuple[str, str, float]] = []
    print(f"\nHighly correlated parameter pairs (|r| > {correlation_threshold}):")
    high_pairs = []
    for i in range(len(parameters_og_list)):
        for j in range(i+1, len(parameters_og_list)):
            val = corr_matrix[i, j]
            if not np.isnan(val) and abs(val) > correlation_threshold:
                high_pairs.append((parameters_og_list[i], parameters_og_list[j], val))
                # note: use the parameter names here, not indexing into the threshold
                print(f"  {parameters_og_list[i]:<15} <--> {parameters_og_list[j]:<15} | r = {val:.4f}")
    if not high_pairs:
        print("  None found.")

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
    if iteration is None:
        return [None] * len(parameters_og_list)

    # Determine indices of calibrated parameters
    indices = [parameters_og_list.index(p) for p in params_list]
    FIM_sub = FIM[np.ix_(indices, indices)]
    cov_sub = pinv(FIM_sub)

    theta = np.array([parameters[p] for p in params_list])
    std_err = np.sqrt(np.diag(cov_sub))
    std_err[std_err == 0] = np.nan
    t_vals = theta / std_err
    print("\nComputed t-values for calibrated parameters:")
    print(f"{'Parameter':<15}{'θ':>12}{'SE':>12}{'t-value':>12}")

    # Print detailed t-values
    for p, th, se, tv in zip(params_list, theta, std_err, t_vals):
        print(f"{p:<15}{th:12.6f}{se:12.6f}{tv:12.2f}")

    # Build complete list including None for fixed parameters
    t_values_complete: List[Optional[float]] = []
    for lbl in parameters_og_list:
        if lbl in params_list:
            t_values_complete.append(float(t_vals[params_list.index(lbl)]))
        else:
            t_values_complete.append(None)
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