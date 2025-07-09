import numpy as np
import pandas as pd
import copy
from numpy.linalg import inv
from statsmodels.stats.stattools import durbin_watson # type: ignore

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns # type: ignore

from DAE_Systems_Simulations import simulate_model


def sim_plus_minus(key, x0, parameters, constants, time_stamps, var_names, perturbation =  None, base_val = None, delta = None, type = None):
    params_plus = copy.deepcopy(parameters)
    params_minus = copy.deepcopy(parameters)
    if delta is None:  
        params_plus[key] = base_val * (1 + perturbation)
        params_minus[key] = base_val * (1 - perturbation)

    if perturbation is None:
        params_plus[key] += delta
        params_minus[key] -= delta

    sim_plus = simulate_model(simulation_type='normal', 
                                x0=x0, 
                                parameters=params_plus, 
                                constants=constants, 
                                time=time_stamps)
    if sim_plus is None:
        print(f"!!!!!!!!!!!!!               Simulation with parameter {key} perturbed up failed. Please check the parameters and initial conditions.")
    
    sim_minus = simulate_model(simulation_type='normal', 
                                x0=x0, 
                                parameters=params_minus, 
                                constants=constants, 
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

def compute_FIM(x0, parameters, constants, time_stamps, weights_exp, correlation_threshold, var_names, params_list, delta=1e-4, type = None):
    """
    Function to compute the Fisher Information Matrix (FIM) for the model parameters.
    Parameters:
        - x0: array, initial conditions for the model.
        - parameters_og: dict, original parameters of the model.
        - constants: dict, constants used in the model.
        - time_stamps: array, time points for the simulation.
        - weights_exp: array, weights for the experimental data.
        - correlation_threshold: float, threshold for identifying highly correlated parameters.
    Returns:
        - FIM: array, Fisher Information Matrix.
    
    This function computes the FIM by perturbing each parameter and calculating the sensitivity of the model outputs.
    It also computes the correlation matrix of the parameters based on the FIM and identifies highly correlated parameter pairs.
    It visualizes the correlation matrix using a heatmap and prints the highly correlated parameter pairs.
    The function also prints the condition number and rank of the FIM, as well as its eigenvalues.
    It returns the FIM as a numpy array.
    """

    print(" ")
    print("                                            >>>>>>>>>> FIM Analysis <<<<<<<<<<                                            ")
    print(" ")


    params_keys = list(parameters.keys())
    n_params = len(params_keys)
    n_outputs = weights_exp.shape[0] * weights_exp.shape[1]

    J = np.zeros((n_outputs, n_params))

    i = 0
    for key in params_keys:
        sim_plus_minus_results = sim_plus_minus(key, x0, parameters, constants, time_stamps, var_names, delta = delta)
            
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

    compute_correlation_matrix(FIM, parameters, correlation_threshold)
    df_t_values = None
    
    if type not in ['initial', 'Initial']:
        df_t_values = compute_t_values(parameters, params_list, FIM)
        
    return [FIM, df_t_values]

def compute_correlation_matrix(FIM, parameters, correlation_threshold):
    params_keys = list(parameters.keys())
    FIM_inv = inv(FIM)
    corr_matrix = np.zeros_like(FIM)

    for i in range(len(params_keys)):
        for j in range(len(params_keys)):
            try:
                corr_matrix[i, j] = FIM_inv[i, j] / np.sqrt(FIM_inv[i, i] * FIM_inv[j, j])
                #to catch invalid srqt operations
            except ValueError:
                print(f"Invalid sqrt operation for indices ({i}, {j})")
                corr_matrix[i, j] = None

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        corr_matrix,
        xticklabels=params_keys,
        yticklabels=params_keys,
        annot=True,
        fmt=".2f",
        cmap=sns.diverging_palette(270, 330, as_cmap=True),
        center=0
    )
    plt.title("Parameter correlation matrix")
    plt.tight_layout()
    plt.show()
    

    highly_correlated_pairs = []

    for i in range(len(params_keys)):
        for j in range(i + 1, len(params_keys)):
            corr_val = corr_matrix[i, j]
            if abs(corr_val) > correlation_threshold:
                highly_correlated_pairs.append((params_keys[i], params_keys[j], corr_val))

    print(f"Highly correlated parameter pairs (|ρ| > {correlation_threshold}):\n")
    for p1, p2, corr in highly_correlated_pairs:
        print(f"{p1:10s} <--> {p2:10s} | correlation: {corr:.4f}")
    

def compute_t_values(parameters, params_list, FIM):
    """
    Function to compute t-values for the parameters based on the Fisher Information Matrix (FIM).
    Parameters:
        - parameters: dict, model parameters.
        - params_list: list, list of parameter names to compute t-values for.
        - FIM: array, Fisher Information Matrix.
    Returns:
        - df_t_values: DataFrame, containing t-values for the specified parameters.
    
    """

    print(" ")
    print("                                              >>>>>>>>>> t-values <<<<<<<<<<                                              ")
    print(" ")
    params_adjusted = copy.deepcopy(parameters)
    params_keys = list(parameters.keys())

    adjusted_indices = [params_keys.index(k) for k in params_list]

    FIM_adj = FIM[np.ix_(adjusted_indices, adjusted_indices)]
    Cov_adj = inv(FIM_adj)

    theta_adj = np.array([params_adjusted[k] for k in params_list])
    try:
        std_errors = np.sqrt(np.diag(Cov_adj))
    except RuntimeError as e:
        print("Error in computing standard errors:", e)
        return None
    t_values = theta_adj / std_errors
    
    t_dict = {}


    for k, theta, se, t in zip(params_list, theta_adj, std_errors, t_values):
        print(f"{k:<10}: θ = {theta:.6f}, SE = {se:.6f}, t-value = {t:.2f}")
        t_dict['t_value_'+k] = t
    df_t_values = pd.DataFrame([t_dict])
    return df_t_values


def compute_sensitivity(x0, parameters, constants, time_stamps, perturbation, var_names):
    """
    Function to compute the sensitivity of the parameters of the model to each state variable
    Parameters:
        - x0: array, initial conditions for the model.
        - parameters_og: dict, original parameters of the model.
        - constants: dict, constants used in the model.
        - time_stamps: array, time points for the simulation.
        - perturbation: float, perturbation factor for sensitivity analysis.
        - var_names: list, names of the state variables to analyze.
    Returns:
        - sensitivity_df: DataFrame, containing the sensitivity values for each parameter and state variable.
    
    """

    print(" ")
    print("                                        >>>>>>>>>> Sensitivity Analysis <<<<<<<<<<                                        ")
    print(" ")
    param_keys = list(parameters.keys())

    sensitivity_df = pd.DataFrame(index = param_keys, columns=var_names)
    model_sim_sensitivity = simulate_model(simulation_type='normal', 
                        x0=x0, 
                        parameters=parameters, 
                        constants=constants, 
                        time=time_stamps)
    if model_sim_sensitivity is None:
        print("!!!!!!!!!!!!!               Simulation for model sensitivity failed. Please check the parameters and initial conditions.")
        return None
    Y_base = []
    for var in var_names:
        Y_base.append(model_sim_sensitivity[var])

    for key in param_keys:
        base_val = parameters[key]
        if base_val == 0 or np.isnan(base_val):
            continue

        sim_plus_minus_results = sim_plus_minus(key, x0, parameters, constants, time_stamps, var_names, perturbation=perturbation, base_val=base_val)
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

    # print(sensitivity_df)

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



def residuals_equations(y_val, y_sim, params_list):

    n = len(y_val)
    k = len(params_list) 

    y_val_range = np.max(y_val) - np.min(y_val)

    res = y_val - y_sim

    rmse = np.sqrt(np.mean(res**2))

    nmrse = np.sqrt(np.mean(res**2)) / y_val_range
    
    mape = np.mean(np.abs(res/ y_val)) * 100

    rss = np.sum(res**2)

    aic = 2 * k + n * np.log(rss / n)

    bic = k * np.log(n) + n * np.log(rss / n)

    return {'res':res,
            'rmse': rmse,
            'nmrse': nmrse,
            'mape': mape,
            'aic': aic,
            'bic': bic}









    
