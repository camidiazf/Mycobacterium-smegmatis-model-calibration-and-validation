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


def compute_FIM(x0, parameters_og, constants, time_stamps, weights_exp, correlation_threshold, delta=1e-4,):
    """
    Function to compute the Fisher Information Matrix (FIM) for the model parameters.
    """
    print(" ")
    print(" ")
    print("                                -------------- FIM Analysis --------------")


    params_keys = list(parameters_og.keys())
    n_params = len(params_keys)
    n_outputs = weights_exp.shape[0] * weights_exp.shape[1]

    J = np.zeros((n_outputs, n_params))

    i = 0
    for key in params_keys:
        params_plus = copy.deepcopy(parameters_og)
        params_minus = copy.deepcopy(parameters_og)
        params_plus[key] += delta
        params_minus[key] -= delta

        sim_plus = simulate_model(x0, params_plus, constants, time_stamps)
        sim_minus = simulate_model(x0, params_minus, constants, time_stamps)

        sim_plus = np.vstack([sim_plus['X'].values, sim_plus['C'].values, sim_plus['N'].values, sim_plus['pH'].values]).T
        sim_minus = np.vstack([sim_minus['X'].values, sim_minus['C'].values, sim_minus['N'].values, sim_minus['pH'].values]).T
        dY_dp = (sim_plus - sim_minus) / (2 * delta)
        dY_dp_weighted = dY_dp * weights_exp

        J[:, i] = dY_dp_weighted.flatten()
        i += 1

    FIM = J.T @ J


    eigvals = np.linalg.eigvalsh(FIM)
    cond_number = np.linalg.cond(FIM)
    rank = np.linalg.matrix_rank(FIM)

    # print(f"FIM condition number: {cond_number:.2e}")
    # print(f"FIM rank: {rank}")
    # print(f"FIM eigenvalues: {eigvals}")

    FIM_inv = inv(FIM)
    corr_matrix = np.zeros_like(FIM)

    for i in range(len(params_keys)):
        for j in range(len(params_keys)):
            corr_matrix[i, j] = FIM_inv[i, j] / np.sqrt(FIM_inv[i, i] * FIM_inv[j, j])

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
        print(f"{p1:20s} <--> {p2:20s} | correlation: {corr:.4f}")


    return FIM


def compute_t_values(parameters, params_list, FIM, type = None):

    print(" ")
    print(" ")
    print("                                -------------- t-values --------------")
    params_adjusted = copy.deepcopy(parameters)
    params_keys = list(parameters.keys())

    adjusted_indices = [params_keys.index(k) for k in params_list]

    FIM_adj = FIM[np.ix_(adjusted_indices, adjusted_indices)]
    Cov_adj = inv(FIM_adj)

    theta_adj = np.array([params_adjusted[k] for k in params_list])
    std_errors = np.sqrt(np.diag(Cov_adj))
    t_values = theta_adj / std_errors

    print(" ")
    print(" ")
    
    t_dict = {}


    for k, theta, se, t in zip(params_list, theta_adj, std_errors, t_values):
        print(f"{k:<10}: θ = {theta:.6f}, SE = {se:.6f}, t-value = {t:.2f}")
        t_dict['t_value_'+k] = t
    df_t_values = pd.DataFrame([t_dict])
    return df_t_values


def compute_sensitivity(x0, parameters_og, constants, time_stamps, perturbation, var_names):
    """
    Function to compute the sensitivity of the parameters of the model to each state variable.
    """

    print("                                -------------- Sensitivity Analysis --------------")
    param_keys = list(parameters_og.keys())

    sensitivity_df = pd.DataFrame(index = param_keys, columns=var_names)
    sim = simulate_model(x0, parameters_og, constants, time_stamps)
    Y_base = [sim['X'], sim['C'], sim['N'], sim['pH']]

    for key in param_keys:
        base_val = parameters_og[key]
        if base_val == 0 or np.isnan(base_val):
            continue

        params_plus = copy.deepcopy(parameters_og)
        params_minus = copy.deepcopy(parameters_og)
        params_plus[key] = base_val * (1 + perturbation)
        params_minus[key] = base_val * (1 - perturbation)

        sim_plus = simulate_model(x0, params_plus, constants, time_stamps)
        sim_minus = simulate_model(x0, params_minus, constants, time_stamps)
        Y_plus = [sim_plus['X'], sim_plus['C'], sim_plus['N'], sim_plus['pH']]
        Y_minus = [sim_minus['X'], sim_minus['C'], sim_minus['N'], sim_minus['pH']]


        for i, var in enumerate(var_names):
            delta_Y = Y_plus[i] - Y_minus[i]
            rel_Y = delta_Y / (2 * perturbation * Y_base[i])
            mean_S = np.mean(np.abs(rel_Y))
            sensitivity_df.loc[key, var] = mean_S


    sensitivity_df = sensitivity_df.astype(float)
    sensitivity_df['Mean'] = sensitivity_df.mean(axis=1)
    sensitivity_sorted = sensitivity_df.sort_values('Mean', ascending=False)
    top5_df = sensitivity_sorted.head(5)
    sensitivity_df.drop(columns='Mean', inplace=True)

    # print(sensitivity_df)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharey=False)

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


def plotting_comparison(condition, original_sol, params_list = None, parameters_updated = None, new_sol = None, type = None):
    colors = {
        'X': '#66C2A6',
        'C': '#FD8D62',
        'N': '#8DA0CB',
        'CO2': '#FED92F',
        'O2': '#A7D854',
        'pH': '#E78AC3',
        'mu': '#B3B3B3'
        }
    
    def darken_color(color, factor=0.6):
        rgb = mcolors.to_rgb(color)
        return tuple(factor * c for c in rgb)

    system_data = system_info(condition)

    constants = system_data['constants']
    x0_sim_v = system_data['x0_sim_v']
    time_stamps_sim = system_data['time_stamps_sim']
    df_val = system_data['df_val']
    t_exp_v = system_data['t_exp_v']

    X_exp_v = df_val['Biomass (g/L)']
    C_exp_v = df_val['Glycerol (g/L)']
    N_exp_v = df_val['Ammonia (g/L)']
    ph_exp_v = df_val['pH']

    parameters_og = system_data['parameters']

    if type in ['initial', 'Initial']:

        print("VISUAL COMPARISON WITN VALIDATION DATA USING ORIGINAL PARAMETERS, SIMULATED IN VALIDATION TIMEFRAME AND VALIDATION INITIAL CONDITIONS")
        original_sol = simulate_model(x0_sim_v, parameters_og, constants, time_stamps_sim)

        fig, axes = plt.subplots(4, 1, figsize=(8, 10))
        
        axes[0].scatter(t_exp_v, X_exp_v, marker='o', label="Biomass exp validation",color=colors['X'])
        axes[0].plot(original_sol['t'], original_sol['X'], '-', label="Biomass original",color=colors['X'])
        axes[0].set_xlabel("Time (h)")
        axes[0].set_ylabel("Concentration (g/L)")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].scatter(t_exp_v, C_exp_v, marker='o', label="Glycerol exp validation",color=colors['C'])
        axes[1].plot(original_sol['t'], original_sol['C'], '-', label="Glycerol original",color=colors['C'])
        axes[1].set_xlabel("Time (h)")
        axes[1].set_ylabel("Concentration (g/L)")
        axes[1].legend()
        axes[1].grid(True)

        axes[2].scatter(t_exp_v, N_exp_v, marker= 'o', label="Ammonia exp validation",color=colors['N'])
        axes[2].plot(original_sol['t'], original_sol['N'], '-', label="Ammonia original",color=colors['N'])
        axes[2].set_xlabel("Time (h)")
        axes[2].set_ylabel("Concentration (g/L)")
        axes[2].legend()
        axes[2].grid(True)

        axes[3].scatter(t_exp_v, ph_exp_v, marker='o', label="pH exp validation",color=colors['pH'])
        axes[3].plot(original_sol['t'], original_sol['pH'], '-', label="pH original",color=colors['pH'])
        axes[3].set_xlabel("Time (h)")
        axes[3].set_ylabel("Concentration (g/L)")
        axes[3].legend()
        axes[3].grid(True)

        fig.suptitle(f'Model Prediction vs. Validation Data', fontsize=18)
        plt.tight_layout()
        plt.show()


    
    else:

        print("VISUAL COMPARISON WITN VALIDATION DATA USING ORIGINAL PARAMETERS AMF UPDATED PARAMETERS, SIMULATED IN VALIDATION TIMEFRAME AND VALIDATION INITIAL CONDITIONS")

        # VALIDATION DATA
        original_sol = simulate_model(x0_sim_v, parameters_og, constants, time_stamps_sim)
        new_sol = simulate_model(x0_sim_v, parameters_updated, constants, time_stamps_sim)

        
        fig, axes = plt.subplots(4, 1, figsize=(8, 10))
        
        axes[0].scatter(t_exp_v, X_exp_v, marker='o', label="Biomass exp validation",color=colors['X'])
        axes[0].plot(time_stamps_sim, original_sol['X'], '-', label="Biomass original",color=colors['X'])
        axes[0].plot(time_stamps_sim, new_sol['X'], '--', label="Biomass new",color=darken_color(colors['X']))
        axes[0].set_xlabel("Time (h)")
        axes[0].set_ylabel("Concentration (g/L)")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].scatter(t_exp_v, C_exp_v, marker='o', label="Glycerol exp validation",color=colors['C'])
        axes[1].plot(time_stamps_sim, original_sol['C'], '-', label="Glycerol original",color=colors['C'])
        axes[1].plot(time_stamps_sim, new_sol['C'], '--', label="Glycerol new",color=darken_color(colors['C']))
        axes[1].set_xlabel("Time (h)")
        axes[1].set_ylabel("Concentration (g/L)")
        axes[1].legend()
        axes[1].grid(True)

        axes[2].scatter(t_exp_v, N_exp_v, marker= 'o', label="Ammonia exp validation",color=colors['N'])
        axes[2].plot(time_stamps_sim, original_sol['N'], '-', label="Ammonia original",color=colors['N'])
        axes[2].plot(time_stamps_sim, new_sol['N'], '--', label="Ammonia new",color=darken_color(colors['N']))
        axes[2].set_xlabel("Time (h)")
        axes[2].set_ylabel("Concentration (g/L)")
        axes[2].legend()
        axes[2].grid(True)

        axes[3].scatter(t_exp_v, ph_exp_v, marker='o', label="pH exp validation",color=colors['pH'])
        axes[3].plot(time_stamps_sim, original_sol['pH'], '-', label="pH original",color=colors['pH'])
        axes[3].plot(time_stamps_sim, new_sol['pH'], '--', label="pH new",color=darken_color(colors['pH']))
        axes[3].set_xlabel("Time (h)")
        axes[3].set_ylabel("Concentration (g/L)")
        axes[3].legend()
        axes[3].grid(True)

        fig.suptitle(f'Model Prediction fitting {params_list} vs. Validation Data', fontsize=18)
        plt.tight_layout()
        plt.show()

def residuals(y_val, y_sim):
    res = y_val - y_sim
    return res

def rmse(res):
        return np.sqrt(np.mean(res**2))

def mape(y_val, res):
    return np.mean(np.abs(res/ y_val)) * 100

def nrmse(y_val_range, res): #range based
    return rmse(res) / y_val_range

def residuals_squared_sum(res):
    return np.sum(res**2)

def aic_bic(res, y_exp, params_list):
    n = len(y_exp)
    k = len(params_list)  
    rss = residuals_squared_sum(res)
    aic = 2 * k + n * np.log(rss / n)
    bic= k * np.log(n) + n * np.log(rss / n)

    return [aic, bic]
