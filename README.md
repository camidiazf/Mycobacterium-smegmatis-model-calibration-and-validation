# DAE-Based Bioprocess Modeling and Analysis

## Overview

DAE-Based Bioprocess Modeling and Analysis is a comprehensive Python framework designed to automate the end-to-end workflow for differential–algebraic equation (DAE) models in bioprocess engineering. By centralizing all system definitions, experimental data, and solver settings in a single `System_info.py` file, you can:

- **Define & Simulate**: Quickly encode your bioprocess kinetics, mass balances, and algebraic restraints in `DAE_Systems_Simulations.py` and run forward integrations or parameter sweeps with CasADi’s efficient IDAS solver.  
- **Calibrate**: Specify any subset of model parameters to fit against experimental measurements, along with user-defined lower/upper bounds and perturbation schemes. The PSO optimizer in `RUN_functions.py` handles all the heavy lifting, iterating across combinations and recording every result.  
- **Validate & Analyze**: Automatically compare model predictions to validation data, compute goodness-of-fit metrics (AIC, BIC, RMSE, MAPE), perform statistical tests (Anderson–Darling, Durbin–Watson), assemble the Fisher Information Matrix (FIM), and derive sensitivity, identifiability, and t-value statistics.  
- **Report**: All numeric outputs—parameter estimates, residuals, sensitivity matrices, and more—are collated into a user-specified Excel file for easy downstream analysis, while built-in plotting routines generate time-course graphs, residual and Q–Q plots, heatmaps, and summary figures.  

Because the pipeline exhaustively explores parameter permutations, total runtime scales rapidly with the number of parameters and granularity of bounds. For large calibration campaigns, consider starting with coarser bounds or fewer parameters, then refining once promising regions are identified.

---

## Table of Contents

* [Features](#features)
* [Quick Start](#quick-start)
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Usage](#usage)

  * [Local Interactive (Jupyter)](#local-interactive-jupyter)
  * [Google Colab Notebook](#google-colab-notebook)
  * [Python Script API](#python-script-api)
* [File Descriptions](#file-descriptions)
* [Dependencies](#dependencies)

---

## Features

* **Automated DAE Simulation**
  Define your system in `Dae_Systems_Simulations.py` as well as the relevant data on `System_info.py`, and run forward integration or batch parameter sweeps with a single command.
* **Parameter Calibration & Validation**
  Fit model parameters to experimental data using evolutionary algorithms or local optimizers, then validate model performance and export statistical metrics.
* **Sensitivity Analysis & FIM**
  Compute local sensitivities, Fisher Information Matrices (FIM), confidence intervals, and parameter identifiability metrics.
* **Reporting & Visualization**
  Generate publication-ready plots (time courses, residuals, sensitivity heatmaps) and compile Excel summaries of all runs in `Excel_Results/`.
* **Interactive Notebooks**
  Step through each stage in `Main.ipynb` (local) or `Main_Collab.ipynb` (Google Colab) for hands-on exploration.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/camidiazf/DAE_System_Model_Calibration_and_Validation.git
cd DAE_System_Model_Calibration_and_Validation

# 2. (Optional) Create and activate a virtualenv
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Tip:** Use a virtual environment to isolate dependencies.

---

## Repository Structure

```
.
├── Analysis_functions.py                        # Post-processing: t-values, CIs, goodness-of-fit, residual stats
├── Aux_Functions.py                             # Helper utilities: plotting routines, statistical tools, FIM, I/O
├── Dae_Systems_Simulations.py                   # CasADi-based DAE definitions, solver setup, integration functions
├── Experimental_data.xlsx                       # Tabular experimental measurements for calibration/validation
├── Main.ipynb                                   # Jupyter notebook: step-by-step local pipeline walkthrough
├── Main_Collab.ipynb                            # Google Colab notebook: reproduce pipeline in the cloud
├── Mycobacterium_smegmatis_Calibration.ipynb   # Results for Model Calibrations
├── RUN_functions.py                             # Functions that run main processes and call other python files.
├── System_info.py                               # User configurations: parameter names, bounds, ICs, solver tolerances
├── requirements.txt                             # Python package requirements (pip-format)
└── README.md                                    # Project overview and instructions (this file)
```

---

## Installation if Running on Computer

Install all required packages via pip:

```bash
pip install -r requirements.txt
```

Packages include (but are not limited to):

* `casadi`
* `numpy`, `pandas`, `scipy`
* `matplotlib`, `seaborn`
* `statsmodels`, `mealpy`
* `openpyxl`

---

## Usage

### Local Interactive (Jupyter)

1. Launch Jupyter Lab/Notebook:

   ```bash
   jupyter lab
   ```
2. Open `Main.ipynb`.
3. Follow each cell to:

   * Configure your system in `System_info.py`
   * Run calibration & simulation
   * Visualize results and export to Excel

### Google Colab Notebook

1. Open `Main_Collab.ipynb` in [Google Colab](https://colab.research.google.com/).
2. (Optional) Mount your Google Drive for persistent storage.
3. Run all cells to reproduce the full pipeline without local setup.

---

## File and Function Overview

Below is a concise description of each Python module in the current version of the repo, along with the functions it contains.


### `Analysis_functions.py`  
High-level routines for validating and analyzing a calibrated model against experimental data.

- **`validation_analysis`**  
  Runs a full residual analysis on validation data: simulates the model, computes residual statistics (RMSE, AIC/BIC), performs Anderson–Darling and Durbin–Watson tests, and plots residual and Q–Q plots.

- **`parameter_analysis`**  
  Computes parameter sensitivity and identifiability: assembles the Fisher Information Matrix (FIM), extracts the correlation matrix and t-values, and returns them along with the sensitivity DataFrame.

- **`plotting_comparison`**  
  Plots time-series comparisons between experimental validation data and both the original and updated model simulations for each state variable.


### `Aux_Functions.py`  
Low-level utilities for cost evaluation, perturbation, and statistical computations used during calibration and analysis.

- **`define_cost_function`**  
  Builds the objective function for calibration by summing squared errors between simulated and experimental PE data.

- **`sim_plus_minus`**  
  Runs two simulations per parameter (± perturbation or Δ) and returns the “plus” and “minus” outputs for sensitivity/FIM.

- **`residuals_equations`**  
  Calculates residuals, RMSE, NRMSE, MAPE—and, if requested, AIC and BIC—from experimental vs. simulated data.

- **`compute_FIM`**  
  Constructs the Fisher Information Matrix via finite-difference Jacobians, then derives eigenvalues, condition number, correlation matrix, and t-values.

- **`compute_correlation_matrix`**  
  Inverts the FIM to yield parameter correlations, plots a heatmap, and prints highly correlated pairs above a threshold.

- **`compute_t_values`**  
  Given an FIM and calibrated parameters, computes standard errors and t-values for each parameter.

- **`compute_sensitivity`**  
  Performs local sensitivity analysis by perturbing each parameter, computes relative sensitivities for all states, and plots bar charts.

- **`format_number`**  
  Utility to format scalars or single-element lists to a fixed number of decimal places (returns `None` unchanged).


### `DAE_Systems_Simulations.py`  
Defines the model’s DAE and provides the driver that compiles and runs it via CasADi.

- **`DAE_system`**  
  Encodes the differential–algebraic equations of the bioprocess (growth, substrate consumption, pH algebraic constraint).

- **`DAE_system_calibrating`**  
  Variant of `DAE_system` that accepts an array of calibration parameters and overwrites the global parameter set before forming the DAE.

- **`simulate_model`**  
  Builds and invokes the CasADi `idas` integrator for either “normal” or “calibrating” modes, returns a Pandas DataFrame of all states plus derived variables (pH, H⁺, µ).


### `RUN_functions.py`  
The main orchestration layer: loops over parameter combinations, calls PSO, and aggregates all results into Excel.

- **`suppress_all_output`**  
  Context manager to silence PSO optimizer stdout/stderr during calibration.

- **`RUN_PARAMETERS_ITERATIONS`**  
  Top-level driver that runs each scenario through PSO, according to the numer of iterations set by the user, and writes results to an Excel file.

- **`RUN_INITIAL`**  
  Executes a single analysis pass using the original parameter set, formats and returns its results for inclusion as the “baseline” row.

- **`RUN_SCENARIO`**  
  Given a set of parameters, bounds, and iteration count, repeatedly calls `RUN_PSO_CALIBRATION`, collects all outputs, and triggers summary plotting.

- **`RUN_PSO_CALIBRATION`**  
  Defines and runs the Particle Swarm Optimization for one calibration iteration, updates parameters, runs analysis, and returns the results dictionary.

- **`RUN_ANALYSIS`**  
  Wraps `validation_analysis` and `parameter_analysis` into one call, ensuring both residual and t-value/correlation outputs are returned.

- **`RUN_SUMMARY_ANALYSIS`**  
  After all iterations for a given parameter combo, aggregates lists of sensitivities, residuals, and correlations, then plots overall mean±σ summaries.


### `System_info.py`  
Static configuration: experimental data import, parameter dictionaries, initial conditions, time grids, plotting colors, solver settings, and the single `system_info` dictionary that drives every simulation and analysis function.

_No standalone functions; this module only defines data structures and constants._  


---

## Dependencies

All dependencies are pinned in `requirements.txt`:

```
casadi
numpy
pandas
scipy
matplotlib
seaborn
statsmodels
mealpy
openpyxl
```

---

## Citations & References


- Apiyo, D., Mouton, J. M., Louw, C., Sampson, S. L., & Louw, T. M. (2022). Dynamic mathematical model development and validation of _in vitro_ _Mycobacterium smegmatis_ growth under nutrient- and pH-stress. *Journal of Theoretical Biology*, 532, 110921. https://doi.org/10.1016/j.jtbi.2021.110921 

- **de Witt Redlich, M., Díaz Figueroa, C. I., Lagos Silva, L. B., Sánchez Toledo, C. A., & Taunton Muzio, F. (2025).**  
  _Enhanced modeling of a Mycobacterium smegmatis batch cultivation._  
  Unpublished manuscript.

- New manuscript in progress.
---

## Acknowledgments

- **Experimental data** for calibration and validation were graciously provided by Dr. D. Apiyo (personal communication).  

---
## Notes
- Working on adding folders for images made, to save them for each iteration
- Working on adding more solver options
- Some function comments have to be updated
