# DAE-Based Bioprocess Modeling and Analysis

This repository contains a suite of Python scripts and utilities for simulating, calibrating, and analyzing a dynamic bioprocess system using Differential-Algebraic Equations (DAEs). The codebase is designed for parameter estimation, sensitivity analysis, and visualization of simulation results, leveraging experimental data for model validation.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Main Features](#main-features)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

The repository models the growth dynamics of a bioprocess (e.g., microbial or cell culture) using DAEs. It provides tools for:
- Simulating system dynamics given initial conditions and parameters
- Calibrating model parameters against experimental data
- Performing sensitivity and Fisher Information Matrix (FIM) analyses
- Visualizing and comparing simulation and experimental results

## Repository Structure

```
.
├── Analysis_funcions.py         # Parameter analysis and validation functions
├── Aux_functions.py             # Utility functions: statistics, plotting, sensitivity, FIM, etc.
├── Dae_System_run.py            # Main entry point for running simulations and calibrations
├── Dae_Systems_Simulations.py   # DAE system definition and simulation routines
├── Experimental_data.xlsx       # Experimental data for calibration/validation
├── Main.ipynb                   # Example notebook for interactive use
├── System_info.py               # System configuration, parameters, and initial conditions
```

## Main Features

- **DAE System Simulation:** Simulates bioprocess dynamics using symbolic computation (CasADi).
- **Parameter Estimation:** Calibrates model parameters using Particle Swarm Optimization (PSO) or other optimizers.
- **Sensitivity Analysis:** Quantifies the impact of parameters on model outputs.
- **FIM Analysis:** Computes parameter identifiability and correlations.
- **Visualization:** Plots simulation results vs. experimental data for validation.
- **Statistical Metrics:** Computes RMSE, MAPE, AIC/BIC, and more.

## Installation

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <repo_folder>
   ```

2. Install dependencies (recommend using a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

   **Main dependencies:**
   - numpy
   - pandas
   - matplotlib
   - seaborn
   - scipy
   - statsmodels
   - casadi
   - mealpy

## Usage

### 1. Interactive Notebook

Open `Main.ipynb` for a step-by-step example of model simulation, calibration, and analysis.

### 2. Script Execution

The main entry point for running simulations and calibrations is `Dae_System_run.py`. Example usage:

```python
from Dae_System_run import RUN_MAIN

# Example parameters (customize as needed)
iteration = 100
condition = "default"
perturbation = 0.01
correlation_threshold = 0.85
params_list = ["k_C", "k_N", ...]  # Parameters to calibrate
lb = [0.01, 0.001, ...]            # Lower bounds
ub = [0.2, 0.1, ...]               # Upper bounds

RUN_MAIN(iteration, condition, perturbation, correlation_threshold, params_list, lb, ub)
```

## File Descriptions

- **System_info.py**  
  Defines system parameters, constants, variable names, and initial conditions for simulations and experiments.

- **Dae_Systems_Simulations.py**  
  Contains the DAE system definition (`DAE_system`, `DAE_system_calibrating`) and simulation routines (`simulate_model`, `simulate_model_calibrating`).

- **Dae_System_run.py**  
  Main orchestration script. Handles parameter estimation, runs simulations, and manages optimization routines.

- **Aux_functions.py**  
  Utility functions for:
  - Statistical metrics (RMSE, MAPE, AIC/BIC)
  - Sensitivity and FIM analysis
  - Residuals and t-value computations
  - Plotting and visualization

- **Analysis_funcions.py**  
  Functions for parameter analysis and model validation.

- **Experimental_data.xlsx**  
  Contains experimental data for model calibration and validation.

- **Main.ipynb**  
  Jupyter notebook for interactive exploration, running simulations, and visualizing results.

## Dependencies

A sample `requirements.txt` (please update according to your environment):

```
numpy
pandas
matplotlib
seaborn
scipy
statsmodels
casadi
mealpy
openpyxl
```

---

**Note:**  
- Adjust parameter lists, bounds, and conditions in your scripts as needed for your specific system.
- For further customization or advanced analysis, refer to the docstrings and comments within each script.

- Codes still being updated
- Paper and origin of experimental data to be published shortly

---
