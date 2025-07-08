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
├── Analysis_functions.py        # Parameter analysis and validation functions
├── Aux_Functions.py             # Utility functions: statistics, plotting, sensitivity, FIM, etc.
├── Dae_System_run.py            # Main entry point for running simulations and calibrations
├── Dae_Systems_Simulations.py   # DAE system definition and simulation routines
├── Experimental_data.xlsx       # Experimental data for calibration/validation
├── Main.ipynb                   # Interactive notebook for local execution
├── Main_Collab.ipynb             # Google Collab notebook to reproduce the full pipeline
├── System_info.py               # System configuration, parameters, and initial conditions
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Main Features

- **DAE System Simulation:** Simulate bioprocess dynamics using symbolic computation (CasADi).
- **Parameter Estimation:** Calibrate model parameters using Particle Swarm Optimization (PSO) or other optimizers.
- **Sensitivity Analysis:** Quantify the impact of parameters on model outputs.
- **FIM Analysis:** Compute parameter identifiability and correlations.
- **Visualization:** Plot simulation results vs. experimental data for validation.
- **Statistical Metrics:** Compute RMSE, MAPE, AIC/BIC, and more.

## Installation

1. Clone the repository:

   ```bash
   git clone <repo_url>
   cd <repo_folder>
   ```

2. Install dependencies (using a virtual environment is recommended):

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Interactive Notebook (Local)

Open `Main.ipynb` for a step-by-step example of model simulation, calibration, and analysis.

### 2. Google Collab

Open `Main_Collab.ipynb` in Google Collab. This notebook will:

1. Clone the repository
2. Install all required packages
3. Configure imports
4. Run the full calibration and analysis pipeline in the cloud

### 3. Script Execution

You can also invoke the pipeline directly from Python scripts:

```python
from Dae_System_run import RUN_MAIN

# Example parameters (customize as needed)
iteration = 100
condition = "default"
perturbation = 0.01
correlation_threshold = 0.85
params_list = ["k_C", "k_N", ...]  # Parameters to calibrate
lb = [0.01, 0.001, ...]                # Lower bounds
ub = [0.2, 0.1, ...]                   # Upper bounds

RUN_MAIN(iteration, condition, perturbation, correlation_threshold, params_list, lb, ub)
```

## File Descriptions

- **System\_info.py**\
  Defines system parameters, constants, variable names, and initial conditions for simulations and experiments.

- **Dae\_Systems\_Simulations.py**\
  Contains the DAE system definitions and simulation routines (`simulate_model`, `simulate_model_calibrating`).

- **Dae\_System\_run.py**\
  Main orchestration script. Handles parameter estimation, simulation runs, and optimization workflows.

- **Aux\_Functions.py**\
  Utility routines for statistical metrics, sensitivity and FIM analysis, residuals, t-value computations, and plotting.

- **Analysis\_functions.py**\
  Additional functions for parameter analysis and model validation.

- **Main.ipynb**\
  Jupyter notebook for interactive exploration and local execution of the calibration pipeline.

- **Main\_Collab.ipynb**\
  Google Collab notebook to run the entire pipeline in a cloud environment, with automated setup steps.

- **Experimental\_data.xlsx**\
  Experimental measurements used for model calibration and validation.

- **requirements.txt**\
  Lists all Python dependencies required to run the code.

## Dependencies

The `requirements.txt` should include:

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

**Notes:**

- Update parameter lists, bounds, and configurations in your scripts as needed for your specific system.
- Experimental data origin and publication details will be added soon.
