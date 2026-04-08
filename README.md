# Numerical Methods for Interpreting Small Punch Test Results for the Assessment and Qualification of Nuclear Fusion Reactor Materials: A Data-Driven Approach Using Finite Element Modelling and Machine Learning Surrogate Modelling

Code repository for the DPhil project

Saleem Lubbad
The University of Oxford
2025

## Overview

This repository contains the computational framework developed for the analysis and modelling of the Small Punch Test (SPT), including finite element Abaqus simulations, data extraction, inverse analysis, and machine learning surrogate modelling.

The workflow integrates Abaqus-based finite element modelling with data-driven techniques to enable both forward simulation and inverse identification of material behaviour from SPT load–deflection responses.

The repository is structured around five main components:

### Finite Element Utilities (FE_utility)
Provides core functions for automating Abaqus simulations, including model setup, job submission, and results extraction.
### Load–Deflection Curve Analysis (LDC_Analysis)
Extracts physically meaningful features (e.g. slopes, inflexion points, characteristic forces) from SPT Load-Deflection Curve in accordance with relevant standards (EN 10371 and ASTM E3205-20).
### Sensitivity Analysis (SPT_sensitivity_analyses)
Evaluates the influence of modelling parameters such as friction and mesh density on SPT responses, supporting model robustness and validation.
### Inverse Analysis (SPT_inverse_analysis)
Implements optimisation-based methods to identify material parameters by matching simulation and experimental SPT responses through automated finite element model updating (FEMU).
### Surrogate Modelling (GPR_surrogate_model_training)
Develops Gaussian Process Regression (GPR) models to capture the relationship between input parameters and SPT responses

## Thesis Reference

If you use or refer to this code, please cite the thesis and the archived code version.

**Thesis:**  
Saleem Lubbad, Numerical Methods for Interpreting Small Punch Test Results for the Assessment and Qualification of Nuclear Fusion Reactor Materials: A Data-Driven Approach Using Finite Element Modelling and Machine Learning Surrogate Modelling, The University of Oxford, 2025.

**Archived code version:**  
Zenodo DOI: 

## Repository Structure

```text
├── FE_utility/
│   └── selected_FE_utility_functions.py
│
├── GPR_surrogate_model_training/
│   └── GPR_Training.py
│
├── LDC_Analysis/
│   └── analyse_LDC.py
│
├── SPT_inverse_analysis/
│   └── SPT_inverse_analysis.py
│
├── SPT_sensitivity_analyses/
│   └── SPT_sensitivity_functions.py
│
└── README.md
