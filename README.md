HPGe Crystal Growth — ML Modeling Repository
Overview

This repository contains data preparation, machine learning, and analysis workflows developed for studying and predicting detector-grade yield in High-Purity Germanium (HPGe) crystal growth. The primary objective is to learn correlations between crystal growth conditions, impurity evolution, and the final detector-grade usable portion of grown crystals.

All modeling, preprocessing, and training were carried out on the USD Lawrence Supercomputing Cluster under an iterative experimental research workflow closely coupled with real crystal growth data. The repository reflects a research-driven development process rather than a fixed production software package.

The codebase is actively maintained and periodically updated as new experimental data, additional physical features, and improved preprocessing strategies become available. The configuration used to generate the results reported in the associated manuscript corresponds to a fixed dataset and script state. Later updates are intended to improve robustness, extend functionality, and support future data integration; therefore, minor variations in numerical performance metrics may occur when running newer versions of the code.

The broader long-term goal of this work is to enable data-guided optimization of HPGe crystal growth and eventually integrate machine learning with physically grounded impurity segregation and crystal growth modeling.

Data Description

The repository includes multiple stages of dataset evolution reflecting progressive refinement and restructuring of experimental data.

crystalgrowth_rawdata.xlsx
Contains the original raw experimental crystal growth logs collected during crystal growth runs. These logs include time-resolved process parameters, impurity information, and growth conditions prior to preprocessing and structuring.

Data_exp.xlsx
Represents an early structured dataset produced during the initial stages of data cleaning and feature organization. This file was used for early preprocessing experimentation and feature exploration.

Inital_data.csv
Processed dataset used for model training and validation. This file represents a cleaned and structured version of experimental data suitable for machine learning pipelines.

Updated_Data_with_Exponential_Format.csv
Contains impurity concentrations expressed in exponential form. This transformation was explored to evaluate sensitivity of models to impurity representation and numerical scaling behavior.

_Data_with_inital_Exponential_Format.csv
Earlier experimental version of the exponential impurity representation used during preliminary modeling stages.

All experimental data were originally stored in .xlsx format and later converted manually into .csv format for compatibility with machine learning and numerical pipelines.

The dataset represents a small, high-cost experimental regime, where each crystal corresponds to a full physical growth experiment. As a result, dataset size is inherently limited, and careful preprocessing, feature handling, and validation strategies are required.

Computing Environment

All model training, preprocessing, and analysis were executed on the USD Lawrence Supercomputing Cluster. The workflows are designed to support:

CPU-based training (default)
Optional GPU acceleration for neural network training
Large-scale experimental data handling
Iterative experimentation and model refinement
GPU usage can be enabled or disabled within the training scripts depending on hardware availability.
Some extended workflow scripts and full-scale pipelines remain within the HPC environment and are not fully included in this public repository.

Software Requirements

Core dependencies:

Python
TensorFlow / Keras
NumPy
Pandas
Scikit-learn
Optional tools (depending on module usage):
SHAP (model interpretability)
Matplotlib / visualization libraries
Additional scientific Python packages as required by analysis scripts
Environment configuration may vary depending on HPC setup.

Data and Script Evolution

This repository reflects an iterative experimental research workflow rather than a static finalized software release. Over time:

The dataset has expanded and undergone restructuring.
Feature definitions have evolved as physical understanding improved.
Scripts have been updated to maintain compatibility with new data formats.
Multiple exploratory modeling approaches were tested and refined.
Some scripts included here represent intermediate modeling stages and exploratory research attempts. When working with updated datasets, the latest preprocessing pipeline should always be used to ensure consistency.

The results presented in this repository were obtained after extensive trial-and-error experimentation, including repeated training under varying configurations to ensure stability and consistency of behavior. Scripts continue to evolve as new data become available, and debugging, feature engineering, and structural modifications are applied as needed based on dataset structure.

HPGe crystal growth is a complex physical process involving impurity transport, segregation, thermodynamic gradients, and material-dependent behavior. As a result:

Data segregation is non-trivial.
Dataset size is inherently limited.
Predictive modeling in rare high-yield regimes remains challenging.
Current models demonstrate strong predictive accuracy in the dominant low-yield regime, while larger uncertainty appears in rare high-yield crystals due to severe data imbalance and lack of certain physically informative features (e.g., output impurity evolution).

Ongoing dataset expansion and model refinement are expected to improve predictive capability in future work.

Impurity Representation
Multiple impurity modeling approaches were explored during experimentation:
Raw impurity concentration representation
Exponential-scaled impurity representation

These representations were evaluated to understand sensitivity of models to numerical scaling and impurity distribution behavior. Both datasets are preserved for comparison and analysis.

Execution

Training scripts can be executed in:

CPU mode (default)

GPU mode (recommended for large neural network training)

Execution mode and configuration parameters are controlled within individual scripts. Users may modify training parameters, batch size, and GPU settings depending on hardware availability and dataset size.

Script Availability and Notes

If any scripts appear missing, incomplete, or inconsistent, please contact:

athul.prem@coyotes.usd.edu

Some auxiliary workflow scripts and full-scale training pipelines are maintained within the USD HPC environment and may not be included in this public repository. In several cases, smaller scripts included here are extracted from larger HPC workflow files and adapted for reduced or curated datasets.

Most scripts in this repository are configured for relatively small datasets. Since raw experimental data are provided, users may reconstruct extended preprocessing pipelines and retrain models using larger combined datasets to achieve improved predictive performance.

Molecular Dynamics and Physical Modeling

Exploratory molecular dynamics simulations related to crystal growth physics were performed using LAMMPS, with visualization carried out in VMD. At present, these simulations represent toy-model physical prototypes, as fully validated interatomic potentials for impurity–germanium systems are not yet available. Consequently, these MD simulations are not intended to represent fully predictive physical models but rather exploratory physical investigations.

Future work will focus on:

Development of physically accurate impurity–germanium potentials
Improved modeling of impurity segregation physics
Stronger coupling between physical modeling and machine learning
Physics-informed predictive frameworks
Reproducibility Note

Minor variations in numerical results may occur due to:
Dataset evolution and restructuring
Random initialization in neural network training
Hardware differences (CPU vs GPU execution)
Continuous script refinement and experimental updates

This repository reflects an active research workflow, and numerical outputs should be interpreted within that context.


