HPGe Crystal Growth — ML Modeling Repository
Overview

This repository contains data preparation, machine learning, and analysis workflows developed for studying and predicting detector-grade yield in High-Purity Germanium (HPGe) crystal growth. All modeling, preprocessing, and training were carried out on the USD Lawrence Supercomputing Cluster under iterative experimental development.

The codebase is actively maintained and periodically updated as new experimental data, additional features, and improved preprocessing strategies become available. The version used to generate the results reported in the associated manuscript corresponds to a fixed experimental configuration. Subsequent updates are intended to improve robustness, extend functionality, and support future data integration; therefore, minor variations in performance metrics may occur when running newer versions of the code.

Data Description

crystalgrowth_rawdata.xlsx
Original raw experimental crystal growth logs.

Data_exp.xlsx
Initial structured dataset used during early preprocessing and feature preparation.

Inital_data.csv
Processed dataset used for model training and testing.

Updated_Data_with_Exponential_Format.csv
Dataset where impurity values are expressed in exponential representation to evaluate sensitivity and model behavior.

_Data_with_inital_Exponential_Format.csv
Earlier experimental version of the exponential impurity representation.

All experimental data were originally stored in .xlsx format and manually converted into .csv for compatibility with the machine learning pipelines.

Computing Environment

All experiments and model training were executed on the USD Lawrence Supercomputing Cluster. Scripts support both CPU execution and optional GPU acceleration. GPU usage can be enabled or disabled within the training scripts depending on availability.

Software Requirements
Python
TensorFlow / Keras
NumPy
Pandas
Scikit-learn
Additional optional libraries may be required depending on analysis modules.

Data and Script Evolution

This repository reflects an iterative experimental workflow. The dataset has expanded over time, and scripts have been updated to remain compatible with newer data formats and feature structures. Some scripts represent exploratory modeling stages and intermediate research attempts. When working with updated data, the latest preprocessing pipeline should always be used.
 The results presented in this repository were obtained after extensive trial-and-error experimentation, including running scripts multiple times under different configurations to ensure stability and consistency. Scripts are continuously updated to accommodate newer datasets and test runs. As the data structure evolves, debugging, feature additions, and other modifications can be implemented accordingly based on dataset usage and format. Crystal growth is a complex physical process, and data segregation is inherently challenging. Limited availability of large detector-grade datasets can affect predictability in certain regimes. However, with continued data expansion and model refinement, future versions are expected to achieve improved predictive performance. Current models show strong accuracy in the dominant low-yield regime and larger error in rare high-yield crystals due to dataset imbalance and absence of output-impurity features.

Impurity Representation

Multiple impurity modeling approaches were explored. Some datasets use raw impurity concentration values, while others use exponential-transformed representations. Both are preserved for comparison and sensitivity analysis.

Execution

Training scripts can run in CPU or GPU mode. CPU execution is default; GPU is recommended for large training runs. Configuration is controlled inside the scripts.

Script Availability and Notes

If any scripts appear missing or incomplete, please contact athul.prem@coyotes.usd.edu
. Some auxiliary or large-scale workflow scripts are maintained within the USD HPC environment and may not be included in this public repository. In certain cases, smaller scripts provided here are extracted from larger HPC pipeline files and adapted for limited datasets. Most scripts in this repository are configured for relatively small curated datasets. Since raw experimental data are included, users may reconstruct extended preprocessing pipelines and retrain models to improve predictive performance as dataset size increases.

Molecular Dynamics and Physical Modeling

Exploratory molecular dynamics simulations related to crystal growth physics were performed using LAMMPS, with visualization carried out in VMD. These simulations currently represent toy-model physical prototypes, as fully validated interatomic potentials for impurity–germanium systems are not yet available.Development of accurate potentials and physically realistic MD models is planned for future work. This will enable stronger coupling between data-driven prediction and underlying crystal growth physics.

*Reproducibility Note

Minor variations in numerical results may occur due to:

Dataset evolution

Random initialization of neural network training

Hardware differences (CPU vs GPU)

Ongoing script refinement


