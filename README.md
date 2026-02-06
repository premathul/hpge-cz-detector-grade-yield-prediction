HPGe Crystal Growth — ML Modeling Repository
Overview

This repository contains data preparation, machine learning, and analysis workflows developed for studying and predicting detector-grade yield in High-Purity Germanium (HPGe) crystal growth. All modeling, preprocessing, and training were carried out on the USD Lawrence Supercomputing Cluster under iterative experimental development.

Data Description

The file crystalgrowth_rawdata.xlsx contains the original raw experimental crystal growth logs. The file Data_exp.xlsx represents the initial structured dataset used during early preprocessing and feature preparation. The file Inital_data.csv is the processed version of the initial dataset and was used for model training and testing.

The file Updated_Data_with_Exponential_Format.csv contains impurity values expressed in exponential form, which were tested to evaluate sensitivity and model behavior. The file _Data_with_inital_Exponential_Format.csv corresponds to an earlier version of this exponential representation used during initial experimentation.

All experimental data were originally stored in .xlsx format and were manually converted into .csv format for compatibility with the training pipelines.

Modeling and Analysis

The script hpge_bilstm_attention_inital.py implements the initial Bidirectional LSTM with Attention model. This script was designed for early prediction experiments and includes a basic architecture with preliminary feature handling. The script is periodically updated as the dataset structure evolves and new experimental data are added.

The script cross_validation.py contains the cross-validation training pipeline. It is adapted to the dataset structure and feature configuration and is updated regularly to incorporate newer data inputs and feature modifications.

The directory shap_analysis_on_model contains SHAP-based interpretability analysis used to understand feature influence in trained LSTM models.

Computing Environment

All experiments and model training were executed on the USD Lawrence Supercomputing Cluster. The scripts support both CPU execution and optional GPU acceleration. GPU usage can be enabled or disabled within the training scripts depending on availability.

Software

The scripts require Python and TensorFlow, along with standard scientific libraries such as NumPy, Pandas, and Scikit-learn. 

Data and Script Evolution

This repository reflects an iterative experimental workflow. The dataset has expanded over time, and scripts have been updated to remain compatible with newer data formats and feature structures. Some scripts represent exploratory modeling stages and intermediate research attempts. When working with updated data, the latest preprocessing pipeline should always be used. 
Note: The results presented in this repository were obtained after extensive trial-and-error experimentation, including running the scripts multiple times under different configurations to ensure stability and consistency. The scripts are continuously updated to accommodate newer datasets and test runs. As the data structure evolves, debugging, feature additions, and other modifications can be implemented accordingly based on dataset usage and format. Crystal growth is a complex and challenging physical process, and data segregation is inherently difficult. The limited availability of large detector-grade datasets can sometimes affect model predictability. However, with ongoing data expansion and model refinement, future versions of the framework are expected to achieve significantly improved predictive performance. The model exhibits strong predictive accuracy in the dominant low-yield regime and larger error in rare high-yield crystals due to severe data imbalance and absence of output-impurity feature.

Impurity Representation

Different impurity modeling approaches were explored during experimentation. Some datasets use raw impurity concentration values, while others use exponential-transformed representations. Both approaches are preserved for comparison and analysis.

Execution

Training scripts can run in CPU mode or GPU mode. CPU execution is the default, while GPU execution is recommended for larger training runs. Configuration is controlled within the scripts.

