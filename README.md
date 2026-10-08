# Ml_analysis environment snapshot

Exported from the existing Conda environment named exactly `Ml_analysis` on macOS Apple Silicon. 
## Files

- `requirements.txt`: Python packages currently reported by the environment's Python/pip, pinned to installed versions.
- `requirement.txt`: Same content, using the filename requested.
- `environment.yml`: Conda packages and pip packages reported by Conda, without the original machine's absolute prefix.
- `conda-explicit-osx-arm64.txt`: Exact Conda package URLs and builds for Apple Silicon macOS. This does not include pip packages.
- `conda-packages.json`: Conda's full package inventory, including channels and build strings.
- `pip-check.txt`: Existing package compatibility issues reported by `pip check`.
- `Ml_analysis_env.tar`: Full archive of the original environment directory, including installed binaries, libraries, package resources, and metadata. It is a backup of files, not a portable Conda installer.

## Recreate

For the closest practical recreation on Apple Silicon macOS, run:

```sh
conda env create -f environment.yml
conda activate Ml_analysis
```

`requirements.txt` can be used with pip, but pip alone does not install the Conda native libraries and tools in this environment. The environment already contains version conflicts; see `pip-check.txt`. Exports record the current state and do not establish that a fresh install will solve or run correctly.

The Conda environment directory contains installed software and package resources. It has no apparent project notebooks or data files outside installed package directories. Project data stored elsewhere cannot be identified from the environment name alone.

Versions used:
| Package | Version |
|---|---:|
| Python | 3\.9.15 |
| TensorFlow | 2\.10.0 |
| Keras | 2\.10.0 |
| TensorFlow Addons | 0\.23.0 |
| PyTorch (`torch`) | 2\.8.0 |
| torchvision | 0\.14.1a0 |
| torchaudio | 0\.13.1 |
| NumPy | 1\.26.4 |
| SciPy | 1\.13.0 |
| pandas | 2\.2.3 |
| scikit-learn | 1\.3.1 |
| XGBoost | 2\.1.1 |
| LightGBM | 4\.6.0 |
| CatBoost | 1\.2.10 |
| SHAP | 0\.46.0 |
| Matplotlib | 3\.8.4 |
| Seaborn | 0\.13.2 |
| Optuna | 4\.1.0 |
| TabPFN | 8\.0.7 |

