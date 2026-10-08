# Ml_analysis environment snapshot

Exported from the existing Conda environment named exactly `Ml_analysis` on macOS Apple Silicon. The similarly named `ML_analysis` and `ml_analysis` environments are separate and were not included.

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
