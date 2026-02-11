#!/usr/bin/env python3
"""
Native XGBoost CV baseline for HPGe detector-grade yield prediction.

Protocol:
- Load CZ logs
- Group by SheetName -> per-crystal sequence
- Convert to fixed-length tabular features (mean/std/min/max per feature)
- Outer 5-fold CV
- On each fold: internal train/val split
- Manual param grid search with xgb.train + early stopping on val MAE
- Retrain using best params and best num_boost_round
- Evaluate on outer test fold
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import KFold, train_test_split, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)


def load_and_preprocess_data(file_path: str = "strict_full_converted_data.csv"):
    df = pd.read_csv(file_path)
    df = df.sort_values(["SheetName", "Time (Sec)"]).reset_index(drop=True)

    feature_cols = [
        "Power(W)",
        "Growth Rate (gm/sec)",
        "No. of net impurity atoms added",
        "Number of net impurity of previous crystal added",
    ]
    target_col = "Detector grade portion (%)"

    
    impurity_cols = [
        "No. of net impurity atoms added",
        "Number of net impurity of previous crystal added",
    ]
    for col in impurity_cols:
        if col in df.columns:
            df[col] = np.log1p(np.abs(df[col]).astype(np.float64) + 1e-10)

    sequences = []
    targets = []

    for crystal_name, group in df.groupby("SheetName"):
        if not all(c in group.columns for c in feature_cols + [target_col]):
            continue

        X_seq = group[feature_cols].astype(np.float32).values
        if np.any(np.isnan(X_seq)) or np.any(np.isinf(X_seq)):
            continue

        y = float(group[target_col].astype(np.float32).max())  # OK if constant per crystal
        if 0.0 <= y <= 100.0 and len(X_seq) >= 3:
            sequences.append(X_seq)
            targets.append(y)

    if len(sequences) == 0:
        raise RuntimeError("No valid crystal sequences found. Check CSV columns and data.")

    
    X_tabular = []
    for seq in sequences:
        feats = []
        for j in range(seq.shape[1]):
            v = seq[:, j]
            feats.extend([np.mean(v), np.std(v), np.min(v), np.max(v)])
        X_tabular.append(feats)

    X_tabular = np.asarray(X_tabular, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    return X_tabular, y


def train_one_model(X_train, y_train, X_val, y_val, params, num_boost_round=5000, es_rounds=50):
    """
    Train with native xgb.train + early stopping on validation MAE.
    Returns: booster, best_val_mae, best_iteration (1-based boosting rounds)
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    watchlist = [(dtrain, "train"), (dval, "val")]

    
    train_params = dict(params)
    train_params["objective"] = "reg:squarederror"
    train_params["eval_metric"] = "mae"
    train_params["seed"] = SEED
    train_params["nthread"] = -1

    booster = xgb.train(
        params=train_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=watchlist,
        early_stopping_rounds=es_rounds,
        verbose_eval=False,
    )

    
    best_score = float(booster.best_score) if booster.best_score is not None else np.inf
    best_iter = int(booster.best_iteration) + 1  # best_iteration is 0-based

    return booster, best_score, best_iter


def run_xgboost_cv_native(X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    mae_scores = []
    rmse_scores = []

    # Grid 
    param_grid = {
        "max_depth": [3, 6, 9],
        "eta": [0.01, 0.1, 0.3],          # 'eta' is learning_rate in native API
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [1.0],        # keep fixed unless you want to tune
        "min_child_weight": [1.0],        # keep fixed unless you want to tune
    }

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train_val, X_test = X[train_idx], X[test_idx]
        y_train_val, y_test = y[train_idx], y[test_idx]

        # internal split for tuning/early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=SEED
        )

        best_params = None
        best_val_mae = np.inf
        best_boost_round = None

        # Manual grid search
        for params in ParameterGrid(param_grid):
            booster, val_mae, best_iter = train_one_model(
                X_train, y_train, X_val, y_val,
                params=params,
                num_boost_round=5000,
                es_rounds=50
            )
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_params = params
                best_boost_round = best_iter

        
        dtrainval = xgb.DMatrix(np.vstack([X_train, X_val]), label=np.hstack([y_train, y_val]))
        dtest = xgb.DMatrix(X_test)

        final_params = dict(best_params)
        final_params["objective"] = "reg:squarederror"
        final_params["eval_metric"] = "mae"
        final_params["seed"] = SEED
        final_params["nthread"] = -1

        final_booster = xgb.train(
            params=final_params,
            dtrain=dtrainval,
            num_boost_round=best_boost_round,
            evals=[(dtrainval, "train")],
            verbose_eval=False,
        )

        y_pred = final_booster.predict(dtest)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        mae_scores.append(mae)
        rmse_scores.append(rmse)

        print(f"Fold {fold}: MAE = {mae:.3f}%, RMSE = {rmse:.3f}%")
        print(f"  Best params: {best_params}, best_boost_round={best_boost_round}, val_MAE={best_val_mae:.3f}%")

    mean_mae = float(np.mean(mae_scores))
    std_mae = float(np.std(mae_scores))
    mean_rmse = float(np.mean(rmse_scores))
    std_rmse = float(np.std(rmse_scores))

    return mean_mae, std_mae, mean_rmse, std_rmse


def main():
    print("XGBoost (native) for HPGe Yield Prediction")
    print("=" * 50)
    print(f"XGBoost version: {getattr(xgb, '__version__', 'unknown')}")

    X, y = load_and_preprocess_data("strict_full_converted_data.csv")
    print(f"Dataset shape: {X.shape}")
    print(f"Target range: {y.min():.1f}% to {y.max():.1f}%")

    print("\nRunning 5-fold cross-validation (native xgb.train + early stopping)...")
    mean_mae, std_mae, mean_rmse, std_rmse = run_xgboost_cv_native(X, y, n_folds=5)

    print("\nResults:")
    print(f"Mean MAE : {mean_mae:.3f}% ± {std_mae:.3f}%")
    print(f"Mean RMSE: {mean_rmse:.3f}% ± {std_rmse:.3f}%")


if __name__ == "__main__":
    main()

