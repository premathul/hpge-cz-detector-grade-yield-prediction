
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

def load_and_preprocess_data_alternative(file_path='strict_full_converted_data.csv'):
    print("\n" + "="*60)
    print("PREPROCESSING")
    print("="*60)

    df = pd.read_csv(file_path)
    df = df.sort_values(['SheetName', 'Time (Sec)']).reset_index(drop=True)

    # Paper features
    feature_cols = [
        'Power(W)',
        'Growth Rate (gm/sec)',
        'No. of net impurity atoms added',
        'Number of net impurity of previous crystal added'
    ]

    available_features = [f for f in feature_cols if f in df.columns]
    print(f"Using features: {available_features}")

    # Apply log1p transformation to impurity columns
    impurity_cols = [
        'No. of net impurity atoms added',
        'Number of net impurity of previous crystal added'
    ]

    for col in impurity_cols:
        if col in df.columns:
            # Use absolute value for transformation
            df[col] = np.log1p(np.abs(df[col]) + 1e-10)

    sequences = []
    targets = []
    crystal_names = []

    # Create sequences with MEDIAN target (more robust than max)
    print(f"\nCreating sequences for each crystal (using MEDIAN target)...")
    for crystal_name, group in df.groupby('SheetName'):
        X = group[available_features].astype(np.float32).values

        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            continue

        #Make sure the data aligns with the workflow
        y = group['Detector grade portion (%)'].astype(np.float32).median()

        # More permissive filter initially
        if len(X) >= 3:
            sequences.append(X)
            targets.append(y)
            crystal_names.append(crystal_name)

    print(f"Initial: {len(sequences)} crystals")

    
    y_array = np.array(targets)

    
    Q1 = np.percentile(y_array, 25)
    Q3 = np.percentile(y_array, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"\nTarget statistics before filtering:")
    print(f"  Min: {y_array.min():.2f}%")
    print(f"  Q1: {Q1:.2f}%")
    print(f"  Median: {np.median(y_array):.2f}%")
    print(f"  Q3: {Q3:.2f}%")
    print(f"  Max: {y_array.max():.2f}%")
    print(f"  IQR: {IQR:.2f}%")
    print(f"  Outlier bounds: [{lower_bound:.2f}%, {upper_bound:.2f}%]")

    # Filter outliers
    filtered_sequences = []
    filtered_targets = []
    filtered_crystals = []

    for i, (seq, target, name) in enumerate(zip(sequences, targets, crystal_names)):
        if lower_bound <= target <= upper_bound:
            filtered_sequences.append(seq)
            filtered_targets.append(target)
            filtered_crystals.append(name)
        else:
            print(f"  Excluded {name}: target={target:.2f}% (outlier)")

    print(f"\nAfter outlier filtering: {len(filtered_sequences)} crystals")
    print(f"Target range: {min(filtered_targets):.1f}% to {max(filtered_targets):.1f}%")
    print(f"Target mean: {np.mean(filtered_targets):.2f}% ± {np.std(filtered_targets):.2f}%")

    
    X_tabular = []
    for seq in filtered_sequences:
        sample_features = []
        for feature_idx in range(seq.shape[1]):
            feature_vals = seq[:, feature_idx]
            sample_features.extend([
                np.mean(feature_vals),
                np.std(feature_vals),
                np.min(feature_vals),
                np.max(feature_vals)
            ])
        X_tabular.append(sample_features)

    return np.array(X_tabular), np.array(filtered_targets), filtered_crystals, available_features

def run_regression_cv_with_ridge(X, y, n_folds=5, n_seeds=5, alpha=1.0):
    """Run cross-validation with Ridge regression"""
    all_mae = []
    all_rmse = []

    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}")

        np.random.seed(seed)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        seed_mae = []
        seed_rmse = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            # Split data
            X_train_val, X_test = X[train_idx], X[test_idx]
            y_train_val, y_test = y[train_idx], y[test_idx]

            # Further split for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.2, random_state=seed
            )

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            # Train Ridge regression (regularized)
            model = Ridge(alpha=alpha, random_state=seed)
            model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred = np.clip(y_pred, 0, 100)

            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            seed_mae.append(mae)
            seed_rmse.append(rmse)

            print(f"  Fold {fold}: MAE = {mae:.3f}%, RMSE = {rmse:.3f}%")

        all_mae.extend(seed_mae)
        all_rmse.extend(seed_rmse)

        print(f"  Seed average: MAE = {np.mean(seed_mae):.3f}%, RMSE = {np.mean(seed_rmse):.3f}%")

    mean_mae = np.mean(all_mae)
    std_mae = np.std(all_mae)
    mean_rmse = np.mean(all_rmse)
    std_rmse = np.std(all_rmse)

    return mean_mae, std_mae, mean_rmse, std_rmse

def main():
    print("\n" + "="*70)
    print("LINEAR REGRESSION FOR HPGe YIELD PREDICTION")
    print(" Median targets + Ridge regression")
    print("="*70)

    # Load with alternative preprocessing
    X, y, crystal_names, feature_names = load_and_preprocess_data_alternative()

    if len(crystal_names) < 10:
        print("\nERROR: Too few crystals after filtering!")
        return

    print(f"\nFinal dataset: {len(crystal_names)} crystals")
    print(f"X shape: {X.shape}")

    # Try different regression approaches
    print("\n" + "="*60)
    print("1. STANDARD LINEAR REGRESSION")
    print("="*60)

    # Simple linear regression
    mean_mae_lr, std_mae_lr, mean_rmse_lr, std_rmse_lr = run_regression_cv_with_ridge(
        X, y, n_folds=5, n_seeds=5, alpha=0.0  # alpha=0 gives standard linear regression
    )

    print("\n" + "="*60)
    print("2. RIDGE REGRESSION (alpha=1.0)")
    print("="*60)

    # Ridge regression
    mean_mae_ridge, std_mae_ridge, mean_rmse_ridge, std_rmse_ridge = run_regression_cv_with_ridge(
        X, y, n_folds=5, n_seeds=5, alpha=1.0
    )


    print("\nThis Implementation:")
    print(f"  Linear Regression:  MAE = {mean_mae_lr:.3f}% ± {std_mae_lr:.3f}%")
    print(f"                      RMSE = {mean_rmse_lr:.3f}% ± {std_rmse_lr:.3f}%")
    print(f"  Ridge Regression:   MAE = {mean_mae_ridge:.3f}% ± {std_mae_ridge:.3f}%")
    print(f"                      RMSE = {mean_rmse_ridge:.3f}% ± {std_rmse_ridge:.3f}%")

  
    best_mae = min(mean_mae_lr, mean_mae_ridge)
    best_rmse = min(mean_rmse_lr, mean_rmse_ridge)

    print(f"\nBest MAE achieved: {best_mae:.3f}%")
    print(f"Best RMSE achieved: {best_rmse:.3f}%")

if __name__ == "__main__":
    main()


