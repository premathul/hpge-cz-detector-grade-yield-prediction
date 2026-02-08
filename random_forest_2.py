"""
Random Forest for HPGe Detector-Grade Yield Prediction
Implements methodology from Section 4.1 of the paper.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

def load_and_preprocess_data(file_path='strict_full_converted_data.csv'):
    """Load and preprocess data as per paper methodology."""
    df = pd.read_csv(file_path)
    df = df.sort_values(['SheetName', 'Time (Sec)']).reset_index(drop=True)
    
    # Paper features
    feature_cols = [
        'Power(W)',
        'Growth Rate (gm/sec)',
        'No. of net impurity atoms added',
        'Number of net impurity of previous crystal added'
    ]
    
    # Apply log1p transformation to impurity columns
    impurity_cols = ['No. of net impurity atoms added', 
                    'Number of net impurity of previous crystal added']
    for col in impurity_cols:
        df[col] = np.log1p(np.abs(df[col]) + 1e-10)
    
    sequences = []
    targets = []
    
    # Create sequences
    for crystal_name, group in df.groupby('SheetName'):
        X = group[feature_cols].astype(np.float32).values
        
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            continue
        
        y = group['Detector grade portion (%)'].astype(np.float32).max()
        
        if 0 <= y <= 100 and len(X) >= 3:
            sequences.append(X)
            targets.append(y)
    
    # Convert to tabular features (mean, std, min, max)
    X_tabular = []
    for seq in sequences:
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
    
    return np.array(X_tabular), np.array(targets)

def run_random_forest_cv(X, y, n_folds=5):
    """Run 5-fold cross-validation for Random Forest."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    mae_scores = []
    rmse_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        # Split data
        X_train_val, X_test = X[train_idx], X[test_idx]
        y_train_val, y_test = y[train_idx], y[test_idx]
        
        # Further split for validation (80/20)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=SEED
        )
        
        # Hyperparameter grid (paper values)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=SEED),
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Predict on test set
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        
        print(f"Fold {fold}: MAE = {mae:.3f}%, RMSE = {rmse:.3f}%")
        print(f"  Best params: {grid_search.best_params_}")
    
    # Calculate mean and standard deviation
    mean_mae = np.mean(mae_scores)
    std_mae = np.std(mae_scores)
    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    
    return mean_mae, std_mae, mean_rmse, std_rmse

def main():
    print("Random Forest for HPGe Yield Prediction")
    print("="*50)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Target range: {y.min():.1f}% to {y.max():.1f}%")
    
    # Run cross-validation
    print("\nRunning 5-fold cross-validation with hyperparameter tuning...")
    mean_mae, std_mae, mean_rmse, std_rmse = run_random_forest_cv(X, y)
    
    print("\nResults:")
    print(f"Mean MAE: {mean_mae:.3f}% ± {std_mae:.3f}%")
    print(f"Mean RMSE: {mean_rmse:.3f}% ± {std_rmse:.3f}%")

if __name__ == "__main__":
    main()
