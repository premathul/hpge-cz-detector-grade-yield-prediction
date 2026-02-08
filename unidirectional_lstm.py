"""
Unidirectional LSTM for HPGe Detector-Grade Yield Prediction
Implements methodology from Section 4.1 of the paper.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

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
    
    # Pad sequences for LSTM
    max_len = max(len(s) for s in sequences)
    n_features = sequences[0].shape[1]
    
    X_padded = np.zeros((len(sequences), max_len, n_features), dtype=np.float32)
    for i, seq in enumerate(sequences):
        X_padded[i, :len(seq), :] = seq
    
    return X_padded, np.array(targets)

def build_unidirectional_lstm(seq_len, n_features):
    """Build Unidirectional LSTM model as per paper architecture."""
    inputs = Input(shape=(seq_len, n_features))
    
    # Masking layer for variable-length sequences
    x = Masking(mask_value=0.0)(inputs)
    
    # LSTM layers (paper: 128, 64, 32 units)
    x = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
    x = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
    x = LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(x)
    
    # Dense layers (paper: 128, 64, 32 units)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile with paper parameters
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=Huber(delta=1.0),
        metrics=['mae']
    )
    
    return model

def run_unidirectional_lstm_cv(X, y, n_folds=5, n_seeds=5):
    """Run cross-validation with multiple random seeds."""
    all_mae = []
    all_rmse = []
    
    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}")
        
        # Set random seeds
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        seed_mae = []
        seed_rmse = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            # Split data
            X_train_val, X_test = X[train_idx], X[test_idx]
            y_train_val, y_test = y[train_idx], y[test_idx]
            
            # Further split for validation (80/20)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.2, random_state=seed
            )
            
            # Normalize data
            scaler = MinMaxScaler()
            n_train, seq_len, n_features = X_train.shape
            
            X_train_2d = X_train.reshape(-1, n_features)
            X_val_2d = X_val.reshape(-1, n_features)
            X_test_2d = X_test.reshape(-1, n_features)
            
            X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_train, seq_len, n_features)
            X_val_scaled = scaler.transform(X_val_2d).reshape(X_val.shape)
            X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape)
            
            # Build and train model
            model = build_unidirectional_lstm(seq_len, n_features)
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6)
            ]
            
            model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=3000,
                batch_size=8,
                callbacks=callbacks,
                verbose=0
            )
            
            # Predict on test set
            y_pred = model.predict(X_test_scaled, verbose=0).flatten()
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            seed_mae.append(mae)
            seed_rmse.append(rmse)
            
            print(f"  Fold {fold}: MAE = {mae:.3f}%, RMSE = {rmse:.3f}%")
        
        all_mae.extend(seed_mae)
        all_rmse.extend(seed_rmse)
    
    # Calculate mean and standard deviation
    mean_mae = np.mean(all_mae)
    std_mae = np.std(all_mae)
    mean_rmse = np.mean(all_rmse)
    std_rmse = np.std(all_rmse)
    
    return mean_mae, std_mae, mean_rmse, std_rmse

def main():
    print("Unidirectional LSTM for HPGe Yield Prediction")
    print("="*50)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Target range: {y.min():.1f}% to {y.max():.1f}%")
    
    # Run cross-validation
    print("\nRunning 5-fold cross-validation with 5 random seeds...")
    mean_mae, std_mae, mean_rmse, std_rmse = run_unidirectional_lstm_cv(X, y)
    
    print("\nResults:")
    print(f"Mean MAE: {mean_mae:.3f}% ± {std_mae:.3f}%")
    print(f"Mean RMSE: {mean_rmse:.3f}% ± {std_rmse:.3f}%")

if __name__ == "__main__":
    main()
