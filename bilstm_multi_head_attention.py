"""
HPGe Detector-Grade Yield Prediction
BiLSTM with Multi-Head Attention Model for Crystal Growth Optimization
The model updates regularly as the features and the data structure of the crystal growth adds
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import warnings
warnings.filterwarnings('ignore')

# Environment configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# TensorFlow and Keras imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization, 
    Bidirectional, Add, Masking, MultiHeadAttention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import Huber
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ==================== CONFIGURATION ====================
CONFIG = {
    'csv_path': 'strict_full_converted_data.csv',
    'output_dir': 'model_results',
    'n_folds': 5,
    'n_seeds': 5,
    'lstm_units': [512, 256, 128],
    'attention_heads': 6,
    'attention_dim': 128,
    'dense_units': [512, 256, 128],
    'batch_size': 8,
    'learning_rate': 1e-3,
    'huber_delta': 1.0,
    'epochs': 3000,
    'l2_reg': 1e-4,
    'dropout_rate': 0.2,
}

# ==================== DATA PREPARATION ====================
def prepare_dataset(file_path):
    """Load and preprocess crystal growth time-series data."""
    df = pd.read_csv(file_path)
    
    # Sort by crystal and time
    df = df.sort_values(['SheetName', 'Time (Sec)']).reset_index(drop=True)
    
    # Define feature columns
    feature_cols = [
        'Power(W)',
        'Growth Rate (gm/sec)',
        'No. of net impurity atoms added',
        'Number of net impurity of previous crystal added'
    ]
    
    # Apply logarithmic transformation to impurity columns
    impurity_cols = ['No. of net impurity atoms added', 
                    'Number of net impurity of previous crystal added']
    
    for col in impurity_cols:
        if col in df.columns:
            df[col] = np.log1p(np.abs(df[col]) + 1e-10)
    
    # Group by crystal and create sequences
    sequences = []
    targets = []
    crystal_names = []
    
    for crystal_name, group in df.groupby('SheetName'):
        X = group[feature_cols].astype(np.float32).values
        
        # Validate data quality
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            continue
        
        # Target is maximum detector grade percentage for the crystal
        y = group['Detector grade portion (%)'].astype(np.float32).max()
        
        if 0 <= y <= 100 and len(X) >= 3:
            sequences.append(X)
            targets.append(y)
            crystal_names.append(crystal_name)
    
    return sequences, np.array(targets), crystal_names, feature_cols

def pad_sequences(sequences):
    """Convert variable-length sequences to fixed-length padded arrays."""
    lengths = [len(s) for s in sequences]
    max_len = max(lengths)
    n_features = sequences[0].shape[1]
    
    X = np.zeros((len(sequences), max_len, n_features), dtype=np.float32)
    mask = np.zeros((len(sequences), max_len), dtype=bool)
    
    for i, seq in enumerate(sequences):
        X[i, :len(seq), :] = seq
        mask[i, :len(seq)] = True
    
    return X, mask, np.array(lengths)

# ==================== MODEL ARCHITECTURE ====================
def build_model(seq_len, n_features):
    """Build BiLSTM with Multi-Head Attention sequence model."""
    inputs = Input(shape=(seq_len, n_features))
    
    # Mask invalid timesteps
    x = Masking(mask_value=0.0)(inputs)
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(
        512, return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.l2(CONFIG['l2_reg']),
        dropout=CONFIG['dropout_rate'],
        recurrent_dropout=CONFIG['dropout_rate']
    ))(x)
    
    x = Bidirectional(LSTM(
        256, return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.l2(CONFIG['l2_reg']),
        dropout=CONFIG['dropout_rate'],
        recurrent_dropout=CONFIG['dropout_rate']
    ))(x)
    
    x = Bidirectional(LSTM(
        128, return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.l2(CONFIG['l2_reg']),
        dropout=CONFIG['dropout_rate'],
        recurrent_dropout=CONFIG['dropout_rate']
    ))(x)
    
    # Multi-head attention mechanism
    attention = MultiHeadAttention(
        num_heads=CONFIG['attention_heads'],
        key_dim=CONFIG['attention_dim']
    )(x, x)
    
    # Residual connection with batch normalization
    residual = Add()([attention, x])
    norm = BatchNormalization()(residual)
    
    # Final BiLSTM layer
    x = Bidirectional(LSTM(
        64, return_sequences=False,
        kernel_regularizer=tf.keras.regularizers.l2(CONFIG['l2_reg']),
        dropout=CONFIG['dropout_rate'],
        recurrent_dropout=CONFIG['dropout_rate']
    ))(norm)
    
    # Dense prediction head
    x = Dense(512, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(CONFIG['l2_reg']))(x)
    x = Dropout(CONFIG['dropout_rate'])(x)
    
    x = Dense(256, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(CONFIG['l2_reg']))(x)
    x = Dropout(CONFIG['dropout_rate'])(x)
    
    x = Dense(128, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(CONFIG['l2_reg']))(x)
    x = Dropout(CONFIG['dropout_rate'])(x)
    
    outputs = Dense(1, activation='linear')(x)
    
    # Compile model
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(
        learning_rate=CONFIG['learning_rate'],
        clipnorm=1.0,
        epsilon=1e-7
    )
    model.compile(
        optimizer=optimizer,
        loss=Huber(delta=CONFIG['huber_delta']),
        metrics=['mae']
    )
    
    return model

# ==================== CROSS-VALIDATION ====================
def scale_sequences(X_train, X_val, X_test):
    """Apply RobustScaler to sequences without data leakage."""
    original_shapes = [X_train.shape, X_val.shape, X_test.shape]
    
    # Reshape for scaling
    X_train_2d = X_train.reshape(-1, X_train.shape[2])
    X_val_2d = X_val.reshape(-1, X_val.shape[2])
    X_test_2d = X_test.reshape(-1, X_test.shape[2])
    
    # Handle any NaN or Inf values
    X_train_2d = np.nan_to_num(X_train_2d, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Fit scaler on training data only
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_2d)
    
    # Transform validation and test data
    X_val_scaled = scaler.transform(X_val_2d)
    X_test_scaled = scaler.transform(X_test_2d)
    
    # Reshape back to original dimensions
    X_train_scaled = X_train_scaled.reshape(original_shapes[0])
    X_val_scaled = X_val_scaled.reshape(original_shapes[1])
    X_test_scaled = X_test_scaled.reshape(original_shapes[2])
    
    return X_train_scaled, X_val_scaled, X_test_scaled

def run_cross_validation(sequences, targets, crystal_names):
    """Execute k-fold cross-validation with multiple random seeds."""
    X, mask, lengths = pad_sequences(sequences)
    
    # Results storage
    all_results = []
    all_predictions = []
    
    for seed in range(CONFIG['n_seeds']):
        print(f"Random seed {seed + 1}/{CONFIG['n_seeds']}")
        
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        kf = KFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=seed)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            print(f"  Fold {fold}: ", end="")
            
            # Data partitioning
            X_train_val, X_test = X[train_idx], X[test_idx]
            y_train_val, y_test = targets[train_idx], targets[test_idx]
            
            train_idx2, val_idx = train_test_split(
                np.arange(len(X_train_val)), 
                test_size=0.2, 
                random_state=seed,
                shuffle=True
            )
            
            X_train, X_val = X_train_val[train_idx2], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx2], y_train_val[val_idx]
            
            print(f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            # Data scaling
            X_train_scaled, X_val_scaled, X_test_scaled = scale_sequences(
                X_train, X_val, X_test
            )
            
            # Model training
            model = build_model(X.shape[1], X.shape[2])
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=50,
                    restore_best_weights=True,
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-5,
                    verbose=0
                ),
                ModelCheckpoint(
                    f"{CONFIG['output_dir']}/model_seed{seed}_fold{fold}.keras",
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=0
                )
            ]
            
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=CONFIG['epochs'],
                batch_size=CONFIG['batch_size'],
                callbacks=callbacks,
                verbose=0
            )
            
            # Model evaluation
            y_pred = model.predict(X_test_scaled, verbose=0).flatten()
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"MAE={mae:.3f}, RMSE={rmse:.3f}")
            
            # Record results
            all_results.append({
                'seed': seed,
                'fold': fold,
                'mae': float(mae),
                'rmse': float(rmse),
                'epochs': len(history.history['loss'])
            })
            
            for i, idx in enumerate(test_idx):
                all_predictions.append({
                    'seed': seed,
                    'fold': fold,
                    'crystal': crystal_names[idx],
                    'actual': float(y_test[i]),
                    'predicted': float(y_pred[i]),
                    'error': float(abs(y_test[i] - y_pred[i]))
                })
    
    return all_results, all_predictions, X.shape

# ==================== RESULTS ANALYSIS ====================
def analyze_and_save_results(results, predictions, dataset_shape):
    """Process, analyze, and save experimental results."""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    results_df = pd.DataFrame(results)
    preds_df = pd.DataFrame(predictions)
    
    # Calculate performance metrics
    mae_values = results_df['mae'].values
    rmse_values = results_df['rmse'].values
    
    avg_mae = np.mean(mae_values)
    std_mae = np.std(mae_values)
    avg_rmse = np.mean(rmse_values)
    
    best_mae = results_df['mae'].min()
    
    # Display results
    print(f"\nExperimental Results")
    print("="*60)
    print(f"Dataset: {dataset_shape[0]} crystals, {dataset_shape[2]} features")
    print(f"Validation: {CONFIG['n_folds']}-fold CV with {CONFIG['n_seeds']} random seeds")
    print(f"\nPerformance Metrics:")
    print(f"  Average MAE:  {avg_mae:.3f} ± {std_mae:.3f}")
    print(f"  Average RMSE: {avg_rmse:.3f}")
    print(f"  Best fold MAE:  {best_mae:.3f}")
    
    # Save data files
    results_df.to_csv(f"{CONFIG['output_dir']}/cross_validation_results.csv", index=False)
    preds_df.to_csv(f"{CONFIG['output_dir']}/crystal_predictions.csv", index=False)
    
    # Save summary
    summary = {
        'average_mae': float(avg_mae),
        'mae_standard_deviation': float(std_mae),
        'average_rmse': float(avg_rmse),
        'best_mae': float(best_mae),
        'n_crystals': int(dataset_shape[0]),
        'n_features': int(dataset_shape[2]),
        'max_sequence_length': int(dataset_shape[1]),
        'n_folds': CONFIG['n_folds'],
        'n_seeds': CONFIG['n_seeds']
    }
    
    with open(f"{CONFIG['output_dir']}/experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results_df, preds_df, summary

def create_result_visualizations(predictions_df, results_df, summary):
    """Generate analytical visualizations of model performance."""
    try:
        # Aggregate crystal-level statistics
        crystal_stats = predictions_df.groupby('crystal').agg({
            'actual': 'first',
            'predicted': ['mean', 'std'],
            'error': ['mean', 'std']
        }).reset_index()
        
        crystal_stats.columns = ['crystal', 'actual', 'pred_mean', 'pred_std', 'error_mean', 'error_std']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Actual vs Predicted
        ax1 = axes[0, 0]
        ax1.errorbar(crystal_stats['actual'], crystal_stats['pred_mean'],
                    yerr=crystal_stats['pred_std'], fmt='o', capsize=5, alpha=0.7)
        ax1.plot([0, 40], [0, 40], 'k--', alpha=0.5)
        ax1.set_xlabel('Actual Detector Grade (%)')
        ax1.set_ylabel('Predicted Detector Grade (%)')
        ax1.set_title('Model Predictions vs Actual Values')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: MAE distribution
        ax2 = axes[0, 1]
        ax2.hist(results_df['mae'], bins=12, edgecolor='black', alpha=0.7)
        ax2.axvline(x=summary['average_mae'], color='red', linestyle='--', 
                   label=f'Mean: {summary["average_mae"]:.2f}')
        ax2.set_xlabel('Mean Absolute Error (Percentage Points)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Cross-Validation MAE Distribution')
        ax2.legend()
        
        # Plot 3: Fold performance comparison
        ax3 = axes[1, 0]
        results_df['fold_label'] = results_df['seed'].astype(str) + '-' + results_df['fold'].astype(str)
        ax3.bar(range(len(results_df)), results_df['mae'])
        ax3.set_xticks(range(len(results_df)))
        ax3.set_xticklabels(results_df['fold_label'], rotation=90)
        ax3.set_xlabel('Validation Fold (Seed-Fold)')
        ax3.set_ylabel('MAE')
        ax3.set_title('Performance Across Validation Folds')
        
        # Plot 4: Error analysis by crystal
        ax4 = axes[1, 1]
        crystal_stats_sorted = crystal_stats.sort_values('actual')
        ax4.bar(range(len(crystal_stats_sorted)), crystal_stats_sorted['error_mean'],
               yerr=crystal_stats_sorted['error_std'], capsize=3, alpha=0.7)
        ax4.set_xticks(range(len(crystal_stats_sorted)))
        ax4.set_xticklabels(crystal_stats_sorted['crystal'], rotation=90)
        ax4.set_xlabel('Crystal Identifier')
        ax4.set_ylabel('Mean Prediction Error (%)')
        ax4.set_title('Prediction Accuracy by Crystal')
        
        plt.tight_layout()
        plt.savefig(f"{CONFIG['output_dir']}/performance_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Visualization generation skipped: {str(e)}")

# ==================== MAIN EXECUTION ====================
def main():
    """Execute the complete modeling pipeline."""
    print("="*70)
    print("HPGe CRYSTAL YIELD PREDICTION MODEL")
    print("="*70)
    
    # Data preparation
    sequences, targets, crystal_names, feature_names = prepare_dataset(CONFIG['csv_path'])
    
    # Cross-validation experiment
    results, predictions, dataset_shape = run_cross_validation(
        sequences, targets, crystal_names
    )
    
    # Results analysis
    results_df, predictions_df, summary = analyze_and_save_results(
        results, predictions, dataset_shape
    )
    
    # Visualization
    create_result_visualizations(predictions_df, results_df, summary)
    
    # Final report
    print(f"\nModel Performance Summary")
    print("="*70)
    print(f"Best achieved MAE: {summary['best_mae']:.3f}")
    print(f"Average MAE across all folds: {summary['average_mae']:.3f} ± {summary['mae_standard_deviation']:.3f}")
    print(f"Total crystals analyzed: {summary['n_crystals']}")
    print(f"Features per timestep: {summary['n_features']}")
    
    # Optional feature addition note
    if 'Output net impurity' not in feature_names:
        print(f"\nNote: Additional process variables such as 'Output net impurity'")
        print("could potentially improve model accuracy if available.")

if __name__ == "__main__":
    main()
