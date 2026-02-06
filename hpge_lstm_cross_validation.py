import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Constants
DEFAULT_CSV_PATH = "example_data.csv"
DEFAULT_OUTDIR = "results_hpge"
GROUP_COL = "SheetName"
LABEL_COL = "Detector grade portion (%)"

# Set CPU only to avoid CUDA errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Data processing
def prepare_sequences(df, group_col, label_col):
    df = df.copy()
    
    # Coerce numeric columns
    for col in df.columns:
        if col not in [group_col, label_col] and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    # Find time column
    time_candidates = ["Time (Sec)", "Time (sec)", "Time (s)", "Time", "time"]
    time_col = next((c for c in time_candidates if c in df.columns), None)
    
    # Sort data
    sort_cols = [group_col]
    if time_col and time_col in df.columns:
        sort_cols.append(time_col)
    df = df.sort_values(sort_cols).reset_index(drop=True)
    
    # Build sequences per crystal
    feature_cols = [c for c in df.columns if c not in [group_col, label_col]]
    X_list, y_list, groups = [], [], []
    
    for name, group in df.groupby(group_col):
        # Convert to float64 first to avoid overflow, then to float32
        features = group[feature_cols].astype(np.float64).values.astype(np.float32)
        
        if len(features) < 2:
            continue
            
        label = group[label_col].astype(np.float32).max()
        
        if 0 <= label <= 100:
            X_list.append(features)
            y_list.append(label)
            groups.append(str(name))
    
    return X_list, np.array(y_list, dtype=np.float32), groups, feature_cols

def pad_sequences(X_list):
    lengths = [x.shape[0] for x in X_list]
    max_len = max(lengths)
    n_features = X_list[0].shape[1]
    
    X_padded = np.zeros((len(X_list), max_len, n_features), dtype=np.float32)
    mask = np.zeros((len(X_list), max_len), dtype=bool)
    
    for i, x in enumerate(X_list):
        t = x.shape[0]
        X_padded[i, :t, :] = x
        mask[i, :t] = True
    
    return X_padded, mask, np.array(lengths)

# Scaling
def fit_minmax(X_train, mask_train):
    valid = X_train[mask_train]
    fmin = np.min(valid, axis=0)
    fmax = np.max(valid, axis=0)
    
    # Avoid division by zero
    diff = fmax - fmin
    zero_mask = np.abs(diff) < 1e-12
    fmax[zero_mask] = fmin[zero_mask] + 1.0
    
    return fmin.astype(np.float32), fmax.astype(np.float32)

def apply_minmax(X, fmin, fmax):
    diff = fmax - fmin
    # Add small epsilon to avoid division by zero
    diff = np.where(np.abs(diff) < 1e-12, 1.0, diff)
    # Suppress warnings for this operation
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (X - fmin) / diff
    return result

# Model architecture
def build_model(seq_len, n_features):
    inputs = tf.keras.Input(shape=(seq_len, n_features))
    mask = tf.keras.Input(shape=(seq_len,), dtype=tf.bool)
    
    # Masking layer
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    
    # BiLSTM layers
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(512, return_sequences=True, 
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                           dropout=0.2, recurrent_dropout=0.1))(x)
    
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=True,
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                           dropout=0.2, recurrent_dropout=0.1))(x)
    
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True,
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                           dropout=0.2, recurrent_dropout=0.1))(x)
    
    # Create attention mask - using Lambda layers with explicit output shapes
    # Expand mask dimensions for attention
    mq = Lambda(lambda x: tf.expand_dims(x, axis=1), 
                output_shape=lambda s: (s[0], 1, s[1]))(mask)
    mv = Lambda(lambda x: tf.expand_dims(x, axis=2),
                output_shape=lambda s: (s[0], s[1], 1))(mask)
    attn_mask = Lambda(lambda x: tf.logical_and(x[0], x[1]),
                       output_shape=lambda s: (s[0][0], s[0][1], s[1][2]))([mq, mv])
    
    # Multi-head attention
    attn = tf.keras.layers.MultiHeadAttention(num_heads=6, key_dim=128)(x, x, attention_mask=attn_mask)
    x = tf.keras.layers.Add()([attn, x])
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Final layers
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=False,
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                           dropout=0.2, recurrent_dropout=0.1))(x)
    
    x = tf.keras.layers.Dense(512, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(128, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    
    model = tf.keras.Model(inputs=[inputs, mask], outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0, epsilon=1e-7)
    model.compile(optimizer=optimizer, 
                  loss=tf.keras.losses.Huber(delta=1.0),
                  metrics=['mae'])
    
    return model

# Training and evaluation
def run_cross_validation(X, mask, lengths, y, groups, outdir, 
                         seeds=(0, 1, 2, 3, 4), n_folds=5, 
                         epochs=3000, batch_size=8):
    
    os.makedirs(outdir, exist_ok=True)
    
    n_samples, seq_len, n_features = X.shape
    all_predictions = []
    all_metrics = []
    
    for seed in seeds:
        set_seed(seed)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_samples)), 1):
            # Split train into train/validation
            train_idx, val_idx = train_test_split(
                train_idx, test_size=0.2, random_state=seed, shuffle=True)
            
            # Prepare data
            X_train, M_train, y_train = X[train_idx], mask[train_idx], y[train_idx]
            X_val, M_val, y_val = X[val_idx], mask[val_idx], y[val_idx]
            X_test, M_test, y_test = X[test_idx], mask[test_idx], y[test_idx]
            
            # Scale data
            fmin, fmax = fit_minmax(X_train, M_train)
            X_train_s = apply_minmax(X_train, fmin, fmax)
            X_val_s = apply_minmax(X_val, fmin, fmax)
            X_test_s = apply_minmax(X_test, fmin, fmax)
            
            # Build and train model
            model = build_model(seq_len, n_features)
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(outdir, f'best_seed{seed}_fold{fold}.keras'),
                    monitor='val_loss', save_best_only=True)
            ]
            
            model.fit(
                [X_train_s, M_train], y_train,
                validation_data=([X_val_s, M_val], y_val),
                epochs=epochs, batch_size=batch_size,
                callbacks=callbacks, verbose=0
            )
            
            # Evaluate
            y_pred = model.predict([X_test_s, M_test], batch_size=batch_size, verbose=0).flatten()
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            all_metrics.append({
                'seed': seed, 'fold': fold,
                'mae': mae, 'rmse': rmse,
                'n_train': len(train_idx),
                'n_val': len(val_idx),
                'n_test': len(test_idx)
            })
            
            for i, idx in enumerate(test_idx):
                all_predictions.append({
                    'seed': seed, 'fold': fold,
                    'SheetName': groups[idx],
                    'Actual': float(y[idx]),
                    'Predicted': float(y_pred[i]),
                    'SeqLen': int(lengths[idx])
                })
            
            print(f'Seed {seed}, Fold {fold}: MAE={mae:.3f}, RMSE={rmse:.3f}')
    
    # Save results
    pred_df = pd.DataFrame(all_predictions)
    metrics_df = pd.DataFrame(all_metrics)
    
    pred_df.to_csv(os.path.join(outdir, 'predictions.csv'), index=False)
    metrics_df.to_csv(os.path.join(outdir, 'metrics.csv'), index=False)
    
    # Calculate summary statistics
    summary = {
        'MAE_mean': float(metrics_df['mae'].mean()),
        'MAE_std': float(metrics_df['mae'].std()),
        'RMSE_mean': float(metrics_df['rmse'].mean()),
        'RMSE_std': float(metrics_df['rmse'].std()),
        'n_samples': n_samples,
        'n_features': n_features,
        'max_seq_len': seq_len,
        'n_folds': n_folds,
        'seeds': list(seeds)
    }
    
    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate plot
    try:
        import matplotlib.pyplot as plt
        
        agg = pred_df.groupby('SheetName').agg({
            'Actual': 'first',
            'Predicted': ['mean', 'std']
        }).reset_index()
        
        agg.columns = ['SheetName', 'Actual', 'Predicted_mean', 'Predicted_std']
        
        plt.figure(figsize=(16, 8))
        plt.errorbar(range(len(agg)), agg['Predicted_mean'], 
                    yerr=agg['Predicted_std'], fmt='o', label='Predicted ± std')
        plt.plot(range(len(agg)), agg['Actual'], 's-', label='Actual')
        plt.xlabel('Crystal Index')
        plt.ylabel('Detector Grade Portion (%)')
        plt.title('Actual vs Predicted Detector Grade Percentage')
        plt.legend()
        plt.xticks(range(len(agg)), agg['SheetName'], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'predictions_plot.png'), dpi=150)
        plt.close()
    except ImportError:
        print('Matplotlib not available, skipping plot generation')
    
    print(f'\nResults saved to {outdir}')
    print(f'Average MAE: {summary["MAE_mean"]:.3f} ± {summary["MAE_std"]:.3f}')
    print(f'Average RMSE: {summary["RMSE_mean"]:.3f} ± {summary["RMSE_std"]:.3f}')
    
    return summary

# Main execution
def main():
    import argparse
    import warnings
    
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default=DEFAULT_CSV_PATH)
    parser.add_argument('--outdir', default=DEFAULT_OUTDIR)
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.csv_path)
    
    # Prepare sequences
    X_list, y, groups, features = prepare_sequences(df, GROUP_COL, LABEL_COL)
    X, mask, lengths = pad_sequences(X_list)
    
    print(f'Loaded {len(X_list)} crystals')
    print(f'Sequence shape: {X.shape}')
    print(f'Features: {len(features)}')
    print(f'Features used: {features}')
    
    # Convert seeds
    seeds = tuple(int(s) for s in args.seeds.split(','))
    
    # Run cross-validation
    summary = run_cross_validation(
        X, mask, lengths, y, groups,
        outdir=args.outdir,
        seeds=seeds,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()
