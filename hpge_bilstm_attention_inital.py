import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import MultiHeadAttention
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.losses import Huber

file_path = "strict_full_converted_data.csv"
data = pd.read_csv(file_path)
for col in data.columns:
    if data[col].dtype == 'object' and col != "SheetName":
        print(f"Warning: Converting non-numeric column {col} to numeric.")
        data[col] = pd.to_numeric(data[col], errors='coerce')

data.dropna(inplace=True)
data = data.sort_values(by="SheetName", ascending=True).reset_index(drop=True)
grouped_data, targets, grouped_sheets = [], [], []

for sheet_name, group in data.groupby("SheetName"):
    group_features = group.drop(columns=["SheetName", "Detector grade portion (%)"], errors='ignore')

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(group_features)

    target_value = group["Detector grade portion (%)"].max()

    # Filter corrupted targets (sanity check)
    if target_value < 100:
        grouped_data.append(features_scaled)
        targets.append(target_value)
        grouped_sheets.append(sheet_name)


X_list = np.array(grouped_data, dtype=object)
y_array = np.array(targets, dtype=np.float32)

max_sequence_length = max([x.shape[0] for x in X_list])
X_padded = tf.keras.preprocessing.sequence.pad_sequences(X_list, maxlen=max_sequence_length, padding="post", dtype="float32")


input_layer = Input(shape=(max_sequence_length, X_padded.shape[2]))

lstm_1 = Bidirectional(LSTM(512, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-4), dropout=0.2, recurrent_dropout=0.1))(input_layer)
lstm_2 = Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-4), dropout=0.2, recurrent_dropout=0.1))(lstm_1)
lstm_3 = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-4), dropout=0.2, recurrent_dropout=0.1))(lstm_2)

attention = MultiHeadAttention(num_heads=6, key_dim=128)(lstm_3, lstm_3)
residual = Add()([attention, lstm_3])
norm = BatchNormalization()(residual)

lstm_4 = Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4), dropout=0.2, recurrent_dropout=0.1))(norm)

dense_1 = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(lstm_4)
dropout_1 = Dropout(0.2)(dense_1)

dense_2 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(dropout_1)
dropout_2 = Dropout(0.2)(dense_2)

dense_3 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(dropout_2)
dropout_3 = Dropout(0.2)(dense_3)

output_layer = Dense(1, activation='linear')(dropout_3)
model = Model(inputs=input_layer, outputs=output_layer)


optimizer = Adam(learning_rate=0.001, clipnorm=1.0, epsilon=1e-7)
model.compile(optimizer=optimizer, loss=Huber(delta=1.0), metrics=["mae"])

model_path = "final_optimized_model_withoutimpurity.keras"
callbacks = [
    EarlyStopping(monitor="loss", patience=50, restore_best_weights=True),
    ReduceLROnPlateau(monitor="loss", factor=0.5, patience=10, min_lr=1e-5),
    ModelCheckpoint(model_path, monitor="loss", save_best_only=True, verbose=1)
]

history = model.fit(
    X_padded, y_array,
    epochs=3000,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

def predict_and_sort(model, X_padded, grouped_sheets, y_array):
    y_pred = model.predict(X_padded).flatten()
    y_true = y_array.flatten()

    df = pd.DataFrame({
        "SheetName": grouped_sheets,
        "Actual": y_true,
        "Predicted": y_pred
    }).sort_values(by="SheetName").reset_index(drop=True)

    print(df)

    plt.figure(figsize=(16, 8))
    plt.plot(df["SheetName"], df["Actual"], label="Actual", marker="o")
    plt.plot(df["SheetName"], df["Predicted"], label="Predicted", marker="x")
    plt.xticks(rotation=90)
    plt.xlabel("Crystal (SheetName)")
    plt.ylabel("Detector Grade Portion (%)")
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df

predict_and_sort(model, X_padded, grouped_sheets, y_array)
