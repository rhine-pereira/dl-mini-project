import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras

# Ensure saved model dir exists
os.makedirs('model/saved', exist_ok=True)

# Load data prepared by phase 1
print("Loading data...")
X_train_res = np.load('data/processed/X_train_res.npy')
y_train_res = np.load('data/processed/y_train_res.npy')
le = joblib.load('model/saved/label_encoder.joblib')

n_features = X_train_res.shape[2]
n_classes = len(le.classes_)

inputs = keras.Input(shape=(1, n_features))
x = keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu')(inputs)
x = keras.layers.MaxPooling1D(pool_size=1)(x)
x = keras.layers.GRU(units=32)(x)
outputs = keras.layers.Dense(n_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

assert model.count_params() < 10_000, f"Parameter budget exceeded: {model.count_params()}"

history = model.fit(
    X_train_res, y_train_res,
    epochs=20,
    batch_size=256,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
)

model.save('model/saved/ids_model.keras')
print("Model saved to model/saved/ids_model.keras")
