import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

model = tf.keras.models.load_model('model/saved/ids_model.keras')
scaler = joblib.load('model/saved/scaler.joblib')
le = joblib.load('model/saved/label_encoder.joblib')

df_test = pd.read_csv('data/processed/test.csv')
X_test = df_test.drop(columns=['Label', 'label_enc']).values
y_test = df_test['label_enc'].values

X_test_scaled = scaler.transform(X_test)
X_test_scaled = X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1])

print("Evaluating...")
y_pred_probs = model.predict(X_test_scaled)
y_pred = y_pred_probs.argmax(axis=1)

report = classification_report(y_test, y_pred, target_names=le.classes_)
print(report)
