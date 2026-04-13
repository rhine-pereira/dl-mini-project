import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Ensure output directories exist
os.makedirs('data/processed', exist_ok=True)
os.makedirs('model/saved', exist_ok=True)

# 1.1 Ingest CICIDS2017
raw_csv_files = glob.glob('data/raw/*.csv')
if not raw_csv_files:
    raise FileNotFoundError("No CSV files found in data/raw/")

print(f"Loading {len(raw_csv_files)} files...")
df_list = [pd.read_csv(f, skipinitialspace=True) for f in raw_csv_files]
df = pd.concat(df_list, ignore_index=True)
print(f"Loaded total dataset shape: {df.shape}")

# 1.2 Clean
drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp']
# Note: In CICIDS2017 some columns come with leading/trailing whitespaces, skipinitialspace handles most but let's strip names just in case
df.columns = df.columns.str.strip()

# Only drop columns that actually exist in the dataframe to avoid KeyError
cols_to_drop = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=cols_to_drop)

# Replace inf with NaN and drop rows with NaN
df = df.replace([np.inf, -np.inf], np.nan).dropna()
print(f"Dataset shape after cleaning: {df.shape}")

# 1.3 Encode labels
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['Label'])
joblib.dump(le, 'model/saved/label_encoder.joblib')
print(f"Labels encoded: {le.classes_}")

# 1.4 Strict train/test split
X = df.drop(columns=['Label', 'label_enc']).values
y = df['label_enc'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Lock test set into a csv
# reconstruct dataframe for test set to save identically
feature_cols = df.drop(columns=['Label', 'label_enc']).columns
df_test = pd.DataFrame(X_test, columns=feature_cols)
df_test['Label'] = le.inverse_transform(y_test)
df_test['label_enc'] = y_test
df_test.to_csv('data/processed/test.csv', index=False)
print("Locked test set saved to data/processed/test.csv")

# 1.5 Scale (training set only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'model/saved/scaler.joblib')
print("Scaler saved to model/saved/scaler.joblib")

# For SMOTE we need X_train_scaled, but to reshape to Conv1D let's continue
# 1.6 SMOTE (training set only)
print("Applying SMOTE...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

# 1.7 Reshape for Conv1D
X_train_res = X_train_res.reshape(-1, 1, X_train_res.shape[1])

# Save the training set for convenience or memory management between scripts
np.save('data/processed/X_train_res.npy', X_train_res)
np.save('data/processed/y_train_res.npy', y_train_res)
print(f"Final training set shape: {X_train_res.shape}")
print("Phase 1 completed successfully.")
