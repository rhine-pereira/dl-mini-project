import asyncio
import json
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from api.inference import IDSInference

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = IDSInference('model/saved/ids_model_int8.tflite')
scaler = joblib.load('model/saved/scaler.joblib')
le = joblib.load('model/saved/label_encoder.joblib')
test_df = pd.read_csv('data/processed/test.csv')

FEATURE_COLS = [c for c in test_df.columns if c not in ('Label', 'label_enc')]

@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    # Continuous loop around the test dataframe for the dashboard
    while True:
        for _, row in test_df.iterrows():
            try:
                features = scaler.transform([row[FEATURE_COLS].values])[0]
                class_idx, latency_ms = model.predict(features)
                label = le.classes_[class_idx]
                is_attack = label != 'BENIGN'

                await websocket.send_text(json.dumps({
                    "label":      label,
                    "is_attack":  is_attack,
                    "latency_ms": round(latency_ms, 4),
                }))
                await asyncio.sleep(0.01)   # 10ms pacing — adjust for demo speed
            except Exception as e:
                # Handle client disconnects gracefully
                break
