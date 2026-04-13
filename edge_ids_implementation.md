# Edge IDS — Coding Agent Implementation Plan

## Project Overview

Build a lightweight, quantized intrusion detection system (IDS) that:
- Trains a Conv1D + GRU model on CICIDS2017 network flow data
- Achieves ≥98% accuracy with <10,000 parameters
- Quantizes to INT8 via TensorFlow Lite for edge deployment
- Serves real-time predictions over WebSocket via FastAPI
- Displays live traffic, latency, and alerts in a Next.js dashboard

---

## Repository Structure

```
edge-ids/
├── data/
│   ├── raw/                        # Drop CICIDS2017 CSVs here
│   ├── processed/
│   │   ├── train.csv
│   │   └── test.csv                # Locked — never modified after split
│   └── prepare_data.py
├── model/
│   ├── train.py
│   ├── evaluate.py
│   ├── quantize.py
│   └── saved/
│       ├── ids_model.keras
│       └── ids_model_int8.tflite
├── api/
│   ├── main.py                     # FastAPI app
│   ├── inference.py                # TFLite inference wrapper
│   └── requirements.txt
└── dashboard/                      # Next.js app
    ├── app/
    │   └── page.tsx
    ├── components/
    │   ├── TrafficLog.tsx
    │   ├── LatencyChart.tsx
    │   └── AlertBanner.tsx
    ├── hooks/
    │   └── useWebSocket.ts
    └── package.json
```

---

## Phase 1 — Data Isolation & Balancing

**File:** `data/prepare_data.py`

### Step 1.1 — Ingest CICIDS2017

- Load all weekly CSV files from `data/raw/` using `pandas.read_csv()`
- Concatenate into a single DataFrame
- The dataset has ~2.8M rows and 79 columns

### Step 1.2 — Clean

Drop or rename these non-feature columns before any processing:
```
'Flow ID', 'Source IP', 'Source Port', 'Destination IP',
'Destination Port', 'Protocol', 'Timestamp'
```
- Replace `inf` and `-inf` with `NaN`, then drop rows containing `NaN`
- The target column is `'Label'` — keep it

### Step 1.3 — Encode labels

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['Label'])
# Save le.classes_ to disk (joblib) for use in the API later
```

### Step 1.4 — Strict train/test split

```python
from sklearn.model_selection import train_test_split
X = df.drop(['Label', 'label_enc'], axis=1).values
y = df['label_enc'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Lock the test set immediately.** Save to `data/processed/test.csv` and do not touch it again until final evaluation.

### Step 1.5 — Scale (training set only)

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # use fit from train only
# Save scaler with joblib for use in the API
```

### Step 1.6 — SMOTE (training set only)

Apply only to `X_train_scaled` / `y_train`:

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
```

Target: minority attack classes (e.g. Botnet, Infiltration) upsampled to match BENIGN volume.

### Step 1.7 — Reshape for Conv1D

Conv1D expects `(samples, timesteps, features)`. Treat each flow as a single timestep:

```python
X_train_res = X_train_res.reshape(-1, 1, X_train_res.shape[1])
X_test_scaled = X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1])
```

### Outputs

| File | Purpose |
|---|---|
| `data/processed/train.csv` | Balanced, scaled training features + labels |
| `data/processed/test.csv` | Raw scaled test features + labels (locked) |
| `model/saved/scaler.joblib` | Fitted StandardScaler |
| `model/saved/label_encoder.joblib` | Fitted LabelEncoder |

---

## Phase 2 — Lightweight Model Architecture

**File:** `model/train.py`

### Architecture

```python
import tensorflow as tf
from tensorflow import keras

n_features = X_train_res.shape[2]   # typically 72 after cleaning
n_classes  = len(le.classes_)

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
```

**GRU vs LSTM rationale:** GRU has 2 gates (reset, update) vs LSTM's 3 (input, forget, output). This reduces parameter count by ~30% with no meaningful accuracy loss on network flow sequences.

### Parameter budget check

After `model.summary()`, assert:
```python
assert model.count_params() < 10_000, "Parameter budget exceeded"
```

Tune filter/unit counts down if needed. With 16 filters + 32 GRU units the model typically lands at ~6,000–8,000 params.

### Training

```python
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
```

### Evaluation

**File:** `model/evaluate.py`

```python
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(X_test_scaled).argmax(axis=1)
print(classification_report(y_test, y_pred, target_names=le.classes_))
```

Document per-class F1 scores. Overall accuracy must be ≥98% on the locked test set.

---

## Phase 3 — Edge Optimization (INT8 Quantization)

**File:** `model/quantize.py`

### Representative dataset

The TFLite converter needs a sample of training data to calibrate INT8 ranges:

```python
import numpy as np

def representative_dataset():
    for i in range(0, min(500, len(X_train_res)), 1):
        sample = X_train_res[i:i+1].astype(np.float32)
        yield [sample]
```

### Conversion

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('model/saved/ids_model.keras')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open('model/saved/ids_model_int8.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Document the tradeoff

After conversion, run evaluation again with the TFLite interpreter and record:

| Metric | FP32 model | INT8 model |
|---|---|---|
| Accuracy | X.XX% | X.XX% |
| Model size | X MB | X MB (4× smaller) |
| Avg inference latency | X ms | X ms |

Expected: ~1% accuracy drop, 4× size reduction, significant latency improvement on CPU.

---

## Phase 4 — Real-Time Simulation Backend

**File:** `api/main.py`, `api/inference.py`

### Dependencies

```
# api/requirements.txt
fastapi
uvicorn[standard]
numpy
pandas
joblib
tflite-runtime      # or tensorflow if tflite-runtime unavailable
```

### TFLite inference wrapper

**File:** `api/inference.py`

```python
import numpy as np
import tflite_runtime.interpreter as tflite
import time

class IDSInference:
    def __init__(self, model_path: str):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, sample: np.ndarray) -> tuple[int, float]:
        """Returns (class_index, latency_ms)."""
        inp = sample.reshape(1, 1, -1).astype(np.float32)

        start = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output_details[0]['index'])
        latency_ms = (time.perf_counter() - start) * 1000

        return int(raw.argmax()), latency_ms
```

### FastAPI WebSocket endpoint

**File:** `api/main.py`

```python
import asyncio
import json
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, WebSocket
from api.inference import IDSInference

app      = FastAPI()
model    = IDSInference('model/saved/ids_model_int8.tflite')
scaler   = joblib.load('model/saved/scaler.joblib')
le       = joblib.load('model/saved/label_encoder.joblib')
test_df  = pd.read_csv('data/processed/test.csv')

FEATURE_COLS = [c for c in test_df.columns if c not in ('Label', 'label_enc')]

@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    for _, row in test_df.iterrows():
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
```

### Run

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Phase 5 — Next.js Presentation Dashboard

### Setup

```bash
npx create-next-app@latest dashboard --typescript --tailwind --app
cd dashboard
npm install recharts
```

### WebSocket hook

**File:** `hooks/useWebSocket.ts`

```typescript
import { useEffect, useRef, useState } from 'react'

export interface Packet {
  label:      string
  is_attack:  boolean
  latency_ms: number
  ts:         number          // client-side timestamp
}

export function useWebSocket(url: string) {
  const [packets, setPackets] = useState<Packet[]>([])
  const [latestAlert, setLatestAlert] = useState<string | null>(null)
  const ws = useRef<WebSocket | null>(null)

  useEffect(() => {
    ws.current = new WebSocket(url)
    ws.current.onmessage = (e) => {
      const data = JSON.parse(e.data) as Omit<Packet, 'ts'>
      const packet: Packet = { ...data, ts: Date.now() }

      setPackets(prev => [packet, ...prev].slice(0, 500))  // keep last 500
      if (data.is_attack) setLatestAlert(data.label)
    }
    return () => ws.current?.close()
  }, [url])

  return { packets, latestAlert, clearAlert: () => setLatestAlert(null) }
}
```

### Alert banner

**File:** `components/AlertBanner.tsx`

```tsx
export function AlertBanner({ label, onDismiss }: { label: string | null, onDismiss: () => void }) {
  if (!label) return null
  return (
    <div className="fixed top-0 inset-x-0 z-50 bg-red-600 text-white px-6 py-3 flex justify-between items-center">
      <span className="font-semibold">⚠ Attack detected: {label}</span>
      <button onClick={onDismiss} className="text-white/80 hover:text-white">Dismiss</button>
    </div>
  )
}
```

### Traffic log

**File:** `components/TrafficLog.tsx`

```tsx
import { Packet } from '@/hooks/useWebSocket'

export function TrafficLog({ packets }: { packets: Packet[] }) {
  return (
    <div className="overflow-auto h-72 border rounded-lg text-sm font-mono">
      <table className="w-full">
        <thead className="sticky top-0 bg-white dark:bg-zinc-900">
          <tr>
            <th className="text-left px-3 py-2">Time</th>
            <th className="text-left px-3 py-2">Label</th>
            <th className="text-right px-3 py-2">Latency (ms)</th>
          </tr>
        </thead>
        <tbody>
          {packets.map((p, i) => (
            <tr key={i} className={p.is_attack ? 'bg-red-50 dark:bg-red-950' : ''}>
              <td className="px-3 py-1 text-zinc-400">{new Date(p.ts).toLocaleTimeString()}</td>
              <td className={`px-3 py-1 font-medium ${p.is_attack ? 'text-red-600' : 'text-green-600'}`}>{p.label}</td>
              <td className="px-3 py-1 text-right text-zinc-500">{p.latency_ms.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

### Latency chart

**File:** `components/LatencyChart.tsx`

```tsx
'use client'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { Packet } from '@/hooks/useWebSocket'

export function LatencyChart({ packets }: { packets: Packet[] }) {
  const data = [...packets].reverse().slice(-100).map((p, i) => ({
    i,
    ms: p.latency_ms,
  }))

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data}>
        <XAxis dataKey="i" hide />
        <YAxis unit="ms" width={48} />
        <Tooltip formatter={(v: number) => [`${v.toFixed(3)} ms`, 'Latency']} />
        <Line type="monotone" dataKey="ms" dot={false} stroke="#6366f1" strokeWidth={1.5} />
      </LineChart>
    </ResponsiveContainer>
  )
}
```

### Main page

**File:** `app/page.tsx`

```tsx
'use client'
import { useWebSocket } from '@/hooks/useWebSocket'
import { AlertBanner }  from '@/components/AlertBanner'
import { TrafficLog }   from '@/components/TrafficLog'
import { LatencyChart } from '@/components/LatencyChart'

export default function Page() {
  const { packets, latestAlert, clearAlert } = useWebSocket('ws://localhost:8000/ws/stream')

  return (
    <main className="min-h-screen p-6 space-y-6">
      <AlertBanner label={latestAlert} onDismiss={clearAlert} />

      <h1 className="text-2xl font-semibold">Edge IDS — Live Monitor</h1>

      <section>
        <h2 className="text-sm font-medium text-zinc-500 mb-2">Inference latency (last 100 packets)</h2>
        <LatencyChart packets={packets} />
      </section>

      <section>
        <h2 className="text-sm font-medium text-zinc-500 mb-2">Live traffic log</h2>
        <TrafficLog packets={packets} />
      </section>
    </main>
  )
}
```

### Run

```bash
npm run dev    # http://localhost:3000
```

---

## Key Constraints Summary

| Constraint | Target | How enforced |
|---|---|---|
| Model parameters | < 10,000 | `assert model.count_params() < 10_000` |
| Test set integrity | Never modified | Split before any transforms; SMOTE on train only |
| Scaler fit | Train set only | `scaler.fit_transform(X_train)`, `scaler.transform(X_test)` |
| Overall accuracy | ≥ 98% | Evaluated on locked test.csv after training |
| Quantization loss | ~1% acceptable | Document FP32 vs INT8 metrics explicitly |
| Inference latency | Sub-millisecond | Measured per packet with `time.perf_counter()` |

---

## Execution Order

```
1. python data/prepare_data.py
2. python model/train.py
3. python model/evaluate.py         # verify ≥98% on test set
4. python model/quantize.py
5. uvicorn api.main:app --port 8000
6. cd dashboard && npm run dev
```
