# Edge IDS (Intrusion Detection System)

A lightweight, quantized intrusion detection system (IDS) optimized for network edge deployment.

## Features

- **Lightweight Model:** Trains a Conv1D + GRU model on the CICIDS2017 network flow data achieving ≥98% accuracy with <10,000 parameters.
- **Edge Deployment Ready:** Uses INT8 quantization via TensorFlow Lite for an optimized, low-latency deployment on resource-constrained devices (~4x size reduction).
- **Real-time Inference:** FastAPI-based WebSocket backend for sub-millisecond live predictions.
- **Live Monitor Dashboard:** Next.js presentation dashboard for real-time traffic monitoring, latency tracking, and instant attack alerts.

## Quick Start

### 1. Dependencies

- Python 3.9+
- Node.js 18+

### 2. Data Preparation

Download the CICIDS2017 dataset and place the raw CSVs into the `data/raw/` directory.

```bash
# Prepare, clean, balance (SMOTE), and scale the data
python data/prepare_data.py
```

### 3. Model Training & Quantization

```bash
# Train the Conv1D+GRU model (saves to model/saved/ids_model.keras)
python model/train.py

# Evaluate on the locked test set (expect >= 98% accuracy)
python model/evaluate.py

# Quantize the model to INT8 for edge deployment
python model/quantize.py
```

### 4. Running the Real-Time API

Launch the FastAPI WebSocket inference server:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 5. Running the Dashboard

Start the Next.js presentation dashboard:
```bash
cd dashboard
npm install
npm run dev
```

Visit `http://localhost:3000` in your browser to view the live edge IDS monitoring dashboard.

## Project Structure

| Directory | Description |
|-----------|-------------|
| `data/`   | Data ingestion, cleaning, scaling, and balancing logic. |
| `model/`  | Model architecture, training loops, evaluation metrics, and TFLite quantization. |
| `api/`    | Real-time FastAPI backend using `tflite-runtime` for WebSocket streaming. |
| `dashboard/` | Next.js web application visualizing live inferences, latencies, and alerts. |

## License

MIT
