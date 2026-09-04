# GPU Fault Classifier

A PyTorch-based classifier for GPU/node incidents, with an interactive web interface for predictions.

## Features

Classifies GPU/node incidents into **6 categories**:
- `normal` - healthy GPU state
- `driver_issue` - NVIDIA driver problems
- `hardware_issue` - GPU hardware failures (thermal, ECC errors, Xid errors)
- `memory_pressure` - GPU memory exhaustion
- `kubernetes_issue` - K8s pod scheduling/availability problems
- `node_issue` - underlying node-level failures

**Input features** (8 total):
| Feature | Description |
|---------|-------------|
| `temperature` | GPU temperature in Celsius |
| `memory_used_pct` | GPU memory utilization % |
| `ecc_errors` | ECC correctable error count |
| `xid_errors` | Xid error count |
| `pod_ready` | Kubernetes pod ready status (0/1) |
| `driver_ok` | Driver health status (0/1) |
| `cuda_visible` | CUDA visibility (0/1) |
| `node_ready` | Node ready status (0/1) |

## Quick Start on Google Colab

The easiest way to run this project is via Google Colab:

### 1. Upload the Notebook
Upload `gpu_fault_classifier.ipynb` to your Google Drive.

### 2. Open in Colab
Right-click the file → Open with → Google Colaboratory

### 3. Enable GPU (Optional)
```
Runtime → Change runtime type → GPU
```
If no GPU is available, the model will run on CPU.

### 4. Run All Cells
```
Runtime → Run all
```

### 5. Use the Interface
- After Step 6, click the **Gradio public link** to open the interactive interface
- Or use the embedded preview panel on the right side of Colab
- Adjust sliders/checkboxes and click **Submit** to get predictions

### 6. Stop the Interface
When done:
```
Runtime → Manage sessions → Terminate
```
Or simply restart the runtime:
```
Runtime → Restart runtime
```

## Local Development

### Installation

```bash
git clone <repo-url>
cd gpu-fault-classifier
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### macOS (Apple Silicon)

macOS does not provide NVIDIA CUDA. The training script automatically uses
Apple Metal (MPS) when available and falls back to CPU otherwise. Python 3.11
or 3.12 is recommended for the broadest package compatibility.

```bash
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Do not install NVIDIA drivers or CUDA on macOS.

The project currently pins NumPy below 2.0 because some PyTorch builds still
expect the NumPy 1.x C ABI.

### Generate Data
```bash
python generate_data.py
```

### Train Model
```bash
python train.py
```

### Start REST API
```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8800
```

The API loads `artifacts/model.pt` at startup, so run `python train.py` first.
Using `python -m uvicorn` ensures that Uvicorn is launched from the active
virtual environment rather than from a different Conda installation.

### Start Web UI

The local Gradio page provides the same interactive prediction experience as
the notebook:

```bash
python gradio_app.py --host 127.0.0.1 --port 7860
```

Open http://127.0.0.1:7860 in a browser. To create a temporary public link,
add `--share`:

```bash
python gradio_app.py --host 127.0.0.1 --port 7860 --share
```

### API Usage
```bash
# Health check
curl http://127.0.0.1:8800/health

# Make a prediction
curl -X POST http://127.0.0.1:8800/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "temperature": 92,
    "memory_used_pct": 80,
    "ecc_errors": 8,
    "xid_errors": 2,
    "pod_ready": 1,
    "driver_ok": 1,
    "cuda_visible": 1,
    "node_ready": 1
  }'
```

### API Documentation
Visit http://127.0.0.1:8800/docs for Swagger UI.

### Run Both Services

FastAPI and Gradio are independent services and can run at the same time. Open
two terminal windows, activate `.venv` in both, and run:

```bash
# Terminal 1: REST API
python -m uvicorn api:app --host 127.0.0.1 --port 8800

# Terminal 2: Web UI
python gradio_app.py --host 127.0.0.1 --port 7860
```

Then use:

```text
REST API: http://127.0.0.1:8800/docs
Web UI:   http://127.0.0.1:7860
```

Stop either service with `Ctrl+C`.

### Troubleshooting

If the error mentions `ModuleNotFoundError: No module named 'fastapi'`, the
wrong Python environment is being used. Run:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8800
```

If PyTorch reports that it was compiled for NumPy 1.x, reinstall the pinned
NumPy version:

```bash
source .venv/bin/activate
python -m pip install --force-reinstall "numpy<2"
```

## Project Structure

| File | Description |
|------|-------------|
| `gpu_fault_classifier.ipynb` | **Colab notebook** - complete pipeline with Gradio UI |
| `train.py` | Training script - trains model, evaluates, saves artifacts |
| `generate_data.py` | Data generation - creates synthetic labeled dataset |
| `api.py` | REST API entry point - FastAPI server |
| `gradio_app.py` | Local Gradio web interface |
| `requirements.txt` | Python dependencies |
| `artifacts/model.pt` | Trained model (generated after training) |
| `data/gpu_faults.jsonl` | Training data (generated after data gen) |

## Model Architecture

- Input: 8 features (normalized)
- Hidden: 32 units + ReLU + Dropout(0.15)
- Output: 6 classes
- Optimizer: AdamW (lr=0.003, weight_decay=1e-4)
- Loss: CrossEntropyLoss
- Epochs: 60, Batch size: 64

## For Production Use

The dataset is synthetic and intentionally simple. Replace `data/gpu_faults.jsonl` with labeled observations from:

- `nvidia-smi` (temperature, memory usage)
- ECC error logs
- Xid/UE errors from kernel logs
- Kubernetes Events (pod readiness)
- Driver/CUDA version info
- Node health signals (kubelet, node-problem-detector)

Then retrain with:
```bash
python generate_data.py  # if you have real data in JSONL format
python train.py
```
