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
uvicorn api:app --host 0.0.0.0 --port 8000
```

### API Usage
```bash
# Health check
curl http://127.0.0.1:8000/health

# Make a prediction
curl -X POST http://127.0.0.1:8000/predict \
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
Visit http://localhost:8000/docs for Swagger UI.

## Project Structure

| File | Description |
|------|-------------|
| `gpu_fault_classifier.ipynb` | **Colab notebook** - complete pipeline with Gradio UI |
| `train.py` | Training script - trains model, evaluates, saves artifacts |
| `generate_data.py` | Data generation - creates synthetic labeled dataset |
| `api.py` | REST API entry point - FastAPI server |
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
