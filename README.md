# GPU Fault Classifier

A small PyTorch project for classifying GPU/node incidents and exposing the classifier as a REST tool for an AI Agent.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python train.py
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Test

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H 'content-type: application/json' \
  -d '{"temperature":92,"memory_used_pct":80,"ecc_errors":8,"xid_errors":2,"pod_ready":1,"driver_ok":1,"cuda_visible":1,"node_ready":1}'
```

The dataset is synthetic and intentionally simple; replace `data/gpu_faults.jsonl` with labelled observations from `nvidia-smi`, Xid/ECC logs, Kubernetes Events, driver/CUDA versions, and node health signals for a real model.
