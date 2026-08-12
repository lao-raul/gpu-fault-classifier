import json, random
from pathlib import Path

LABELS = ["normal", "driver_issue", "hardware_issue", "memory_pressure", "kubernetes_issue", "node_issue"]

def sample(label):
    x = {"temperature": random.gauss(55, 8), "memory_used_pct": random.gauss(45, 15),
         "ecc_errors": 0, "xid_errors": 0, "pod_ready": 1, "driver_ok": 1,
         "cuda_visible": 1, "node_ready": 1}
    if label == "driver_issue": x.update(driver_ok=0, cuda_visible=0)
    elif label == "hardware_issue": x.update(temperature=random.gauss(88, 5), ecc_errors=random.randint(2, 20), xid_errors=random.randint(1, 8))
    elif label == "memory_pressure": x.update(memory_used_pct=random.gauss(94, 3))
    elif label == "kubernetes_issue": x.update(pod_ready=0, cuda_visible=0)
    elif label == "node_issue": x.update(node_ready=0, pod_ready=0)
    return {"features": x, "label": label}

def main():
    random.seed(42); rows = [sample(label) for label in LABELS for _ in range(300)]
    random.shuffle(rows); Path("data").mkdir(exist_ok=True)
    Path("data/gpu_faults.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

if __name__ == "__main__": main()

