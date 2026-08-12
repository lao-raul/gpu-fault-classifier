import json, os
from pathlib import Path
import numpy as np, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

FEATURES = ["temperature", "memory_used_pct", "ecc_errors", "xid_errors", "pod_ready", "driver_ok", "cuda_visible", "node_ready"]
LABELS = ["normal", "driver_issue", "hardware_issue", "memory_pressure", "kubernetes_issue", "node_issue"]

class Classifier(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Dropout(.15), nn.Linear(32, 6))
    def forward(self, x): return self.net(x)

def main():
    if not Path("data/gpu_faults.jsonl").exists(): os.system("python generate_data.py")
    rows = [json.loads(x) for x in Path("data/gpu_faults.jsonl").read_text().splitlines()]
    X = np.array([[r["features"][f] for f in FEATURES] for r in rows], dtype="float32"); y = np.array([LABELS.index(r["label"]) for r in rows])
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=.3, stratify=y, random_state=42); Xv, Xte, yv, yte = train_test_split(Xtmp, ytmp, test_size=.5, stratify=ytmp, random_state=42)
    scaler = StandardScaler().fit(Xtr); Xtr, Xv, Xte = [scaler.transform(x).astype("float32") for x in (Xtr, Xv, Xte)]
    model = Classifier(); opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1e-4); loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=64, shuffle=True)
    for _ in range(60):
        model.train()
        for xb, yb in loader: opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad(): pred = model(torch.tensor(Xte)).argmax(1).numpy()
    print("accuracy=", accuracy_score(yte, pred), "macro_f1=", f1_score(yte, pred, average="macro")); print(classification_report(yte, pred, target_names=LABELS))
    Path("artifacts").mkdir(exist_ok=True); torch.save({"state_dict": model.state_dict(), "mean": scaler.mean_, "scale": scaler.scale_, "features": FEATURES, "labels": LABELS}, "artifacts/model.pt")

if __name__ == "__main__": main()

