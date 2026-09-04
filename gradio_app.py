from pathlib import Path
import argparse

import gradio as gr
import numpy as np
import torch

from train import Classifier


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "model.pt"


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}. Run `python train.py` first.")

    bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = Classifier()
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle


model, bundle = load_model()


def predict(temperature, memory_used_pct, ecc_errors, xid_errors,
            pod_ready, driver_ok, cuda_visible, node_ready):
    observation = {
        "temperature": temperature,
        "memory_used_pct": memory_used_pct,
        "ecc_errors": ecc_errors,
        "xid_errors": xid_errors,
        "pod_ready": int(pod_ready),
        "driver_ok": int(driver_ok),
        "cuda_visible": int(cuda_visible),
        "node_ready": int(node_ready),
    }

    x = np.array([[observation[f] for f in bundle["features"]]], dtype="float32")
    x = ((x - bundle["mean"]) / bundle["scale"]).astype("float32")

    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(x)), dim=1)[0].numpy()

    order = probabilities.argsort()[::-1]
    label = bundle["labels"][int(order[0])]
    confidence = float(probabilities[order[0]])

    rows = [f"### 预测结果：`{label}`", f"**置信度：{confidence:.1%}**", "", "| 类别 | 概率 |", "|---|---:|"]
    rows.extend(
        f"| `{bundle['labels'][int(i)]}` | {probabilities[i]:.1%} |"
        for i in order
    )
    return "\n".join(rows)


def build_interface():
    with gr.Blocks(title="GPU Fault Classifier") as interface:
        gr.Markdown(
            "# 🚀 GPU Fault Classifier\n"
            "输入 GPU、驱动、CUDA、Pod 和节点状态，预测当前最可能的故障类别。"
        )

        with gr.Row():
            with gr.Column():
                temperature = gr.Slider(-20, 150, value=55, step=1, label="🌡️ 温度 (°C)")
                memory_used_pct = gr.Slider(0, 100, value=45, step=1, label="💾 显存使用率 (%)")
                ecc_errors = gr.Slider(0, 50, value=0, step=1, label="⚠️ ECC 错误数")
                xid_errors = gr.Slider(0, 20, value=0, step=1, label="⚠️ XID 错误数")
            with gr.Column():
                pod_ready = gr.Checkbox(value=True, label="✅ Pod Ready")
                driver_ok = gr.Checkbox(value=True, label="✅ Driver OK")
                cuda_visible = gr.Checkbox(value=True, label="✅ CUDA Visible")
                node_ready = gr.Checkbox(value=True, label="✅ Node Ready")

        predict_button = gr.Button("开始预测", variant="primary")
        result = gr.Markdown()

        predict_button.click(
            fn=predict,
            inputs=[temperature, memory_used_pct, ecc_errors, xid_errors,
                    pod_ready, driver_ok, cuda_visible, node_ready],
            outputs=result,
        )

        gr.Examples(
            examples=[
                [55, 45, 0, 0, True, True, True, True],
                [88, 50, 15, 5, True, True, True, True],
                [58, 95, 0, 0, True, True, True, True],
                [55, 40, 0, 0, False, False, False, True],
                [55, 40, 0, 0, False, True, True, False],
                [55, 40, 0, 0, False, True, True, True],
            ],
            inputs=[temperature, memory_used_pct, ecc_errors, xid_errors,
                    pod_ready, driver_ok, cuda_visible, node_ready],
            label="示例场景",
        )

        gr.Markdown(
            "**类别说明：** `normal` 正常，`driver_issue` 驱动问题，"
            "`hardware_issue` 硬件问题，`memory_pressure` 显存压力，"
            "`kubernetes_issue` Kubernetes 问题，`node_issue` 节点问题。\n\n"
            "> 当前模型使用合成数据训练，仅适合演示。"
        )

    return interface


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the GPU fault classifier UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()
    build_interface().launch(server_name=args.host, server_port=args.port, share=args.share)
