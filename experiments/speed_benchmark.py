"""
Inference Speed (FPS/Latency) Benchmark Suite
--------------------------------------------------
Purpose:
    Measures the inference speed (Latency per image and FPS) for all SOTA 
    models using the same hardware environment. Essential for the 
    "Real-time Performance" section of the SCI paper.

Usage:
    python experiments/speed_benchmark.py
"""

import os
import sys
import time
import torch
import pandas as pd

# Ensure local 'ultralytics' is used (Must be before any ultralytics imports!)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from ultralytics import YOLO, RTDETR, YOLOv10

def benchmark_speed():
    # 统一测试参数
    DATA_YAML = "agriyolo.yaml"
    IMG_SIZE = 640
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    WARMUP = 10  # 预热次数
    ITERATIONS = 50 # 测试次数
    
    # 待测试模型列表 (指向 SOTA 对比实验训练好的最佳权重)
    # 路径格式: SOTA_Comparisons/<ModelName>/weights/best.pt
    MODELS_TO_TEST = [
        {"name": "AgriYOLO",  "path": "SOTA_Comparisons/AgriYOLO/weights/best.pt",  "type": "v10"},
        {"name": "YOLOv10s",  "path": "SOTA_Comparisons/YOLOv10s/weights/best.pt",  "type": "v10"},
        {"name": "YOLOv8s",   "path": "SOTA_Comparisons/YOLOv8s/weights/best.pt",   "type": "v8"},
        {"name": "YOLOv9c",   "path": "SOTA_Comparisons/YOLOv9c/weights/best.pt",   "type": "v9"},
        {"name": "YOLOv5s",   "path": "SOTA_Comparisons/YOLOv5s/weights/best.pt",   "type": "v5"},
        {"name": "RT_DETR_l", "path": "SOTA_Comparisons/RT_DETR_l/weights/best.pt", "type": "rtdetr"}
    ]

    results = []

    print(f"🚀 Starting speed benchmark on {DEVICE}...")

    for m in MODELS_TO_TEST:
        print(f"Testing {m['name']}...")
        
        if not os.path.exists(m['path']):
            print(f"⚠️ Warning: Weights not found at {m['path']}. Skipping...")
            continue

        # 加载模型
        try:
            if m["type"] == "rtdetr": model = RTDETR(m["path"])
            elif m["type"] == "v10": model = YOLOv10(m["path"])
            else: model = YOLO(m["path"])
            
            model.to(DEVICE)
            
            # 创建虚拟输入
            dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
            
            # 1. 预热
            for _ in range(WARMUP):
                _ = model(dummy_input, verbose=False)
            
            # 2. 正式测速
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(ITERATIONS):
                _ = model(dummy_input, verbose=False)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            # 计算平均耗时 (ms) 和 FPS
            avg_latency = ((end_time - start_time) / ITERATIONS) * 1000
            fps = 1000 / avg_latency
            
            results.append({
                "Model": m["name"],
                "Latency (ms)": round(avg_latency, 2),
                "FPS": round(fps, 1)
            })
            print(f"   Done: {avg_latency:.2f} ms | {fps:.1} FPS")
            
        except Exception as e:
            print(f"   Failed to test {m['name']}: {e}")

    # 保存结果
    if results:
        df = pd.DataFrame(results)
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        df.to_csv(os.path.join(log_dir, "speed_benchmark.csv"), index=False)
        print(f"\n✅ Benchmark results saved to {log_dir}/speed_benchmark.csv")
        print(df.to_markdown(index=False))

if __name__ == "__main__":
    benchmark_speed()
