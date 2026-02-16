"""
[NEW] Dataset Insight Tool: Target Size Distribution
--------------------------------------------------
Purpose:
    Analyzes the distribution of bounding box sizes (relative to image size)
    in the training/test sets. This provides quantitative evidence for 
    the "Small Object Challenge" claim in the SCI paper.

Usage:
    python experiments/dataset_analysis.py
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fix for Windows MKL/Fortran Runtime Error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def analyze_dataset():
    data_yaml = r"E:\Desktop\datasets\Crop\data.yaml"
    if not os.path.exists(data_yaml):
        print(f"❌ {data_yaml} not found.")
        return

    with open(data_yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    base_path = cfg.get('path', './data/Crop')
    # 我们分析 Test 集最为严谨
    labels_dir = os.path.join(base_path, "test", "labels")
    
    if not os.path.exists(labels_dir):
        print(f"❌ Label directory not found: {labels_dir}")
        return

    print(f"📊 Analyzing target sizes in: {labels_dir}...")
    
    areas = []
    aspect_ratios = []
    
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(labels_dir, filename), 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        w, h = float(parts[3]), float(parts[4])
                        # 在 YOLO 格式下，w 和 h 是相对于 1.0 的
                        # 面积 = w * h
                        areas.append(w * h)
                        aspect_ratios.append(w / (h + 1e-6))

    if not areas:
        print("⚠️ No annotations found.")
        return

    # 绘制直方图
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # 按照 COCO 标准定义微小目标: 面积 < 32^2 (在 640x640 下约等于 0.0025)
    coco_small_thresh = (32/640)**2 
    
    sns.histplot(areas, bins=50, kde=True, color='skyblue')
    plt.axvline(coco_small_thresh, color='red', linestyle='--', label=f'COCO Small Threshold ({coco_small_thresh:.4f})')
    
    # 计算微小目标占比
    small_count = sum(1 for a in areas if a < coco_small_thresh)
    total_count = len(areas)
    small_percent = (small_count / total_count) * 100

    plt.title(f"Target Size Distribution (Small Objects: {small_percent:.1f}%)", fontsize=14)
    plt.xlabel("Relative Area (Width * Height)")
    plt.ylabel("Frequency")
    plt.legend()
    
    save_dir = "picture"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "dataset_size_distribution.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "dataset_size_distribution.png"), dpi=300, bbox_inches='tight')
    
    print(f"✅ Distribution plot saved to {save_dir}")
    print(f"📊 Total objects: {total_count}")
    print(f"📊 Small objects (<32^2): {small_count} ({small_percent:.1f}%)")

if __name__ == "__main__":
    analyze_dataset()
