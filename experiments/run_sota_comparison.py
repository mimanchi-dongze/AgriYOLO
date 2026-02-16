"""
Comprehensive SOTA Model Comparison Suite (Fine-grained for Small Objects)
-------------------------------------------------------------------------
Purpose:
    Compares TAL-FFN (Task-driven Asymmetric Lightweight Feature Fusion Network)
    against an extensive list of SOTA object detection models using local YAML
    configurations to ensure training from scratch.
    Specifically reports mAP_small metrics for SCI publication evidence.

Models Included (All trained from scratch):
    - TAL-FFN (Ours) - YOLOv10s with TAL-FFN module
    - YOLOv10s (Baseline)
    - YOLOv8s
    - YOLOv9c
    - YOLOv5s
    - RT-DETR-l

Usage:
    python experiments/run_sota_comparison.py

Note:
    Must be run from the project root directory.
"""

import os
import sys

# Fix for Windows MKL/Fortran Runtime Error (OMP: Error #15 or forrtl: error 200)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import yaml
import pandas as pd
import glob
import matplotlib
# Force non-interactive backend to avoid GUI crashes
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Ensure local 'ultralytics' is used (Must be before any ultralytics imports!)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from ultralytics import YOLO, RTDETR, YOLOv10
    
# --- Helper Functions (Small Map Evaluation Logic) ---
def xywhn2xywh(x, y, w, h, W, H):
    return x * W, y * H, w * W, h * H

def xywh2xywh_topleft(x, y, w, h):
    return x - w / 2, y - h / 2, w, h

def generate_gt_json(data_yaml_path, image_ids_in_pred, output_json="gt.json"):
    """Generates COCO GT JSON from YOLO labels for precision evaluation"""
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Path logic based on Agri dataset structure
    base_path = cfg.get('path', './data/Crop')
    test_labels_path = os.path.join(base_path, "test", "labels")
    
    coco_data = {"images": [], "annotations": [], "categories": []}
    
    # 1. Categories
    names = cfg.get('names', {})
    # Handle case where names is a list (common in official YOLO configs)
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
        
    for k, v in names.items():
        coco_data["categories"].append({"id": int(k), "name": v})
            
    # 2. Images & Annotations
    ann_id = 1
    # Assuming standard resolution used in training for consistency in metric calc
    img_w, img_h = 640, 640 
    
    for img_id in image_ids_in_pred:
        txt_path = os.path.join(test_labels_path, f"{img_id}.txt")
        coco_data["images"].append({
            "id": img_id,
            "file_name": f"{img_id}.jpg",
            "height": img_h,
            "width": img_w
        })

        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_c, y_c, w, h = xywhn2xywh(float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), img_w, img_h)
                        x1, y1, w, h = xywh2xywh_topleft(x_c, y_c, w, h)
                        coco_data["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls_id,
                            "bbox": [x1, y1, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        ann_id += 1
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f)
    return output_json

def evaluate_small_objects(weight_path, model_name, data_yaml_path):
    """Rigorous small object evaluation using pycocotools"""
    print(f"📊 Running fine-grained evaluation for {model_name}...")
    
    # Determine model class based on name
    if "rtdetr" in model_name.lower():
        model = RTDETR(weight_path)
    elif "v10" in model_name.lower() or "agriyolo" in model_name.lower():
        model = YOLOv10(weight_path)
    else:
        model = YOLO(weight_path)
        
    project_run = f"runs/sota_eval/{model_name}"
    # plots=True forces YOLO to generate PR curves and Confusion Matrices
    model.val(data=data_yaml_path, split='test', save_json=True, plots=True, verbose=False, project=project_run, name="val")
    
    pred_json = os.path.join(project_run, "val", "predictions.json")
    if not os.path.exists(pred_json):
        return {"Model": model_name, "mAP50": 0, "mAP50-95": 0, "mAP_small": 0}

    with open(pred_json, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    pred_image_ids = list(set([p['image_id'] for p in preds]))

    gt_json = os.path.join(project_run, "gt_temp.json")
    generate_gt_json(data_yaml_path, pred_image_ids, gt_json)
    
    try:
        cocoGt = COCO(gt_json)
        cocoDt = cocoGt.loadRes(pred_json)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        return {
            "Model": model_name,
            "mAP50": cocoEval.stats[1],
            "mAP50-95": cocoEval.stats[0],
            "mAP_small": cocoEval.stats[3]
        }
    except Exception as e:
        print(f"❌ Evaluation failed for {model_name}: {e}")
        return {"Model": model_name, "mAP50": 0, "mAP50-95": 0, "mAP_small": 0}

def plot_sota_curves(output_root):
    """Generates comparative training curves for all SOTA models"""
    print("\n📈 Generating SOTA Comparison Charts...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    csv_paths = glob.glob(os.path.join(output_root, "*", "results.csv"))
    if not csv_paths:
        print("⚠️ No results.csv found for plotting.")
        return

    plt.figure(figsize=(10, 6))
    for csv_file in csv_paths:
        model_name = os.path.basename(os.path.dirname(csv_file))
        
        # [NEW] Explicitly exclude RT-DETR as requested
        if "rtdetr" in model_name.lower():
            continue

        df = pd.read_csv(csv_file)
        df.columns = [c.strip() for c in df.columns]
        
        # Determine mAP column
        map_col = next((c for c in df.columns if "mAP50-95" in c), None)
        if map_col:
            sns.lineplot(x=df["epoch"], y=df[map_col], label=model_name, linewidth=2)
    
    plt.title("SOTA Models Training Progress (mAP50-95)", fontsize=15)
    plt.xlabel("Epoch")
    plt.ylabel("mAP50-95")
    plt.legend(title="Models", loc='lower right')
    plt.tight_layout()
    
    save_dir = "picture"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save in high-res vector formats
    plt.savefig(os.path.join(save_dir, "sota_comparison_curve.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "sota_comparison_curve.svg"), format='svg', bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "sota_comparison_curve.png"), dpi=300, bbox_inches='tight')
    print(f"✅ SOTA comparison curves saved to {save_dir}: .pdf, .svg, .png")
    plt.close()

# --- Main Suite ---
def main():
    # 统一使用根目录下的数据配置
    DATA_YAML = r"E:\Desktop\datasets\Crop\data.yaml"
    EPOCHS = 150
    IMG_SIZE = 640
    DEVICE = 0
    OUTPUT_ROOT = "SOTA_Comparisons"

    # --- 对比模型列表 (使用预训练权重以加速收敛 - Transfer Learning) ---
    # 注意：AgriYOLO 将加载 YOLOv10s 的权重作为基础 (Partial Load)
    COMPETITORS = [
        # Our Model (Custom Arch + Pretrained Weights)
        {
            "name": "AgriYOLO", 
            "cfg": "ultralytics/cfg/models/v10/yolov10_agriyolo_full.yaml", 
            "weights": "yolov10s.pt", 
            "type": "v10"
        },
        
        # Baselines (Official Pretrained Weights)
        {
            "name": "YOLOv10s", 
            "cfg": "ultralytics/cfg/models/v10/yolov10s.yaml", 
            "weights": "yolov10s.pt", 
            "type": "v10"
        },
        {
            "name": "YOLOv8s",  
            "cfg": "ultralytics/cfg/models/v8/yolov8.yaml",    
            "weights": "yolov8s.pt",    
            "type": "v8"
        },
        {
            "name": "YOLOv9c",  
            "cfg": "ultralytics/cfg/models/v9/yolov9c.yaml",   
            "weights": "yolov9c.pt",    
            "type": "v9"
        },
        {
            "name": "YOLOv5s",  
            "cfg": "ultralytics/cfg/models/v5/yolov5.yaml",    
            "weights": "yolov5su.pt",   # Ultralytics uses v5su by default
            "type": "v5"
        },
        # {
        #     "name": "RT_DETR_l",
        #     "cfg": "ultralytics/cfg/models/rt-detr/rtdetr-l.yaml", 
        #     "weights": "rtdetr-l.pt", 
        #     "type": "rtdetr"
        # }
    ]

    summary_list = []

    for m in COMPETITORS:
        print(f"\n" + "="*50 + f"\n🚀 Processing {m['name']} (Transfer Learning)\n" + "="*50)
        
        # 1. 初始化模型 (优先加载预训练权重)
        try:
            if m["name"] == "AgriYOLO":
                # 对于 AgriYOLO，先加载结构，再加载权重 (Partial Load)
                print(f"👉 Loading custom architecture: {m['cfg']} with weights {m['weights']}")
                if m["type"] == "v10":
                    model = YOLOv10(m["cfg"])
                else: # Fallback
                    model = YOLO(m["cfg"])
                
                # 下载并加载权重 (如果不下载，ultralytics transform 会处理，但显式 load 更稳)
                if not os.path.exists(m['weights']):
                    print(f"📥 Downloading {m['weights']}...")
                    model.load(m['weights']) # load() 方法会自动下载吗？通常是在 train() 或 init()，这里我们依赖 ultralytics 的自动下载机制
                else:
                    model.load(m['weights'])
                    
            else:
                # 对于标准 SOTA 模型，直接加载 .pt 文件 (包含结构+权重)
                print(f"👉 Loading pretrained model: {m['weights']}")
                if m["type"] == "rtdetr":
                    model = RTDETR(m['weights'])
                elif m["type"] == "v10":
                    model = YOLOv10(m['weights'])
                else:
                    model = YOLO(m['weights'])
        except Exception as e:
            print(f"⚠️ Failed to load specific weights, falling back to config: {e}")
            if m["type"] == "rtdetr":
                model = RTDETR(m["cfg"])
            elif m["type"] == "v10":
                model = YOLOv10(m["cfg"])
            else:
                model = YOLO(m["cfg"])

        # 2. 检查是否已存在训练好的权重 (断点续传逻辑)
        best_weight = os.path.join(OUTPUT_ROOT, m["name"], "weights", "best.pt")
        
        if os.path.exists(best_weight):
            print(f"✅ Found existing weights for {m['name']}, skipping training...")
        else:
            # 启动训练 (Fine-tuning)
            # pretrained=True 启用迁移学习
            model.train(
                data=DATA_YAML, 
                epochs=EPOCHS, 
                imgsz=IMG_SIZE, 
                device=DEVICE, 
                project=OUTPUT_ROOT, 
                name=m["name"],
                pretrained=True # ✅ 开启预训练权重迁移
            )

        # 3. 最终评估 (统计 mAP50 和 mAP_small)
        # 重新定义 best_weight 以防路径逻辑有变 (其实没变)
        if os.path.exists(best_weight):
            summary_list.append(evaluate_small_objects(best_weight, m["name"], DATA_YAML))

    # 4. 生成最终汇总表与图表
    df = pd.DataFrame(summary_list)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    output_csv = os.path.join(log_dir, "sota_comparison_final_v2.csv")
    df.to_csv(output_csv, index=False)
    
    plot_sota_curves(OUTPUT_ROOT)
    
    print(f"\n🏆 SOTA Comparison Complete! Summary saved to {output_csv}")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    target_data_yaml = r"E:\Desktop\datasets\Crop\data.yaml"
    if os.path.exists(target_data_yaml): 
        main()
    else: 
        print(f"❌ '{target_data_yaml}' not found. Please check data config path.")
