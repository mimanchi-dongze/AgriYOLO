# AgriYOLO: Task-driven Asymmetric Lightweight Feature Fusion Network

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Official Implementation of AgriYOLO**  
*A specialized object detection model for tiny crop disease lesion detection*

🚀 [Quick Start](#-quick-start) | 📊 [Results](#-results) | 🏗️ [Architecture](#️-architecture)

</div>

---

## 🌾 Highlights

AgriYOLO introduces **TAL-FFN** (Task-driven Asymmetric Lightweight Feature Fusion Network), an innovative architecture designed specifically for detecting tiny crop disease lesions in complex agricultural environments.

**Key Innovations:**
- **🎯 ADSA (Asymmetric Depth Allocation Strategy)**: Allocates more computational resources to shallow layers (P2) for enhanced small object detection
- **🔄 CADFM (Context-Aware Dynamic Fusion Mechanism)**: Adaptive feature fusion with content-driven weighting
- **⚡ DSConv (Depthwise Separable Convolution)**: Lightweight convolution blocks for efficient feature extraction
- **🎨 SimAM (Simple Attention Module)**: Parameter-free attention mechanism for enhanced feature expression

**Performance:**
- **98.90% mAP50** with **22.9% fewer parameters** than YOLOv10s baseline
- Real-time inference at **71.7 FPS** on RTX 4090
- Superior small object detection capability (optimized for lesions < 32×32 pixels)

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/AgriYOLO.git
cd AgriYOLO

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Training

```bash
# Train AgriYOLO on your custom dataset
yolo train model=ultralytics/cfg/models/v10/yolov10s_TAL_FFN.yaml \
           data=your_dataset.yaml \
           epochs=300 \
           imgsz=640 \
           batch=16
```

### Inference

```bash
# Run inference on images
yolo predict model=weights/agriyolo.pt \
             source=path/to/images \
             conf=0.25 \
             imgsz=640

# Run inference on video
yolo predict model=weights/agriyolo.pt \
             source=path/to/video.mp4
```

### Validation

```bash
# Evaluate model performance
yolo val model=weights/agriyolo.pt \
         data=your_dataset.yaml \
         imgsz=640
```

---

## 📊 Results

### Comparison with State-of-the-Art

| Model | mAP50 (%) | mAP50-95 (%) | Params (M) | GFLOPs | FPS |
|-------|-----------|--------------|------------|--------|-----|
| YOLOv5s | 98.29 | 95.52 | 2.65 | 3.92 | 163.9 |
| YOLOv8s | 97.40 | 94.55 | 3.16 | 4.43 | 149.2 |
| YOLOv9c | 98.19 | 96.10 | 25.59 | 52.01 | 60.0 |
| YOLOv10s | 98.73 | 95.71 | 8.09 | 12.44 | 82.4 |
| **AgriYOLO (Ours)** | **98.90** | **95.95** | **6.24** | 21.17 | 71.7 |

### Ablation Study

| Configuration | mAP50 (%) | Params (M) | Description |
|--------------|-----------|------------|-------------|
| Baseline (YOLOv10s + PANet) | 98.70 | 8.09 | Original architecture |
| + ADSA (P2 head) | 98.80 | 6.50 | Add P2 detection head |
| + TAL-FFN | 98.80 | 6.42 | Replace PANet with TAL-FFN |
| + SimAM (AgriYOLO) | **98.90** | **6.24** | Final model with attention |

---

## 🏗️ Architecture

### TAL-FFN Overview

```
         ┌─────────────────────────────────────┐
         │    Backbone (CSPDarknet + SimAM)   │
         └──┬──────┬──────┬──────┬────────────┘
            │      │      │      │
           C2     C3     C4     C5
           │      │      │      │
           └──────┴──────┴──────┘
                  │
          ┌───────▼───────┐
          │   TAL-FFN     │
          │ ┌───────────┐ │
          │ │   ADSA    │ │  ← Asymmetric depth allocation
          │ │  (P2 focus)│ │
          │ └─────┬─────┘ │
          │ ┌─────▼─────┐ │
          │ │  CADFM    │ │  ← Dynamic fusion weights
          │ └─────┬─────┘ │
          │ ┌─────▼─────┐ │
          │ │  DSConv   │ │  ← Lightweight convolution
          │ └───────────┘ │
          └───────┬───────┘
                  │
          ┌───────▼───────┐
          │  SimAM + Detect│
          │  (P2,P3,P4,P5) │
          └────────────────┘
```

For detailed architecture diagrams, see [svg/](svg/).

---

## 📁 Project Structure

```
AgriYOLO/
├── ultralytics/
│   ├── nn/
│   │   └── modules/
│   │       └── dynamic_bifpn.py      # TAL-FFN implementation (ADSA + CADFM + DSConv)
│   └── cfg/
│       └── models/v10/
│           └── yolov10s_TAL_FFN.yaml # Model configuration
│
├── experiments/
│   ├── ablation_study.py             # Ablation experiments
│   ├── run_sota_comparison.py        # SOTA comparison
│   └── speed_benchmark.py            # Performance benchmarking
│
├── visualize/
│   ├── plot_radar_chart.py           # Performance visualization
│   ├── plot_bubble_chart.py          # Speed-accuracy trade-off
│   └── tal_ffn_visualizer.py         # Architecture visualization
│
├── svg/                               # Architecture diagrams
│   ├── agriyolo_architecture.drawio.svg
│   ├── TAL_FFN_Internal_Mechanism.drawio.svg
│   ├── CADFM_Detail.drawio.svg
│   └── ADSA_Strategy.drawio.svg
│
├── requirements.txt
└── README.md
```

---

## 🔬 Experiments

### Run Ablation Study

```bash
python experiments/ablation_study.py
```

### Reproduce SOTA Comparison

```bash
python experiments/run_sota_comparison.py --models yolov5s yolov8s yolov10s agriyolo
```

### Speed Benchmark

```bash
python experiments/speed_benchmark.py --model weights/agriyolo.pt --device cuda:0
```

---

## 📈 Visualization

### Generate Performance Plots

```bash
# Radar chart (multi-dimensional performance)
python visualize/plot_radar_chart.py

# Bubble chart (speed-accuracy-complexity trade-off)
python visualize/plot_bubble_chart.py

# Architecture visualization
python visualize/tal_ffn_visualizer.py
```

---

## 🎓 Citation

**Paper status**: Under preparation

If you use AgriYOLO in your research, please cite this repository:

```bibtex
@misc{agriyolo2024,
  title={AgriYOLO: Task-driven Asymmetric Lightweight Feature Fusion Network for Tiny Crop Disease Lesion Detection},
  author={mimanchi-dongze},
  year={2024},
  howpublished={\url{https://github.com/mimanchi-dongze/AgriYOLO}},
  note={GitHub repository}
}
```

*BibTeX will be updated once the paper is published.*

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

This work is built upon the excellent [Ultralytics YOLOv10](https://github.com/THU-MIG/yolov10) framework. We thank the authors for their outstanding contributions to the object detection community.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or collaborations, please contact:
- **Email**: mimanchi-dongze@users.noreply.github.com
- **Issues**: [GitHub Issues](https://github.com/mimanchi-dongze/AgriYOLO/issues)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

</div>
