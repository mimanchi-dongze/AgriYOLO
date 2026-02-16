# AgriYOLO: 任务驱动非对称轻量化特征融合网络

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**AgriYOLO 官方实现**  
*专为农作物微小病斑检测设计的目标检测模型*

🚀 [快速开始](#-快速开始) | 📊 [实验结果](#-实验结果) | 🏗️ [网络架构](#️-网络架构)

</div>

---

## 🌾 核心亮点

AgriYOLO 提出了 **TAL-FFN**（Task-driven Asymmetric Lightweight Feature Fusion Network，任务驱动非对称轻量化特征融合网络），这是一种专门针对复杂农业环境下微小病斑检测任务设计的创新架构。

**主要创新：**
- **🎯 ADSA（非对称深度分配策略）**：将更多计算资源分配给浅层特征（P2层），增强小目标检测能力
- **🔄 CADFM（上下文感知动态融合机制）**：内容驱动的自适应特征融合，根据输入动态调整融合权重
- **⚡ DSConv（深度可分离卷积）**：轻量化卷积模块，高效提取特征
- **🎨 SimAM（简单注意力模块）**：无参数注意力机制，增强特征表达能力

**性能表现：**
- **98.90% mAP50**，参数量相比 YOLOv10s 基线**减少 22.9%**
- 在 RTX 4090 上实时推理速度达 **71.7 FPS**
- 专为小目标优化（针对 < 32×32 像素的病斑）

---

## 📦 安装

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+（用于 GPU 加速）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/yourusername/AgriYOLO.git
cd AgriYOLO

# 安装依赖
pip install -r requirements.txt
```

---

## 🚀 快速开始

### 训练

```bash
# 在自定义数据集上训练 AgriYOLO
yolo train model=ultralytics/cfg/models/v10/yolov10s_TAL_FFN.yaml \
           data=your_dataset.yaml \
           epochs=300 \
           imgsz=640 \
           batch=16
```

### 推理

```bash
# 对图像进行推理
yolo predict model=weights/agriyolo.pt \
             source=path/to/images \
             conf=0.25 \
             imgsz=640

# 对视频进行推理
yolo predict model=weights/agriyolo.pt \
             source=path/to/video.mp4
```

### 验证

```bash
# 评估模型性能
yolo val model=weights/agriyolo.pt \
         data=your_dataset.yaml \
         imgsz=640
```

---

## 📊 实验结果

### 与主流算法对比

| 模型 | mAP50 (%) | mAP50-95 (%) | 参数量 (M) | GFLOPs | FPS |
|-------|-----------|--------------|------------|--------|-----|
| YOLOv5s | 98.29 | 95.52 | 2.65 | 3.92 | 163.9 |
| YOLOv8s | 97.40 | 94.55 | 3.16 | 4.43 | 149.2 |
| YOLOv9c | 98.19 | 96.10 | 25.59 | 52.01 | 60.0 |
| YOLOv10s | 98.73 | 95.71 | 8.09 | 12.44 | 82.4 |
| **AgriYOLO（本文）** | **98.90** | **95.95** | **6.24** | 21.17 | 71.7 |

### 消融实验

| 配置 | mAP50 (%) | 参数量 (M) | 说明 |
|--------------|-----------|------------|-------------|
| 基线（YOLOv10s + PANet） | 98.70 | 8.09 | 原始架构 |
| + ADSA（P2 检测头） | 98.80 | 6.50 | 添加 P2 高分辨率检测层 |
| + TAL-FFN | 98.80 | 6.42 | 用 TAL-FFN 替换 PANet |
| + SimAM（AgriYOLO 完整版） | **98.90** | **6.24** | 添加无参数注意力机制 |

---

## 🏗️ 网络架构

### TAL-FFN 总体结构

```
         ┌─────────────────────────────────────┐
         │  骨干网络（CSPDarknet + SimAM）    │
         └──┬──────┬──────┬──────┬────────────┘
            │      │      │      │
           C2     C3     C4     C5
           │      │      │      │
           └──────┴──────┴──────┘
                  │
          ┌───────▼───────┐
          │   TAL-FFN     │
          │ ┌───────────┐ │
          │ │   ADSA    │ │  ← 非对称深度分配
          │ │  (P2聚焦) │ │
          │ └─────┬─────┘ │
          │ ┌─────▼─────┐ │
          │ │  CADFM    │ │  ← 动态融合权重
          │ └─────┬─────┘ │
          │ ┌─────▼─────┐ │
          │ │  DSConv   │ │  ← 轻量化卷积
          │ └───────────┘ │
          └───────┬───────┘
                  │
          ┌───────▼───────┐
          │  SimAM + 检测 │
          │  (P2,P3,P4,P5)│
          └────────────────┘
```

详细架构图见 [svg/](svg/) 目录。

---

## 📁 项目结构

```
AgriYOLO/
├── ultralytics/
│   ├── nn/
│   │   └── modules/
│   │       └── dynamic_bifpn.py      # TAL-FFN 实现（ADSA + CADFM + DSConv）
│   └── cfg/
│       └── models/v10/
│           └── yolov10s_TAL_FFN.yaml # 模型配置文件
│
├── experiments/
│   ├── ablation_study.py             # 消融实验
│   ├── run_sota_comparison.py        # 主流算法对比
│   └── speed_benchmark.py            # 性能基准测试
│
├── visualize/
│   ├── plot_radar_chart.py           # 性能雷达图可视化
│   ├── plot_bubble_chart.py          # 速度-精度权衡气泡图
│   └── tal_ffn_visualizer.py         # 架构可视化工具
│
├── svg/                               # 架构图（SVG 格式）
│   ├── agriyolo_architecture.drawio.svg
│   ├── TAL_FFN_Internal_Mechanism.drawio.svg
│   ├── CADFM_Detail.drawio.svg
│   └── ADSA_Strategy.drawio.svg
│
├── requirements.txt
└── README.md
```

---

## 🔬 实验复现

### 运行消融实验

```bash
python experiments/ablation_study.py
```

### 复现主流算法对比

```bash
python experiments/run_sota_comparison.py --models yolov5s yolov8s yolov10s agriyolo
```

### 速度基准测试

```bash
python experiments/speed_benchmark.py --model weights/agriyolo.pt --device cuda:0
```

---

## 📈 可视化工具

### 生成性能图表

```bash
# 雷达图（多维性能对比）
python visualize/plot_radar_chart.py

# 气泡图（速度-精度-复杂度权衡）
python visualize/plot_bubble_chart.py

# 架构可视化
python visualize/tal_ffn_visualizer.py
```

---

## 🎓 引用

**论文状态**：准备投稿中

如果您在研究中使用了 AgriYOLO，请引用本仓库：

```bibtex
@misc{agriyolo2024,
  title={AgriYOLO: Task-driven Asymmetric Lightweight Feature Fusion Network for Tiny Crop Disease Lesion Detection},
  author={mimanchi-dongze},
  year={2024},
  howpublished={\url{https://github.com/mimanchi-dongze/AgriYOLO}},
  note={GitHub repository}
}
```

*论文发表后将更新正式引用格式。*

---

## 📄 开源协议

本项目基于 [MIT 协议](LICENSE) 开源。

---

## 🙏 致谢

本工作基于优秀的 [Ultralytics YOLOv10](https://github.com/THU-MIG/yolov10) 框架构建。感谢作者对目标检测领域的杰出贡献。

---

## 🤝 贡献

欢迎贡献代码！请随时提交 Pull Request。

---

## 📧 联系方式

如有问题或合作意向，请联系：
- **邮箱**: mimanchi-dongze@users.noreply.github.com
- **Issues**: [GitHub Issues](https://github.com/mimanchi-dongze/AgriYOLO/issues)

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给个 Star！**

</div>
