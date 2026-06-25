# Fire Detection System

A deep learning-based fire detection system comparing **YOLOv8n**, **YOLOv11m**, and **RT-DETR-L** for real-time fire and smoke detection, with a full Flask web application supporting image, video, and live camera detection.

## Overview

This project builds and evaluates three object detection models on a curated fire/smoke dataset, then deploys the best-performing models into a web-based detection system. The goal was to compare detection accuracy, robustness, and real-time performance to identify the optimal model for different deployment scenarios (edge devices vs. server-grade hardware).

## Dataset

- **19,363** raw images collected from public datasets (17,363 images) and targeted web crawling (2,000 images covering low-light, urban neon interference, and cloud/sunset false-positive scenarios)
- After cleaning (perceptual hashing for deduplication, removal of corrupted/irrelevant samples, semantic relabeling of false positives like lanterns and traffic lights), the final dataset contains **17,653 images**
- Split: 14,209 train / 1,722 validation / 1,722 test (8:1:1)
- Preprocessing: format standardization (JPEG conversion), Letterbox resizing to 640×640, CLAHE contrast enhancement, HSV augmentation

**Dataset credit**: Built primarily from public fire/smoke detection datasets available on GitHub/Kaggle, supplemented with web-crawled images for edge cases. See `Acknowledgements` below.

## Models & Results

All models trained on **NVIDIA Tesla T4** with a 200-epoch budget, SGD optimizer with cosine annealing (lr=0.01), input size 640×640. YOLOv11m and RT-DETR-L used early stopping to prevent overfitting (best checkpoints selected based on validation performance); YOLOv8n trained for the full 200 epochs.

| Model | Params | GFLOPs | FPS | mAP50 (test) | mAP50 (noisy, σ=10) |
|---|---|---|---|---|---|
| **YOLOv8n** | 3.0M | 8.1 | **222.2** | 0.775 | 0.687 |
| **YOLOv11m** | 20.0M | 67.7 | 52.6 | 0.782 | 0.720 |
| **RT-DETR-L** | 32M | 103.4 | 24.6 | 0.778 | 0.700 |

**Key findings**:
- **YOLOv8n** wins on speed (222 FPS) and deployability — best fit for edge devices (Jetson Nano, Raspberry Pi) and real-time monitoring where latency matters most
- **YOLOv11m** wins on accuracy and noise robustness (smallest accuracy drop under Gaussian noise) — best fit for scenarios with adequate compute where precision matters most
- **RT-DETR-L** offers transformer-based global attention, useful for capturing long-range spatial context (e.g. diffuse smoke), at the cost of inference speed

### Model Weights

- **YOLOv8n**: included directly in this repository (`weights/`)
- **YOLOv11m** (188MB, exceeds GitHub's 100MB file limit): [Download from Google Drive](https://drive.google.com/drive/folders/16JnnlR3jzfvokBq3LbmK4xv4V0-jaFL6?usp=sharing)
- **RT-DETR-L** (exceeds GitHub's 100MB file limit): [Download from Google Drive](https://drive.google.com/drive/folders/1MHZS9m_OgeW2l2ZmEas8AIumlTcuB3N6?usp=sharing)

## System Features

Built with **Python + Flask**, the web application includes:

- **Image detection** — batch upload and analysis of static images (default model: YOLOv11m, prioritizing accuracy)
- **Video detection** — frame-by-frame analysis of uploaded video files with async frontend rendering via Canvas overlay
- **Live camera detection** — real-time webcam monitoring (default model: YOLOv8n, prioritizing speed) with browser desktop notifications on fire detection (30s anti-spam cooldown)
- Invite-code based registration system (designed for deployment in controlled/institutional environments)

## Tech Stack

- **Models**: YOLOv8n, YOLOv11m, RT-DETR-L (Ultralytics)
- **Backend**: Python, Flask, OpenCV
- **Frontend**: HTML5, JavaScript, Canvas API
- **Training hardware**: NVIDIA Tesla T4

## Acknowledgements

- Public fire/smoke detection datasets sourced from open repositories (GitHub/Kaggle community datasets)
- Pretrained YOLOv11n weights (Ultralytics) used for semi-automated annotation of crawled images
- Built upon the Ultralytics YOLO framework and the RT-DETR architecture

## License

MIT License — see [LICENSE](LICENSE) for details.

This is an academic/portfolio project. Dataset credits belong to their original authors; only the training code, evaluation pipeline, and web application in this repository are original work by the author.

---

# 火灾检测系统

一个基于深度学习的火灾检测系统,对比了 **YOLOv8n**、**YOLOv11m** 和 **RT-DETR-L** 三种模型在实时火灾与烟雾检测任务上的表现,并配套实现了完整的 Flask 网页应用,支持图片检测、视频检测和摄像头实时检测。

## 项目概述

本项目在自建的火灾/烟雾数据集上训练并评估了三种目标检测模型,对比检测精度、鲁棒性和实时性能,从而为不同部署场景(边缘设备 vs 服务器级硬件)选出最优模型,并将最终模型部署进一套完整的网页检测系统。

## 数据集

- 累计采集 **19,363** 张原始图像,其中 **17,363** 张来自公开数据集,**2,000** 张通过网络爬虫定向采集(覆盖夜间暗光、城市霓虹灯干扰、火烧云等容易引起误判的场景)
- 经过数据清洗(感知哈希去重、剔除损坏/无关样本、语义纠偏剔除红灯笼/交通灯等误标样本)后,最终数据集为 **17,653** 张图像
- 按 8:1:1 划分:训练集 14,209 张 / 验证集 1,722 张 / 测试集 1,722 张
- 预处理:格式统一转码为JPEG、Letterbox等比例缩放至640×640、CLAHE对比度增强、HSV色彩增强

**数据集来源说明**:数据集主要基于GitHub/Kaggle上的公开火灾检测数据集构建,并补充了网络爬取的边缘场景图像,详见下方"致谢"部分。

## 模型与结果

所有模型均在 **NVIDIA Tesla T4** 上训练,配置200轮训练上限,SGD优化器配合余弦退火学习率(初始lr=0.01),输入尺寸640×640。其中YOLOv11m和RT-DETR-L采用了早停机制以避免过拟合(根据验证集表现选取最优权重);YOLOv8n则完整训练了200轮。

| 模型 | 参数量 | GFLOPs | FPS | mAP50(测试集) | mAP50(加噪σ=10) |
|---|---|---|---|---|---|
| **YOLOv8n** | 3.0M | 8.1 | **222.2** | 0.775 | 0.687 |
| **YOLOv11m** | 20.0M | 67.7 | 52.6 | 0.782 | 0.720 |
| **RT-DETR-L** | 32M | 103.4 | 24.6 | 0.778 | 0.700 |

**核心结论**:
- **YOLOv8n** 速度最快(222 FPS)、部署成本最低,最适合边缘设备(Jetson Nano、树莓派等)和对实时性要求高的监测场景
- **YOLOv11m** 精度最高、抗噪鲁棒性最强(加噪后精度衰减最小),最适合算力充足、对精度要求高的场景
- **RT-DETR-L** 基于Transformer的全局注意力机制,擅长捕捉烟雾弥散这类长程空间特征,但推理速度是三者中最慢的

### 模型权重下载

- **YOLOv8n**:直接包含在本仓库中(`weights/`目录)
- **YOLOv11m**(188MB,超过GitHub单文件100MB限制):[从Google Drive下载](https://drive.google.com/drive/folders/16JnnlR3jzfvokBq3LbmK4xv4V0-jaFL6?usp=sharing)
- **RT-DETR-L**(超过GitHub单文件100MB限制):[从Google Drive下载](https://drive.google.com/drive/folders/1MHZS9m_OgeW2l2ZmEas8AIumlTcuB3N6?usp=sharing)

## 系统功能

基于 **Python + Flask** 搭建,网页应用包含:

- **图片检测** —— 支持批量上传静态图片进行分析(默认加载YOLOv11m,优先保证精度)
- **视频检测** —— 对上传的视频文件逐帧分析,前端通过Canvas异步叠加渲染检测框
- **摄像头实时检测** —— 实时监控摄像头画面(默认加载YOLOv8n,优先保证速度),检测到火情后通过浏览器桌面通知推送告警(设有30秒防重复提醒机制)
- 基于邀请码的注册登录机制(适配机构内部部署场景)

## 技术栈

- **模型**:YOLOv8n、YOLOv11m、RT-DETR-L(基于Ultralytics框架)
- **后端**:Python、Flask、OpenCV
- **前端**:HTML5、JavaScript、Canvas API
- **训练硬件**:NVIDIA Tesla T4

## 致谢

- 公开火灾/烟雾检测数据集来源于GitHub/Kaggle社区
- 使用了Ultralytics提供的YOLOv11n预训练权重,用于对爬取图像进行半自动化标注
- 项目基于Ultralytics YOLO框架与RT-DETR架构构建

## 许可证

MIT License —— 详见 [LICENSE](LICENSE) 文件。

本项目为学术/作品展示性质项目。数据集版权归原作者所有,本仓库中的训练代码、评估流程和网页应用部分为原创工作。
