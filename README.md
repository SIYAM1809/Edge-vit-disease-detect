# 🌱 Edge-ViT-FSL: Cross-Domain Crop Disease Diagnosis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Edge-ViT-FSL** is a resource-efficient AI system designed to identify crop diseases in "wild" environments. It utilizes a **MobileViT-XS** backbone enhanced with a **Saliency-Guided Attention Module (SGAM)**, achieving robust performance even when trained on limited laboratory data and tested on unseen field images.

## 🚀 Live Demo
**[Click Here to Launch the App](https://siyam1809-edge-vit-disease-detect-app-xr57lq.streamlit.app/)**

---

## 🖥️ Interface Preview
The system provides real-time inference with **Explainable AI (XAI)**, overlaying a saliency heatmap to visualize the specific lesions driving the diagnosis.

![App Interface Demo](App Demo/Agri_img1.png)
![App Interface Demo](App Demo/aAgri_img2.png)


---

## 🔑 Key Innovations

* **Cross-Domain Robustness:** Tackles the "Domain Shift" problem by training on high-quality lab images (*PlantVillage*) and generalizing to noisy, real-world field data (*CCMT*).
* **Saliency-Guided Attention:** A custom Sigmoid-based module spatially gates feature maps, forcing the model to focus on pathology (lesions/spots) rather than background noise.
* **Edge-Optimized:** Designed for CPU-only inference. The model fits under **10MB** and runs at **~22 FPS**, making it viable for mobile and IoT deployment in agriculture.
* **Few-Shot Learning:** Capable of generalizing to new disease classes with minimal support samples.

## 📊 Performance Metrics

| Metric | Result | Note |
| :--- | :--- | :--- |
| **Accuracy (Cross-Domain)** | **73.93%** | Tested on 600 episodes (5-way 5-shot) |
| **Inference Speed** | **22.13 FPS** | Standard CPU environment |
| **Latency** | **45.19 ms** | Per image |
| **Model Parameter Size** | **< 10 MB** | MobileViT-XS Backbone |

> *Note: While standard ResNet50 baselines drop to ~55% accuracy on this cross-domain task, Edge-ViT-FSL maintains robust performance due to feature alignment and saliency masking.*

## 🛠️ Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/Edge-ViT-FSL.git](https://github.com/your-username/Edge-ViT-FSL.git)
cd Edge-ViT-FSL
