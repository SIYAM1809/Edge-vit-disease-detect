🌱 Edge-ViT-FSL: Cross-Domain Crop Disease Diagnosis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Edge-ViT-FSL** is a resource-efficient AI system designed to identify crop diseases in "wild" environments. It utilizes a **MobileViT-XS** backbone enhanced with a **Saliency-Guided Attention Module**, achieving robust performance even when trained on limited laboratory data.

## 🚀 Key Features
* **Cross-Domain Robustness:** Trained on *PlantVillage* (Lab), tested on *CCMT* (Wild). Achieves **73.93% Accuracy** on unseen domains.
* **Edge-Optimized:** Runs purely on CPU with **~45ms latency** (22 FPS).
* **Explainable AI:** Generates real-time Saliency Heatmaps to pinpoint disease lesions.
* **Few-Shot Learning:** Capable of generalizing from minimal examples.

## 📊 Performance
| Metric | Result |
| :--- | :--- |
| **Accuracy (Cross-Domain)** | **73.93%** |
| **Inference Speed (CPU)** | **22.13 FPS** |
| **Model Size** | **< 10 MB** |

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/Edge-ViT-FSL.git](https://github.com/your-username/Edge-ViT-FSL.git)
   cd Edge-ViT-FSL
