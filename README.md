# 🏥 Dr. AI (v6.0): Dual-Brain Pneumonia Detector

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Medical AI](https://img.shields.io/badge/Medical-Imaging-green)

## 📖 Overview
**Dr. AI** is an advanced diagnostic system designed to detect Pneumonia from Chest X-Rays. It features a **Dual-Brain Architecture**:
1. **The Gatekeeper:** Rejects invalid images (bones, hands, objects).
2. **The Specialist:** Detects Pneumonia and explains decisions using Heatmaps (Grad-CAM).

## 🧠 Architecture
* **Model 1 (Gatekeeper):** ResNet18 trained on 4,000 images (Lungs vs Non-Lungs).
* **Model 2 (Specialist):** DenseNet121 trained on 5,856 Chest X-Rays (Kermany Dataset).

## 📊 Performance
* **Gatekeeper Accuracy:** ~99%
* **Specialist Accuracy:** ~94%

## 🚀 Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Download models from Releases.
3. Run the bot: `python src/bot.py`

## 🔗 Credits
Datasets by Paul Mooney (Lungs) and Tawsifur Rahman (Bones).
