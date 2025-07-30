
# 🎥 AI-Based Video Content Moderation System

This project implements an AI-powered video content moderation pipeline using deep learning techniques. It automatically analyzes videos, extracts features from each frame using a pretrained **ResNet-50 CNN**, and uses an **LSTM** (Long Short-Term Memory) model to learn temporal patterns across frames to classify the video as **Safe** or **Inappropriate**.

---

## 📌 Features

- 🔍 Frame-wise feature extraction using **ResNet-50**
- 🧠 Temporal sequence modeling using **LSTM**
- ✅ Binary classification: Safe (`0`) vs Inappropriate (`1`)
- 🧪 Modular, extensible design for adding your own training datasets
- 📈 Reduces manual review effort and improves policy compliance

---

## 🗂️ Project Structure

```
video_moderation/
│
├── data/                      # Input videos (.mp4)
├── frames/                    # Extracted frames from video
│
├── model/
│   ├── cnn_extractor.py       # ResNet50-based feature extractor
│   └── lstm_classifier.py     # LSTM for temporal classification
│
├── utils/
│   ├── video_utils.py         # Frame extraction helper
│   └── visualize.py           # Optional: plot confidence scores
│
├── main.py                    # Inference pipeline (no training yet)
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation
```

---

## 📽️ How It Works

### ➤ Data Flow Diagram

```
   Input Video
       │
[Frame Extraction: 1 fps]
       │
[ResNet-50 CNN]  → 2048-D feature per frame
       │
[Sequence of Frame Features]
       │
[LSTM Model] → Learns temporal behavior
       │
[Classifier] → Safe or Inappropriate
```

---

## 🔧 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Required Python packages:
- `torch`, `torchvision`
- `opencv-python`
- `numpy`
- `matplotlib`
- `tqdm`

---

## 🚀 How to Run

1. Place a sample video inside the `data/` folder:
   ```bash
   data/sample_sports_video.mp4
   ```

2. Run the main pipeline:
   ```bash
   python main.py
   ```

3. Output:
   - Extracts 1 frame/sec
   - Prints model prediction: **Safe ✅** or **Inappropriate ❌**

---

## 🧠 What’s Special About ResNet-50?

- Converts each video frame into a **2048-dimensional feature vector**
- Captures semantic features like objects, people, scenes
- Acts as a **feature extractor** for the LSTM
- Can be replaced with other CNNs (EfficientNet, DINO, etc.)

---

## 📦 Next Steps (Future Work)

- [ ] Integrate live webcam or video streaming input
- [ ] Support multi-class moderation categories
- [ ] Build a web API using FastAPI or Flask

---

## 📚 Related Concepts

- CNN (ResNet): For spatial understanding of individual frames
- LSTM: For learning temporal patterns across video frames
- Feature extraction vs classification
- Transfer learning and sequence modeling
- Safe content detection in sports / general video moderation

---

## 🤖 Authors

- Developed by SRI VIDHYA
- Designed for ML video moderation research and prototyping

---

## 📄 License

MIT License — feel free to use and modify for educational or commercial use.

---
