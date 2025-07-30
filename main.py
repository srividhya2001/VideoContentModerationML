import os
import torch
from model.cnn_extractor import CNNFeatureExtractor
from model.lstm_classifier import LSTMVideoClassifier
from utils.video_utils import extract_frames

def load_sequence_features(frame_dir, extractor):
    features = []
    frame_files = sorted(os.listdir(frame_dir))
    for frame_file in frame_files:
        feat = extractor.extract_features(os.path.join(frame_dir, frame_file))
        features.append(feat)
    return torch.tensor([features], dtype=torch.float32)  # Shape: (1, seq_len, 2048)

def main():
    video_path = "data/test2.mp4"
    frame_dir = "frames/"
    num_frames = extract_frames(video_path, frame_dir)
    print(f"Extracted {num_frames} frames.")

    extractor = CNNFeatureExtractor()
    model = LSTMVideoClassifier()
    model.eval()

    input_tensor = load_sequence_features(frame_dir, extractor)
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()

    print("Prediction:", "Safe ✅" if prediction == 0 else "Inappropriate ❌")

if __name__ == "__main__":
    main()
