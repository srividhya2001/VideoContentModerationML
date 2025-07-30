import cv2
import os

def extract_frames(video_path, output_dir, fps=1):
    """
    Extracts frames from the video at 1 frame per second.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % int(cap.get(cv2.CAP_PROP_FPS)) == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1
    cap.release()
    return saved
