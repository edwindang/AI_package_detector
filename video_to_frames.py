import cv2
import os

video_path = 'videos/warehouse_1.mp4'
output_dir = 'frames_output'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_rate = 50  # Extract every 5th frame

frame_id = 0
saved_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_rate == 0:
        cv2.imwrite(f"{output_dir}/frame_{saved_id:05d}.jpg", frame)
        saved_id += 1

    frame_id += 1

cap.release()

# You’ll now have a folder of images like: frame_00001.jpg, frame_00002.jpg, etc.