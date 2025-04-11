import cv2
import numpy as np
from ultralytics import YOLO
import pyautogui

screen_width, screen_height = pyautogui.size()

# Optional: Set a margin (so window doesn’t go edge-to-edge)
margin = 100
max_width = screen_width - margin
max_height = screen_height - margin

# Load YOLOv8 model (use a custom trained model for best results)
model = YOLO("yolov8n.pt")  # Replace with your own trained model if available

# Load video
video_path = "videos/IMG_1781.mp4"
cap = cv2.VideoCapture(video_path)

# Define virtual counting line (y-coordinates or x, depending on orientation)
line_offset = 10 # may remove
count = 0
object_ids_crossed = set()

# Simple tracking state
tracked_objects = {}

# Play/Pause state
is_playing = True

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global is_playing
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is within button area
        if 50 <= x <= 150 and 80 <= y <= 120:
            is_playing = not is_playing

# Create window and set mouse callback
cv2.namedWindow("Package Counter")
cv2.setMouseCallback("Package Counter", mouse_callback)

# Read first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()

while cap.isOpened():
    # Only read a new frame if we're playing
    if is_playing:
        ret, new_frame = cap.read()
        if not ret:
            break
        frame = new_frame

    frame_width=frame.shape[1]
    line_position = frame_width-100
    # Run detection (only when playing)
    if is_playing:
        results = model(frame, verbose=False)[0]
        detections = results.boxes.data.cpu().numpy()

        current_ids = {}

        for i, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
            if conf < 0.5:
                continue

            # For demo, draw boxes
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Assign a temporary object ID
            # Simple ID matching by proximity
            matched_id = None
            for tid, prev_cx in tracked_objects.items():
                if abs(cx - prev_cx) < 50:  # 50-pixel threshold
                    matched_id = tid
                    break

            if matched_id is None:
                matched_id = f"id_{len(tracked_objects)}"

            obj_id = matched_id

            # Store last known position
            if obj_id in tracked_objects:
                prev_cx = tracked_objects[obj_id]
                # Detect crossing from left to right
                if prev_cx < line_position and cx >= line_position:
                    print("detect object cross")
                    if obj_id not in object_ids_crossed:
                        count += 1
                        object_ids_crossed.add(obj_id)
                        print("added object", obj_id)
            tracked_objects[obj_id] = cx
            current_ids[obj_id] = True

            # Draw box and ID
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Clean up old tracked objects
        tracked_objects = {k: v for k, v in tracked_objects.items() if k in current_ids}

    # Draw counting line
    cv2.line(frame, (line_position, 0), (line_position, frame.shape[0]), (0, 255, 255), 2)

    # Show count
    cv2.putText(frame, f"Count: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Draw play/pause button
    button_color = (0, 0, 255) if is_playing else (0, 255, 0)
    cv2.rectangle(frame, (50, 80), (150, 120), button_color, -1)
    button_text = "Pause" if is_playing else "Play"
    cv2.putText(frame, button_text, (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    h, w = frame.shape[:2]

    # Compute scale that fits within max dimensions while preserving aspect ratio
    scale = min(max_width / w, max_height / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_frame = cv2.resize(frame, (new_w, new_h))

    cv2.imshow("Package Counter", resized_frame)
    key = cv2.waitKey(1) & 0xFF
    
    # Add keyboard controls as well
    if key == ord("q"):
        break
    elif key == ord("p") or key == ord(" "):  # 'p' or spacebar to toggle play/pause
        is_playing = not is_playing

cap.release()
cv2.destroyAllWindows()
