import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (use a custom trained model for best results)
model = YOLO("yolov8n.pt")  # Replace with your own trained model if available

# Load video
video_path = "videos/IMG_1781.MP4"
cap = cv2.VideoCapture(video_path)

# Define virtual counting line (y-coordinates or x, depending on orientation)
line_offset = 10
count = 0
object_ids_crossed = set()

# Simple tracking state
tracked_objects = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_width=frame.shape[1]
    line_position = frame_width-100
    # Run detection
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
  # In production, use a tracker like Deep SORT

        # Store last known position
        if obj_id in tracked_objects:
            print("first loop")
            prev_cx = tracked_objects[obj_id]
            # Detect crossing from left to right
            if prev_cx < line_position and cx >= line_position:
                print("2nd loop")
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

    cv2.imshow("Package Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
