import cv2
from ultralytics import YOLO

# Load your model
model = YOLO("yolov8n.pt")

# Print all class names
class_names = model.names
print("Model classes:")
for class_id, class_name in class_names.items():
    print(f"{class_id}: {class_name}")

# Check for package-related classes
package_related = ['box', 'suitcase', 'backpack', 'handbag', 'cardboard', 'package']
found_classes = [name for id, name in class_names.items() 
                if any(pkg in name.lower() for pkg in package_related)]

print("\nPackage-related classes found:")
print(found_classes if found_classes else "No package-related classes found")