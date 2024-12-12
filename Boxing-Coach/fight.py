import cv2
import torch
from ultralytics import YOLO  # Import YOLO from ultralytics

# Load the YOLO model
model_path = "best.pt"
model = YOLO(model_path)

# Define the label mappings if needed
label_map = {0: "No Fight", 1: "Fight"}

# Capture video
cap = cv2.VideoCapture(0)  # Change to video file path if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)  # Directly pass frame for prediction

    # Parse results
    for result in results:
        for box in result.boxes:
            conf = box.conf.item()  # Confidence score
            cls = int(box.cls.item())  # Class label

            # Only consider "Fight" detections if needed
            label = label_map.get(cls, "Unknown")
            if label == "Fight":
                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Fight Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
