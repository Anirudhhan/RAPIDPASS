from ultralytics import YOLO
import cv2

# Load YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")  # lightweight & fast

# Video source (0 for webcam OR path to video)
cap = cv2.VideoCapture("traffic_video.mp4")  
# cap = cv2.VideoCapture(0)

# Vehicle classes from COCO dataset
VEHICLE_CLASSES = ["car", "bus", "truck", "van"]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])

            if class_name in VEHICLE_CLASSES and confidence > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_name.upper()} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Trigger message (NOT FINAL CONFIRMATION)
                cv2.putText(frame, "POTENTIAL EMERGENCY VEHICLE",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)

    cv2.imshow("Step 1 - Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
