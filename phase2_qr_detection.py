import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
qr_detector = cv2.QRCodeDetector()

cap = cv2.VideoCapture("traffic_video_qr.mp4")

global_emergency_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    global_emergency_detected = False

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if label in ["car", "truck", "bus"] and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                vehicle_roi = frame[y1:y2, x1:x2]
                if vehicle_roi.size == 0:
                    continue

                is_emergency_vehicle = False

                data, bbox, _ = qr_detector.detectAndDecode(vehicle_roi)

                if data:
                    is_emergency_vehicle = True
                    global_emergency_detected = True

                    cv2.putText(
                        frame,
                        "EMERGENCY VEHICLE VERIFIED",
                        (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                color = (0, 0, 255) if is_emergency_vehicle else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

    if global_emergency_detected:
        cv2.putText(
            frame,
            "STATUS: EMERGENCY DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Phase 2 - Emergency Vehicle Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
