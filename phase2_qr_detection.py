import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load YOLO model
model = YOLO("yolov8n.pt")

# QR Detector
qr_detector = cv2.QRCodeDetector()

# Video input
cap = cv2.VideoCapture("traffic_video_qr.mp4")

# Temporal tracking for siren detection (reduces false positives)
siren_history = defaultdict(list)
HISTORY_LENGTH = 5  # Number of frames to track

# Permanent emergency vehicle registry
verified_emergency_vehicles = {}  # vehicle_id: {'count': confirmations, 'last_seen': frame_number}
CONFIRMATION_THRESHOLD = 3  # Need 3 confirmations to mark as emergency permanently
EXPIRY_FRAMES = 30  # Remove vehicle from registry if not seen for 30 frames (~1 second at 30fps)


def detect_siren_light(roi, vehicle_id):
    """
    Detect red/blue flashing siren lights using HSV color thresholding
    with temporal consistency and balanced sensitivity
    """
    if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
        return False

    # Focus on top portion of vehicle (where sirens are typically located)
    height = roi.shape[0]
    top_roi = roi[0:int(height * 0.5), :]  # Top 50% of vehicle

    if top_roi.size == 0:
        return False

    hsv = cv2.cvtColor(top_roi, cv2.COLOR_BGR2HSV)

    # Balanced red color range
    red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))

    # Balanced blue color range
    blue = cv2.inRange(hsv, (100, 120, 80), (140, 255, 255))

    siren_mask = red1 + red2 + blue
    siren_pixels = cv2.countNonZero(siren_mask)

    # Calculate percentage of bright pixels in ROI
    total_pixels = top_roi.shape[0] * top_roi.shape[1]
    siren_percentage = (siren_pixels / total_pixels) * 100

    # Balanced thresholds - more lenient
    detected = siren_pixels > 100 and siren_percentage > 1.0
    
    # Add to history
    siren_history[vehicle_id].append(detected)
    if len(siren_history[vehicle_id]) > HISTORY_LENGTH:
        siren_history[vehicle_id].pop(0)
    
    # More lenient temporal requirement
    if len(siren_history[vehicle_id]) >= 2:
        recent_detections = sum(siren_history[vehicle_id][-2:])
        return recent_detections >= 1  # At least 1 out of last 2 frames
    
    return detected  # Return immediate detection if not enough history


def get_vehicle_id(box):
    """Generate a more stable ID based on bounding box center with larger tolerance"""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    # Round to nearest 100 pixels for better tracking tolerance
    return f"{center_x // 100}_{center_y // 100}"


frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame)
    global_emergency_detected = False
    
    # Track which vehicles are seen in current frame
    current_frame_vehicles = set()

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            # Higher confidence threshold
            if label in ["car", "truck", "bus"] and conf > 0.6:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Ensure bounding box is valid
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                vehicle_roi = frame[y1:y2, x1:x2]

                if vehicle_roi.size == 0:
                    continue

                is_emergency_vehicle = False
                siren_detected = False
                vehicle_id = get_vehicle_id(box)
                
                # Mark this vehicle as seen in current frame
                current_frame_vehicles.add(vehicle_id)

                # Check if this vehicle is already permanently verified
                if vehicle_id in verified_emergency_vehicles:
                    # Update last seen frame
                    verified_emergency_vehicles[vehicle_id]['last_seen'] = frame_count
                    
                    is_emergency_vehicle = True
                    global_emergency_detected = True
                    
                    cv2.putText(
                        frame,
                        "VERIFIED EMERGENCY VEHICLE",
                        (x1, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"Confirmations: {verified_emergency_vehicles[vehicle_id]['count']}",
                        (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )
                else:
                    # Step 1: QR Code detection (primary check)
                    qr_data, _, _ = qr_detector.detectAndDecode(vehicle_roi)

                    # Step 2: Only check siren if QR code is detected
                    if qr_data:
                        # QR detected - this is potentially an emergency vehicle
                        # Now check if siren is active
                        siren_detected = detect_siren_light(vehicle_roi, vehicle_id)
                        
                        if siren_detected:
                            # Both QR and siren confirmed
                            # Increment confirmation counter
                            if vehicle_id not in verified_emergency_vehicles:
                                verified_emergency_vehicles[vehicle_id] = {
                                    'count': 0,
                                    'last_seen': frame_count
                                }
                            
                            verified_emergency_vehicles[vehicle_id]['count'] += 1
                            verified_emergency_vehicles[vehicle_id]['last_seen'] = frame_count
                            
                            # Check if reached confirmation threshold
                            if verified_emergency_vehicles[vehicle_id]['count'] >= CONFIRMATION_THRESHOLD:
                                is_emergency_vehicle = True
                                global_emergency_detected = True
                                
                                cv2.putText(
                                    frame,
                                    "EMERGENCY VERIFIED - PERMANENT",
                                    (x1, y1 - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 0, 255),
                                    2
                                )
                            else:
                                # Still building confirmations
                                cv2.putText(
                                    frame,
                                    f"VERIFYING... ({verified_emergency_vehicles[vehicle_id]['count']}/{CONFIRMATION_THRESHOLD})",
                                    (x1, y1 - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 165, 255),  # Orange
                                    2
                                )
                            
                            cv2.putText(
                                frame,
                                "SIREN ACTIVE",
                                (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 0, 0),
                                2
                            )
                        else:
                            # QR detected but no siren - show warning
                            cv2.putText(
                                frame,
                                "QR DETECTED - NO SIREN",
                                (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 165, 255),  # Orange
                                2
                            )

                # Bounding box color
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
    
    # Clean up expired vehicles (not seen for EXPIRY_FRAMES)
    expired_vehicles = []
    for vehicle_id, data in verified_emergency_vehicles.items():
        if frame_count - data['last_seen'] > EXPIRY_FRAMES:
            expired_vehicles.append(vehicle_id)
    
    for vehicle_id in expired_vehicles:
        del verified_emergency_vehicles[vehicle_id]
        if vehicle_id in siren_history:
            del siren_history[vehicle_id]

    # Global emergency status
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
    else:
        cv2.putText(
            frame,
            "STATUS: NORMAL TRAFFIC",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

    cv2.imshow("Emergency Vehicle Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()