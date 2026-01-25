import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

# Load YOLO model
model = YOLO("yolov8n.pt")

# QR Detector
qr_detector = cv2.QRCodeDetector()

# Video input
cap = cv2.VideoCapture("traffic_video_qr.mp4")

# Temporal tracking for siren detection
siren_history = defaultdict(list)
HISTORY_LENGTH = 5

# Permanent emergency vehicle registry
verified_emergency_vehicles = {}
CONFIRMATION_THRESHOLD = 3
EXPIRY_FRAMES = 30

# ==================== PHASE 3: TRAFFIC SIGNAL CONTROLLER ====================
class TrafficSignalController:
    def __init__(self):
        self.current_state = "RED"  # Start with RED and stay RED until emergency
        self.emergency_mode = False
        self.emergency_start_time = None
        self.emergency_duration = 15  # seconds to hold green for emergency
        self.state_start_time = time.time()
        self.emergency_cleared = False
        
    def update(self, emergency_detected):
        """Update traffic signal state based on emergency detection"""
        current_time = time.time()
        
        if emergency_detected and not self.emergency_mode:
            # Emergency vehicle detected - switch to GREEN immediately
            self.emergency_mode = True
            self.emergency_start_time = current_time
            self.current_state = "GREEN"
            self.state_start_time = current_time
            self.emergency_cleared = False
            return True  # Signal changed
            
        elif self.emergency_mode:
            # In emergency mode
            elapsed = current_time - self.emergency_start_time
            
            if elapsed >= self.emergency_duration:
                # Emergency duration expired, return to RED
                self.emergency_mode = False
                self.emergency_cleared = True
                self.current_state = "YELLOW"
                self.state_start_time = current_time
                return True
            else:
                # Keep green for emergency vehicle
                return False
                
        elif self.emergency_cleared:
            # After emergency, transition YELLOW → RED and stay RED
            elapsed = current_time - self.state_start_time
            
            if self.current_state == "YELLOW" and elapsed >= 3:
                self.current_state = "RED"
                self.state_start_time = current_time
                self.emergency_cleared = False  # Reset flag
                return True
                
        # Otherwise stay RED (waiting for next emergency)
        return False
    
    def get_signal_color(self):
        """Return BGR color tuple for current signal"""
        colors = {
            "RED": (0, 0, 255),
            "YELLOW": (0, 255, 255),
            "GREEN": (0, 255, 0)
        }
        return colors[self.current_state]
    
    def get_status_text(self):
        """Return status description"""
        if self.emergency_mode:
            remaining = max(0, self.emergency_duration - (time.time() - self.emergency_start_time))
            return f"EMERGENCY MODE - {remaining:.1f}s remaining"
        elif self.emergency_cleared and self.current_state == "YELLOW":
            return "CLEARING EMERGENCY - RETURNING TO RED"
        else:
            return "WAITING FOR EMERGENCY VEHICLE"

# Initialize traffic controller
traffic_controller = TrafficSignalController()

# ==================== PHASE 2: EMERGENCY DETECTION FUNCTIONS ====================
def detect_siren_light(roi, vehicle_id):
    """Detect red/blue flashing siren lights"""
    if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
        return False

    height = roi.shape[0]
    top_roi = roi[0:int(height * 0.5), :]

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

    total_pixels = top_roi.shape[0] * top_roi.shape[1]
    siren_percentage = (siren_pixels / total_pixels) * 100

    detected = siren_pixels > 100 and siren_percentage > 1.0
    
    siren_history[vehicle_id].append(detected)
    if len(siren_history[vehicle_id]) > HISTORY_LENGTH:
        siren_history[vehicle_id].pop(0)
    
    if len(siren_history[vehicle_id]) >= 2:
        recent_detections = sum(siren_history[vehicle_id][-2:])
        return recent_detections >= 1
    
    return detected


def get_vehicle_id(box):
    """Generate stable ID based on bounding box center"""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return f"{center_x // 100}_{center_y // 100}"


def draw_traffic_signal(frame, controller):
    """Draw traffic signal display on frame"""
    # Signal box position (top-right corner)
    signal_x = frame.shape[1] - 200
    signal_y = 20
    
    # Background box
    cv2.rectangle(frame, (signal_x - 10, signal_y - 10), 
                  (signal_x + 170, signal_y + 140), (50, 50, 50), -1)
    cv2.rectangle(frame, (signal_x - 10, signal_y - 10), 
                  (signal_x + 170, signal_y + 140), (255, 255, 255), 2)
    
    # Title
    cv2.putText(frame, "TRAFFIC SIGNAL", (signal_x, signal_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw three lights
    light_radius = 20
    light_x = signal_x + 80
    
    # Red light
    red_color = (0, 0, 255) if controller.current_state == "RED" else (50, 50, 50)
    cv2.circle(frame, (light_x, signal_y + 50), light_radius, red_color, -1)
    cv2.circle(frame, (light_x, signal_y + 50), light_radius, (255, 255, 255), 2)
    
    # Yellow light
    yellow_color = (0, 255, 255) if controller.current_state == "YELLOW" else (50, 50, 50)
    cv2.circle(frame, (light_x, signal_y + 85), light_radius, yellow_color, -1)
    cv2.circle(frame, (light_x, signal_y + 85), light_radius, (255, 255, 255), 2)
    
    # Green light
    green_color = (0, 255, 0) if controller.current_state == "GREEN" else (50, 50, 50)
    cv2.circle(frame, (light_x, signal_y + 120), light_radius, green_color, -1)
    cv2.circle(frame, (light_x, signal_y + 120), light_radius, (255, 255, 255), 2)


# ==================== MAIN LOOP ====================
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame)
    global_emergency_detected = False
    
    current_frame_vehicles = set()

    # ========== PHASE 2: Emergency Vehicle Detection ==========
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if label in ["car", "truck", "bus"] and conf > 0.6:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                vehicle_roi = frame[y1:y2, x1:x2]

                if vehicle_roi.size == 0:
                    continue

                is_emergency_vehicle = False
                siren_detected = False
                vehicle_id = get_vehicle_id(box)
                
                current_frame_vehicles.add(vehicle_id)

                # Check if already verified
                if vehicle_id in verified_emergency_vehicles:
                    verified_emergency_vehicles[vehicle_id]['last_seen'] = frame_count
                    
                    is_emergency_vehicle = True
                    global_emergency_detected = True
                    
                    cv2.putText(frame, "VERIFIED EMERGENCY VEHICLE", (x1, y1 - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Confirmations: {verified_emergency_vehicles[vehicle_id]['count']}", 
                                (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    # QR Code detection
                    qr_data, _, _ = qr_detector.detectAndDecode(vehicle_roi)

                    if qr_data:
                        # Siren detection
                        siren_detected = detect_siren_light(vehicle_roi, vehicle_id)
                        
                        if siren_detected:
                            if vehicle_id not in verified_emergency_vehicles:
                                verified_emergency_vehicles[vehicle_id] = {
                                    'count': 0,
                                    'last_seen': frame_count
                                }
                            
                            verified_emergency_vehicles[vehicle_id]['count'] += 1
                            verified_emergency_vehicles[vehicle_id]['last_seen'] = frame_count
                            
                            if verified_emergency_vehicles[vehicle_id]['count'] >= CONFIRMATION_THRESHOLD:
                                is_emergency_vehicle = True
                                global_emergency_detected = True
                                
                                cv2.putText(frame, "EMERGENCY VERIFIED - PERMANENT", (x1, y1 - 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            else:
                                cv2.putText(frame, 
                                    f"VERIFYING... ({verified_emergency_vehicles[vehicle_id]['count']}/{CONFIRMATION_THRESHOLD})",
                                    (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                            
                            cv2.putText(frame, "SIREN ACTIVE", (x1, y1 - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        else:
                            cv2.putText(frame, "QR DETECTED - NO SIREN", (x1, y1 - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                # Draw bounding box
                color = (0, 0, 255) if is_emergency_vehicle else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Clean up expired vehicles
    expired_vehicles = [vid for vid, data in verified_emergency_vehicles.items() 
                        if frame_count - data['last_seen'] > EXPIRY_FRAMES]
    
    for vehicle_id in expired_vehicles:
        del verified_emergency_vehicles[vehicle_id]
        if vehicle_id in siren_history:
            del siren_history[vehicle_id]

    # ========== PHASE 3: Traffic Signal Control ==========
    traffic_controller.update(global_emergency_detected)
    
    # Draw traffic signal
    draw_traffic_signal(frame, traffic_controller)
    
    # Draw system status
    status_y = 40
    if global_emergency_detected:
        cv2.putText(frame, "STATUS: EMERGENCY DETECTED", (20, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "STATUS: NORMAL TRAFFIC", (20, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    # Draw traffic controller status
    cv2.putText(frame, traffic_controller.get_status_text(), (20, status_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw signal state
    signal_color = traffic_controller.get_signal_color()
    cv2.putText(frame, f"SIGNAL: {traffic_controller.current_state}", (20, status_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, signal_color, 2)

    cv2.imshow("Phase 3 - Complete Traffic Management System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()