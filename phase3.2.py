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
        # Four directions: North, South, East, West
        self.signals = {
            'North': 'GREEN',
            'South': 'GREEN',
            'East': 'RED',
            'West': 'RED'
        }
        self.normal_cycle_start = time.time()
        self.normal_cycle_duration = 20  # 30 seconds per cycle
        self.emergency_mode = False
        self.emergency_start_time = None
        self.emergency_countdown = 3  # 3 second countdown before turning green
        self.emergency_duration = 15  # Hold green for 15 seconds
        self.emergency_direction = None
        self.in_countdown = False
        self.countdown_start = None
        
    def update(self, emergency_detected, direction='North'):
        """Update traffic signal state based on emergency detection"""
        current_time = time.time()
        
        if emergency_detected and not self.emergency_mode and not self.in_countdown:
            # Emergency vehicle detected - start 3 second countdown
            self.in_countdown = True
            self.countdown_start = current_time
            self.emergency_direction = direction
            
            # Turn all to RED during countdown
            for dir_name in self.signals:
                self.signals[dir_name] = 'RED'
            return True
            
        elif self.in_countdown:
            # In countdown phase
            elapsed = current_time - self.countdown_start
            
            if elapsed >= self.emergency_countdown:
                # Countdown complete - turn emergency direction and opposite to GREEN
                self.in_countdown = False
                self.emergency_mode = True
                self.emergency_start_time = current_time
                
                # Get opposite direction
                opposites = {'North': 'South', 'South': 'North', 'East': 'West', 'West': 'East'}
                opposite_dir = opposites[self.emergency_direction]
                
                # Turn emergency direction and opposite to GREEN
                for dir_name in self.signals:
                    if dir_name == self.emergency_direction or dir_name == opposite_dir:
                        self.signals[dir_name] = 'GREEN'
                    else:
                        self.signals[dir_name] = 'RED'
                return True
            else:
                # Still counting down
                return False
            
        elif self.emergency_mode:
            # In emergency mode
            elapsed = current_time - self.emergency_start_time
            
            if elapsed >= self.emergency_duration:
                # Emergency duration expired, return to normal cycle
                self.emergency_mode = False
                self.emergency_direction = None
                self.normal_cycle_start = current_time
                
                # Reset to normal: North-South GREEN, East-West RED
                self.signals['North'] = 'GREEN'
                self.signals['South'] = 'GREEN'
                self.signals['East'] = 'RED'
                self.signals['West'] = 'RED'
                return True
            else:
                # Keep emergency configuration
                return False
        else:
            # Normal traffic cycle - alternate between North-South and East-West
            elapsed = current_time - self.normal_cycle_start
            
            if elapsed >= self.normal_cycle_duration:
                # Switch directions
                self.normal_cycle_start = current_time
                
                if self.signals['North'] == 'GREEN':
                    # Switch to East-West
                    self.signals['North'] = 'RED'
                    self.signals['South'] = 'RED'
                    self.signals['East'] = 'GREEN'
                    self.signals['West'] = 'GREEN'
                else:
                    # Switch to North-South
                    self.signals['North'] = 'GREEN'
                    self.signals['South'] = 'GREEN'
                    self.signals['East'] = 'RED'
                    self.signals['West'] = 'RED'
                return True
                
        return False
    
    def get_signal_color(self, direction):
        """Return BGR color tuple for signal"""
        colors = {
            "RED": (0, 0, 255),
            "YELLOW": (0, 255, 255),
            "GREEN": (0, 255, 0)
        }
        return colors[self.signals[direction]]
    
    def get_status_text(self):
        """Return status description"""
        if self.in_countdown:
            remaining = max(0, self.emergency_countdown - (time.time() - self.countdown_start))
            return f"EMERGENCY COUNTDOWN - {remaining:.1f}s - CLEARING INTERSECTION"
        elif self.emergency_mode:
            remaining = max(0, self.emergency_duration - (time.time() - self.emergency_start_time))
            return f"EMERGENCY MODE - {self.emergency_direction} - {remaining:.1f}s"
        else:
            # Show which directions are green
            green_dirs = [dir_name for dir_name, state in self.signals.items() if state == 'GREEN']
            return f"NORMAL OPERATION - {' & '.join(green_dirs)} GREEN"


def create_intersection_view(controller, width=600, height=600):
    """Create a top-down view of 4-way intersection with traffic signals"""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)  # Dark background
    
    center_x, center_y = width // 2, height // 2
    road_width = 120
    lane_width = road_width // 2
    
    # Draw roads
    # Horizontal road
    cv2.rectangle(canvas, (0, center_y - road_width//2), 
                  (width, center_y + road_width//2), (60, 60, 60), -1)
    # Vertical road
    cv2.rectangle(canvas, (center_x - road_width//2, 0), 
                  (center_x + road_width//2, height), (60, 60, 60), -1)
    
    # Draw lane markings
    dash_length = 20
    gap_length = 15
    
    # Horizontal center line
    for x in range(0, width, dash_length + gap_length):
        cv2.line(canvas, (x, center_y), (x + dash_length, center_y), (255, 255, 255), 2)
    
    # Vertical center line
    for y in range(0, height, dash_length + gap_length):
        cv2.line(canvas, (center_x, y), (center_x, y + dash_length), (255, 255, 255), 2)
    
    # Draw crosswalks
    stripe_width = 8
    num_stripes = 8
    
    # North crosswalk
    for i in range(num_stripes):
        x = center_x - road_width//2 + i * (road_width // num_stripes)
        cv2.rectangle(canvas, (x, center_y - road_width//2 - 30), 
                     (x + stripe_width, center_y - road_width//2), (255, 255, 255), -1)
    
    # South crosswalk
    for i in range(num_stripes):
        x = center_x - road_width//2 + i * (road_width // num_stripes)
        cv2.rectangle(canvas, (x, center_y + road_width//2), 
                     (x + stripe_width, center_y + road_width//2 + 30), (255, 255, 255), -1)
    
    # East crosswalk
    for i in range(num_stripes):
        y = center_y - road_width//2 + i * (road_width // num_stripes)
        cv2.rectangle(canvas, (center_x + road_width//2, y), 
                     (center_x + road_width//2 + 30, y + stripe_width), (255, 255, 255), -1)
    
    # West crosswalk
    for i in range(num_stripes):
        y = center_y - road_width//2 + i * (road_width // num_stripes)
        cv2.rectangle(canvas, (center_x - road_width//2 - 30, y), 
                     (center_x - road_width//2, y + stripe_width), (255, 255, 255), -1)
    
    # Draw traffic signals at each corner
    signal_positions = {
        'North': (center_x + 80, center_y - road_width//2 - 140),
        'South': (center_x - 80, center_y + road_width//2 + 40),
        'East': (center_x + road_width//2 + 60, center_y + 80),
        'West': (center_x - road_width//2 - 90, center_y - 80)
    }
    
    for direction, (sx, sy) in signal_positions.items():
        # Signal pole background
        cv2.rectangle(canvas, (sx - 5, sy - 5), (sx + 45, sy + 115), (40, 40, 40), -1)
        cv2.rectangle(canvas, (sx - 5, sy - 5), (sx + 45, sy + 115), (180, 180, 180), 2)
        
        # Draw three lights vertically
        light_radius = 15
        light_spacing = 35
        
        # Red light (top)
        red_color = (0, 0, 255) if controller.signals[direction] == 'RED' else (60, 60, 60)
        cv2.circle(canvas, (sx + 20, sy + 20), light_radius, red_color, -1)
        cv2.circle(canvas, (sx + 20, sy + 20), light_radius, (200, 200, 200), 2)
        
        # Yellow light (middle)
        yellow_color = (0, 255, 255) if controller.signals[direction] == 'YELLOW' else (60, 60, 60)
        cv2.circle(canvas, (sx + 20, sy + 20 + light_spacing), light_radius, yellow_color, -1)
        cv2.circle(canvas, (sx + 20, sy + 20 + light_spacing), light_radius, (200, 200, 200), 2)
        
        # Green light (bottom)
        green_color = (0, 255, 0) if controller.signals[direction] == 'GREEN' else (60, 60, 60)
        cv2.circle(canvas, (sx + 20, sy + 20 + light_spacing * 2), light_radius, green_color, -1)
        cv2.circle(canvas, (sx + 20, sy + 20 + light_spacing * 2), light_radius, (200, 200, 200), 2)
        
        # Direction label - positioned better for each direction
        if direction == 'North':
            label_pos = (sx - 15, sy - 15)
        elif direction == 'South':
            label_pos = (sx - 15, sy + 135)
        elif direction == 'East':
            label_pos = (sx + 55, sy + 65)
        else:  # West
            label_pos = (sx - 60, sy + 65)
            
        cv2.putText(canvas, direction, label_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw center intersection box
    cv2.rectangle(canvas, (center_x - 15, center_y - 15), 
                 (center_x + 15, center_y + 15), (100, 100, 100), -1)
    
    # Add title
    cv2.putText(canvas, "4-WAY INTERSECTION", (width//2 - 120, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add status
    status = controller.get_status_text()
    cv2.putText(canvas, status, (20, height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return canvas


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

    red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
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


# ==================== MAIN LOOP ====================
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame)
    global_emergency_detected = False
    
    # Resize video frame to fit left side
    video_height = 600
    aspect_ratio = frame.shape[1] / frame.shape[0]
    video_width = int(video_height * aspect_ratio)
    frame_resized = cv2.resize(frame, (video_width, video_height))

    # ========== PHASE 2: Emergency Vehicle Detection ==========
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if label in ["car", "truck", "bus"] and conf > 0.6:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Scale coordinates to resized frame
                scale_x = video_width / frame.shape[1]
                scale_y = video_height / frame.shape[0]
                x1_s = int(x1 * scale_x)
                y1_s = int(y1 * scale_y)
                x2_s = int(x2 * scale_x)
                y2_s = int(y2 * scale_y)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                vehicle_roi = frame[y1:y2, x1:x2]

                if vehicle_roi.size == 0:
                    continue

                is_emergency_vehicle = False
                vehicle_id = get_vehicle_id(box)

                # Check if already verified
                if vehicle_id in verified_emergency_vehicles:
                    verified_emergency_vehicles[vehicle_id]['last_seen'] = frame_count
                    
                    is_emergency_vehicle = True
                    global_emergency_detected = True
                    
                    cv2.putText(frame_resized, "EMERGENCY", (x1_s, y1_s - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # QR Code detection
                    qr_data, _, _ = qr_detector.detectAndDecode(vehicle_roi)

                    if qr_data:
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
                                
                                cv2.putText(frame_resized, "EMERGENCY VERIFIED", (x1_s, y1_s - 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            else:
                                cv2.putText(frame_resized, 
                                    f"VERIFY ({verified_emergency_vehicles[vehicle_id]['count']}/{CONFIRMATION_THRESHOLD})",
                                    (x1_s, y1_s - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                # Draw bounding box
                color = (0, 0, 255) if is_emergency_vehicle else (0, 255, 0)
                cv2.rectangle(frame_resized, (x1_s, y1_s), (x2_s, y2_s), color, 2)
                cv2.putText(frame_resized, f"{label}", (x1_s, y1_s - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Clean up expired vehicles
    expired_vehicles = [vid for vid, data in verified_emergency_vehicles.items() 
                        if frame_count - data['last_seen'] > EXPIRY_FRAMES]
    
    for vehicle_id in expired_vehicles:
        del verified_emergency_vehicles[vehicle_id]
        if vehicle_id in siren_history:
            del siren_history[vehicle_id]

    # ========== PHASE 3: Traffic Signal Control ==========
    traffic_controller.update(global_emergency_detected, direction='North')
    
    # Create intersection view
    intersection_view = create_intersection_view(traffic_controller)
    
    # Combine video and intersection side by side
    combined_width = video_width + intersection_view.shape[1]
    combined_height = max(video_height, intersection_view.shape[0])
    combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    
    # Place video on left
    combined_frame[0:video_height, 0:video_width] = frame_resized
    
    # Place intersection on right
    combined_frame[0:intersection_view.shape[0], 
                   video_width:video_width+intersection_view.shape[1]] = intersection_view
    
    # Add status overlay on video
    if global_emergency_detected:
        cv2.putText(combined_frame, "EMERGENCY DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(combined_frame, "NORMAL TRAFFIC", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Phase 3 - Traffic Management System", combined_frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()