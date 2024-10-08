import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics import YOLO
from scipy.spatial import distance

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Use the YOLOv8 model (e.g., 'yolov8n.pt' for the nano version)
vehicle_classes = [2, 3, 7]
def calculate_speed(prev_position, cur_position, fps):
    """Estimate speed based on distance traveled between frames."""
    pixel_distance = distance.euclidean(prev_position, cur_position)
    real_distance = pixel_distance / 144 
    speed = real_distance * fps * 0.056  # Convert to mph
    return speed

# Define known colors and their ranges (in BGR format)
color_ranges = {
    # 'Red': ([0, 0, 100], [80, 80, 255]),
    # 'Blue': ([100, 0, 0], [255, 80, 80]),
    'White': ([200, 200, 200], [255, 255, 255]),
    'Black': ([0, 0, 0], [140, 140, 140]),
    'Gray': ([150, 150, 150], [190, 190, 190])
}

def detect_color(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Focus on a central region of the bounding box to avoid background colors
    h, w, _ = image.shape
    center_h, center_w = h // 4, w // 4  # Use the central quarter of the bounding box
    center_region = image[center_h:h-center_h, center_w:w-center_w]
    
    # Calculate the mean color of the center region
    average_color = np.mean(center_region, axis=(0, 1))  # Get the mean color in the BGR space
    b, g, r = average_color

    # Check the color ranges to match with a predefined color
    detected_color = 'Unknown'
    
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        
        # Count how many channels (B, G, R) fall within the range
        matches = sum([lower[0] <= b <= upper[0],  # Check if blue is in range
                       lower[1] <= g <= upper[1],  # Check if green is in range
                       lower[2] <= r <= upper[2]])  # Check if red is in range
        
        # If at least 2 out of 3 components (B, G, or R) match, assign the color
        if matches >= 2:
            detected_color = color_name
            break

    return detected_color

# Load the video or start the camera
video_path = "data\I94-US20-35.1.mp4"  # Replace with video file or use 0 for webcam
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
vehicle_tracks = {}
# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or failed to capture frame.")
        break

    # Get vehicle detections using YOLOv8
    results = model(frame)
    
    for result in results:
        boxes = result.boxes  # Bounding boxes of detected objects
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.cpu().numpy()  # Confidence score
            vehicle_type = int(box.cls.cpu().numpy())  # Class index
            
            # Filter for vehicles (you may use COCO dataset classes for vehicles, e.g., car, truck, etc.)
            if vehicle_type in vehicle_classes:  # Cars, motorcycles, buses, trucks (COCO class indices)
                # Calculate vehicle's center point for speed estimation
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                center = (center_x, center_y)

                # Track the vehicle's position to estimate speed
                if vehicle_type in vehicle_tracks:
                    speed = calculate_speed(vehicle_tracks[vehicle_type][-1], center, fps)
                    direction = "Left" if center[0] < vehicle_tracks[vehicle_type][-1][0] else "Right"
                else:
                    speed = 0
                    direction = "Unknown"

                # Save the track
                if vehicle_type not in vehicle_tracks:
                    vehicle_tracks[vehicle_type] = []
                vehicle_tracks[vehicle_type].append(center)
                
                # Extract the bounding box area (vehicle region)
                vehicle_region = frame[y1:y2, x1:x2]
                
                # Detect the color of the vehicle in the bounding box
                vehicle_color = detect_color(vehicle_region)
                
                # Draw bounding box and color label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Speed: {int(speed)} mph, Color: {vehicle_color}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
    # Display the frame with bounding boxes and color labels
    cv2.imshow('Vehicle Color Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
