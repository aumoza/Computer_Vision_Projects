import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

def get_dominant_color(img):
    # Focus on the middle of the box (the jersey)
    height, width, _ = img.shape
    jersey_crop = img[int(height*0.2):int(height*0.8), int(width*0.2):int(width*0.8)]
    
    # Reshape for K-means
    data = jersey_crop.reshape(-1, 3).astype(np.float32)
    
    # Use OpenCV K-Means to find the main color
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, flags)
    
    # Return the color of the largest cluster
    return centers[0].astype(int)

video_path = # Update this path to your video file
cap = cv2.VideoCapture(video_path)

cv2.namedWindow("TacticalVision", cv2.WINDOW_NORMAL)
cv2.resizeWindow("TacticalVision", 960, 540)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, classes=[0])
    
    for result in results:
        for box in result.boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop player
            player_img = frame[y1:y2, x1:x2]
            
            if player_img.size > 0:
                color = get_dominant_color(player_img)
                
                # Draw a custom circle above their head with the team color
                cv2.circle(frame, (int((x1+x2)/2), y1 - 10), 5, color.tolist(), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 2)

    cv2.imshow("TacticalVision", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

