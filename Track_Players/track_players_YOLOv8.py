from ultralytics import YOLO
import cv2

# 1. Load a pre-trained model (YOLOv8 is very stable)
model = YOLO('yolov8n.pt') 

# 2. Open video file
video_path = # Update this path to your video file
cap = cv2.VideoCapture(video_path)

# 1. Create a window and make it resizable
cv2.namedWindow("TacticalVision", cv2.WINDOW_NORMAL)

# 2. Resize the window to something manageable (e.g., 960x540)
cv2.resizeWindow("TacticalVision", 960, 540)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Detect "People" (Class 0 in COCO dataset)
    results = model(frame, classes=[0]) 

    # 4. Show the results
    annotated_frame = results[0].plot()
    cv2.imshow("TacticalVision", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()