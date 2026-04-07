import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 1. Setup
model = YOLO('yolov8n.pt')
video_path = # Update this path to your video file
cap = cv2.VideoCapture(video_path)
cv2.namedWindow("TacticalVision", cv2.WINDOW_NORMAL)
cv2.resizeWindow("TacticalVision", 960, 540)

# Data storage for trajectories
trajectories = {"Team A": {}, "Team B": {}} 
team_colors = {} # Maps tracker_id to a stable Team name

def get_team(img):
    # Same logic as before: focus on jersey
    h, w, _ = img.shape
    crop = img[int(h*0.3):int(h*0.6), int(w*0.3):int(w*0.6)]
    avg_color = crop.mean(axis=(0,1))
    # Simple logic: If Red channel > Blue channel, it's Team A (modify for your video!)
    return "Team A" if avg_color[2] > avg_color[0] else "Team B"

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 2. Tracking (persist=True is the key)
        results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, y2 # Bottom-center (feet)
                
                # 3. Identify Team once and remember it
                if track_id not in team_colors:
                    player_crop = frame[y1:y2, x1:x2]
                    team_colors[track_id] = get_team(player_crop)
                
                team = team_colors[track_id]
                
                # 4. Store Coordinate
                if track_id not in trajectories[team]:
                    trajectories[team][track_id] = []
                trajectories[team][track_id].append((cx, cy))
                
                # Draw Visuals
                color = (0, 0, 255) if team == "Team A" else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("TacticalVision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

except KeyboardInterrupt:
    print("Saving trajectories...")

finally:
    cap.release()
    cv2.destroyAllWindows()

    # 5. Final Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    for team, ax, title in zip(["Team A", "Team B"], [ax1, ax2], ["Home Team Path", "Away Team Path"]):
        for track_id, path in trajectories[team].items():
            if len(path) > 10: # Only plot players who were on screen for a while
                path_np = np.array(path)
                ax.plot(path_np[:, 0], path_np[:, 1], label=f"ID {track_id}")
        ax.set_title(title)
        ax.invert_yaxis() # Invert because image (0,0) is top-left
    
    plt.show()