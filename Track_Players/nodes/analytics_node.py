import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils.utils import get_jersey_color, calculate_tangent, plot_color_clusters, plot_trajectories

def analytics_worker(detect_q):
    # trajectories = {"Team A": {id: [(x,y), ...]}, "Team B": {id: [(x,y), ...]}}
    trajectories = {"Team A": {}, "Team B": {}}
    team_assignments = {}

    # Calibration for visualization
    calibration_samples = []
    team_centres = None
    calibration_frames = 50
    frame_count = 0

    # Create a window for visualization
    cv2.namedWindow("TacticalVision", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TacticalVision", 960, 540)
    print("Analytics Node Initialized.")

    try:
        while True:
            data = detect_q.get()
            if data is None: break
            frame_count += 1
            frame, results = data
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = int((x1 + x2) / 2), y2 # Foot position

                    # Phase 1: Calibration (First N frames to determine team colors)
                    if frame_count < calibration_frames:
                        color = get_jersey_color(frame, (x1, y1, x2, y2))
                        if color is not None:
                            calibration_samples.append(color)
                            cv2.putText(frame, "CALIBRATING TEAMS...", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            
                        if track_id not in team_assignments:
                            team_assignments[track_id] = "Team A" if cx < frame.shape[1] / 2 else "Team B"
                            
                        if len(calibration_samples) >= 20: # Arbitrary threshold to start clustering
                            kmeans = KMeans(n_clusters=2, random_state=0).fit(calibration_samples)
                            team_centres = kmeans.cluster_centers_

                    else:
                        if team_centres is not None:
                            if track_id not in team_assignments:
                                color = get_jersey_color(frame, (x1, y1, x2, y2))
                                if color is not None:
                                    # Assign team based on closest color center using Euclidean distance
                                    dist_to_team_a = np.linalg.norm(color - team_centres[0])
                                    dist_to_team_b = np.linalg.norm(color - team_centres[1])
                                    assigned_team = "Team A" if dist_to_team_a < dist_to_team_b else "Team B"
                                    team_assignments[track_id] = assigned_team
                    
                
                    team = team_assignments[track_id]
                    color = get_jersey_color(frame, (x1, y1, x2, y2))
                    if color is None: color = (0,0,0) # Default to black if color extraction fails
                    color = tuple(map(int, color)) # Convert to int for OpenCV

                    # 2. Update Trajectory for THIS specific ID
                    if track_id not in trajectories[team]:
                        trajectories[team][track_id] = []
                    trajectories[team][track_id].append((cx, cy))

                    # 3. Calculate and Draw Motion Tangent (Velocity Vector)
                    path = trajectories[team][track_id]
                    tangent = calculate_tangent(path)
                    if tangent is not None:
                        end_pt = (int(cx + tangent[0]), int(cy + tangent[1]))
                        cv2.arrowedLine(frame, (cx, cy), end_pt, (0, 255, 255), 2, tipLength=0.3)

                    # 4. Drawing Bounding Boxes & IDs
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{track_id} {team}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("TacticalVision", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    finally:
        cv2.destroyAllWindows()

        # 5. Final Plotting Logic
        plot_color_clusters(calibration_samples, team_centres)
        plot_trajectories(trajectories)
        