import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from utils.utils import get_jersey_color, calculate_tangent

def analytics_worker(detect_q):
    trajectories = {"Team A": {}, "Team B": {}}
    team_assignments = {}
    calibration_samples = []
    team_centres = None
    calibration_frames = 50
    frame_count = 0

    cv2.namedWindow("TacticalVision (GPU Accelerated)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TacticalVision (GPU Accelerated)", 960, 540)

    try:
        while True:
            data = detect_q.get()
            if data is None: break
                
            gpu_tensor, results = data
            frame_count += 1

            # 1. ZERO-COPY: Convert PyTorch CUDA tensor directly to OpenCV GPU Storage
            # Requires torch tensor to be in (H, W, C) layout and uint8 type
            gpu_tensor_hwc = (gpu_tensor.squeeze(0).permute(1, 2, 0) * 255).to(torch.uint8)
            
            # This creates a GPU Matrix pointing directly to the PyTorch memory allocation
            gpu_frame = cv2.cuda.GpuMat(
                gpu_tensor_hwc.size(0), gpu_tensor_hwc.size(1), 
                cv2.CV_8UC3, gpu_tensor_hwc.data_ptr()
            )

            # 2. GPU Accelerated Color Conversion (RGB -> BGR for OpenCV display)
            gpu_display_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_RGB2BGR)

            # --- Extract Data for Analytics (Requires CPU mapping for Bounding Boxes) ---
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)

                # NOTE: For drawing logic and K-Means jersey sampling, OpenCV still 
                # requires basic rendering instructions on a host image. 
                # We download ONLY the final annotated visual matrix for the monitor display.
                display_frame = gpu_display_frame.download()

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = int((x1 + x2) / 2), y2

                    # Team Calibration Logic
                    if frame_count < calibration_frames:
                        color = get_jersey_color(display_frame, (x1, y1, x2, y2))
                        if color is not None: calibration_samples.append(color)
                        if track_id not in team_assignments:
                            team_assignments[track_id] = "Team A" if cx < display_frame.shape[1] / 2 else "Team B"
                        if len(calibration_samples) >= 20 and team_centres is None:
                            team_centres = KMeans(n_clusters=2, random_state=0).fit(calibration_samples).cluster_centers_
                    else:
                        if team_centres is not None and track_id not in team_assignments:
                            color = get_jersey_color(display_frame, (x1, y1, x2, y2))
                            if color is not None:
                                dist_to_a = np.linalg.norm(color - team_centres[0])
                                dist_to_b = np.linalg.norm(color - team_centres[1])
                                team_assignments[track_id] = "Team A" if dist_to_a < dist_to_b else "Team B"

                    team = team_assignments.get(track_id, "Team A")
                    
                    if track_id not in trajectories[team]: trajectories[team][track_id] = []
                    trajectories[team][track_id].append((cx, cy))

                    # Vector Math and Rendering (drawn on the downloaded display frame)
                    path = trajectories[team][track_id]
                    tangent = calculate_tangent(path)
                    if tangent is not None:
                        end_pt = (int(cx + tangent[0]), int(cy + tangent[1]))
                        cv2.arrowedLine(display_frame, (cx, cy), end_pt, (0, 255, 255), 2)

                    box_color = (0, 0, 255) if team == "Team A" else (255, 0, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(display_frame, f"ID:{track_id} {team}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            else:
                display_frame = gpu_display_frame.download()

            # Render frame on screen
            cv2.imshow("TacticalVision (GPU Accelerated)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                
    finally:
        cv2.destroyAllWindows()