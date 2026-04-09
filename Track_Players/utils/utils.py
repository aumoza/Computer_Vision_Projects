import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_jersey_color(frame, bbox):
    """
    Extracts the average color from the torso area of the detected player.
    bbox: (x1, y1, x2, y2)
     Returns: Average BGR color as a numpy array
    """
    
    x1, y1, x2, y2 = map(int, bbox)
    height, width = y2 - y1, x2 - x1
    # Focus on the torso (middle 40% of the box)
    jersey_crop = frame[y1+int(height*0.3):y1+int(height*0.7), 
                        x1+int(width*0.3):x1+int(width*0.7)]
    
    if jersey_crop.size == 0: return None
    return np.mean(jersey_crop, axis=(0, 1)) # BGR Average

def calculate_tangent(path, scale=5):
    """
    Calculates the motion tangent (velocity vector) based on the last two points in the trajectory.
    """
    
    if len(path) < 2:
        return None
    # Calculate vector from previous point to current point
    pt1 = np.array(path[-2])
    pt2 = np.array(path[-1])
    vector = pt2 - pt1
    return vector * scale

def plot_color_clusters(samples, centers):
    """
    Plots the collected jersey colors in 3D RGB space.
    samples: List or array of BGR values
    centers: The two [B, G, R] centroids from KMeans
    """
    
    samples_np = np.array(samples)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Convert BGR to RGB for plotting (Matplotlib expects RGB)
    # Also normalize to 0-1 range for the 'c' argument
    colors_rgb = samples_np[:, [2, 1, 0]] / 255.0

    # Plot raw samples
    ax.scatter(samples_np[:, 2], samples_np[:, 1], samples_np[:, 0], 
               c=colors_rgb, s=15, alpha=0.6, label="Player Samples")

    # Plot Cluster Centers as big stars
    for i, center in enumerate(centers):
        center_rgb = center[[2, 1, 0]] / 255.0
        ax.scatter(center[2], center[1], center[0], 
                   c=[center_rgb], s=300, marker='*', edgecolors='black', 
                   label=f"Team {i} Center")

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('3D Jersey Color Distribution')
    plt.legend()
    plt.show()

def plot_trajectories(trajectories):
    """
    Plots the trajectories of players for both teams.
    trajectories: Dict with team names as keys and dict of track_id to list of (x, y) positions as values
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    for team, ax, title in zip(["Team A", "Team B"], [ax1, ax2], ["Home Team Path", "Away Team Path"]):
        for track_id, path in trajectories[team].items():
            if len(path) > 15: # Filter out "noise" detections
                path_np = np.array(path)
                ax.plot(path_np[:, 0], path_np[:, 1], alpha=0.7, label=f"ID {track_id}")
        
        ax.set_title(title)
        ax.set_xlabel("Field Width (px)")
        ax.set_ylabel("Field Depth (px)")
        ax.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    plt.show()