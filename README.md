# Computer_Vision_Projects

TacticalVision: 
AI-Powered Sports Analytics & Trajectory MappingTacticalVision is a computer vision pipeline designed to transform standard broadcast sports footage into actionable tactical data. By combining state-of-the-art object detection (YOLO) with Multi-Object Tracking (ByteTrack) and color-based team classification, the system extracts player trajectories and generates spatial movement heatmaps.

🚀 Key FeaturesPersistent Multi-Object Tracking:
Uses ByteTrack to maintain player identities across occlusions and fast movements.
Automated Team Classification: Implements K-Means clustering on jersey color histograms to distinguish between Home, Away, and Referee groups.
Stateful ID Management: Resolves lighting/shadow flickers by caching team assignments per tracker ID.Trajectory Analytics: Generates post-match movement graphs for each team to analyze spatial coverage and formation.

🛠 Tech StackCore Model: 
OLOv8 (Ultralytics) for real-time person detection.Tracking Engine: ByteTrack (Multi-Object Tracking).Image Processing: OpenCV for color space manipulation and ROI extraction.Analytics: Matplotlib for trajectory plotting.Language: Python 3.10.

📊 MethodologyTo ensure professional-grade accuracy, the pipeline follows these steps:Detection & ROI: Detects "Person" class and crops the torso region (30%–60% of bounding box height) to isolate the jersey.Color Profiling: Converts crops to RGB and uses K-Means clustering to identify the dominant jersey color, ignoring pitch green.Tracking: Assigns a unique tracker_id to each player. This ID is mapped to a team color once to ensure visual consistency throughout the video.

📄 LicenseThis project is licensed under the Apache 2.0 License - see the LICENSE file for details.
