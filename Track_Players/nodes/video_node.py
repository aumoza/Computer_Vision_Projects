import cv2

def video_node(video_path, raw_q):
    cap = cv2.VideoCapture(video_path)
    print("Video Node Initialized (Feeding Queue).")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Just put the frame in the queue and move on
        raw_q.put(frame)

    raw_q.put(None)
    cap.release()