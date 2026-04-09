from ultralytics import YOLO

def detection_worker(in_queue, out_queue):
    model = YOLO('yolov8n.pt')
    print("Detection Node Initialized.")

    while True:
        frame = in_queue.get()
        if frame is None: 
            out_queue.put(None) # Pass the end signal!
            break 

        results = model.track(frame, imgsz=1280,persist=True, classes=[0], verbose=False)
        
        # We pass the results as they are. 
        # Note: If you want to use the color in Analytics, 
        # you should ideally calculate it there or pass it in a custom dict.
        out_queue.put((frame, results))