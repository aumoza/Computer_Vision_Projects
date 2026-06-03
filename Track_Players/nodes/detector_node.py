from ultralytics import YOLO
import torch

def detection_worker(in_queue, out_queue):
    # CUDA Context gets initialized cleanly inside the child process
    model = YOLO('yolov8s.pt').to('cuda')
    print(f"Detection Node Initialized on: {torch.cuda.get_device_name(0)}")

    while True:
        frame = in_queue.get()
        if frame is None: 
            out_queue.put(None)
            break 
        
        # YOLO runs tracking directly on the shared GPU memory frame
        results = model.track(
            frame, 
            imgsz=640, # Match video_node resize dimensions for optimal performance
            device='cuda', 
            half=True, 
            persist=True, 
            verbose=False
        )
        
        # We pass the frame and results. Results contain CPU bounding-box pointers
        frame_ipc = frame.clone() # Clone again to ensure clean GPU memory for IPC
        out_queue.put((frame_ipc, results))