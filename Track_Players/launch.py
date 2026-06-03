import multiprocessing as mp
import sys
from nodes.video_node import video_node
from nodes.detector_node import detection_worker
from nodes.analytics_node import analytics_worker

def launch_system():
    # Maxsize=1 handles backpressure, forcing the decoder to wait for YOLO
    raw_q = mp.Queue(maxsize=1)
    detect_q = mp.Queue(maxsize=1)
    
    video_path = r"C:\Users\aumoz\.cache\kagglehub\datasets\atomscott\teamtrack\versions\6\teamtrack\teamtrack\basketball_side\test\videos\Q4_side_60-90.mp4" 
    processes = [
        mp.Process(target=video_node, args=(video_path, raw_q), name="VideoNode"),
        mp.Process(target=detection_worker, args=(raw_q, detect_q), name="DetectorNode"),
        mp.Process(target=analytics_worker, args=(detect_q,), name="AnalyticsNode")
    ]

    try:
        for p in processes:
            p.daemon = True 
            p.start()
        
        print("TacticalVision Live. Press Ctrl+C to exit.")
        for p in processes:
            p.join()

    except KeyboardInterrupt:
        print("\nCtrl+C detected, shutting down...")
    finally:
        for p in processes:
            if p.is_alive():
                print(f"Stopping {p.name}...")
                p.terminate() 
                p.join(timeout=1)
        print("All processes shutdown successfully.")
        sys.exit(0)

if __name__ == "__main__":
    # Crucial for CUDA IPC (Inter-Process Communication)
    mp.set_start_method('spawn', force=True)
    launch_system()