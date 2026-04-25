import multiprocessing as mp
import cv2
import signal
import sys
from nodes.video_node import video_node
from nodes.detector_node import detection_worker
from nodes.analytics_node import analytics_worker

def launch_system():
    # 1. Setup shared communication
    raw_q = mp.Queue(maxsize=1)
    detect_q = mp.Queue(maxsize=1)
    
    video_path = r"" # Set your video path here
    processes = [
        mp.Process(target=video_node, args=(video_path, raw_q), name="VideoNode"),
        mp.Process(target=detection_worker, args=(raw_q, detect_q), name="DetectorNode"),
        mp.Process(target=analytics_worker, args=(detect_q,), name="AnalyticsNode")
    ]

    try:
        # Start all nodes
        for p in processes:
            p.daemon = True # Allows processes to exit if main exits
            p.start()
        
        print("TacticalVision Live. Press 'q' in the window or Ctrl+C here to exit.")

        # Keep main thread alive to monitor processes
        for p in processes:
            p.join()

    except KeyboardInterrupt:
        print("\n Ctrl+C detected, shutdown...")
    
    finally:
        # 3. Cleanup: Kill all child processes immediately
        for p in processes:
            if p.is_alive():
                print(f"Stopping {p.name}...")
                p.terminate() 
                p.join(timeout=1) # Wait a second for it to die
        
        print("All processes shutdown successfully.")
        sys.exit(0)

if __name__ == "__main__":
    # Crucial for Windows/Mac
    mp.set_start_method('spawn', force=True)
    launch_system()