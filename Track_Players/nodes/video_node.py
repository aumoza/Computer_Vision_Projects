import torch
import PyNvVideoCodec as nvc
import torchvision.transforms.functional as F
import time
import queue

def video_node(video_path, raw_q):
    nv_dmx = nvc.CreateDemuxer(filename=video_path)
    nv_dec = nvc.CreateDecoder(gpuid=0, codec=nv_dmx.GetNvCodecId())

    print(f"NVDEC Hardware Decoding: {nv_dmx.Width()}x{nv_dmx.Height()}")

    for packet in nv_dmx:
        for surface in nv_dec.Decode(packet):
            # 1. Decode directly to GPU
            frame_tensor = torch.from_dlpack(surface).to('cuda') 

            # 2. Reshape from (H, W*C) to (H, W, C)
            if frame_tensor.dim() == 2:
                h, w_c = frame_tensor.shape
                w = w_c // 3
                frame_tensor = frame_tensor.reshape(h, w, 3)

            # 3. Process frame
            frame = frame_tensor.permute(2, 0, 1).float() / 255.0
            frame = F.resize(frame, [640, 640])
            frame = frame.unsqueeze(0)
            
            # 4. Clone to get a clean IPC handle
            frame_ipc = frame.clone()
            
            # 5. SAFE NON-BLOCKING QUEUE PUSH
            # Instead of a hard .put() which blocks the thread and ruins CUDA IPC,
            # we loop with a tiny sleep if the queue is full.
            while True:
                try:
                    raw_q.put_nowait(frame_ipc)
                    break # Successfully put in queue, break the loop
                except queue.Full:
                    # Detector is still processing. Sleep briefly to give it breathing room
                    # without locking up the underlying OS semaphore.
                    time.sleep(0.005) 

            # 6. FORCE GARBAGE COLLECTION OF CUDA MEMORY
            # Free up the massive 4K surface pointers and local tensors immediately
            del frame_tensor
            del frame
            
        # Periodically empty PyTorch's internal VRAM cache memory pool
        torch.cuda.empty_cache()
            
    # Send poison pill when video ends
    raw_q.put(None)