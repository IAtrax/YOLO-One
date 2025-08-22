import torch
import time

def benchmark_formats():
    batch_size = 8
    height, width = 640, 640
    channels = 64
    iterations = 150
    
    # Channel-first (NCHW) - Format PyTorch natif
    tensor_nchw = torch.randn(batch_size, channels, height, width).cuda().half()
    conv_nchw = torch.nn.Conv2d(channels, 128, 3, padding=1).cuda().half()
    
    # Channel-last (NHWC) 
    tensor_nhwc = torch.randn(batch_size, height, width, channels).cuda().half()
    
    # Warmup
    for _ in range(10):
        _ = conv_nchw(tensor_nchw)
    
    # Test NCHW (Channel-first)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        out = conv_nchw(tensor_nchw)
    torch.cuda.synchronize()
    nchw_time = time.time() - start
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        temp = tensor_nhwc.permute(0, 3, 1, 2)  # NHWC -> NCHW
        out = conv_nchw(temp)
        out = out.permute(0, 2, 3, 1)  # NCHW -> NHWC
    torch.cuda.synchronize()
    nhwc_time = time.time() - start
    
    print(f"NCHW (Channel-first): {nchw_time:.3f}s")
    print(f"NHWC (Channel-last): {nhwc_time:.3f}s")
    print(f"Speedup: {nhwc_time/nchw_time:.2f}x plus lent avec NHWC")

benchmark_formats()