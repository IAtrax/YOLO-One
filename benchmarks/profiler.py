import torch
import argparse
from torch.profiler import profile, record_function, ProfilerActivity
import sys
from pathlib import Path
try:
    from thop import profile as thop_profile
    thop_available = True
except ImportError:
    thop_available = False

# Add project root to path to allow direct script execution
sys.path.append(str(Path(__file__).parent.parent))

from yolo_one.models.yolo_one_model import YoloOne

def profile_model(args):
    """
    Profiles the YOLO-One model to identify performance bottlenecks.
    """
    device = torch.device(args.device)
    use_half = (not args.no_half) and device.type == 'cuda'
    use_compile = args.compile and device.type == 'cuda'
    use_channels_last = device.type == 'cuda' # Channels-last is now default for CUDA
    print(f"🚀 Starting Profiler on device: {device}")
    print(f"Model Type: {args.model_type}, Model Size: {args.model_size}, Batch: {args.batch_size}, Input: {args.input_size}, FP16: {use_half}, Compile: {use_compile}, Channels Last: {use_channels_last}")

    # 1. Load Model
    print("\n[1/5] Loading model...")
    try:
        if args.model_type == 'yolo_one':
            model = YoloOne(model_size=args.model_size).to(device)
        elif args.model_type == 'yolov8':
            # We load the raw nn.Module from ultralytics to directly compare its
            # compatibility with torch.compile against YOLO-One.
            try:
                from ultralytics import YOLO
            except ImportError:
                print("❌ Error: 'ultralytics' package not found. Please run 'pip install ultralytics'")
                return
            print("Loading standard YOLOv8 model from ultralytics...")
            # .model gives us the underlying nn.Module
            # NOTE: Profiling the raw YOLOv8 nn.Module like this is a "pure" test of the
            # architecture's compatibility with standard PyTorch tools like torch.compile.
            # The results may be slower than those reported by Ultralytics' own benchmark tools,
            # which use a highly integrated and optimized full pipeline (not just the model).
            # This comparison highlights the "compile-friendliness" of an architecture.
            model = YOLO(f'yolov8{args.model_size}.pt').model.to(device)


        if use_half:
            model.half()
        model.eval()

        if use_channels_last:
            print("Converting model to channels-last memory format...")
            model = model.to(memory_format=torch.channels_last)

        # 2. Create Dummy Input for FLOPs calculation (before compilation)
        # thop needs a model before it's compiled
        flops_input = torch.randn(
            1, 3, args.input_size, args.input_size,
            device=device,
            dtype=torch.float16 if use_half else torch.float32
        )
        if use_channels_last:
            flops_input = flops_input.to(memory_format=torch.channels_last)

        # Calculate FLOPs and Params using thop
        if thop_available:
            # Note: thop might not support all custom ops or compiled models perfectly.
            # It's best to run it on the un-compiled model.
            total_macs, total_params = thop_profile(model, inputs=(flops_input, ), verbose=False)
            # FLOPs is approx. 2 * MACs
            gflops = (total_macs * 2) / 1e9
            print(f"📦 Model Complexity: {total_params/1e6:.2f}M params, {gflops:.2f} GFLOPs")
        else:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"📦 Model Parameters: {total_params/1e6:.2f}M (thop not installed, cannot calculate FLOPs)")

        if use_compile:
            print("\n[+] Compiling model with torch.compile()... (this may take a moment)")
            try:
                # We compile the full model for both types. This will highlight how
                # well each architecture works with PyTorch 2.0+ optimizations.
                # YOLO-One should compile cleanly, while YOLOv8 may have graph breaks.
                print(f"   Compiling full {args.model_type} model...")
                model = torch.compile(model, mode="reduce-overhead")
                print("✅ Model compiled successfully.")
            except Exception as e:
                print(f"⚠️ Warning: Model compilation failed: {e}. Continuing without compilation.")

        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Create Dummy Input
    print("[2/5] Creating dummy input tensor...")
    try:
        dummy_input = torch.randn(
            args.batch_size, 3, args.input_size, args.input_size,
            device=device,
            dtype=torch.float16 if use_half else torch.float32
        )
        if use_channels_last:
            dummy_input = dummy_input.to(memory_format=torch.channels_last)
        print("✅ Dummy input created.")
    except Exception as e:
        print(f"❌ Error creating input tensor (is CUDA out of memory?): {e}")
        return

    # 3. Warmup
    print("[3/5] Warming up the model...")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    for _ in range(args.warmup_steps):
        with torch.no_grad():
            _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print("✅ Warmup complete.")

    # 4. Profile Execution
    print(f"[4/5] Profiling for {args.profile_steps} steps...")
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(args.profile_steps):
            with torch.no_grad():
                with record_function("model_inference"):
                    _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize() # Ensure accurate timing
    print("✅ Profiling complete.")

    # 5. Print Results
    print("\n[5/5] Analyzing Profiler Results...")
    print("=" * 80)

    # Use 'cuda_time_total' if on GPU, otherwise 'cpu_time_total'
    time_metric = 'cuda_time_total' if device.type == 'cuda' else 'cpu_time_total'
    # For our high-level components (Backbone, Neck, Head), we need the total time and memory,
    # not the 'self' time/memory, as they are just wrappers.
    memory_metric = 'cuda_memory_usage' if device.type == 'cuda' else 'cpu_memory_usage'
    # Get event averages
    events = prof.key_averages()

    # Find total inference time from our custom marker
    total_inference_event = [e for e in events if e.key == 'model_inference']
    if not total_inference_event:
        print("❌ Error: 'model_inference' marker not found. Cannot calculate percentages.")
        print(events.table(sort_by=time_metric, row_limit=20))
        return

    total_time_us = getattr(total_inference_event[0], time_metric)

    # --- Build a simplified summary table ---
    # Only show component breakdown for YOLO-One, as YOLOv8 doesn't have our custom markers
    summary_data = []
    component_events = [e for e in events if "YoloOne::" in e.key]

    for event in component_events:
        component_name = event.key.replace("YoloOne::", "")
        component_time_us = getattr(event, time_metric)
        # Similarly, for memory, we want the total memory allocated within the block.
        percentage = (component_time_us / total_time_us * 100) if total_time_us > 0 else 0
        memory_mb = getattr(event, memory_metric) / (1024 * 1024)
        summary_data.append({
            "Component": component_name,
            "Time (ms)": component_time_us / 1000,
            "Percentage (%)": percentage,
            "Memory (MB)": memory_mb,
        })

    if summary_data:
        summary_data.sort(key=lambda x: x['Time (ms)'], reverse=True)

    # --- Print the simplified table ---
    print("📊 Simplified Performance Summary")
    print("-" * 80)
    if args.model_type == 'yolo_one' and component_events:
        print(f"{'Component':<20} | {'Time (ms)':>12} | {'Percentage (%)':>15} | {'Memory (MB)':>15}")
        print("-" * 80)
        for item in summary_data:
            print(f"{item['Component']:<20} | {item['Time (ms)']:>12.3f} | {item['Percentage (%)']:>15.2f} | {item['Memory (MB)']:>15.2f}")
    print("-" * 80)
    print(f"{'Total Inference':<20} | {total_time_us / 1000:>12.3f} | {'100.00':>15} |")
    print("=" * 80)

    # Save trace if requested
    if args.trace_file:
        print(f"\n💾 Saving profiler trace to {args.trace_file}...")
        prof.export_chrome_trace(args.trace_file)
        print(f"✅ Trace saved. You can view it in Chrome by navigating to chrome://tracing")


    # --- Optional Detailed View ---
    if args.detailed:
        print("\n🔬 Detailed Operator-level View (for advanced debugging, sorted by self-time)")
        # For this view, sorting by 'self' time is useful to find individual costly operators.
        self_time_metric = 'self_' + time_metric
        print(events.table(sort_by=self_time_metric, row_limit=20))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO-One Model Profiler")
    parser.add_argument('--model-type', type=str, default='yolo_one', choices=['yolo_one', 'yolov8'], help='Type of model to profile.')
    parser.add_argument('--model-size', type=str, default='nano', help="Model size (e.g., 'nano' for yolo_one, 'n' for yolov8).")
    parser.add_argument('--input-size', type=int, default=640, help='Input image size (square).')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for profiling.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run on (cuda or cpu).')
    parser.add_argument('--warmup-steps', type=int, default=10, help='Number of warmup iterations before profiling.')
    parser.add_argument('--profile-steps', type=int, default=20, help='Number of profiling iterations.')
    parser.add_argument('--trace-file', type=str, default=None, help='Optional: Path to save a Chrome trace file (e.g., trace.json).')
    parser.add_argument('--detailed', action='store_true', help='Show detailed operator-level profiling tables.')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile() for optimization on CUDA.')
    parser.add_argument('--no-half', action='store_true', help='Disable half-precision (FP16) for profiling on CUDA (FP16 is enabled by default).')

    args = parser.parse_args()
    profile_model(args)