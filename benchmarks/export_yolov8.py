import argparse
from ultralytics import YOLO
from pathlib import Path

def export_model(args):
    """
    Exports a YOLOv8 model to an optimized TorchScript format using torch.compile.
    This creates a static, high-performance version of the model suitable for fair benchmarking.
    """
    model_name = f'yolov8{args.model_size}'
    export_filename = Path(f'{model_name}_torchscript.pt')
    
    print(f"🚀 Loading YOLOv8 model: {model_name}.pt")
    model = YOLO(f'{model_name}.pt')

    print(f"\n🔥 Exporting to optimized TorchScript format ({export_filename})...")
    print("   This uses 'torch.compile' internally via the 'optimize=True' flag and may take a moment.")
    print(f"   Targeting device: {args.device}")
    
    # The 'optimize=True' flag uses torch.compile. Specifying the device is crucial
    # to prevent baking in CPU-specific ops (like XNNPACK) when the target is GPU.
    # The export function returns the path to the created file. We don't specify the name here.
    default_export_path_str = model.export(format='torchscript', optimize=True, half=True, device=args.device)
    default_export_path = Path(default_export_path_str)

    # Rename the exported file to our desired consistent name for the profiler
    if default_export_path.exists() and default_export_path != export_filename:
        print(f"   Renaming '{default_export_path}' to '{export_filename}' for consistency.")
        default_export_path.rename(export_filename)
    elif not export_filename.exists():
        print(f"❌ Error: Export failed, file '{export_filename}' not found.")
    
    print(f"\n✅ Export complete! Optimized model saved to '{export_filename}'")
    print("   You can now use this file for benchmarking with the profiler.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Optimized Exporter for Fair Benchmarking")
    parser.add_argument('--model-size', type=str, default='n', help="Model size (e.g., 'n', 's', 'm').")
    parser.add_argument('--device', type=str, default='cuda', help="Device to export for (e.g., 'cuda', 'cpu').")
    args = parser.parse_args()
    export_model(args)