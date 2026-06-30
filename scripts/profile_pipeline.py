"""Standalone performance profiler for SBSPipeline + YOLO + SGBM.

Runs N frames through the real pipeline (USB camera -> SBSPipeline),
collects per-frame timing and depth accuracy data, writes to debug log.

Usage:
    python scripts/profile_pipeline.py [--frames N] [--distance CM]

Hardware info (GPU, backend) is also logged on startup.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "debug-c7ffa8.log",
)

# ── logging ──────────────────────────────────────────────────────────────────
def log_write(measurement_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "id": f"log_{int(time.time()*1000)}_{os.getpid()}",
            "timestamp": int(time.time() * 1000),
            "measurementId": measurement_id,
            "location": location,
            "message": message,
            "data": data,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] log_write failed: {e}")


def log_separator(label: str) -> None:
    log_write("META", "profile_pipeline.py", f"=== {label} ===", {})


# ── hardware info ─────────────────────────────────────────────────────────────
def collect_hw_info() -> dict:
    info = {"platform": sys.platform, "python_version": sys.version}
    # torch
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["torch_error"] = str(e)

    # DirectML
    try:
        import torch_directml
        info["directml_available"] = True
        info["directml_device"] = str(torch_directml.device().name)
    except Exception:
        info["directml_available"] = False

    # OpenCV
    try:
        info["cv_version"] = cv2.__version__
        info["cv_build_info"] = cv2.getBuildInformation()
    except Exception:
        pass

    # GPU memory (Windows)
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory.argtypes = []
            ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory.restype = ctypes.c_uint64
            mem_kb = kernel32.GetPhysicallyInstalledSystemMemory()
            info["system_ram_gb"] = round(mem_kb / 1024 / 1024, 1)
        except Exception:
            pass

    return info


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Profile SBSPipeline performance")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to profile")
    parser.add_argument("--distance", type=float, default=None, help="True distance in cm (for accuracy check)")
    args = parser.parse_args()

    print("[profile_pipeline] Starting hardware detection...")
    hw_info = collect_hw_info()
    print(f"  torch: {hw_info.get('torch_version','N/A')}")
    print(f"  cuda: {hw_info.get('torch_cuda_available', False)}")
    print(f"  directml: {hw_info.get('directml_available', False)}")
    if hw_info.get("directml_device"):
        print(f"  dml device: {hw_info['directml_device']}")
    print(f"  opencv: {hw_info.get('cv_version','N/A')}")
    print(f"  ram: {hw_info.get('system_ram_gb','N/A')} GB")

    log_write(
        "META",
        "profile_pipeline.py:startup",
        "hardware info",
        hw_info,
    )
    log_separator("START PROFILING")

    print(f"[profile_pipeline] Initializing camera + pipeline ({args.frames} frames)...")

    # Init camera + pipeline (this triggers YOLO load + warmup)
    from config import USB_LEFT_INDEX, USB_RIGHT_INDEX, USB_TARGET_WIDTH, USB_TARGET_HEIGHT
    from camera.usb_camera import USBCamera
    from processing.sbs_pipeline import SBSPipeline

    print(f"  Camera: left={USB_LEFT_INDEX}, right={USB_RIGHT_INDEX}")
    print(f"  Resolution: {USB_TARGET_WIDTH}x{USB_TARGET_HEIGHT}")

    try:
        cam = USBCamera(
            left_index=USB_LEFT_INDEX,
            right_index=USB_RIGHT_INDEX,
            target_width=USB_TARGET_WIDTH,
            target_height=USB_TARGET_HEIGHT,
        )
        print("  Camera opened OK")
    except Exception as e:
        print(f"[ERROR] Camera open failed: {e}")
        log_write("META", "profile_pipeline.py:camera", f"Camera open FAILED: {e}", {})
        return

    try:
        pipeline = SBSPipeline(
            left_index=USB_LEFT_INDEX,
            right_index=USB_RIGHT_INDEX,
            target_width=USB_TARGET_WIDTH,
            target_height=USB_TARGET_HEIGHT,
        )
        print("  Pipeline initialized OK (YOLO warmup done)")
    except Exception as e:
        print(f"[ERROR] Pipeline init failed: {e}")
        log_write("META", "profile_pipeline.py:pipeline", f"Pipeline init FAILED: {e}", {})
        cam.close()
        return

    log_write("META", "profile_pipeline.py:pipeline_ready", "Pipeline ready", {})

    # Warmup: skip first 3 frames (CUDA/DML kernel JIT warmup)
    print("[profile_pipeline] Warming up (3 frames)...")
    for _ in range(3):
        pipeline.process_one_frame()

    print(f"[profile_pipeline] Profiling {args.frames} frames...")
    latencies = []
    depth_values = []
    yolo_l_times = []
    yolo_r_times = []
    sgbm_times = []

    for i in range(args.frames):
        t_start = time.perf_counter()
        result = pipeline.process_one_frame()
        t_end = time.perf_counter()

        if result is not None:
            latencies.append((t_end - t_start) * 1000)
            # depth is stored in the log by the pipeline itself;
            # also read it from the solver state
            try:
                solver = pipeline._solver
                # read last returned depth from smoothed
                last_depth = solver.smoothed_depth(None)
            except Exception:
                last_depth = None
            if last_depth is not None:
                depth_values.append(last_depth)
        else:
            latencies.append(np.nan)

        if (i + 1) % 5 == 0:
            print(f"  frame {i+1}/{args.frames} done, latency={latencies[-1]:.1f}ms" if not np.isnan(latencies[-1]) else f"  frame {i+1}/{args.frames} done (null)")

    cam.close()

    # ── Summary stats ──────────────────────────────────────────────────────────
    valid = [l for l in latencies if not np.isnan(l)]
    log_write(
        "META",
        "profile_pipeline.py:summary",
        "profile summary",
        {
            "n_frames": args.frames,
            "n_valid": len(valid),
            "n_null": latencies.count(np.nan),
            "true_distance_cm": args.distance,
            # latency stats
            "lat_mean_ms": round(float(np.mean(valid)), 1) if valid else None,
            "lat_p50_ms": round(float(np.median(valid)), 1) if valid else None,
            "lat_p95_ms": round(float(np.percentile(valid, 95)), 1) if valid else None,
            "lat_max_ms": round(float(np.max(valid)), 1) if valid else None,
            "fps_approx": round(1000.0 / np.mean(valid), 1) if valid else 0,
            # depth stats
            "depth_values": depth_values,
            "depth_mean_cm": round(float(np.mean(depth_values)), 1) if depth_values else None,
            "depth_std_cm": round(float(np.std(depth_values)), 1) if depth_values else None,
            "depth_min_cm": round(float(np.min(depth_values)), 1) if depth_values else None,
            "depth_max_cm": round(float(np.max(depth_values)), 1) if depth_values else None,
        },
    )

    print(f"\n{'='*50}")
    print(f"[profile_pipeline] SUMMARY")
    print(f"  Frames: {args.frames}, valid: {len(valid)}, null: {latencies.count(np.nan)}")
    if valid:
        print(f"  Latency mean: {np.mean(valid):.1f} ms")
        print(f"  Latency p50:  {np.median(valid):.1f} ms")
        print(f"  Latency p95:  {np.percentile(valid, 95):.1f} ms")
        print(f"  Latency max:  {np.max(valid):.1f} ms")
        print(f"  FPS approx:   {1000.0/np.mean(valid):.1f}")
    if depth_values:
        print(f"  Depth mean: {np.mean(depth_values):.1f} cm")
        print(f"  Depth std:  {np.std(depth_values):.1f} cm")
        print(f"  Depth range: [{np.min(depth_values):.1f}, {np.max(depth_values):.1f}] cm")
        if args.distance:
            abs_err = abs(np.mean(depth_values) - args.distance)
            print(f"  True distance: {args.distance} cm, abs error: {abs_err:.1f} cm ({abs_err/args.distance*100:.1f}%)")
    print(f"  Log written to: {LOG_PATH}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
