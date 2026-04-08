import os
import csv
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

warnings.filterwarnings("ignore")


def discover_weights(weights_root: Path, mode: str):
    if mode == "best":
        files = sorted(weights_root.rglob("weights/best.pt"))
    elif mode == "last":
        files = sorted(weights_root.rglob("weights/last.pt"))
    elif mode == "both":
        files = sorted(weights_root.rglob("weights/best.pt")) + sorted(weights_root.rglob("weights/last.pt"))
    elif mode == "all_pt":
        files = sorted(weights_root.rglob("weights/*.pt"))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    uniq = []
    seen = set()
    for p in files:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def pick_device(device_arg: str):
    if device_arg.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if device_arg == "":
        return torch.device("cuda:0")
    return torch.device(f"cuda:{device_arg}")


def trim_stats(ms_list, low=5, high=95):
    arr = np.array(ms_list, dtype=np.float64)
    p_low = np.percentile(arr, low)
    p_high = np.percentile(arr, high)
    trimmed = arr[(arr >= p_low) & (arr <= p_high)]
    if len(trimmed) == 0:
        trimmed = arr
    avg = float(np.mean(trimmed))
    std = float(np.std(trimmed))
    fps = 1000.0 / avg if avg > 0 else 0.0
    return avg, std, fps, len(arr) - len(trimmed)


def benchmark_one(
    weights_path: str,
    device: torch.device,
    imgsz: int = 640,
    channels: int = 6,
    batch_size: int = 1,
    warmup_runs: int = 100,
    test_runs: int = 1000,
    use_fp16: bool = False,
):
    yolo_model = YOLO(weights_path)
    model = yolo_model.model.to(device).eval()

    dtype = torch.float16 if (use_fp16 and device.type == "cuda") else torch.float32
    dummy = torch.randn(batch_size, channels, imgsz, imgsz, device=device, dtype=dtype)

    if dtype == torch.float16:
        model = model.half()

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(dummy)

    timings_ms = []

    with torch.inference_mode():
        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for _ in range(test_runs):
                torch.cuda.synchronize(device)
                starter.record()
                _ = model(dummy)
                ender.record()
                torch.cuda.synchronize(device)
                timings_ms.append(float(starter.elapsed_time(ender)))
        else:
            for _ in range(test_runs):
                t0 = time.perf_counter()
                _ = model(dummy)
                t1 = time.perf_counter()
                timings_ms.append((t1 - t0) * 1000.0)

    avg_ms, std_ms, fps, dropped = trim_stats(timings_ms, 5, 95)
    return avg_ms, std_ms, fps, dropped


def write_md(md_path: Path, rows, args, total_sec):
    lines = []
    lines.append("# Batch Speed Benchmark Report")
    lines.append("")
    lines.append(f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- weights_root: {args.weights_root}")
    lines.append(f"- mode: {args.mode}")
    lines.append(f"- device: {args.device if args.device != '' else 'auto'}")
    lines.append(f"- imgsz: {args.imgsz}")
    lines.append(f"- channels: {args.channels}")
    lines.append(f"- batch_size: {args.batch}")
    lines.append(f"- warmup_runs: {args.warmup}")
    lines.append(f"- test_runs: {args.testtime}")
    lines.append(f"- fp16: {args.half}")
    lines.append(f"- total_models: {len(rows)}")
    lines.append(f"- total_elapsed_sec: {total_sec:.2f}")
    lines.append("")
    lines.append("## Leaderboard (Latency asc)")
    lines.append("")
    lines.append("| Rank | Exp | Weight | Device | Avg Latency(ms) | Std(ms) | FPS | Dropped(5%-95%) | Model Size(MB) | OK | Error |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|:---:|---|")

    for i, r in enumerate(rows, 1):
        lines.append(
            "| {rank} | {exp} | {wname} | {dev} | {lat:.4f} | {std:.4f} | {fps:.2f} | {drop} | {size:.2f} | {ok} | {err} |".format(
                rank=i,
                exp=r["exp_name"],
                wname=r["weight_name"],
                dev=r["device"],
                lat=r["avg_latency_ms"] if r["avg_latency_ms"] is not None else -1.0,
                std=r["std_ms"] if r["std_ms"] is not None else -1.0,
                fps=r["fps"] if r["fps"] is not None else -1.0,
                drop=r["dropped_count"] if r["dropped_count"] is not None else -1,
                size=r["model_size_mb"] if r["model_size_mb"] is not None else -1.0,
                ok="Y" if r["ok"] else "N",
                err=(r["error"].replace("|", "/") if r["error"] else "-"),
            )
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser("Batch speed benchmark for all weights")
    parser.add_argument("--weights_root", type=str, default="/root/autodl-tmp/MM-MOE/runs/myDualDataV4")
    parser.add_argument("--mode", type=str, default="best", choices=["best", "last", "both", "all_pt"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--testtime", type=int, default=1000)
    parser.add_argument("--device", type=str, default="0", help="0/1/... or cpu, empty means auto")
    parser.add_argument("--half", action="store_true", default=False)
    parser.add_argument("--out_dir", type=str, default="runs/val/myDualDataV4_speed_batch")
    parser.add_argument("--out_name", type=str, default="all_weights_speed")
    args = parser.parse_args()

    weights_root = Path(args.weights_root)
    weights = discover_weights(weights_root, args.mode)
    if len(weights) == 0:
        print(f"[ERROR] no weights found under: {weights_root}, mode={args.mode}")
        return

    out_dir = Path(args.out_dir) / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = pick_device(args.device)
    rows = []
    t_all = time.time()

    print("=" * 100)
    print(f"Found {len(weights)} weights. Benchmark device: {dev}")
    print("=" * 100)

    for i, w in enumerate(weights, 1):
        exp_name = w.parent.parent.name if w.parent.name == "weights" else w.parent.name
        weight_name = w.name
        model_size_mb = os.path.getsize(w) / 1024 / 1024

        row = {
            "index": i,
            "exp_name": exp_name,
            "weight_name": weight_name,
            "weight_path": str(w),
            "device": str(dev),
            "avg_latency_ms": None,
            "std_ms": None,
            "fps": None,
            "dropped_count": None,
            "model_size_mb": model_size_mb,
            "ok": False,
            "error": "",
        }

        print(f"[{i}/{len(weights)}] {w}")
        try:
            avg_ms, std_ms, fps, dropped = benchmark_one(
                weights_path=str(w),
                device=dev,
                imgsz=args.imgsz,
                channels=args.channels,
                batch_size=args.batch,
                warmup_runs=args.warmup,
                test_runs=args.testtime,
                use_fp16=args.half,
            )
            row["avg_latency_ms"] = float(avg_ms)
            row["std_ms"] = float(std_ms)
            row["fps"] = float(fps)
            row["dropped_count"] = int(dropped)
            row["ok"] = True
            print(f"  OK: latency={avg_ms:.4f} ms, fps={fps:.2f}")
        except Exception as e:
            row["error"] = str(e)
            print(f"  FAIL: {e}")

        rows.append(row)

    total_sec = time.time() - t_all

    rows_sorted = sorted(
        rows,
        key=lambda x: (x["avg_latency_ms"] if isinstance(x["avg_latency_ms"], (int, float)) else 1e18)
    )

    csv_path = out_dir / "all_speed_results.csv"
    md_path = out_dir / "all_speed_results.md"

    fields = [
        "index", "exp_name", "weight_name", "weight_path", "device",
        "avg_latency_ms", "std_ms", "fps", "dropped_count", "model_size_mb",
        "ok", "error"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow(r)

    write_md(md_path, rows_sorted, args, total_sec)

    print("=" * 100)
    print(f"Done. total={len(rows_sorted)}, elapsed={total_sec:.2f}s")
    print(f"CSV: {csv_path}")
    print(f"MD : {md_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()