import os
import csv
import json
import time
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from ultralytics import YOLO


def pick_metric(results_dict, keys, default=None):
    for k in keys:
        if k in results_dict:
            return results_dict[k]
    return default


def main():
    parser = argparse.ArgumentParser("Batch validate all weights in a directory")
    parser.add_argument("--weights_root", type=str, default="/root/autodl-tmp/MM-MOE/runs/myDualDataV4")
    parser.add_argument("--dataset_yaml", type=str, default="/root/autodl-tmp/MM-MOE/ultralytics/cfg/datasets/myDualData.yaml")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--use_simotm", type=str, default="RGBRGB6C")
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--pattern", type=str, default="best.pt", help="best.pt or last.pt")
    parser.add_argument("--project", type=str, default="runs/val/myDualDataV4_batch")
    parser.add_argument("--name", type=str, default="all_weights_eval")
    parser.add_argument("--device", type=str, default=None, help="e.g. 0 or cpu")
    parser.add_argument("--save_json", action="store_true", default=False)
    parser.add_argument("--plots", action="store_true", default=False)
    args = parser.parse_args()

    weights_root = Path(args.weights_root)
    weight_files = sorted(weights_root.rglob(args.pattern))

    if len(weight_files) == 0:
        print(f"[ERROR] No weight file matched: {args.pattern}")
        print(f"[ERROR] Search root: {weights_root}")
        return

    print("=" * 90)
    print(f"Found {len(weight_files)} weights under: {weights_root}")
    print("=" * 90)

    rows = []
    start_all = time.time()

    for i, w in enumerate(weight_files, 1):
        exp_name = w.parent.parent.name  # .../exp_name/weights/best.pt
        run_name = f"{exp_name}_{args.split}"

        print("\n" + "-" * 90)
        print(f"[{i}/{len(weight_files)}] Evaluating: {w}")
        print("-" * 90)

        one = {
            "index": i,
            "exp_name": exp_name,
            "weight_path": str(w),
            "ok": False,
            "error": "",
            "precision_B": None,
            "recall_B": None,
            "map50_B": None,
            "map50_95_B": None,
            "fitness": None,
            "preprocess_ms": None,
            "inference_ms": None,
            "postprocess_ms": None,
            "elapsed_sec": None,
        }

        t0 = time.time()
        try:
            model = YOLO(str(w))

            kwargs = dict(
                data=args.dataset_yaml,
                split=args.split,
                imgsz=args.imgsz,
                batch=args.batch,
                use_simotm=args.use_simotm,
                channels=args.channels,
                project=args.project,
                name=run_name,
                save_json=args.save_json,
                plots=args.plots,
                verbose=False,
                workers=0
            )
            if args.device is not None:
                kwargs["device"] = args.device

            results = model.val(**kwargs)

            # 通用字典指标
            rd = {}
            if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
                rd = results.results_dict

            one["precision_B"] = pick_metric(rd, ["metrics/precision(B)"])
            one["recall_B"] = pick_metric(rd, ["metrics/recall(B)"])
            one["map50_B"] = pick_metric(rd, ["metrics/mAP50(B)"])
            one["map50_95_B"] = pick_metric(rd, ["metrics/mAP50-95(B)"])
            one["fitness"] = pick_metric(rd, ["fitness"])

            # 兜底：若上面没拿到，尝试 box 属性
            if hasattr(results, "box"):
                if one["map50_B"] is None and hasattr(results.box, "map50"):
                    one["map50_B"] = results.box.map50
                if one["map50_95_B"] is None and hasattr(results.box, "map"):
                    one["map50_95_B"] = results.box.map

            # 速度
            if hasattr(results, "speed") and isinstance(results.speed, dict):
                one["preprocess_ms"] = results.speed.get("preprocess", None)
                one["inference_ms"] = results.speed.get("inference", None)
                one["postprocess_ms"] = results.speed.get("postprocess", None)

            one["ok"] = True

        except Exception as e:
            one["error"] = str(e)

        one["elapsed_sec"] = time.time() - t0
        rows.append(one)

        if one["ok"]:
            print(
                "[OK] "
                f"P={one['precision_B']}, R={one['recall_B']}, "
                f"mAP50={one['map50_B']}, mAP50-95={one['map50_95_B']}, "
                f"infer={one['inference_ms']} ms, elapsed={one['elapsed_sec']:.2f}s"
            )
        else:
            print(f"[FAIL] {one['error']}")

    total_sec = time.time() - start_all

    # 排序：优先按 mAP50-95，再按 mAP50
    def sort_key(x):
        a = x["map50_95_B"] if x["map50_95_B"] is not None else -1.0
        b = x["map50_B"] if x["map50_B"] is not None else -1.0
        return (a, b)

    rows_sorted = sorted(rows, key=sort_key, reverse=True)

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "all_results.csv"
    json_path = out_dir / "all_results.json"

    fields = [
        "index", "exp_name", "weight_path", "ok", "error",
        "precision_B", "recall_B", "map50_B", "map50_95_B", "fitness",
        "preprocess_ms", "inference_ms", "postprocess_ms",
        "elapsed_sec"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow(r)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "weights_root": str(weights_root),
                "count": len(rows_sorted),
                "total_elapsed_sec": total_sec,
                "args": vars(args),
                "results": rows_sorted,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 90)
    print(f"Done. Total models: {len(rows_sorted)}, total time: {total_sec:.2f}s")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print("=" * 90)

    print("\nTop 10 by mAP50-95(B):")
    for k, r in enumerate(rows_sorted[:10], 1):
        print(
            f"{k:02d}. {r['exp_name']:<40} "
            f"mAP50-95={r['map50_95_B']}  mAP50={r['map50_B']}  ok={r['ok']}"
        )


if __name__ == "__main__":
    main()