import os
import re
import csv
import json
import time
import argparse
import warnings
from pathlib import Path

from ultralytics import YOLO

warnings.filterwarnings("ignore")


def pick_metric(results_dict, keys, default=None):
    for k in keys:
        if k in results_dict:
            return results_dict[k]
    return default


def parse_yaml_version_note(yaml_path: Path):
    first_comment = ""
    try:
        with open(yaml_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i > 60:
                    break
                s = line.strip()
                if not s.startswith("#"):
                    continue
                c = s.lstrip("#").strip()
                if c and not first_comment:
                    first_comment = c
                if ":" in c and re.search(r"(?i)\bv\d+(?:[._]\d+)*\b", c):
                    return c
    except Exception:
        return ""
    return first_comment


def discover_weights(weights_root: Path, pattern: str):
    files = sorted(weights_root.rglob(pattern))
    uniq = []
    seen = set()
    for p in files:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def read_model_path_from_args_yaml(args_yaml_path: Path):
    try:
        with open(args_yaml_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.match(r"^\s*model\s*:\s*(.+?)\s*$", line)
                if not m:
                    continue
                val = m.group(1).strip().strip('"').strip("'")
                return val
    except Exception:
        return ""
    return ""


def resolve_model_note_from_weight(weight_path: Path):
    exp_dir = weight_path.parent.parent if weight_path.parent.name == "weights" else weight_path.parent
    args_yaml = exp_dir / "args.yaml"
    if not args_yaml.exists():
        return ""

    model_path = read_model_path_from_args_yaml(args_yaml)
    if not model_path:
        return ""

    model_yaml = Path(model_path)
    if not model_yaml.is_absolute():
        model_yaml = (exp_dir / model_yaml).resolve()

    return parse_yaml_version_note(model_yaml) if model_yaml.exists() else ""


def run_eval_split(yolo_model: YOLO, args, split: str, run_name: str, eval_project: Path):
    one = {
        "precision_B": None,
        "recall_B": None,
        "map50_B": None,
        "map50_95_B": None,
        "fitness": None,
        "elapsed_sec": None,
        "error": "",
    }

    t0 = time.time()
    kwargs = dict(
        data=args.dataset_yaml,
        split=split,
        imgsz=args.imgsz,
        batch=args.eval_batch,
        use_simotm=args.use_simotm,
        channels=args.channels,
        project=str(eval_project),
        name=run_name,
        save_json=args.save_json,
        plots=args.plots,
        verbose=False,
        workers=args.workers,
    )
    if args.device is not None:
        kwargs["device"] = args.device

    try:
        results = yolo_model.val(**kwargs)
        rd = {}
        if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
            rd = results.results_dict

        one["precision_B"] = pick_metric(rd, ["metrics/precision(B)"])
        one["recall_B"] = pick_metric(rd, ["metrics/recall(B)"])
        one["map50_B"] = pick_metric(rd, ["metrics/mAP50(B)"])
        one["map50_95_B"] = pick_metric(rd, ["metrics/mAP50-95(B)"])
        one["fitness"] = pick_metric(rd, ["fitness"])

        if hasattr(results, "box"):
            if one["map50_B"] is None and hasattr(results.box, "map50"):
                one["map50_B"] = results.box.map50
            if one["map50_95_B"] is None and hasattr(results.box, "map"):
                one["map50_95_B"] = results.box.map

    except Exception as e:
        one["error"] = str(e)

    one["elapsed_sec"] = time.time() - t0
    return one


def main():
    parser = argparse.ArgumentParser("Batch val/test eval in one pass")
    parser.add_argument("--weights_root", type=str, required=True, help="目录1：权重目录")
    parser.add_argument("--out_root", type=str, required=True, help="目录2：输出目录(含中间结果)")
    parser.add_argument("--dataset_yaml", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="best.pt", help="best.pt / last.pt / *.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--use_simotm", type=str, default="RGBRGB6C")
    parser.add_argument("--eval_batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0", help="0/1/... or cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save_json", action="store_true", default=False)
    parser.add_argument("--plots", action="store_true", default=False)
    parser.add_argument("--out_name", type=str, default="all_weights_joint_eval")
    args = parser.parse_args()

    weights_root = Path(args.weights_root)
    out_dir = Path(args.out_root) / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_project = out_dir / "tmp_eval"
    eval_project.mkdir(parents=True, exist_ok=True)

    weight_files = discover_weights(weights_root, args.pattern)
    if len(weight_files) == 0:
        print(f"[ERROR] No weight file matched: {args.pattern}")
        print(f"[ERROR] Search root: {weights_root}")
        return

    print("=" * 110)
    print(f"Found {len(weight_files)} weights under: {weights_root}")
    print("Model yaml source: <exp_dir>/args.yaml -> model")
    print(f"Eval device: {args.device}")
    print(f"Output root: {out_dir}")
    print("=" * 110)

    rows = []
    start_all = time.time()

    for i, w in enumerate(weight_files, 1):
        exp_name = w.parent.parent.name if w.parent.name == "weights" else w.parent.name
        run_base = f"{exp_name}_{w.stem}"

        model_version_note = resolve_model_note_from_weight(w)

        one = {
            "index": i,
            "exp_name": exp_name,
            "model_version_note": model_version_note,
            "model_size_mb": os.path.getsize(w) / 1024 / 1024,
            "val_precision_B": None,
            "val_recall_B": None,
            "val_map50_B": None,
            "val_map50_95_B": None,
            "val_fitness": None,
            "val_elapsed_sec": None,
            "val_error": "",
            "test_precision_B": None,
            "test_recall_B": None,
            "test_map50_B": None,
            "test_map50_95_B": None,
            "test_fitness": None,
            "test_elapsed_sec": None,
            "test_error": "",
            "ok": False,
            "error": "",
            "elapsed_sec": None,
        }

        print("\n" + "-" * 110)
        print(f"[{i}/{len(weight_files)}] {w}")
        print("-" * 110)

        t0 = time.time()
        try:
            yolo_model = YOLO(str(w))

            val_res = run_eval_split(yolo_model, args, "val", f"{run_base}_val", eval_project)
            for k, v in val_res.items():
                one[f"val_{k}"] = v

            test_res = run_eval_split(yolo_model, args, "test", f"{run_base}_test", eval_project)
            for k, v in test_res.items():
                one[f"test_{k}"] = v

            if one["val_error"] or one["test_error"]:
                one["error"] = f"val_error={one['val_error']}; test_error={one['test_error']}"
            one["ok"] = not one["error"]

        except Exception as e:
            one["error"] = str(e)

        one["elapsed_sec"] = time.time() - t0
        rows.append(one)

        print(
            "[SUMMARY] "
            f"val_mAP50-95={one['val_map50_95_B']} | "
            f"test_mAP50-95={one['test_map50_95_B']} | "
            f"ok={one['ok']} | elapsed={one['elapsed_sec']:.2f}s"
        )
        if one["error"]:
            print(f"[ERROR] {one['error']}")

    total_sec = time.time() - start_all

    rows_sorted = sorted(
        rows,
        key=lambda x: (
            x["test_map50_95_B"] if isinstance(x["test_map50_95_B"], (int, float)) else -1.0,
            x["val_map50_95_B"] if isinstance(x["val_map50_95_B"], (int, float)) else -1.0,
        ),
        reverse=True,
    )

    csv_path = out_dir / "all_joint_results.csv"
    json_path = out_dir / "all_joint_results.json"

    fields = [
        "index", "exp_name", "model_version_note", "model_size_mb",
        "val_precision_B", "val_recall_B", "val_map50_B", "val_map50_95_B", "val_fitness", "val_elapsed_sec", "val_error",
        "test_precision_B", "test_recall_B", "test_map50_B", "test_map50_95_B", "test_fitness", "test_elapsed_sec", "test_error",
        "ok", "error", "elapsed_sec"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow(r)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "weights_root": str(weights_root),
                "out_root": str(out_dir),
                "count": len(rows_sorted),
                "total_elapsed_sec": total_sec,
                "args": vars(args),
                "results": rows_sorted,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 110)
    print(f"Done. Total models: {len(rows_sorted)}, total time: {total_sec:.2f}s")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print("=" * 110)

    print("\nTop 10 by test mAP50-95(B):")
    for k, r in enumerate(rows_sorted[:10], 1):
        print(
            f"{k:02d}. {r['exp_name']:<40} "
            f"test_mAP50-95={r['test_map50_95_B']}  "
            f"val_mAP50-95={r['val_map50_95_B']}  "
            f"ok={r['ok']}"
        )


if __name__ == "__main__":
    main()