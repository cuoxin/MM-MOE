import argparse
import csv
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch


@dataclass
class TrialResult:
    name: str
    params: Dict
    map5095: float
    collapse_max: float
    inference_ms: float
    fps: float
    score: float
    save_dir: Path


def resolve_actual_save_dir(project_dir: Path, trial_name: str) -> Optional[Path]:
    """Resolve actual run directory, handling Ultralytics auto-suffix naming (e.g., trial, trial2, trial3)."""
    if not project_dir.exists():
        return None

    candidates = [p for p in project_dir.iterdir() if p.is_dir() and p.name.startswith(trial_name)]
    if not candidates:
        return None

    # Prefer folders that already produced results.csv.
    with_results = [p for p in candidates if (p / "results.csv").exists()]
    if with_results:
        return max(with_results, key=lambda p: p.stat().st_mtime)

    # Fall back to latest modified candidate.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_last_map5095(results_csv: Path):
    if not results_csv.exists():
        return None
    with results_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    last = rows[-1]
    for k in ["metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50-95(M)"]:
        if k in last:
            try:
                return float(last[k])
            except Exception:
                pass
    return None


def parse_collapse_from_moe_states(moe_states_txt: Path):
    # Collapse proxy = maximum expert selection ratio (%) observed in file.
    if not moe_states_txt.exists():
        return None

    max_ratio = 0.0
    with moe_states_txt.open("r", encoding="utf-8") as f:
        for line in f:
            if "[Select%]" not in line:
                continue
            # Example: Exp0:  56.7% | Exp1: 17.8% ...
            parts = line.split("|")
            for p in parts:
                p = p.strip()
                if "%" not in p or "Exp" not in p:
                    continue
                try:
                    pct = float(p.split(":")[-1].replace("%", "").strip())
                    max_ratio = max(max_ratio, pct)
                except Exception:
                    continue
    return max_ratio


def benchmark_inference_ms(weights_path: Path, channels: int, imgsz: int, warmup_runs: int, test_runs: int):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ultralytics import YOLO

    if not weights_path.exists():
        return None, None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(weights_path)).model.to(device).eval()
    x = torch.randn(1, channels, imgsz, imgsz, device=device, dtype=torch.float32)

    if device.type == "cuda":
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(x)
            times = []
            for _ in range(test_runs):
                torch.cuda.synchronize()
                starter.record()
                _ = model(x)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))
        avg_ms = sum(times) / max(len(times), 1)
    else:
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(x)
            import time
            times = []
            for _ in range(test_runs):
                t0 = time.perf_counter()
                _ = model(x)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)
        avg_ms = sum(times) / max(len(times), 1)

    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    return avg_ms, fps


def objective_score(
    map5095: float,
    collapse_max: float,
    collapse_threshold: float,
    penalty_alpha: float,
    inference_ms: Optional[float],
    speed_target_ms: float,
    speed_penalty_beta: float,
    collapse_hard_max: float,
):
    if map5095 is None:
        return -1e9
    if collapse_max is None:
        collapse_max = 100.0

    if collapse_hard_max > 0 and collapse_max > collapse_hard_max:
        return -1e9

    collapse_over = max(0.0, collapse_max - collapse_threshold) / 100.0
    speed_over = 0.0
    if inference_ms is not None and speed_target_ms > 0:
        speed_over = max(0.0, inference_ms - speed_target_ms) / speed_target_ms

    return map5095 - penalty_alpha * collapse_over - speed_penalty_beta * speed_over


def sample_log_uniform(rng: random.Random, low: float, high: float):
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def parse_int_list(spec: str) -> List[int]:
    vals = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Empty int list spec: {spec}")
    return vals


def run_trial(
    repo_root: Path,
    train_script: Path,
    base_model_yaml: str,
    data: str,
    project: str,
    trial_name: str,
    train_epochs: int,
    common_args: Dict,
    params: Dict,
) -> TrialResult:
    cmd = [
        "/root/miniconda3/envs/MMMOE/bin/python",
        str(train_script),
        "--base-model-yaml",
        base_model_yaml,
        "--data",
        data,
        "--project",
        project,
        "--name",
        trial_name,
        "--epochs",
        str(train_epochs),
        "--batch",
        str(common_args["batch"]),
        "--imgsz",
        str(common_args["imgsz"]),
        "--workers",
        str(common_args["workers"]),
        "--device",
        str(common_args["device"]),
        "--optimizer",
        str(common_args["optimizer"]),
        "--lr0",
        str(common_args["lr0"]),
        "--close-mosaic",
        str(common_args["close_mosaic"]),
        "--channels",
        str(common_args["channels"]),
        "--use-simotm",
        str(common_args["use_simotm"]),
        "--seed",
        str(common_args["seed"]),
        "--omp-threads",
        str(common_args["omp_threads"]),
        "--num-experts",
        str(params["num_experts"]),
        "--top-k",
        str(params["top_k"]),
        "--shared-experts-nums",
        str(params["shared_experts_nums"]),
        "--pass-through-expert-nums",
        str(params["pass_through_expert_nums"]),
        "--decay-steps",
        str(params["decay_steps"]),
        "--loss-weight",
        str(params["loss_weight"]),
        "--noise-multiplier",
        str(params["noise_multiplier"]),
    ]

    if common_args["amp"]:
        cmd.append("--amp")
    if common_args["deterministic"]:
        cmd.append("--deterministic")

    print(f"\n[Search] Running trial {trial_name}")
    print(f"[Search] Params: {params}")

    logs_dir = repo_root / project / "_trial_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    trial_log = logs_dir / f"{trial_name}.log"

    if common_args["show_train_log"]:
        subprocess.run(cmd, cwd=str(repo_root), check=True)
    else:
        with trial_log.open("w", encoding="utf-8") as lf:
            subprocess.run(
                cmd,
                cwd=str(repo_root),
                check=True,
                stdout=lf,
                stderr=subprocess.STDOUT,
            )

    project_dir = repo_root / project
    save_dir = resolve_actual_save_dir(project_dir, trial_name)
    if save_dir is None:
        save_dir = project_dir / trial_name
    map5095 = parse_last_map5095(save_dir / "results.csv")
    collapse = parse_collapse_from_moe_states(save_dir / "moe_states.txt")
    inference_ms = None
    fps = None
    if common_args["benchmark_speed"]:
        weights = save_dir / "weights" / "best.pt"
        inference_ms, fps = benchmark_inference_ms(
            weights,
            channels=common_args["channels"],
            imgsz=common_args["imgsz"],
            warmup_runs=common_args["speed_warmup_runs"],
            test_runs=common_args["speed_test_runs"],
        )

    score = objective_score(
        map5095,
        collapse,
        collapse_threshold=common_args["collapse_threshold"],
        penalty_alpha=common_args["penalty_alpha"],
        inference_ms=inference_ms,
        speed_target_ms=common_args["speed_target_ms"],
        speed_penalty_beta=common_args["speed_penalty_beta"],
        collapse_hard_max=common_args["collapse_hard_max"],
    )

    if map5095 is None:
        map5095 = -1.0
    if collapse is None:
        collapse = 100.0
    if inference_ms is None:
        inference_ms = -1.0
    if fps is None:
        fps = -1.0

    return TrialResult(
        name=trial_name,
        params=params,
        map5095=map5095,
        collapse_max=collapse,
        inference_ms=inference_ms,
        fps=fps,
        score=score,
        save_dir=save_dir,
    )


def build_stage1_candidates(rng: random.Random, n: int, search_space: Dict) -> List[Dict]:
    candidates = []
    for _ in range(n):
        candidates.append(
            {
                "num_experts": rng.choice(search_space["num_experts"]),
                "top_k": rng.choice(search_space["top_k"]),
                "shared_experts_nums": rng.choice(search_space["shared_experts_nums"]),
                "pass_through_expert_nums": rng.choice(search_space["pass_through_expert_nums"]),
                "decay_steps": rng.choice(search_space["decay_steps"]),
                "loss_weight": sample_log_uniform(
                    rng, search_space["loss_weight_min"], search_space["loss_weight_max"]
                ),
                "noise_multiplier": sample_log_uniform(
                    rng, search_space["noise_multiplier_min"], search_space["noise_multiplier_max"]
                ),
            }
        )
    return candidates


def build_stage2_candidates(rng: random.Random, base: Dict, n: int, search_space: Dict) -> List[Dict]:
    # Local refinement around current best.
    candidates = []
    for _ in range(n):
        p = dict(base)
        p["loss_weight"] = min(0.2, max(0.001, base["loss_weight"] * sample_log_uniform(rng, 0.6, 1.8)))
        p["noise_multiplier"] = min(0.8, max(0.01, base["noise_multiplier"] * sample_log_uniform(rng, 0.6, 1.8)))
        p["decay_steps"] = int(min(6000, max(500, base["decay_steps"] * sample_log_uniform(rng, 0.7, 1.5))))

        if rng.random() < 0.2 and len(search_space["top_k"]) > 1:
            choices = [v for v in search_space["top_k"] if v != p["top_k"]]
            p["top_k"] = rng.choice(choices)
        if rng.random() < 0.2 and len(search_space["num_experts"]) > 1:
            choices = [v for v in search_space["num_experts"] if v != p["num_experts"]]
            p["num_experts"] = rng.choice(choices)
        if rng.random() < 0.2 and len(search_space["shared_experts_nums"]) > 1:
            choices = [v for v in search_space["shared_experts_nums"] if v != p["shared_experts_nums"]]
            p["shared_experts_nums"] = rng.choice(choices)
        if rng.random() < 0.2 and len(search_space["pass_through_expert_nums"]) > 1:
            choices = [v for v in search_space["pass_through_expert_nums"] if v != p["pass_through_expert_nums"]]
            p["pass_through_expert_nums"] = rng.choice(choices)

        candidates.append(p)
    return candidates


def build_stage2_candidates_multi_anchor(rng: random.Random, anchors: List[Dict], n: int, search_space: Dict) -> List[Dict]:
    if not anchors:
        return []

    candidates = []
    for i in range(n):
        base = anchors[i % len(anchors)]
        candidates.extend(build_stage2_candidates(rng, base, 1, search_space))
    return candidates[:n]


def save_summary(path: Path, trials: List[TrialResult]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "score",
                "map50_95",
                "collapse_max_pct",
                "inference_ms",
                "fps",
                "num_experts",
                "top_k",
                "shared_experts_nums",
                "pass_through_expert_nums",
                "decay_steps",
                "loss_weight",
                "noise_multiplier",
                "save_dir",
            ]
        )
        for t in sorted(trials, key=lambda x: x.score, reverse=True):
            writer.writerow(
                [
                    t.name,
                    f"{t.score:.6f}",
                    f"{t.map5095:.6f}",
                    f"{t.collapse_max:.2f}",
                    f"{t.inference_ms:.4f}",
                    f"{t.fps:.2f}",
                    t.params["num_experts"],
                    t.params["top_k"],
                    t.params["shared_experts_nums"],
                    t.params["pass_through_expert_nums"],
                    t.params["decay_steps"],
                    f"{t.params['loss_weight']:.8f}",
                    f"{t.params['noise_multiplier']:.8f}",
                    str(t.save_dir),
                ]
            )


def format_progress_bar(done: int, total: int, width: int = 28) -> str:
    total = max(1, total)
    done = max(0, min(done, total))
    ratio = done / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total} ({ratio * 100:.1f}%)"


def print_topk_partial(trials: List[TrialResult], k: int = 3):
    if not trials:
        return
    topk = sorted(trials, key=lambda x: x.score, reverse=True)[:k]
    print("[Search][TopK] Current best trials:")
    for i, t in enumerate(topk, 1):
        print(
            f"  #{i} {t.name} | score={t.score:.6f} | mAP={t.map5095:.6f} | collapse={t.collapse_max:.2f}% | inf={t.inference_ms:.2f}ms | fps={t.fps:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Purposeful 2-stage search for MMMOE V8 hyperparameters.")
    parser.add_argument("--repo-root", type=str, default="/root/autodl-tmp/MM-MOE")
    parser.add_argument("--train-script", type=str, default="/root/autodl-tmp/MM-MOE/train_RGBRGB_MMMOETopk1_V8.py")
    parser.add_argument("--base-model-yaml", type=str, default="/root/autodl-tmp/MM-MOE/ultralytics/cfg/models/11MMMOE/yolo11-RGBT-moe-backboneV8_0.yaml")
    parser.add_argument("--data", type=str, default="/root/autodl-tmp/MM-MOE/ultralytics/cfg/datasets/myDualDataV.yaml")
    parser.add_argument("--project", type=str, default="runs/search_v8")
    parser.add_argument("--tag", type=str, default="v8_moe")

    parser.add_argument("--stage1-trials", type=int, default=8)
    parser.add_argument("--stage1-epochs", type=int, default=12)
    parser.add_argument("--stage2-trials", type=int, default=6)
    parser.add_argument("--stage2-epochs", type=int, default=40)
    parser.add_argument("--stage2-anchors", type=int, default=3)

    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--use-simotm", type=str, default="RGBRGB6C")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--omp-threads", type=int, default=8)

    parser.add_argument("--collapse-threshold", type=float, default=70.0)
    parser.add_argument("--penalty-alpha", type=float, default=0.1)
    parser.add_argument("--benchmark-speed", action="store_true")
    parser.add_argument("--speed-warmup-runs", type=int, default=20)
    parser.add_argument("--speed-test-runs", type=int, default=60)
    parser.add_argument("--speed-target-ms", type=float, default=16.0)
    parser.add_argument("--speed-penalty-beta", type=float, default=0.05)
    parser.add_argument("--collapse-hard-max", type=float, default=95.0)
    parser.add_argument("--show-train-log", action="store_true")

    parser.add_argument("--space-num-experts", type=str, default="4,6")
    parser.add_argument("--space-top-k", type=str, default="1,2")
    parser.add_argument("--space-shared-experts", type=str, default="0,1")
    parser.add_argument("--space-pass-through", type=str, default="1,2")
    parser.add_argument("--space-decay-steps", type=str, default="1000,2080,4000")
    parser.add_argument("--space-loss-weight-min", type=float, default=0.003)
    parser.add_argument("--space-loss-weight-max", type=float, default=0.1)
    parser.add_argument("--space-noise-min", type=float, default=0.05)
    parser.add_argument("--space-noise-max", type=float, default=0.4)

    args = parser.parse_args()

    rng = random.Random(args.seed)
    repo_root = Path(args.repo_root)
    train_script = Path(args.train_script)

    common_args = {
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "device": args.device,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "close_mosaic": args.close_mosaic,
        "channels": args.channels,
        "use_simotm": args.use_simotm,
        "amp": args.amp,
        "deterministic": args.deterministic,
        "seed": args.seed,
        "omp_threads": args.omp_threads,
        "collapse_threshold": args.collapse_threshold,
        "penalty_alpha": args.penalty_alpha,
        "benchmark_speed": args.benchmark_speed,
        "speed_warmup_runs": args.speed_warmup_runs,
        "speed_test_runs": args.speed_test_runs,
        "speed_target_ms": args.speed_target_ms,
        "speed_penalty_beta": args.speed_penalty_beta,
        "collapse_hard_max": args.collapse_hard_max,
        "show_train_log": args.show_train_log,
    }

    search_space = {
        "num_experts": parse_int_list(args.space_num_experts),
        "top_k": parse_int_list(args.space_top_k),
        "shared_experts_nums": parse_int_list(args.space_shared_experts),
        "pass_through_expert_nums": parse_int_list(args.space_pass_through),
        "decay_steps": parse_int_list(args.space_decay_steps),
        "loss_weight_min": args.space_loss_weight_min,
        "loss_weight_max": args.space_loss_weight_max,
        "noise_multiplier_min": args.space_noise_min,
        "noise_multiplier_max": args.space_noise_max,
    }

    all_trials: List[TrialResult] = []
    total_trials = args.stage1_trials + args.stage2_trials
    completed_trials = 0

    print("[Search] Stage-1: global exploration")
    stage1_candidates = build_stage1_candidates(rng, args.stage1_trials, search_space)
    for i, params in enumerate(stage1_candidates, 1):
        print(f"[Search][Progress] {format_progress_bar(completed_trials, total_trials)}")
        trial_name = f"{args.tag}_s1_{i:02d}"
        result = run_trial(
            repo_root,
            train_script,
            args.base_model_yaml,
            args.data,
            args.project,
            trial_name,
            args.stage1_epochs,
            common_args,
            params,
        )
        all_trials.append(result)
        completed_trials += 1
        print(
            f"[Search][S1] {trial_name}: score={result.score:.6f}, mAP50-95={result.map5095:.6f}, collapse={result.collapse_max:.2f}%, inf={result.inference_ms:.2f}ms, fps={result.fps:.2f}"
        )
        print_topk_partial(all_trials, k=3)

    stage1_sorted = sorted(all_trials, key=lambda x: x.score, reverse=True)
    best_stage1 = stage1_sorted[0]
    anchor_count = max(1, min(args.stage2_anchors, len(stage1_sorted)))
    anchors = [t.params for t in stage1_sorted[:anchor_count]]
    print(f"[Search] Stage-1 best: {best_stage1.name}, params={best_stage1.params}")
    print(f"[Search] Stage-2 anchors: {anchor_count}")

    print("[Search] Stage-2: local refinement around Stage-1 best")
    stage2_candidates = build_stage2_candidates_multi_anchor(rng, anchors, args.stage2_trials, search_space)
    for i, params in enumerate(stage2_candidates, 1):
        print(f"[Search][Progress] {format_progress_bar(completed_trials, total_trials)}")
        trial_name = f"{args.tag}_s2_{i:02d}"
        result = run_trial(
            repo_root,
            train_script,
            args.base_model_yaml,
            args.data,
            args.project,
            trial_name,
            args.stage2_epochs,
            common_args,
            params,
        )
        all_trials.append(result)
        completed_trials += 1
        print(
            f"[Search][S2] {trial_name}: score={result.score:.6f}, mAP50-95={result.map5095:.6f}, collapse={result.collapse_max:.2f}%, inf={result.inference_ms:.2f}ms, fps={result.fps:.2f}"
        )
        print_topk_partial(all_trials, k=3)

    print(f"[Search][Progress] {format_progress_bar(completed_trials, total_trials)}")

    final_sorted = sorted(all_trials, key=lambda x: x.score, reverse=True)
    best = final_sorted[0]
    summary_path = repo_root / args.project / f"{args.tag}_summary.csv"
    save_summary(summary_path, final_sorted)

    print("\n[Search] Done")
    print(f"[Search] Best trial: {best.name}")
    print(f"[Search] Best params: {best.params}")
    print(f"[Search] Best score: {best.score:.6f}")
    print(f"[Search] Best mAP50-95: {best.map5095:.6f}")
    print(f"[Search] Best collapse: {best.collapse_max:.2f}%")
    print(f"[Search] Best inference: {best.inference_ms:.2f} ms, fps={best.fps:.2f}")
    print(f"[Search] Summary: {summary_path}")


if __name__ == "__main__":
    main()
