import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def simplify_name(name: str) -> str:
	s = str(name or "").strip()
	if s.startswith("MMOEV"):
		return s.replace("MMOEV", "V")
	if s.lower().startswith("baseline"):
		s = s.replace("baseline", "BL").replace("Baseline", "BL")
		return s.replace("_", "-")
	s = s.replace("_", "-")
	return s[:16] if len(s) > 16 else s


def pareto_mask_maximize(x, y):
	n = len(x)
	mask = [True] * n
	for i in range(n):
		for j in range(n):
			if i == j:
				continue
			dominates = (x[j] >= x[i] and y[j] >= y[i]) and (x[j] > x[i] or y[j] > y[i])
			if dominates:
				mask[i] = False
				break
	return mask


def to_float(v):
	try:
		return float(v)
	except Exception:
		return None


def load_metrics_csv(metrics_csv: Path):
	needed = [
		"exp_name",
		"test_map50_B",
		"test_map50_95_B",
		"val_map50_B",
		"val_map50_95_B",
	]

	rows = []
	with open(metrics_csv, "r", encoding="utf-8-sig", newline="") as f:
		reader = csv.DictReader(f)
		if reader.fieldnames is None:
			raise ValueError("Metrics CSV has no header")
		for c in needed:
			if c not in reader.fieldnames:
				raise ValueError(f"Missing column in metrics CSV: {c}")

		for r in reader:
			exp_name = (r.get("exp_name", "") or "").strip()
			if not exp_name:
				continue
			one = {"exp_name": exp_name}
			for c in needed[1:]:
				one[c] = to_float(r.get(c, ""))
			rows.append(one)
	return rows


def load_speed_csv(speed_csv: Path):
	rows = []
	with open(speed_csv, "r", encoding="utf-8-sig", newline="") as f:
		reader = csv.DictReader(f)
		if reader.fieldnames is None:
			raise ValueError("Speed CSV has no header")

		fps_col = None
		for c in ["fps", "speed_fps"]:
			if c in reader.fieldnames:
				fps_col = c
				break
		if fps_col is None:
			raise ValueError("Missing fps column in speed CSV (expect 'fps' or 'speed_fps')")
		if "exp_name" not in reader.fieldnames:
			raise ValueError("Missing column in speed CSV: exp_name")

		for r in reader:
			exp_name = (r.get("exp_name", "") or "").strip()
			if not exp_name:
				continue
			rows.append({"exp_name": exp_name, "speed_fps": to_float(r.get(fps_col, ""))})
	return rows


def prepare_data(metrics_csv: Path, speed_csv: Path):
	metrics_rows = load_metrics_csv(metrics_csv)
	speed_rows = load_speed_csv(speed_csv)

	fps_by_name = {}
	for r in speed_rows:
		if r["speed_fps"] is not None:
			fps_by_name[r["exp_name"]] = r["speed_fps"]

	rows = []
	for m in metrics_rows:
		exp_name = m["exp_name"]
		if exp_name not in fps_by_name:
			continue
		one = dict(m)
		one["speed_fps"] = fps_by_name[exp_name]
		one["short_name"] = simplify_name(exp_name)
		rows.append(one)

	print(f"[INFO] metrics rows={len(metrics_rows)}, speed rows={len(speed_rows)}, merged rows={len(rows)}")
	return rows


def plot_one(df, metric_col: str, metric_label: str, out_file: Path):
	sub = [r for r in df if r.get(metric_col) is not None and r.get("speed_fps") is not None]
	if len(sub) == 0:
		print(f"[WARN] Skip {metric_col}: no valid rows")
		return

	x = [r["speed_fps"] for r in sub]
	y = [r[metric_col] for r in sub]
	mask = pareto_mask_maximize(x, y)
	pareto = [sub[i] for i, keep in enumerate(mask) if keep]
	pareto = sorted(pareto, key=lambda r: (r["speed_fps"], r[metric_col]))

	plt.figure(figsize=(10, 7), dpi=150)
	plt.scatter(
		[r["speed_fps"] for r in sub],
		[r[metric_col] for r in sub],
		s=36,
		alpha=0.55,
		label="All models",
	)
	plt.scatter(
		[r["speed_fps"] for r in pareto],
		[r[metric_col] for r in pareto],
		s=58,
		alpha=0.95,
		marker="D",
		label="Pareto frontier",
	)

	if len(pareto) >= 2:
		plt.plot([r["speed_fps"] for r in pareto], [r[metric_col] for r in pareto], linewidth=1.8)

	for r in pareto:
		plt.annotate(
			r["short_name"],
			(r["speed_fps"], r[metric_col]),
			textcoords="offset points",
			xytext=(4, 4),
			fontsize=8,
		)

	plt.xlabel("FPS")
	plt.ylabel(metric_label)
	plt.title(f"Pareto Frontier: {metric_label} vs FPS")
	plt.grid(alpha=0.25)
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_file)
	plt.close()
	print(f"[OK] Saved: {out_file}")


def main():
	parser = argparse.ArgumentParser("Plot Pareto frontiers from metrics CSV + speed CSV")
	parser.add_argument(
		"--metrics_csv",
		type=str,
		default="/root/autodl-tmp/MM-MOE/runs/val/final_test/all_weights_joint_eval/all_joint_results.csv",
		help="Path to metrics CSV (all_joint_results.csv)",
	)
	parser.add_argument(
		"--speed_csv",
		type=str,
		default="/root/autodl-tmp/MM-MOE/runs/val/final_test/all_weights_speed/all_speed_results.csv",
		help="Path to speed CSV (all_speed_results.csv)",
	)
	parser.add_argument(
		"--out_dir",
		type=str,
		default="",
		help="Output directory for plots (default: <metrics_csv_dir>/pareto_plots)",
	)
	args = parser.parse_args()

	metrics_csv_path = Path(args.metrics_csv)
	speed_csv_path = Path(args.speed_csv)
	if not metrics_csv_path.exists():
		raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv_path}")
	if not speed_csv_path.exists():
		raise FileNotFoundError(f"Speed CSV not found: {speed_csv_path}")

	out_dir = Path(args.out_dir) if args.out_dir else (metrics_csv_path.parent / "pareto_plots")
	out_dir.mkdir(parents=True, exist_ok=True)

	df = prepare_data(metrics_csv_path, speed_csv_path)

	plot_one(df, "test_map50_B", "test mAP50", out_dir / "pareto_test_map50_vs_fps.png")
	plot_one(df, "test_map50_95_B", "test mAP50-95", out_dir / "pareto_test_map50_95_vs_fps.png")
	plot_one(df, "val_map50_B", "val mAP50", out_dir / "pareto_val_map50_vs_fps.png")
	plot_one(df, "val_map50_95_B", "val mAP50-95", out_dir / "pareto_val_map50_95_vs_fps.png")

	print("[DONE] Generated 4 Pareto plots.")


if __name__ == "__main__":
	main()
