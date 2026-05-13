import argparse
import csv
import math
import sys
from pathlib import Path

import torch

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "OmegaConf is not installed. Install it with `pip install omegaconf hydra-core` "
        "and then rerun this script."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Energies.targets import create_target_from_config
from Testing.run_eval import run_evaluation, save_projection_scatter_plots
from Training.run_training import _build_mixture, _build_training_state, _resolve_device
from Training.train import train_and_save


DEFAULT_SEEDS = [40, 41, 42]
DEFAULT_SAMPLERS = ["nets", "ais"]
DEFAULT_DIMENSION_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train linear interpolation on the 16d 2-GMM with covariance scales [1, 3] across "
            "multiple seeds, then evaluate the final saved checkpoints."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--samplers", nargs="+", default=DEFAULT_SAMPLERS, choices=["nets", "ais"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-walkers", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--modal-loss-weight", type=float, default=1e5)
    parser.add_argument("--modal-loss-end-fraction", type=float, default=0.4)
    parser.add_argument("--loss-type", default="manual", choices=["manual", "ctds"])
    parser.add_argument("--eval-num-samples", type=int, default=512)
    parser.add_argument("--eval-true-sample-count", type=int, default=512)
    parser.add_argument("--eval-dt", type=float, default=0.02)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--eval-epsilon", type=float, default=1.0)
    parser.add_argument("--plot-every", type=int, default=0)
    parser.add_argument("--checkpoint-name", default="final")
    parser.add_argument(
        "--dimension-pairs",
        nargs="*",
        default=[f"{a},{b}" for a, b in DEFAULT_DIMENSION_PAIRS],
        help='Pairs like "0,1" "2,3" used for 2D projection plots.',
    )
    parser.add_argument(
        "--sweep-name",
        default="16d_2gmm_covs_1_3_linear_multiseed",
        help="Output directory name under mean_interpolation/ for the aggregate sweep summaries.",
    )
    parser.add_argument(
        "--ctds-root",
        default=None,
        help=(
            "Optional root directory containing CTDS artifacts with ELBO/EUBO summaries. "
            "The script will try to match files by seed/run name and use those values in preference "
            "to eval-derived ELBO/EUBO."
        ),
    )
    return parser.parse_args()


def _parse_dimension_pairs(raw_pairs):
    parsed = []
    for item in raw_pairs:
        left, right = item.split(",", maxsplit=1)
        parsed.append((int(left), int(right)))
    return parsed


def _component_balance_suffix(modal_loss_weight: float) -> str:
    return "_no_component_balance" if float(modal_loss_weight) == 0.0 else ""


def _base_config(args):
    return OmegaConf.create(
        {
            "seed": int(args.seeds[0]),
            "device": args.device,
            "training": {
                "n_walkers": int(args.n_walkers),
                "steps": int(args.train_steps),
                "epsilon": float(args.epsilon),
                "K": int(args.K),
                "modal_loss_weight": float(args.modal_loss_weight),
                "modal_loss_end_fraction": float(args.modal_loss_end_fraction),
                "loss_type": str(args.loss_type),
            },
            "data": {
                "target": "gmm",
                "dim": 16,
                "num_components": 2,
                "covs": None,
                "weights": [0.3, 0.7],
                "means_source": None,
                "apply_mean_layout": True,
                "mean_layout": "pair_shift",
                "layout_scale": 10.0,
                "perturb_mean": 10.0,
                "true_sample_count": int(args.eval_true_sample_count),
                "run_name": "",
                "covariance": {
                    "style": "scaled_identity",
                    "scales": [1.0, 3.0],
                    "diagonal_scales": None,
                    "random_rotation_per_component": True,
                },
            },
            "model": {
                "interpolation_kind": "fixed",
                "beta_max": float(args.beta_max),
                "perturbation": 0.0,
                "use_time_logits": False,
            },
            "eval": {
                "sampler": "nets",
                "checkpoint_name": str(args.checkpoint_name),
                "num_samples": int(args.eval_num_samples),
                "true_sample_count": int(args.eval_true_sample_count),
                "dt": float(args.eval_dt),
                "steps": int(args.eval_steps),
                "epsilon": float(args.eval_epsilon),
                "plot_every": int(args.plot_every),
            },
        }
    )


def _train_once(cfg):
    device = _resolve_device(cfg.device)
    torch.manual_seed(int(cfg.seed))

    target = create_target_from_config(cfg, device)
    if target is None:
        target = _build_mixture(cfg, device)
    state = _build_training_state(cfg, target, device)

    train_and_save(
        dim=cfg.data.dim,
        num_components=cfg.data.num_components,
        mixture=target,
        means=state["means"],
        U_net=state["U_net"],
        modes=state["modes"],
        device=device,
        prior=state["prior"],
        energy_model=state["energy_model"],
        prior_samples=state["prior_samples"],
        n_walkers=cfg.training.n_walkers,
        steps=cfg.training.steps,
        epsilon=cfg.training.epsilon,
        K=cfg.training.K,
        modal_loss_weight=float(cfg.training.get("modal_loss_weight", 0.0)),
        modal_loss_end_fraction=float(cfg.training.get("modal_loss_end_fraction", 0.6)),
        loss_type=str(cfg.training.get("loss_type", "manual")),
        true_modes=state["true_modes"],
        true_samples=state["true_samples"],
        interpolation_kind=cfg.model.interpolation_kind,
        run_name=str(cfg.data.get("run_name", "")) or None,
    )


def _read_csv_rows(path: Path):
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _extract_elbo_eubo_from_csv(path: Path):
    rows = _read_csv_rows(path)
    if not rows:
        return None

    for row in reversed(rows):
        elbo = _to_float(row.get("elbo"))
        eubo = _to_float(row.get("eubo"))
        if elbo is not None or eubo is not None:
            return {"elbo": elbo, "eubo": eubo, "source": str(path)}
    return None


def _extract_elbo_eubo_from_pth(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    candidates = []

    if isinstance(obj, dict):
        candidates.append(obj)
    elif isinstance(obj, list):
        for item in reversed(obj):
            if isinstance(item, dict):
                candidates.append(item)

    for candidate in candidates:
        elbo = _to_float(candidate.get("elbo"))
        eubo = _to_float(candidate.get("eubo"))
        if elbo is not None or eubo is not None:
            return {"elbo": elbo, "eubo": eubo, "source": str(path)}
    return None


def _load_ctds_elbo_eubo(ctds_root: Path | None, run_name: str, seed: int):
    if ctds_root is None or not ctds_root.exists():
        return None

    patterns = [
        f"*{run_name}*",
        f"*seed_{seed}*",
        f"*{seed}*",
    ]
    candidate_files = []
    for pattern in patterns:
        candidate_files.extend(ctds_root.rglob(pattern))

    prioritized = []
    for path in candidate_files:
        if path.is_file() and path.suffix.lower() in {".csv", ".pth"}:
            prioritized.append(path)

    prioritized = sorted(
        set(prioritized),
        key=lambda path: (
            0 if "elbo" in path.name.lower() or "eubo" in path.name.lower() else 1,
            0 if "metrics" in str(path).lower() or "summary" in str(path).lower() else 1,
            len(str(path)),
        ),
    )

    for path in prioritized:
        try:
            if path.suffix.lower() == ".csv":
                match = _extract_elbo_eubo_from_csv(path)
            else:
                match = _extract_elbo_eubo_from_pth(path)
        except Exception:
            continue
        if match is not None:
            return match
    return None


def _row_from_results(seed, sampler, run_name, results, ctds_metrics=None):
    final_metrics = dict(results["final_metrics"])
    elbo = final_metrics.get("elbo", float("nan"))
    eubo = final_metrics.get("eubo", float("nan"))
    elbo_source = "eval"
    eubo_source = "eval"

    if ctds_metrics is not None:
        if ctds_metrics.get("elbo") is not None:
            elbo = ctds_metrics["elbo"]
            elbo_source = ctds_metrics["source"]
        if ctds_metrics.get("eubo") is not None:
            eubo = ctds_metrics["eubo"]
            eubo_source = ctds_metrics["source"]

    row = {
        "seed": int(seed),
        "sampler": sampler,
        "run_name": run_name,
        "checkpoint_name": results["metadata"]["checkpoint_name"],
        "weighted_w2": final_metrics.get("weighted_w2", float("nan")),
        "elbo": elbo,
        "eubo": eubo,
        "elbo_source": elbo_source,
        "eubo_source": eubo_source,
        "ess_count": final_metrics.get("ess_count", float("nan")),
        "mmd": final_metrics.get("mmd", float("nan")),
        "w2": final_metrics.get("w2", float("nan")),
    }

    mode_keys = sorted(
        [key for key in final_metrics if key.startswith("mode_weight_")],
        key=lambda key: int(key.split("_")[-1]),
    )
    for key in mode_keys:
        row[key] = final_metrics[key]
    return row


def _write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    header = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(rows):
    if not rows:
        return []

    metric_keys = [
        key
        for key in rows[0]
        if key not in {"seed", "sampler", "run_name", "checkpoint_name", "elbo_source", "eubo_source"}
    ]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["sampler"], []).append(row)

    aggregate_rows = []
    for sampler, sampler_rows in grouped.items():
        summary = {
            "sampler": sampler,
            "num_seeds": len(sampler_rows),
        }
        for key in metric_keys:
            values = [float(row[key]) for row in sampler_rows if key in row and not math.isnan(float(row[key]))]
            if values:
                mean = sum(values) / len(values)
                var = sum((value - mean) ** 2 for value in values) / len(values)
                summary[f"{key}_mean"] = mean
                summary[f"{key}_std"] = math.sqrt(var)
            else:
                summary[f"{key}_mean"] = float("nan")
                summary[f"{key}_std"] = float("nan")
        aggregate_rows.append(summary)
    return aggregate_rows


def main():
    args = _parse_args()
    dimension_pairs = _parse_dimension_pairs(args.dimension_pairs)
    base_cfg = _base_config(args)
    ctds_root = None if args.ctds_root is None else Path(args.ctds_root)
    balance_suffix = _component_balance_suffix(base_cfg.training.modal_loss_weight)

    sweep_dir = PROJECT_ROOT / "mean_interpolation" / f"{args.sweep_name}{balance_suffix}"
    summary_dir = sweep_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    with (summary_dir / "config.yaml").open("w") as handle:
        handle.write(OmegaConf.to_yaml(base_cfg))
        handle.write(f"seeds: {list(args.seeds)}\n")
        handle.write(f"samplers: {list(args.samplers)}\n")
        handle.write(f"dimension_pairs: {dimension_pairs}\n")
        handle.write(f"ctds_root: {ctds_root}\n")

    all_rows = []
    for seed in args.seeds:
        run_name = f"16d_2gmm_covs_1_3_linear_seed_{seed}{balance_suffix}"
        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        cfg.seed = int(seed)
        cfg.data.run_name = run_name

        print("=" * 80)
        print(f"Training seed {seed} -> run_name={run_name}")
        _train_once(cfg)

        for sampler in args.samplers:
            cfg.eval.sampler = sampler
            print("-" * 80)
            print(f"Evaluating seed {seed} with sampler={sampler} from final checkpoint")
            results = run_evaluation(cfg)
            ctds_metrics = _load_ctds_elbo_eubo(ctds_root, run_name, seed)
            if ctds_metrics is not None:
                print(
                    f"Using CTDS ELBO/EUBO for seed {seed} sampler={sampler} "
                    f"from {ctds_metrics['source']}"
                )
            save_projection_scatter_plots(
                generated_samples=results["final_samples"],
                true_samples=results["true_samples"],
                weights=results["final_weights"],
                save_dir=results["plots_dir"] / "projections",
                dimension_pairs=dimension_pairs,
            )
            all_rows.append(_row_from_results(seed, sampler, run_name, results, ctds_metrics=ctds_metrics))

    aggregate_rows = _aggregate_rows(all_rows)
    _write_csv(all_rows, summary_dir / "per_seed_metrics.csv")
    _write_csv(aggregate_rows, summary_dir / "aggregate_metrics.csv")
    torch.save(all_rows, summary_dir / "per_seed_metrics.pth")
    torch.save(aggregate_rows, summary_dir / "aggregate_metrics.pth")

    print("=" * 80)
    print(f"Saved per-seed metrics to {summary_dir / 'per_seed_metrics.csv'}")
    print(f"Saved aggregate metrics to {summary_dir / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
