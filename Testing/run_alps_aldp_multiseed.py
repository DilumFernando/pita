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
from Utils.aldp_tica import plot_tica_comparison


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
            "Train ALPS on ALDP across multiple seeds, then evaluate the final saved checkpoints."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--samplers", nargs="+", default=DEFAULT_SAMPLERS, choices=["nets", "ais"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-walkers", type=int, default=100)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--K", type=int, default=1000)
    parser.add_argument("--beta-max", type=float, default=1.0)
    beta_max_group = parser.add_mutually_exclusive_group()
    beta_max_group.add_argument(
        "--beta-max-learnable",
        dest="beta_max_learnable",
        action="store_true",
        help="Train beta_max as a learnable ALPS energy parameter.",
    )
    beta_max_group.add_argument(
        "--fixed-beta-max",
        dest="beta_max_learnable",
        action="store_false",
        help="Keep beta_max fixed at --beta-max during training.",
    )
    parser.set_defaults(beta_max_learnable=False)
    parser.add_argument("--modal-loss-weight", type=float, default=0.0)
    parser.add_argument("--modal-loss-end-fraction", type=float, default=0.0)
    parser.add_argument("--loss-type", default="manual", choices=["manual", "ctds"])
    parser.add_argument("--eval-num-samples", type=int, default=512)
    parser.add_argument("--eval-true-sample-count", type=int, default=512)
    parser.add_argument("--eval-dt", type=float, default=0.01)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--eval-epsilon", type=float, default=5.0)
    parser.add_argument("--plot-every", type=int, default=0)
    parser.add_argument("--checkpoint-name", default="final")
    parser.add_argument("--openmm-platform", default="CPU")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--dimension-pairs",
        nargs="*",
        default=[f"{a},{b}" for a, b in DEFAULT_DIMENSION_PAIRS],
        help='Pairs like "0,1" "2,3" used for Cartesian projection plots.',
    )
    parser.add_argument(
        "--sweep-name",
        default="aldp_alps_multiseed",
        help="Output directory name under alps_interpolation/ for aggregate sweep summaries.",
    )
    parser.add_argument(
        "--make-tica",
        action="store_true",
        help="Also save an ALDP TICA comparison plot for each eval run.",
    )
    return parser.parse_args()


def _parse_dimension_pairs(raw_pairs):
    parsed = []
    for item in raw_pairs:
        left, right = item.split(",", maxsplit=1)
        parsed.append((int(left), int(right)))
    return parsed


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
                "target": "aldp",
                "run_name": "",
                "dim": 66,
                "num_components": 6,
                "n_particles": 22,
                "spatial_dim": 3,
                "temperature": 300.0,
                "should_normalize": True,
                "normalization_factor": 0.1640,
                "should_remove_mean": True,
                "train_path": "data/alanine/AL22_temp_300.00/train_split_AL22-10000.npy",
                "val_path": "data/alanine/AL22_temp_300.00/val_split_AL22-10000.npy",
                "test_path": "data/alanine/AL22_temp_300.00/test_split_AL22-10000.npy",
                "pdb_path": "data/pdbs/A_capped.pdb",
                "mode_centers_path": None,
                "num_mode_centers": 6,
                "mode_estimation_samples": 10000,
                "mode_strategy": "canonical_rama",
                "energy_batch_size": 1024,
                "openmm_platform": str(args.openmm_platform),
                "device_index": int(args.device_index),
                "true_sample_count": int(args.eval_true_sample_count),
            },
            "model": {
                "interpolation_kind": "alps",
                "beta_max": float(args.beta_max),
                "beta_max_learnable": bool(args.beta_max_learnable),
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


def _row_from_results(seed, sampler, run_name, results):
    final_metrics = dict(results["final_metrics"])
    row = {
        "seed": int(seed),
        "sampler": sampler,
        "run_name": run_name,
        "checkpoint_name": results["metadata"]["checkpoint_name"],
        "weighted_w2": final_metrics.get("weighted_w2", float("nan")),
        "elbo": final_metrics.get("elbo", float("nan")),
        "eubo": final_metrics.get("eubo", float("nan")),
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
        if key not in {"seed", "sampler", "run_name", "checkpoint_name"}
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


def _save_tica_plot(cfg, results):
    save_path = results["plots_dir"] / "aldp_tica.png"
    plot_tica_comparison(
        generated_samples=results["final_samples"],
        true_samples=results["true_samples"],
        pdb_path=cfg.data.pdb_path,
        save_path=save_path,
        n_particles=int(cfg.data.n_particles),
        spatial_dim=int(cfg.data.spatial_dim),
        normalization_factor=float(cfg.data.normalization_factor),
        normalized=bool(cfg.data.should_normalize),
    )
    return save_path


def main():
    args = _parse_args()
    dimension_pairs = _parse_dimension_pairs(args.dimension_pairs)
    base_cfg = _base_config(args)

    sweep_dir = PROJECT_ROOT / "alps_interpolation" / args.sweep_name
    summary_dir = sweep_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    with (summary_dir / "config.yaml").open("w") as handle:
        handle.write(OmegaConf.to_yaml(base_cfg))
        handle.write(f"seeds: {list(args.seeds)}\n")
        handle.write(f"samplers: {list(args.samplers)}\n")
        handle.write(f"dimension_pairs: {dimension_pairs}\n")
        handle.write(f"make_tica: {bool(args.make_tica)}\n")

    all_rows = []
    for seed in args.seeds:
        run_name = f"aldp_alps_seed_{seed}"
        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        cfg.seed = int(seed)
        cfg.data.run_name = run_name

        print("=" * 80)
        print(f"Training seed {seed} -> run_name={run_name}")
        _train_once(cfg)

        for sampler in args.samplers:
            cfg.eval.sampler = sampler
            print("-" * 80)
            print(f"Evaluating seed {seed} with sampler={sampler} from {cfg.eval.checkpoint_name} checkpoint")
            results = run_evaluation(cfg)
            save_projection_scatter_plots(
                generated_samples=results["final_samples"],
                true_samples=results["true_samples"],
                weights=results["final_weights"],
                save_dir=results["plots_dir"] / "projections",
                dimension_pairs=dimension_pairs,
            )
            if args.make_tica:
                tica_path = _save_tica_plot(cfg, results)
                print(f"Saved ALDP TICA plot to {tica_path}")
            all_rows.append(_row_from_results(seed, sampler, run_name, results))

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
