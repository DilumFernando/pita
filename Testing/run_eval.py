import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

try:
    import hydra
    from omegaconf import OmegaConf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Hydra is not installed. Install it with `pip install hydra-core omegaconf` "
        "and then rerun this script."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Samplers.sampling import NET_Sampler, langevin_sampler, prior_samples as sample_prior_from_modes
from Energies.interpolations import energy_components
from Energies.targets import sample_reference
from Testing.test import artifact_dirs, load_model
from Training.run_training import _model_type_and_kwargs
from Utils.metrics import mmd_rbf, unweighted_w2_from_samples, weighted_w2
from Utils.misc import label_assignment_hard, mode_weights_from_particles
from Utils.plotting import plot_path_metrics, save_weighted_samples_plot


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _stable_weights(log_weights: torch.Tensor) -> torch.Tensor:
    shifted = log_weights - torch.max(log_weights)
    return torch.exp(shifted)


def _logmeanexp(values: torch.Tensor) -> float:
    values = values.view(-1)
    return float((torch.logsumexp(values, dim=0) - math.log(values.numel())).item())


def _elbo_from_log_weights(log_weights: torch.Tensor) -> float:
    return float(log_weights.mean().item())


def _eubo_from_log_weights(log_weights: torch.Tensor) -> float:
    return _logmeanexp(log_weights)


def _ess_count(weights: torch.Tensor, eps: float = 1e-12) -> float:
    normalized = weights / (weights.sum() + eps)
    return float((1.0 / (normalized.pow(2).sum() + eps)).item())


def _default_projection_dimension_pairs(dim: int):
    max_dims = min(dim, 8)
    pairs = []
    for dim_x in range(0, max_dims - 1, 2):
        pairs.append((dim_x, dim_x + 1))
    return pairs or [(0, 1)]


def _resolve_projection_dimension_pairs(cfg, dim: int):
    raw_pairs = cfg.eval.get("projection_dimension_pairs")
    if raw_pairs:
        parsed_pairs = []
        for pair in raw_pairs:
            if isinstance(pair, str):
                left, right = pair.split(",", maxsplit=1)
                parsed_pairs.append((int(left), int(right)))
            else:
                parsed_pairs.append(tuple(int(idx) for idx in pair))
        return parsed_pairs
    return _default_projection_dimension_pairs(dim)


def _build_initial_state(cfg, interpolation_kind, prior, true_modes, device):
    num_samples = int(cfg.eval.num_samples)
    if interpolation_kind == "alps" and prior is not None:
        x0 = prior.distribution.sample(torch.Size((num_samples,))).to(device)
    elif interpolation_kind in {"mean", "learned"} and true_modes is not None:
        x0 = sample_prior_from_modes(true_modes.to(device), num_samples=num_samples, device=device)
    else:
        dim = true_modes.shape[1] if true_modes is not None else int(cfg.data.dim)
        x0 = torch.randn(num_samples, dim, device=device)
    return x0.requires_grad_(True), torch.zeros(num_samples, device=device)


def _path_metrics(
    x_trajectory,
    a_trajectory,
    true_modes,
    true_samples,
    dt,
    means,
    U_net,
    modes,
    mixture,
    prior,
    energy_model,
):
    metrics_history = []
    true_samples_device = None if true_samples is None else true_samples.to(dtype=x_trajectory.dtype)

    for step_idx in range(x_trajectory.shape[0]):
        samples = x_trajectory[step_idx]
        log_weights = a_trajectory[step_idx]
        weights = _stable_weights(log_weights)
        t_tensor = torch.full(
            (samples.shape[0],),
            step_idx * dt,
            device=samples.device,
            dtype=samples.dtype,
        )
        components = energy_components(
            x=samples,
            t=t_tensor,
            means=means,
            U_net=U_net,
            modes=modes,
            target=mixture,
            prior=prior,
            energy_model=energy_model,
        )
        mode_weights = None
        if true_modes is not None:
            mode_weights, _ = mode_weights_from_particles(samples, true_modes, particle_weights=weights)

        step_metrics = {
            "step": int(step_idx),
            "t": float(step_idx * dt),
            "elbo": _elbo_from_log_weights(log_weights),
            "eubo": _eubo_from_log_weights(log_weights),
            "ess_count": _ess_count(weights),
            "U1": float(components["U1"].mean().item()),
            "U_gmm_t": float(components["U_gmm_t"].mean().item()),
            "U_net": float(components["U_net"].mean().item()),
            "U_t": float(components["U_t"].mean().item()),
            "w2": float("nan"),
            "weighted_w2": float("nan"),
            "mmd": float("nan"),
        }

        if mode_weights is not None:
            for mode_idx, value in enumerate(mode_weights):
                step_metrics[f"mode_weight_{mode_idx}"] = float(value.item())

        if true_samples_device is None:
            step_metrics["w2"] = float("nan")
            step_metrics["weighted_w2"] = float("nan")
            step_metrics["mmd"] = float("nan")
        else:
            try:
                step_metrics["w2"] = float(unweighted_w2_from_samples(samples, true_samples_device))
                step_metrics["weighted_w2"] = float(weighted_w2(samples, true_samples_device, weights))
            except ModuleNotFoundError:
                step_metrics["w2"] = float("nan")
                step_metrics["weighted_w2"] = float("nan")
            step_metrics["mmd"] = float(mmd_rbf(samples, true_samples_device).item())
        metrics_history.append(step_metrics)

    return metrics_history


def _save_metrics(metrics_history, metrics_dir):
    csv_path = metrics_dir / "path_metrics.csv"
    pth_path = metrics_dir / "path_metrics_history.pth"
    if not metrics_history:
        return csv_path, pth_path

    header = list(metrics_history[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(metrics_history)

    torch.save(metrics_history, pth_path)
    return csv_path, pth_path


def _save_metadata(metadata, metrics_dir):
    torch.save(metadata, metrics_dir / "eval_metadata.pth")
    with (metrics_dir / "eval_metadata.txt").open("w") as handle:
        for key, value in metadata.items():
            handle.write(f"{key}: {value}\n")


def _save_sample_artifacts(final_samples, final_weights, true_samples, metrics_dir):
    torch.save(final_samples.detach().cpu(), metrics_dir / "generated_samples.pth")
    torch.save(final_weights.detach().cpu(), metrics_dir / "generated_weights.pth")
    if true_samples is not None:
        torch.save(true_samples.detach().cpu(), metrics_dir / "true_samples.pth")


def save_projection_scatter_plots(
    generated_samples: torch.Tensor,
    true_samples: torch.Tensor | None,
    weights: torch.Tensor,
    save_dir: Path,
    dimension_pairs,
    max_points: int = 2000,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    generated = generated_samples.detach().cpu()
    true_cpu = None if true_samples is None else true_samples.detach().cpu()
    weights_cpu = weights.detach().cpu()
    normalized_weights = weights_cpu / (weights_cpu.sum() + 1e-12)

    if generated.shape[0] > max_points:
        gen_idx = torch.randperm(generated.shape[0])[:max_points]
        generated = generated[gen_idx]
        normalized_weights = normalized_weights[gen_idx]
        normalized_weights = normalized_weights / (normalized_weights.sum() + 1e-12)

    if true_cpu is not None and true_cpu.shape[0] > max_points:
        true_idx = torch.randperm(true_cpu.shape[0])[:max_points]
        true_cpu = true_cpu[true_idx]

    for dim_x, dim_y in dimension_pairs:
        if dim_x >= generated.shape[1] or dim_y >= generated.shape[1]:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if true_cpu is not None:
            axes[0].scatter(
                true_cpu[:, dim_x].numpy(),
                true_cpu[:, dim_y].numpy(),
                s=8,
                alpha=0.35,
                c="#d62728",
                edgecolors="none",
            )
            axes[0].set_title(f"True samples ({dim_x}, {dim_y})")
        else:
            axes[0].text(0.5, 0.5, "No true samples", ha="center", va="center")
            axes[0].set_title("True samples unavailable")

        scatter = axes[1].scatter(
            generated[:, dim_x].numpy(),
            generated[:, dim_y].numpy(),
            s=10,
            alpha=0.5,
            c=normalized_weights.numpy(),
            cmap="viridis",
            edgecolors="none",
        )
        axes[1].set_title(f"Generated samples ({dim_x}, {dim_y})")
        fig.colorbar(scatter, ax=axes[1], label="normalized weight")

        for ax in axes:
            ax.set_xlabel(f"dim {dim_x}")
            ax.set_ylabel(f"dim {dim_y}")
            ax.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(save_dir / f"projection_dims_{dim_x}_{dim_y}.png", dpi=200)
        plt.close(fig)


def save_projection_trajectory_plots(
    x_trajectory: torch.Tensor,
    a_trajectory: torch.Tensor,
    true_samples: torch.Tensor | None,
    save_dir: Path,
    dimension_pairs,
    step_interval: int = 5,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    snapshots = {}
    last_step = x_trajectory.shape[0] - 1
    for step_idx in range(x_trajectory.shape[0]):
        if step_idx % step_interval != 0 and step_idx != last_step:
            continue

        step_dir = save_dir / f"step_{step_idx:04d}"
        weights = _stable_weights(a_trajectory[step_idx])
        save_projection_scatter_plots(
            generated_samples=x_trajectory[step_idx],
            true_samples=true_samples,
            weights=weights,
            save_dir=step_dir,
            dimension_pairs=dimension_pairs,
        )
        snapshots[int(step_idx)] = {
            "samples": x_trajectory[step_idx].detach().cpu(),
            "weights": weights.detach().cpu(),
        }

    torch.save(snapshots, save_dir / "projection_snapshots.pth")


def run_evaluation(cfg):
    device = _resolve_device(cfg.device)
    torch.manual_seed(int(cfg.seed))

    print("Running evaluation with config:")
    print(OmegaConf.to_yaml(cfg))
    print(f"Resolved device: {device}")

    interpolation_kind = cfg.model.interpolation_kind
    checkpoint_name = cfg.eval.checkpoint_name
    sampler_name = cfg.eval.sampler

    marker_means = torch.zeros(1) if interpolation_kind == "mean" else None
    marker_u_net = object() if interpolation_kind == "learned" else None
    marker_modes = torch.zeros(1) if interpolation_kind == "alps" else None
    has_model_type = "model_type" in cfg.model or "network_type" in cfg.model
    model_type, model_kwargs = _model_type_and_kwargs(cfg) if has_model_type else (None, None)

    (
        drift,
        _free_energy,
        energy_model,
        potential_net,
        prior,
        mixture_gpu,
        _mixture_cpu,
        means,
        loaded_u_net,
        true_modes,
        _train_plot_dir,
    ) = load_model(
        dim=cfg.data.dim,
        num_components=cfg.data.num_components,
        means=marker_means,
        U_net=marker_u_net,
        modes=marker_modes,
        device=device,
        interpolation_kind=interpolation_kind,
        checkpoint_name=checkpoint_name,
        run_name=str(cfg.data.get("run_name", "")) or None,
        model_type=model_type,
        model_kwargs=model_kwargs,
    )

    true_sample_count = int(
        cfg.eval.get("true_sample_count")
        or cfg.data.get("true_sample_count")
        or cfg.eval.num_samples
    )
    true_samples = sample_reference(mixture_gpu, true_sample_count)
    x0, a0 = _build_initial_state(cfg, interpolation_kind, prior, true_modes, device)

    plot_every = int(cfg.eval.plot_every) if cfg.eval.plot_every is not None else 0
    projection_dimension_pairs = _resolve_projection_dimension_pairs(cfg, int(cfg.data.dim))

    if sampler_name == "ais":
        x_trajectory, a_trajectory = langevin_sampler(
            x0=x0,
            A0=a0,
            dt=float(cfg.eval.dt),
            steps=int(cfg.eval.steps),
            epsilon=float(cfg.eval.epsilon),
            means=means,
            U_net=loaded_u_net,
            modes=true_modes if interpolation_kind == "alps" else None,
            mixture=mixture_gpu,
            prior=prior,
            energy_model=energy_model,
            true_modes=true_modes,
            true_samples=true_samples,
            device=device,
            plot_every=plot_every,
        )
        is_ais = True
    elif sampler_name == "nets":
        x_trajectory, a_trajectory = NET_Sampler(
            drift=drift,
            x0=x0,
            A0=a0,
            dt=float(cfg.eval.dt),
            steps=int(cfg.eval.steps),
            epsilon=float(cfg.eval.epsilon),
            means=means,
            U_net=loaded_u_net,
            modes=true_modes if interpolation_kind == "alps" else None,
            mixture=mixture_gpu,
            prior=prior,
            energy_model=energy_model,
            true_modes=true_modes,
            true_samples=true_samples,
            device=device,
            plot_every=plot_every,
        )
        is_ais = False
    else:
        raise ValueError(f"Unsupported eval sampler: {sampler_name}")

    x_trajectory = x_trajectory.to(device)
    a_trajectory = a_trajectory.to(device)
    true_modes_device = None if true_modes is None else true_modes.to(device)
    metrics_history = _path_metrics(
        x_trajectory=x_trajectory,
        a_trajectory=a_trajectory,
        true_modes=true_modes_device,
        true_samples=true_samples,
        dt=float(cfg.eval.dt),
        means=means,
        U_net=loaded_u_net,
        modes=true_modes_device if interpolation_kind == "alps" else None,
        mixture=mixture_gpu,
        prior=prior,
        energy_model=energy_model,
    )

    dirs = artifact_dirs(
        interpolation_kind,
        cfg.data.dim,
        cfg.data.num_components,
        run_name=str(cfg.data.get("run_name", "")) or None,
    )
    checkpoint_label = checkpoint_name if checkpoint_name not in (None, "", "final") else "final"
    metrics_dir = Path(dirs["metrics"]) / "eval" / sampler_name / checkpoint_label
    plots_dir = Path(dirs["plots"]) / "eval" / sampler_name / checkpoint_label
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "sampler": sampler_name,
        "checkpoint_name": checkpoint_label,
        "interpolation_kind": interpolation_kind,
        "dim": int(cfg.data.dim),
        "num_components": int(cfg.data.num_components),
        "num_samples": int(cfg.eval.num_samples),
        "true_sample_count": true_sample_count,
        "dt": float(cfg.eval.dt),
        "steps": int(cfg.eval.steps),
        "epsilon": float(cfg.eval.epsilon),
        "trajectory_length": int(x_trajectory.shape[0]),
        "projection_dimension_pairs": projection_dimension_pairs,
    }

    csv_path, history_path = _save_metrics(metrics_history, metrics_dir)
    _save_metadata(metadata, metrics_dir)
    plot_path_metrics(metrics_history, str(plots_dir), metadata=metadata)
    save_projection_trajectory_plots(
        x_trajectory=x_trajectory,
        a_trajectory=a_trajectory,
        true_samples=true_samples,
        save_dir=plots_dir / "projections",
        dimension_pairs=projection_dimension_pairs,
        step_interval=5,
    )

    final_samples = x_trajectory[-1]
    final_log_weights = a_trajectory[-1]
    final_weights = _stable_weights(final_log_weights)
    _save_sample_artifacts(final_samples, final_weights, true_samples, metrics_dir)
    if true_modes is not None and true_samples is not None:
        labels = label_assignment_hard(final_samples, true_modes.to(device))
        labels_true = label_assignment_hard(true_samples, true_modes.to(device))
        save_weighted_samples_plot(
            generated_samples=final_samples,
            true_samples=true_samples,
            weights=final_weights,
            labels=labels,
            labels_true=labels_true,
            save_path=str(plots_dir / "final_weighted_samples.png"),
            is_ais=is_ais,
        )

    final_metrics = metrics_history[-1] if metrics_history else {}
    torch.save(final_metrics, metrics_dir / "final_metrics.pth")

    return {
        "csv_path": csv_path,
        "history_path": history_path,
        "metrics_dir": metrics_dir,
        "plots_dir": plots_dir,
        "metadata": metadata,
        "metrics_history": metrics_history,
        "final_metrics": final_metrics,
        "final_samples": final_samples.detach().cpu(),
        "final_weights": final_weights.detach().cpu(),
        "final_log_weights": final_log_weights.detach().cpu(),
        "true_samples": None if true_samples is None else true_samples.detach().cpu(),
        "true_modes": None if true_modes is None else true_modes.detach().cpu(),
        "is_ais": is_ais,
    }


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg):
    results = run_evaluation(cfg)
    final_metrics = results["final_metrics"]
    weighted_w2_value = final_metrics.get("weighted_w2", float("nan"))
    weighted_w2_text = "nan" if math.isnan(weighted_w2_value) else f"{weighted_w2_value:.4f}"
    print(f"Saved eval metrics to {results['csv_path']}")
    print(f"Saved eval history to {results['history_path']}")
    print(f"Saved eval plots to {results['plots_dir']}")
    print(
        f"Final step {final_metrics.get('step', 'n/a')} | "
        f"ELBO {final_metrics.get('elbo', float('nan')):.4f} | "
        f"EUBO {final_metrics.get('eubo', float('nan')):.4f} | "
        f"ESS {final_metrics.get('ess_count', float('nan')):.2f} | "
        f"weighted_w2 {weighted_w2_text} | "
        f"mmd {final_metrics.get('mmd', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()
