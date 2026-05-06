import sys
from pathlib import Path

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

from Testing.test import artifact_dirs, load_model
from Utils.aldp_tica import plot_tica_comparison


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg):
    device = _resolve_device(cfg.device)
    interpolation_kind = cfg.model.interpolation_kind
    checkpoint_name = cfg.eval.checkpoint_name
    sampler_name = cfg.eval.sampler
    run_name = str(cfg.data.get("run_name", "")) or None

    (
        _drift,
        _free_energy,
        _energy_model,
        _potential_net,
        _prior,
        target,
        _target_cpu,
        _means,
        _loaded_u_net,
        _true_modes,
        _train_plot_dir,
    ) = load_model(
        dim=cfg.data.dim,
        num_components=cfg.data.num_components,
        device=device,
        interpolation_kind=interpolation_kind,
        checkpoint_name=checkpoint_name,
        run_name=run_name,
    )

    if getattr(target, "target_type", None) != "aldp":
        raise ValueError("TICA plotting is currently implemented only for the ALDP target.")

    dirs = artifact_dirs(
        interpolation_kind,
        cfg.data.dim,
        cfg.data.num_components,
        run_name=run_name,
    )
    checkpoint_label = checkpoint_name if checkpoint_name not in (None, "", "final") else "final"
    metrics_dir = Path(dirs["metrics"]) / "eval" / sampler_name / checkpoint_label
    plots_dir = Path(dirs["plots"]) / "eval" / sampler_name / checkpoint_label

    generated_path = metrics_dir / "generated_samples.pth"
    true_path = metrics_dir / "true_samples.pth"
    if not generated_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            "Expected eval sample artifacts were not found. Run `Testing/run_eval.py` first."
        )

    generated_samples = torch.load(generated_path, map_location="cpu", weights_only=False)
    true_samples = torch.load(true_path, map_location="cpu", weights_only=False)
    output_path = plots_dir / "aldp_tica.png"

    plot_tica_comparison(
        generated_samples=generated_samples,
        true_samples=true_samples,
        pdb_path=target.pdb_path,
        save_path=output_path,
        n_particles=target.n_particles,
        spatial_dim=target.spatial_dim,
        normalization_factor=target.normalization_factor,
        normalized=target.should_normalize,
    )
    print(f"Saved ALDP TICA plot to {output_path}")
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
