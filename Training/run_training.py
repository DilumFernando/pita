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

from Energies.gmm import (
    LearnableGMM,
    GMMModesEnergy,
    GMMModesEnergyTimeLogits,
    create_gaussian_mixture,
    make_rotated_diagonal_covariances,
    make_rotated_full_covariances,
    make_scaled_identity_covariances,
)
from Energies.targets import create_target_from_config, infer_mode_centers, sample_reference
from Models.models import DiTDriftNet, LogitsNet, PotentialNet
from Samplers.sampling import prior_samples as sample_prior_from_modes
from Training.train import train_and_save
from Utils.constants import means_40


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _optional_tensor(values, device):
    if values is None:
        return None
    return torch.tensor(values, dtype=torch.float32, device=device)


def _apply_mean_layout(mixture, cfg):
    layout = cfg.data.get("mean_layout", "pair_shift")
    if not cfg.data.get("apply_mean_layout", True):
        return mixture

    means = mixture.component_distribution.mean
    num_components, dim = means.shape
    scale = float(cfg.data.get("layout_scale", cfg.data.get("perturb_mean", 10.0)))

    if layout == "none" or num_components == 0:
        return mixture

    if layout == "pair_shift":
        if num_components >= 2 and scale != 0.0:
            means[0] += scale * torch.ones_like(means[0])
            means[1] += -scale * torch.ones_like(means[1])
        return mixture

    if layout == "line":
        offsets = torch.linspace(-scale, scale, num_components, device=means.device, dtype=means.dtype)
        means[:, 0] = means[:, 0] + offsets
        return mixture

    if layout == "circle":
        if dim < 2:
            raise ValueError("circle mean layout requires dim >= 2")
        angles = torch.linspace(0, 2 * torch.pi, num_components + 1, device=means.device, dtype=means.dtype)[:-1]
        means[:, 0] = means[:, 0] + scale * torch.cos(angles)
        means[:, 1] = means[:, 1] + scale * torch.sin(angles)
        return mixture

    raise ValueError(f"Unsupported mean_layout: {layout}")


def _build_mixture(cfg, device):
    preset_means = None
    if cfg.data.means_source == "means_40":
        preset_means = means_40.to(device=device, dtype=torch.float32)

    covariance_cfg = cfg.data.get("covariance", None)
    covs = None
    if covariance_cfg is not None and covariance_cfg.style is not None:
        if covariance_cfg.style == "scaled_identity":
            covs = make_scaled_identity_covariances(
                cfg.data.dim,
                covariance_cfg.scales,
                device=device,
            )
        elif covariance_cfg.style == "rotated_diagonal":
            covs = make_rotated_diagonal_covariances(
                cfg.data.dim,
                covariance_cfg.diagonal_scales,
                random_rotation_per_component=bool(covariance_cfg.random_rotation_per_component),
                device=device,
            )
        elif covariance_cfg.style == "rotated_full":
            covs = make_rotated_full_covariances(
                cfg.data.dim,
                covariance_cfg.base_covariances,
                random_rotation_per_component=bool(covariance_cfg.random_rotation_per_component),
                device=device,
            )
        else:
            raise ValueError(f"Unsupported covariance style: {covariance_cfg.style}")
    else:
        covs = _optional_tensor(cfg.data.covs, device)
    weights = _optional_tensor(cfg.data.weights, device)

    mixture = create_gaussian_mixture(
        cfg.data.dim,
        cfg.data.num_components,
        means=preset_means,
        covs=covs,
        weights=weights,
        device=device,
    )

    if preset_means is None:
        mixture = _apply_mean_layout(mixture, cfg)

    return mixture


def _build_training_state(cfg, target, device):
    interpolation_kind = cfg.model.interpolation_kind
    n_walkers = cfg.training.n_walkers
    dim = cfg.data.dim
    num_components = cfg.data.num_components
    true_modes = infer_mode_centers(target)

    means = None
    modes = None
    U_net = None
    prior = None
    energy_model = None

    if interpolation_kind == "mean":
        if true_modes is None:
            raise ValueError("Mean interpolation requires target modes/means.")
        means = true_modes
        prior_state = sample_prior_from_modes(means, num_samples=n_walkers, device=device)
    elif interpolation_kind == "learned":
        if true_modes is None:
            raise ValueError("Learned mean interpolation requires target modes/means.")
        means = true_modes
        U_net = PotentialNet(dim).to(device)
        prior_state = sample_prior_from_modes(means, num_samples=n_walkers, device=device)
    elif interpolation_kind == "alps":
        if true_modes is None:
            raise ValueError("ALPS interpolation requires target modes/means.")
        beta_max_init = float(cfg.model.get("beta_max_init", cfg.model.get("beta_max", 1.0)))
        beta_max = torch.tensor(beta_max_init, dtype=torch.float32, device=device)
        perturbation = float(cfg.model.perturbation)
        warm_starts = true_modes + perturbation * torch.randn(num_components, dim, device=device)
        init_logits = torch.zeros(num_components, device=device)

        if cfg.model.use_time_logits:
            logits_net = LogitsNet(dim, num_components).to(device)
            energy_model = GMMModesEnergyTimeLogits(warm_starts, beta_max, logits_net)
        else:
            energy_model = GMMModesEnergy(warm_starts, beta_max, init_logits)

        init_covs = (1.0 / beta_max) * torch.ones(num_components, device=device)
        prior = LearnableGMM(means=warm_starts, covs=init_covs, logits=init_logits)
        prior_state = prior.distribution.sample(torch.Size((n_walkers,))).to(device).requires_grad_(True)
        modes = warm_starts
    elif interpolation_kind == "fixed":
        if true_modes is not None:
            prior_state = sample_prior_from_modes(true_modes, num_samples=n_walkers, device=device)
        else:
            prior_state = torch.randn(n_walkers, dim, device=device).requires_grad_(True)
    else:
        raise ValueError(f"Unsupported interpolation_kind: {interpolation_kind}")

    true_sample_count = cfg.data.true_sample_count or n_walkers
    true_samples = sample_reference(target, true_sample_count)

    return {
        "means": means,
        "modes": modes,
        "U_net": U_net,
        "prior": prior,
        "energy_model": energy_model,
        "prior_samples": prior_state,
        "true_modes": true_modes,
        "true_samples": true_samples,
    }


def _build_drift_model(cfg, device):
    drift_cfg = cfg.model.get("drift", None)
    if drift_cfg is None:
        return None

    architecture = str(drift_cfg.get("architecture", "mlp")).lower()
    if architecture == "mlp":
        return None
    if architecture != "dit":
        raise ValueError(f"Unsupported drift architecture: {architecture}")

    if cfg.data.target != "aldp":
        raise ValueError("DiT drift is currently configured for ALDP particle coordinates.")

    return DiTDriftNet(
        dim=int(cfg.data.dim),
        n_particles=int(cfg.data.n_particles),
        spatial_dim=int(cfg.data.spatial_dim),
        hidden_size=int(drift_cfg.get("hidden_size", 192)),
        cond_dim=int(drift_cfg.get("cond_dim", 64)),
        n_heads=int(drift_cfg.get("n_heads", 6)),
        n_blocks=int(drift_cfg.get("n_blocks", 6)),
        dropout=float(drift_cfg.get("dropout", 0.1)),
    ).to(device)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    device = _resolve_device(cfg.device)
    torch.manual_seed(int(cfg.seed))

    print("Running training with config:")
    print(OmegaConf.to_yaml(cfg))
    print(f"Resolved device: {device}")

    target = create_target_from_config(cfg, device)
    if target is None:
        target = _build_mixture(cfg, device)
    state = _build_training_state(cfg, target, device)
    drift_net = _build_drift_model(cfg, device)

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
        drift_net=drift_net,
    )


if __name__ == "__main__":
    main()
