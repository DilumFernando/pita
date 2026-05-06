import os
import torch

from Energies.gmm import LearnableGMM, GMMModesEnergyTimeLogits, create_gaussian_mixture
from Energies.targets import infer_mode_centers, load_target_from_metadata
from Models.models import DriftNet, FreeEnergyNet, LogitsNet, PotentialNet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def interpolation_dir(kind, dim, num_components, run_name=None):
    if run_name:
        return os.path.join(PROJECT_ROOT, f"{kind}_interpolation", run_name)
    return os.path.join(PROJECT_ROOT, f"{kind}_interpolation", f"{dim}d_{num_components}gmm")


def artifact_dirs(kind, dim, num_components, run_name=None):
    run_dir = interpolation_dir(kind, dim, num_components, run_name=run_name)
    return {
        "run": run_dir,
        "models": os.path.join(run_dir, "models"),
        "plots": os.path.join(run_dir, "plots"),
        "metrics": os.path.join(run_dir, "metrics"),
    }


def _infer_interpolation_kind(interpolation_kind, means, U_net, modes):
    if interpolation_kind is not None:
        return interpolation_kind
    if U_net is not None:
        return "learned"
    if means is not None:
        return "mean"
    if modes is not None:
        return "alps"
    return "fixed"


def _resolve_checkpoint_dir(kind, dim, num_components, checkpoint_name, run_name=None):
    dirs = artifact_dirs(kind, dim, num_components, run_name=run_name)
    if checkpoint_name in (None, "", "final"):
        return dirs["models"], dirs
    if checkpoint_name == "best_weighted_w2":
        return os.path.join(dirs["models"], "best_checkpoints", "best_weighted_w2"), dirs
    return os.path.join(dirs["models"], checkpoint_name), dirs


def _load_mixture(mixture_path, device):
    metadata = torch.load(mixture_path, map_location="cpu", weights_only=False)
    target_gpu = load_target_from_metadata(metadata, device=device)
    target_cpu = load_target_from_metadata(metadata, device=torch.device("cpu"))
    return target_gpu, target_cpu


def load_model(
    dim,
    num_components,
    means=None,
    U_net=None,
    modes=None,
    beta_max=None,
    device=None,
    interpolation_kind=None,
    checkpoint_name="final",
    run_name=None,
):
    del beta_max

    kind = _infer_interpolation_kind(interpolation_kind, means, U_net, modes)
    checkpoint_dir, dirs = _resolve_checkpoint_dir(kind, dim, num_components, checkpoint_name, run_name=run_name)

    drift = DriftNet(dim).to(device)
    free_energy = FreeEnergyNet(dim).to(device)
    potential_net = PotentialNet(dim).to(device)
    energy_model = None
    prior = None
    true_modes = None
    loaded_means = None
    loaded_u_net = None

    if checkpoint_name in (None, "", "final"):
        if kind == "learned":
            drift_path = os.path.join(checkpoint_dir, "final_drift_means.pth")
            free_path = os.path.join(checkpoint_dir, "final_free_energy_means.pth")
            potential_path = os.path.join(checkpoint_dir, "final_potential_means.pth")
            mixture_path = os.path.join(checkpoint_dir, "mixture_means.pth")
            prior_path = os.path.join(checkpoint_dir, "prior_means.pth")
        elif kind == "mean":
            drift_path = os.path.join(checkpoint_dir, "final_drift_mean_interp.pth")
            free_path = os.path.join(checkpoint_dir, "final_free_energy_mean_interp.pth")
            potential_path = None
            mixture_path = os.path.join(checkpoint_dir, "mixture_mean_interp.pth")
            prior_path = os.path.join(checkpoint_dir, "prior_mean_interp.pth")
        elif kind == "alps":
            drift_path = os.path.join(checkpoint_dir, "covs_final_drift_alps.pth")
            free_path = os.path.join(checkpoint_dir, "covs_final_free_energy_alps.pth")
            potential_path = os.path.join(checkpoint_dir, "covs_final_potential_alps.pth")
            mixture_path = os.path.join(checkpoint_dir, "covs_mixture_weights_exp.pth")
            prior_path = os.path.join(checkpoint_dir, "covs_prior_alps_exp.pth")
            energy_path = os.path.join(checkpoint_dir, "covs_final_energy_model_alps.pth")
        else:
            drift_path = os.path.join(checkpoint_dir, "final_drift.pth")
            free_path = os.path.join(checkpoint_dir, "final_free_energy.pth")
            potential_path = os.path.join(checkpoint_dir, "final_potential.pth")
            mixture_path = os.path.join(checkpoint_dir, "mixture.pth")
            prior_path = os.path.join(checkpoint_dir, "prior.pth")
    else:
        drift_path = os.path.join(checkpoint_dir, "drift.pth")
        free_path = os.path.join(checkpoint_dir, "free_energy.pth")
        potential_path = os.path.join(checkpoint_dir, "potential.pth")
        mixture_path = os.path.join(checkpoint_dir, "mixture.pth")
        prior_path = os.path.join(checkpoint_dir, "prior.pth")
        energy_path = os.path.join(checkpoint_dir, "energy_model.pth")

    drift.load_state_dict(torch.load(drift_path, map_location=device, weights_only=True))
    free_energy.load_state_dict(torch.load(free_path, map_location=device, weights_only=True))
    drift.eval()
    free_energy.eval()

    if potential_path is not None and os.path.exists(potential_path):
        potential_net.load_state_dict(torch.load(potential_path, map_location=device, weights_only=True))
        potential_net.eval()
        loaded_u_net = potential_net
    else:
        potential_net = None

    mixture_gpu, mixture_cpu = _load_mixture(mixture_path, device)
    true_modes = infer_mode_centers(mixture_gpu)

    if kind in {"learned", "mean"} and true_modes is not None:
        loaded_means = true_modes.requires_grad_(True)
    if kind == "alps":
        if true_modes is None:
            raise ValueError("ALPS checkpoints require target modes.")
        logits_net = LogitsNet(dim, num_components).to(device)
        energy_model = GMMModesEnergyTimeLogits(true_modes, 1.0, logits_net)
        if os.path.exists(energy_path):
            energy_model = torch.load(energy_path, map_location=device, weights_only=False)
        if os.path.exists(prior_path):
            prior = torch.load(prior_path, map_location=device, weights_only=False)
        else:
            covs_ = torch.ones(num_components, dtype=torch.float32, device=device)
            prior = LearnableGMM(means=true_modes, covs=covs_, logits=None)
    elif os.path.exists(prior_path):
        prior = torch.load(prior_path, map_location=device, weights_only=False)

    return (
        drift,
        free_energy,
        energy_model,
        potential_net,
        prior,
        mixture_gpu,
        mixture_cpu,
        loaded_means,
        loaded_u_net,
        true_modes,
        dirs["plots"],
    )
