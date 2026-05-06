from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import torch
from matplotlib.colors import LogNorm

try:
    import deeptime as dt
except ImportError:
    dt = None


SELECTION = "symbol == C or symbol == N or symbol == S"


def distances(xyz: np.ndarray) -> np.ndarray:
    distance_matrix = np.linalg.norm(xyz[:, None, :, :] - xyz[:, :, None, :], axis=-1)
    m, n = np.triu_indices(distance_matrix.shape[-1], k=1)
    return distance_matrix[:, m, n]


def wrap(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.sin(array), np.cos(array)


def tica_features(
    trajectory: md.Trajectory,
    *,
    use_dihedrals: bool = True,
    use_distances: bool = True,
    selection: str = SELECTION,
) -> np.ndarray:
    trajectory = trajectory.atom_slice(trajectory.top.select(selection))
    features = []
    if use_distances:
        features.append(distances(trajectory.xyz))
    if use_dihedrals:
        _, phi = md.compute_phi(trajectory)
        _, psi = md.compute_psi(trajectory)
        _, omega = md.compute_omega(trajectory)
        features.extend([*wrap(phi), *wrap(psi), *wrap(omega)])
    if not features:
        raise ValueError("At least one TICA feature family must be enabled.")
    return np.concatenate(features, axis=-1)


def run_tica(features: np.ndarray, lagtime: int = 500, dim: int = 40):
    if dt is None:
        raise ImportError("`deeptime` is required for TICA plotting. Install it in `pita-aldp`.")
    tica = dt.decomposition.TICA(dim=dim, lagtime=lagtime)
    koopman_estimator = dt.covariance.KoopmanWeightingEstimator(lagtime=lagtime)
    reweighting_model = koopman_estimator.fit(features).fetch_model()
    return tica.fit(features, reweighting_model).fetch_model()


def _to_numpy(samples: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(samples, torch.Tensor):
        return samples.detach().cpu().numpy()
    return np.asarray(samples)


def samples_to_trajectory(
    samples: torch.Tensor | np.ndarray,
    *,
    pdb_path: str,
    n_particles: int = 22,
    spatial_dim: int = 3,
    normalization_factor: float = 0.1640,
    normalized: bool = True,
) -> md.Trajectory:
    xyz = _to_numpy(samples).reshape(-1, n_particles, spatial_dim)
    if normalized:
        xyz = xyz * normalization_factor
    topology = md.load_topology(str(pdb_path))
    return md.Trajectory(xyz, topology)


def plot_tica_comparison(
    generated_samples: torch.Tensor | np.ndarray,
    true_samples: torch.Tensor | np.ndarray,
    *,
    pdb_path: str,
    save_path: str | Path,
    n_particles: int = 22,
    spatial_dim: int = 3,
    normalization_factor: float = 0.1640,
    normalized: bool = True,
    lagtime: int = 500,
    dim: int = 40,
    bins: int = 100,
) -> Path:
    generated_traj = samples_to_trajectory(
        generated_samples,
        pdb_path=pdb_path,
        n_particles=n_particles,
        spatial_dim=spatial_dim,
        normalization_factor=normalization_factor,
        normalized=normalized,
    )
    true_traj = samples_to_trajectory(
        true_samples,
        pdb_path=pdb_path,
        n_particles=n_particles,
        spatial_dim=spatial_dim,
        normalization_factor=normalization_factor,
        normalized=normalized,
    )

    true_features = tica_features(true_traj)
    generated_features = tica_features(generated_traj)

    tica_model = run_tica(true_features, lagtime=lagtime, dim=dim)
    true_tics = tica_model.transform(true_features)
    generated_tics = tica_model.transform(generated_features)

    tics_lims = np.vstack([true_tics[:, :2], generated_tics[:, :2]])
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, tics, title in zip(
        axes,
        (true_tics, generated_tics),
        ("True Samples", "Generated Samples"),
    ):
        ax.hist2d(
            tics[:, 0],
            tics[:, 1],
            bins=bins,
            norm=LogNorm(),
            cmap="viridis",
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlabel("TIC0")
        ax.set_ylabel("TIC1")
        ax.set_xlim(tics_lims[:, 0].min(), tics_lims[:, 0].max())
        ax.set_ylim(tics_lims[:, 1].min(), tics_lims[:, 1].max())

    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path
