from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Energies.gmm import create_gaussian_mixture

try:
    import openmm
    from bgflow import OpenMMBridge, OpenMMEnergy
    from openmm import app
except ImportError:
    openmm = None
    OpenMMBridge = None
    OpenMMEnergy = None
    app = None

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

try:
    import mdtraj as md
except ImportError:
    md = None


ALDP_CANONICAL_MODE_CENTERS_DEG: dict[str, tuple[float, float]] = {
    "alpha_R": (-75.0, -30.0),
    "beta": (-75.0, 150.0),
    "C5": (-140.0, 170.0),
    "alpha_p": (-130.0, -10.0),
    "alpha_L": (60.0, 10.0),
    "alpha_D": (60.0, 180.0),
}


def _remove_mean(samples: torch.Tensor, n_particles: int, spatial_dim: int) -> torch.Tensor:
    shape = samples.shape
    samples = samples.view(-1, n_particles, spatial_dim)
    samples = samples - samples.mean(dim=1, keepdim=True)
    return samples.view(*shape)


def _ensure_tensor(values: Any, device: torch.device | None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=dtype)
    return torch.tensor(values, device=device, dtype=dtype)


@dataclass
class TargetMetadata:
    target_type: str
    config: dict[str, Any]


class AlanineDipeptideTarget:
    def __init__(
        self,
        train_path: str | None,
        val_path: str | None,
        test_path: str,
        *,
        device: torch.device,
        temperature: float = 300.0,
        dimensionality: int = 66,
        n_particles: int = 22,
        spatial_dim: int = 3,
        should_normalize: bool = True,
        normalization_factor: float = 0.1640,
        should_remove_mean: bool = True,
        pdb_path: str | None = None,
        energy_batch_size: int = 2048,
        openmm_platform: str = "CPU",
        device_index: int = 0,
        mode_centers_path: str | None = None,
        num_mode_centers: int = 4,
        mode_estimation_samples: int = 10000,
        mode_strategy: str = "rama_kmeans",
        rama_mode_centers_deg: dict[str, tuple[float, float]] | None = None,
        rama_mode_names: list[str] | None = None,
    ):
        self.target_type = "aldp"
        self.device = device
        self.temperature = float(temperature)
        self.dim = int(dimensionality)
        self.n_particles = int(n_particles)
        self.spatial_dim = int(spatial_dim)
        self.should_normalize = bool(should_normalize)
        self.normalization_factor = float(normalization_factor)
        self.should_remove_mean = bool(should_remove_mean)
        self.energy_batch_size = int(energy_batch_size)
        self.pdb_path = pdb_path
        self.openmm_platform = str(openmm_platform)
        self.device_index = int(device_index)
        self.mode_centers_path = mode_centers_path
        self.num_mode_centers = int(num_mode_centers)
        self.mode_estimation_samples = int(mode_estimation_samples)
        self.mode_strategy = str(mode_strategy)
        self.rama_mode_names = rama_mode_names
        self.rama_mode_centers_deg = rama_mode_centers_deg

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        self._train_data = self._load_split(train_path) if train_path else None
        self._val_data = self._load_split(val_path) if val_path else None
        self._test_data = self._load_split(test_path)
        self._reference_pool = torch.cat(
            [split for split in (self._train_data, self._val_data, self._test_data) if split is not None],
            dim=0,
        )
        self.mode_centers = self._load_or_estimate_mode_centers()
        self._openmm_energy = self._build_openmm_energy()

    def _load_split(self, path: str) -> torch.Tensor:
        array = np.load(path, allow_pickle=True)
        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        return self._maybe_normalize(tensor)

    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.should_remove_mean:
            x = _remove_mean(x, self.n_particles, self.spatial_dim)
        if self.should_normalize:
            x = x / self.normalization_factor
        return x

    def _maybe_unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.should_normalize:
            x = x * self.normalization_factor
        return x

    def _build_openmm_energy(self):
        if self.pdb_path is None:
            return None
        if openmm is None or OpenMMBridge is None or OpenMMEnergy is None or app is None:
            raise ImportError(
                "Alanine dipeptide energy evaluation requires `openmm` and `bgflow` to be installed."
            )

        pdb = app.PDBFile(self.pdb_path)
        forcefield = app.ForceField("amber14-all.xml", "implicit/obc1.xml")
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=2.0 * openmm.unit.nanometer,
            constraints=None,
        )
        integrator = openmm.LangevinMiddleIntegrator(
            self.temperature * openmm.unit.kelvin,
            0.3 / openmm.unit.picosecond,
            1.0 * openmm.unit.femtosecond,
        )

        platform_properties = {}
        if self.openmm_platform.upper() == "CUDA":
            platform_properties = {"Precision": "single", "DeviceIndex": str(self.device_index)}

        try:
            bridge = OpenMMBridge(
                system,
                integrator,
                platform_name=self.openmm_platform,
                platform_properties=platform_properties,
            )
        except TypeError:
            bridge = OpenMMBridge(
                system,
                integrator,
                platform_name=self.openmm_platform,
            )

        return OpenMMEnergy(bridge=bridge)

    def _sample_mode_pool(self) -> torch.Tensor:
        pool = self._reference_pool
        if pool.shape[0] > self.mode_estimation_samples:
            idx = torch.randperm(pool.shape[0], device=pool.device)[: self.mode_estimation_samples]
            pool = pool[idx]
        return pool

    def _estimate_mode_centers_cartesian(self, pool: torch.Tensor) -> torch.Tensor:
        centers = KMeans(n_clusters=self.num_mode_centers, random_state=0, n_init=10).fit(
            pool.detach().cpu().numpy()
        ).cluster_centers_
        return torch.as_tensor(centers, dtype=torch.float32, device=self.device)

    def _estimate_mode_centers_rama(self, pool: torch.Tensor) -> torch.Tensor:
        if md is None:
            raise ImportError("Ramachandran mode estimation requires `mdtraj`.")
        if self.pdb_path is None:
            raise ValueError("Ramachandran mode estimation requires `pdb_path`.")

        xyz = self._maybe_unnormalize(pool).detach().cpu().numpy().reshape(-1, self.n_particles, self.spatial_dim)
        topology = md.load_topology(self.pdb_path)
        traj = md.Trajectory(xyz, topology)

        phi = md.compute_phi(traj)[1].reshape(-1)
        psi = md.compute_psi(traj)[1].reshape(-1)
        valid = ~(np.isnan(phi) | np.isnan(psi))
        if valid.sum() < self.num_mode_centers:
            raise ValueError(
                f"Not enough valid phi/psi samples to estimate {self.num_mode_centers} ALDP modes."
            )

        pool_valid = pool[torch.from_numpy(valid).to(device=pool.device)]
        phi = phi[valid]
        psi = psi[valid]
        angular_features = np.stack([np.cos(phi), np.sin(phi), np.cos(psi), np.sin(psi)], axis=1)
        assignments = KMeans(n_clusters=self.num_mode_centers, random_state=0, n_init=20).fit(angular_features)
        labels = assignments.labels_
        centers = assignments.cluster_centers_

        medoid_indices = []
        for cluster_id in range(self.num_mode_centers):
            cluster_members = np.where(labels == cluster_id)[0]
            if cluster_members.size == 0:
                continue
            cluster_features = angular_features[cluster_members]
            distances = ((cluster_features - centers[cluster_id]) ** 2).sum(axis=1)
            medoid_indices.append(int(cluster_members[np.argmin(distances)]))

        if len(medoid_indices) != self.num_mode_centers:
            raise ValueError(
                f"Expected {self.num_mode_centers} ALDP mode representatives, found {len(medoid_indices)}."
            )
        return pool_valid[medoid_indices].to(device=self.device, dtype=torch.float32)

    def _estimate_mode_centers_canonical_rama(self, pool: torch.Tensor) -> torch.Tensor:
        if md is None:
            raise ImportError("Canonical Ramachandran mode estimation requires `mdtraj`.")
        if self.pdb_path is None:
            raise ValueError("Canonical Ramachandran mode estimation requires `pdb_path`.")
        if self.num_mode_centers != len(ALDP_CANONICAL_MODE_CENTERS_DEG):
            raise ValueError(
                "Canonical ALDP Ramachandran modes define exactly "
                f"{len(ALDP_CANONICAL_MODE_CENTERS_DEG)} centers; got num_mode_centers={self.num_mode_centers}."
            )

        xyz = self._maybe_unnormalize(pool).detach().cpu().numpy().reshape(-1, self.n_particles, self.spatial_dim)
        topology = md.load_topology(self.pdb_path)
        traj = md.Trajectory(xyz, topology)

        phi = md.compute_phi(traj)[1].reshape(-1)
        psi = md.compute_psi(traj)[1].reshape(-1)
        valid = ~(np.isnan(phi) | np.isnan(psi))
        if valid.sum() < self.num_mode_centers:
            raise ValueError(
                f"Not enough valid phi/psi samples to estimate {self.num_mode_centers} ALDP modes."
            )

        mode_names = list(ALDP_CANONICAL_MODE_CENTERS_DEG.keys())
        centers_deg = np.asarray(
            [ALDP_CANONICAL_MODE_CENTERS_DEG[name] for name in mode_names],
            dtype=np.float64,
        )
        centers_rad = np.deg2rad(centers_deg)
        angles = np.stack([phi[valid], psi[valid]], axis=1)
        angular_diff = (angles[:, None, :] - centers_rad[None, :, :] + np.pi) % (2 * np.pi) - np.pi
        squared_distances = (angular_diff**2).sum(axis=2)
        labels = np.argmin(squared_distances, axis=1)

        medoid_indices = []
        for mode_idx in range(len(mode_names)):
            cluster_members = np.where(labels == mode_idx)[0]
            if cluster_members.size == 0:
                medoid_indices.append(int(np.argmin(squared_distances[:, mode_idx])))
            else:
                member_distances = squared_distances[cluster_members, mode_idx]
                medoid_indices.append(int(cluster_members[np.argmin(member_distances)]))

        self.rama_mode_names = mode_names
        self.rama_mode_centers_deg = ALDP_CANONICAL_MODE_CENTERS_DEG.copy()
        pool_valid = pool[torch.from_numpy(valid).to(device=pool.device)]
        return pool_valid[medoid_indices].to(device=self.device, dtype=torch.float32)

    def _load_or_estimate_mode_centers(self) -> torch.Tensor | None:
        if self.mode_centers_path:
            centers = np.load(self.mode_centers_path, allow_pickle=True)
            return torch.as_tensor(centers, dtype=torch.float32, device=self.device)

        if self.num_mode_centers <= 0:
            return None

        if self.mode_strategy == "canonical_rama":
            return self._estimate_mode_centers_canonical_rama(self._reference_pool)

        if KMeans is None:
            raise ImportError(
                "Estimating alanine dipeptide mode centers requires scikit-learn. "
                "Install it or provide `data.mode_centers_path`."
            )

        pool = self._sample_mode_pool()
        if self.mode_strategy == "rama_kmeans":
            try:
                return self._estimate_mode_centers_rama(pool)
            except Exception:
                return self._estimate_mode_centers_cartesian(pool)
        if self.mode_strategy == "cartesian_kmeans":
            return self._estimate_mode_centers_cartesian(pool)
        raise ValueError(f"Unsupported ALDP mode strategy: {self.mode_strategy}")

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        if self._openmm_energy is None:
            raise RuntimeError(
                "Alanine dipeptide target was constructed without OpenMM energy support. "
                "Set `data.pdb_path` and install the required dependencies to evaluate log_prob."
            )

        x = x.to(self.device)
        x_phys = self._maybe_unnormalize(x)
        log_probs = []
        for batch in torch.split(x_phys, self.energy_batch_size, dim=0):
            log_probs.append(-self._openmm_energy.energy(batch).squeeze(-1))
        return torch.cat(log_probs, dim=0)

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        idx = torch.randint(0, self._reference_pool.shape[0], (num_samples,), device=self.device)
        samples = self._reference_pool[idx]
        return samples[0] if num_samples == 1 else samples

    def sample_reference(self, num_samples: int) -> torch.Tensor:
        return self.sample(num_samples)

    def metadata(self) -> dict[str, Any]:
        return {
            "target_type": self.target_type,
            "config": {
                "train_path": self.train_path,
                "val_path": self.val_path,
                "test_path": self.test_path,
                "temperature": self.temperature,
                "dimensionality": self.dim,
                "n_particles": self.n_particles,
                "spatial_dim": self.spatial_dim,
                "should_normalize": self.should_normalize,
                "normalization_factor": self.normalization_factor,
                "should_remove_mean": self.should_remove_mean,
                "pdb_path": self.pdb_path,
                "energy_batch_size": self.energy_batch_size,
                "openmm_platform": self.openmm_platform,
                "device_index": self.device_index,
                "mode_centers_path": self.mode_centers_path,
                "num_mode_centers": self.num_mode_centers,
                "mode_estimation_samples": self.mode_estimation_samples,
                "mode_strategy": self.mode_strategy,
                "rama_mode_centers_deg": self.rama_mode_centers_deg,
                "rama_mode_names": self.rama_mode_names,
            },
        }


def create_target_from_config(cfg, device: torch.device):
    target_name = str(cfg.data.get("target", "gmm")).lower()
    if target_name == "gmm":
        return None

    if target_name == "aldp":
        return AlanineDipeptideTarget(
            train_path=cfg.data.get("train_path"),
            val_path=cfg.data.get("val_path"),
            test_path=cfg.data.test_path,
            device=device,
            temperature=float(cfg.data.get("temperature", 300.0)),
            dimensionality=int(cfg.data.get("dim", 66)),
            n_particles=int(cfg.data.get("n_particles", 22)),
            spatial_dim=int(cfg.data.get("spatial_dim", 3)),
            should_normalize=bool(cfg.data.get("should_normalize", True)),
            normalization_factor=float(cfg.data.get("normalization_factor", 0.1640)),
            should_remove_mean=bool(cfg.data.get("should_remove_mean", True)),
            pdb_path=cfg.data.get("pdb_path"),
            energy_batch_size=int(cfg.data.get("energy_batch_size", 2048)),
            openmm_platform=str(cfg.data.get("openmm_platform", "CPU")),
            device_index=int(cfg.data.get("device_index", 0)),
            mode_centers_path=cfg.data.get("mode_centers_path"),
            num_mode_centers=int(cfg.data.get("num_mode_centers", 4)),
            mode_estimation_samples=int(cfg.data.get("mode_estimation_samples", 10000)),
            mode_strategy=str(cfg.data.get("mode_strategy", "rama_kmeans")),
        )

    raise ValueError(f"Unsupported target type: {target_name}")


def load_target_from_metadata(metadata: dict[str, Any], device: torch.device):
    if metadata is None:
        raise ValueError("Target metadata is required to reconstruct a target.")

    target_type = metadata.get("target_type", "gmm")
    if target_type == "gmm":
        if "config" in metadata:
            params = metadata["config"]
        else:
            # Backward compatibility for older saved GMM artifacts that stored
            # the mixture parameters at the top level instead of under "config".
            params = metadata
        return create_gaussian_mixture(
            dimension=params["dim"],
            num_components=params["num_components"],
            means=_ensure_tensor(params["means"], device),
            covs=_ensure_tensor(params["covs"], device),
            weights=_ensure_tensor(params["weights"], device) if params.get("weights") is not None else None,
            device=device,
        )

    if target_type == "aldp":
        return AlanineDipeptideTarget(device=device, **metadata["config"])

    raise ValueError(f"Unsupported target type in metadata: {target_type}")


def target_payload(target) -> dict[str, Any]:
    if hasattr(target, "target_type") and getattr(target, "target_type") == "aldp":
        return target.metadata()

    return {
        "target_type": "gmm",
        "config": {
            "means": target.component_distribution.loc.detach().cpu(),
            "covs": target.component_distribution.covariance_matrix.detach().cpu(),
            "weights": target.mixture_distribution.probs.detach().cpu(),
            "dim": target.component_distribution.loc.shape[1],
            "num_components": target.component_distribution.loc.shape[0],
        },
    }


def sample_reference(target, num_samples: int) -> torch.Tensor | None:
    if target is None:
        return None
    if hasattr(target, "sample_reference"):
        return target.sample_reference(num_samples)
    if hasattr(target, "sample"):
        return target.sample(torch.Size((num_samples,)))
    return None


def infer_mode_centers(target) -> torch.Tensor | None:
    if target is None:
        return None
    mode_centers = getattr(target, "mode_centers", None)
    if mode_centers is not None:
        return mode_centers.detach().clone()
    component_distribution = getattr(target, "component_distribution", None)
    if component_distribution is None:
        return None
    return component_distribution.mean.detach().clone()
