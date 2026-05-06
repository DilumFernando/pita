import torch
import math
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor
from Utils.misc import beta_schedule


def _inverse_softplus(value: Tensor) -> Tensor:
    value = torch.clamp(value, min=torch.finfo(value.dtype).eps)
    return value + torch.log(-torch.expm1(-value))


def random_rotation_matrix(dimension: int, device=None, dtype=torch.float32) -> Tensor:
    q, r = torch.linalg.qr(torch.randn(dimension, dimension, device=device, dtype=dtype))
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    q = q @ torch.diag(signs)
    return q


def make_scaled_identity_covariances(
    dimension: int,
    scales,
    *,
    device=None,
    dtype=torch.float32,
) -> Tensor:
    scales = torch.as_tensor(scales, device=device, dtype=dtype)
    eye = torch.eye(dimension, device=device, dtype=dtype).unsqueeze(0)
    return scales.view(-1, 1, 1) * eye


def make_rotated_diagonal_covariances(
    dimension: int,
    diagonal_scales,
    *,
    random_rotation_per_component: bool = True,
    device=None,
    dtype=torch.float32,
) -> Tensor:
    diagonal_scales = torch.as_tensor(diagonal_scales, device=device, dtype=dtype)
    if diagonal_scales.ndim != 2 or diagonal_scales.shape[1] != dimension:
        raise ValueError(
            f"diagonal_scales must have shape [num_components, {dimension}], "
            f"got {tuple(diagonal_scales.shape)}"
        )

    covariances = []
    shared_rotation = None
    if not random_rotation_per_component:
        shared_rotation = random_rotation_matrix(dimension, device=device, dtype=dtype)

    for diag_vals in diagonal_scales:
        rotation = shared_rotation
        if rotation is None:
            rotation = random_rotation_matrix(dimension, device=device, dtype=dtype)
        cov = rotation @ torch.diag(diag_vals) @ rotation.transpose(-1, -2)
        covariances.append(cov)

    return torch.stack(covariances, dim=0)


def make_rotated_full_covariances(
    dimension: int,
    base_covariances,
    *,
    random_rotation_per_component: bool = True,
    device=None,
    dtype=torch.float32,
) -> Tensor:
    base_covariances = torch.as_tensor(base_covariances, device=device, dtype=dtype)
    if base_covariances.ndim != 3 or base_covariances.shape[1:] != (dimension, dimension):
        raise ValueError(
            f"base_covariances must have shape [num_components, {dimension}, {dimension}], "
            f"got {tuple(base_covariances.shape)}"
        )

    covariances = []
    shared_rotation = None
    if not random_rotation_per_component:
        shared_rotation = random_rotation_matrix(dimension, device=device, dtype=dtype)

    for base_cov in base_covariances:
        rotation = shared_rotation
        if rotation is None:
            rotation = random_rotation_matrix(dimension, device=device, dtype=dtype)
        cov = rotation @ base_cov @ rotation.transpose(-1, -2)
        covariances.append(cov)

    return torch.stack(covariances, dim=0)


def create_scaled_identity_gaussian_mixture(
    dimension: int,
    scales,
    *,
    means=None,
    weights=None,
    device=None,
) -> D.MixtureSameFamily:
    covs = make_scaled_identity_covariances(dimension, scales, device=device)
    return create_gaussian_mixture(
        dimension,
        len(scales),
        means=means,
        covs=covs,
        weights=weights,
        device=device,
    )


def create_rotated_gaussian_mixture(
    dimension: int,
    diagonal_scales,
    *,
    means=None,
    weights=None,
    random_rotation_per_component: bool = True,
    device=None,
) -> D.MixtureSameFamily:
    covs = make_rotated_diagonal_covariances(
        dimension,
        diagonal_scales,
        random_rotation_per_component=random_rotation_per_component,
        device=device,
    )
    return create_gaussian_mixture(
        dimension,
        covs.shape[0],
        means=means,
        covs=covs,
        weights=weights,
        device=device,
    )


class LearnableGMM(nn.Module):
    def __init__(
        self,
        means: Tensor,  # nmodes x data_dim
        covs: Tensor,  # nmodes 
        logits: Tensor,  # nmodes
        beta: float = 1.0,  # Inverse temperature
    ):
        super().__init__()
        self.nmodes = means.shape[0]
        self.device = means.device
        self.dim = means.shape[1]
        self.beta = beta
        self.register_buffer("covs", covs.view(self.nmodes, 1, 1)* torch.eye(self.dim, device=self.device).view(1, self.dim, self.dim))
        self.register_buffer("means", means)
        if logits is None:
            logits = torch.zeros(self.nmodes, device=self.device)
        self.logits = nn.Parameter(logits)

    @property
    def distribution(self):
        return D.MixtureSameFamily(
            mixture_distribution=D.Categorical(logits=self.logits, validate_args=False),
            component_distribution=D.MultivariateNormal(
                loc=self.means,
                covariance_matrix=self.covs,
                validate_args=False,
            ),
            validate_args=False,
        )

    @property
    def weights(self):
        return torch.softmax(self.logits, dim=0)

    def log_density(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, dim)
        Returns:
            log_density: (batch_size,)
        """
        return self.beta * self.distribution.log_prob(x).view(-1, 1)

    def energy(self, x: Tensor) -> Tensor:
        return -self.log_density(x)

    def raise_to_temperature(self, beta: float) -> "LearnableGMM":
        return LearnableGMM(self.means, self.covs, self.weights, self.beta * beta)
    
    def scale_covariance(self, beta_t: Tensor) -> "LearnableGMM":
        self.covs = self.covs / beta_t[0] + 1e-8
        return LearnableGMM(self.means, self.covs, self.weights, self.beta)

class GMMModesEnergy(nn.Module):
    def __init__(
        self,
        modes: torch.Tensor,
        beta_max: torch.Tensor,
        init_logits: torch.Tensor | None = None,
        beta_max_learnable: bool = True,
    ):
        """
        modes: [C, D] (fixed component means; you can also make them learnable if you want)
        init_logits: [C] optional initialization for mixture logits
        """
        super().__init__()
        self.register_buffer("modes", modes)  # keep fixed; change to nn.Parameter if you want learnable means
        C = modes.shape[0]
        if init_logits is None:
            init_logits = torch.zeros(C, device=modes.device, dtype=modes.dtype)
        # self.logits = nn.Parameter(init_logits)  # learnable mixture weights (via softmax)
        self.register_buffer("logits", init_logits)
        self.beta_max_learnable = bool(beta_max_learnable)
        beta_max = torch.as_tensor(beta_max, device=modes.device, dtype=modes.dtype)
        beta_max_unconstrained = _inverse_softplus(beta_max)
        if self.beta_max_learnable:
            self.beta_max_unconstrained = nn.Parameter(beta_max_unconstrained)
        else:
            self.register_buffer("beta_max_unconstrained", beta_max_unconstrained)

    @property
    def beta_max(self) -> torch.Tensor:
        return F.softplus(self.beta_max_unconstrained)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: [N, D]
        t: [N] or scalar
        returns: Ut [N, 1], beta [N]
        """
        device = x.device
        N, D = x.shape
        C = self.modes.shape[0]

        beta = beta_schedule(t, self.beta_max).to(device)  # should return [N] if t is [N], else scalar; we standardize below
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta, device=device, dtype=x.dtype)
        beta = beta.to(device=device, dtype=x.dtype).view(-1)  # [N] or [1]
        if beta.numel() == 1:
            beta = beta.expand(N)  # [N]

        means_ = self.modes[None, :, :]                         # [1, C, D]
        diff2 = (x[:, None, :] - means_).pow(2).sum(dim=-1)     # [N, C]
        energy = 0.5 * beta[:, None] * diff2                    # [N, C]

        # Learnable mixture weights
        log_w = torch.log_softmax(self.logits, dim=0)           # [C]

        
        logp = torch.logsumexp(log_w[None, :] - energy, dim=1)  # [N]
        Ut = -logp                                                     # [N]
        return Ut

class GMMModesEnergyTimeLogits(nn.Module):
    def __init__(
        self,
        modes: torch.Tensor,
        beta_max: float | torch.Tensor,
        logits_net: nn.Module,
        beta_max_learnable: bool = True,
    ):
        super().__init__()
        self.register_buffer("modes", modes)   # [C, D]
        self.logits_net = logits_net           # [N, C] if t is [N]
        self.beta_max_learnable = bool(beta_max_learnable)
        beta_max = torch.as_tensor(beta_max, device=modes.device, dtype=modes.dtype)
        beta_max_unconstrained = _inverse_softplus(beta_max)
        if self.beta_max_learnable:
            self.beta_max_unconstrained = nn.Parameter(beta_max_unconstrained)
        else:
            self.register_buffer("beta_max_unconstrained", beta_max_unconstrained)

    @property
    def beta_max(self) -> torch.Tensor:
        return F.softplus(self.beta_max_unconstrained)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: [N, D]
        t: [N] or scalar

        Returns:
            U: [N]
        """
        device = x.device
        dtype = x.dtype
        N, D = x.shape
        C = self.modes.shape[0]

        # beta(t): [N]
        beta = beta_schedule(t, beta_max=self.beta_max).to(device=device, dtype=dtype).view(-1)
        if beta.numel() == 1:
            beta = beta.expand(N)
        elif beta.numel() != N:
            raise ValueError(f"beta should have {N} entries or 1 entry, got {beta.shape}")

        # logits_t: [N, C]
        logits_t = self.logits_net(t)
        if logits_t.shape != (N, C):
            raise ValueError(f"logits_net(t) should return shape {(N, C)}, got {tuple(logits_t.shape)}")

        log_w = F.log_softmax(logits_t, dim=-1)   # [N, C]

        # quadratic term
        means_ = self.modes[None, :, :]                     # [1, C, D]
        diff2 = (x[:, None, :] - means_).pow(2).sum(dim=-1)  # [N, C]
        energy = 0.5 * beta[:, None] * diff2               # [N, C]

        # log mixture
        log_q = torch.logsumexp(log_w - energy, dim=1)     # [N]
        U = -log_q
        return U

# class GMMModesEnergyTimeLogits(nn.Module):
#     def __init__(self, modes: torch.Tensor, beta_max: float, logits_net: nn.Module, log_w: None):
#         super().__init__()
#         self.register_buffer("modes", modes)   # [C,D]
#         self.logits_net = logits_net           # outputs [N,C]
#         self.beta_max = float(beta_max)
#         self.log_w = log_w

#     def forward(self, x: torch.Tensor, t: torch.Tensor):
#         """
#         Returns:
#           U_gmm(x,t) = -log sum_k softmax(logits_net(t))_k * exp(-0.5*beta(t)*||x-mu_k||^2)
#           beta: [N]
#         """
#         device = x.device
#         N, D = x.shape
#         C = self.modes.shape[0]

#         # beta(t)
#         beta = beta_schedule(t, beta_max=self.beta_max).to(device=device, dtype=x.dtype).view(-1)
#         if beta.numel() == 1:
#             beta = beta.expand(N)  # [N]
        
#         # logits_t should be [C]
#         # t_in = t.to(device=device, dtype=t.dtype)
#         logits_t = self.logits_net(t)

#         # if logits_t.ndim != 1 or logits_t.shape[0] != C:
#         #     raise ValueError(f"logits_net(t) should return shape [{C}], got {tuple(logits_t.shape)}")

#         self.log_w = F.log_softmax(logits_t, dim=0)   # [N, C]

#         # quadratic term
#         means_ = self.modes[None, :, :]                          # [1,C,D]
#         diff2 = (x[:, None, :] - means_).pow(2).sum(dim=-1)      # [N,C]
#         energy = 0.5 * beta[:, None] * diff2                       # [N,C]

#         # log q_t(x)
#         log_q = torch.logsumexp(self.log_w - energy, dim=1)             # [N]
#         U = -log_q                                               # [N]
#         return U


def create_gaussian_mixture(dimension: int, 
                            num_components: int, 
                            *,
                            eps: float = 1e-4, 
                            device=None, 
                            means=None, 
                            covs=None, 
                            weights=None) -> D.MixtureSameFamily:
    """
    Create an unconstrained, random Mixture of Gaussians in a given dimension

    Parameters
    ----------
    dimension : int
        Dimensionality of each component.
    num_components : int
        Number of Gaussian components.
    eps : float, optional
        Jitter added to the diagonal for numerical stability.
    device : torch.device or None
        Move tensors to this device if given.

    Returns
    -------
    torch.distributions.MixtureSameFamily
    """

    if means is None:
        means = 16*torch.rand(num_components, dimension, device=device) - 8
    else:
        means = means.to(device)
        # print(means.shape)
        
    if (means == 1).all():
        covs = 2*torch.eye(dimension, device=device)
        pass
    else:
        if covs is None:
            covs = 1*torch.eye(dimension, device=device)
        else:
            # covs = (1/covs)*torch.eye(dimension, device=device)
            if covs.ndim == 0:
                # scalar variance shared by all components
                covs = covs*torch.eye(dimension, device=device)
                # covs = covs.expand(num_components)
            elif covs.ndim == 1:
                # (K,) → isotropic diag per component
                covs = covs.view(num_components, 1, 1)* torch.eye(dimension, device=device).view(1, dimension, dimension)
                # print(covs.shape)
            elif covs.ndim == 2:
                # (K, dim) → diagonal per component
                covs = torch.diag_embed(covs)
            elif covs.ndim == 3:
                # (K, dim, dim) → full cov, keep as is
                pass
            else:
                raise ValueError(f"Unsupported covs shape: {covs.shape}")
    

    # Mixture weights  (K,)
    if weights is None:
        if len(means.shape) == 2:
            weights = torch.ones(num_components, device=device)
            # weights = 10*torch.rand(num_components, device=device)
            weights.div_(weights.sum())
        else:
            batch_size = means.shape[0]
            weights = torch.ones(batch_size, num_components, device=device)
            # weights = 10*torch.rand(batch_size, num_components, device=device)
            weights.div_(weights.sum(dim=1, keepdim=True))
    else:
        assert weights.sum() == 1, "The weights must sum to 1"
        weights = weights

    # Build the distribution
    # print(means.shape, covs.shape)
    mvn = D.MultivariateNormal(loc=means, covariance_matrix=covs)
    mix = D.MixtureSameFamily(
        mixture_distribution=D.Categorical(probs=weights),
        component_distribution=mvn,
    )
    return mix
