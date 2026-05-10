import os
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from Utils.misc import mode_weights_from_particles, p_t, normalize_weights


def _weighted_center(values, weights):
    total_weight = np.sum(weights)
    if np.isfinite(total_weight) and total_weight > 0:
        return np.average(values, weights=weights)
    return np.mean(values)


def _resolve_plot_dir(output_dir):
    if os.path.basename(os.path.normpath(output_dir)) == "plots":
        plot_dir = output_dir
    else:
        plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def plot_training_metrics(metrics_history, save_dir, hyperparams=None):
    if not metrics_history:
        return

    steps = [entry["step"] for entry in metrics_history]

    metric_keys = [
        key for key in metrics_history[0]
        if key != "step" and not key.startswith("mode_weight_")
    ]
    for key in metric_keys:
        values = [entry[key] for entry in metrics_history]
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, label=key)
        plt.title(key)
        plt.xlabel("Training Step")
        plt.ylabel(key)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{key}.png"), dpi=200)
        plt.close()

    mode_weight_keys = sorted(
        [key for key in metrics_history[0] if key.startswith("mode_weight_")],
        key=lambda key: int(key.split("_")[-1]),
    )
    if mode_weight_keys:
        plt.figure(figsize=(10, 6))
        for key in mode_weight_keys:
            plt.plot(steps, [entry[key] for entry in metrics_history], label=key)
        plt.title("Mode Weights")
        plt.xlabel("Training Step")
        plt.ylabel("Weight")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "mode_weights.png"), dpi=200)
        plt.close()
    if hyperparams:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axis("off")
        fig.patch.set_facecolor("white")
        ax.text(0.03, 0.95, "Hyperparameters", fontsize=20, fontweight="bold", va="top")
        y = 0.82
        for key, value in hyperparams.items():
            ax.text(0.06, y, f"{key}: {value}", fontsize=14, va="top")
            y -= 0.10
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "hyperparameters.png"), dpi=200)
        plt.close()


def plot_path_metrics(metrics_history, save_dir, metadata=None):
    if not metrics_history:
        return

    os.makedirs(save_dir, exist_ok=True)
    steps = [entry["step"] for entry in metrics_history]

    metric_keys = [
        key for key in metrics_history[0]
        if key != "step" and not key.startswith("mode_weight_")
    ]

    for key in metric_keys:
        values = [entry.get(key, float("nan")) for entry in metrics_history]
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, linewidth=2)
        plt.title(key)
        plt.xlabel("Path Step")
        plt.ylabel(key)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{key}.png"), dpi=200)
        plt.close()

    mode_weight_keys = sorted(
        [key for key in metrics_history[0] if key.startswith("mode_weight_")],
        key=lambda key: int(key.split("_")[-1]),
    )
    if mode_weight_keys:
        for key in mode_weight_keys:
            plt.figure(figsize=(10, 6))
            plt.plot(steps, [entry.get(key, float("nan")) for entry in metrics_history], linewidth=2)
            plt.title(key)
            plt.xlabel("Path Step")
            plt.ylabel("Weight")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{key}.png"), dpi=200)
            plt.close()

        plt.figure(figsize=(10, 6))
        for key in mode_weight_keys:
            plt.plot(steps, [entry.get(key, float("nan")) for entry in metrics_history], linewidth=2, label=key)
        plt.title("mode_weights")
        plt.xlabel("Path Step")
        plt.ylabel("Weight")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "mode_weights.png"), dpi=200)
        plt.close()

    if metadata:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axis("off")
        fig.patch.set_facecolor("white")
        ax.text(0.03, 0.95, "Eval Metadata", fontsize=20, fontweight="bold", va="top")
        y = 0.82
        for key, value in metadata.items():
            ax.text(0.06, y, f"{key}: {value}", fontsize=14, va="top")
            y -= 0.10
            if y < 0.08:
                break
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "eval_metadata.png"), dpi=200)
        plt.close()

@torch.inference_mode
def plot_walkers(
    x,
    t,
    means,
    U_net,
    step,
    modes,
    mixture,
    prior,
    energy_model,
    true_modes,
    true_samples=None,
    *,
    particle_weights: torch.Tensor | None = None,   # <-- ADD THIS
    annotate_modes: bool = True,                    # <-- optional
    grid_res: int = 500,
    gridsize_hex: int = 100,
    levels: int = 50,
    cmap_heat: str = "inferno",
    cmap_contour: str = "#00b3b3",
    alpha_contour: float = 0.20,
    use_contour: bool = True,
    share_color_scale: bool = True,
):
    device = x.device
    dim = x.shape[1]

    if dim == 1:

        pts = x.detach().cpu().view(-1)
        true_samples = true_samples.detach().cpu()

        # plotting range
        size = 20
        x_grid = torch.linspace(-size, size, grid_res, device=device)

        # compute density
        t_tensor = torch.full((x_grid.shape[0],), float(t), device=device)
        p_vals = p_t(
            x_grid.unsqueeze(-1),
            t_tensor,
            means,
            U_net,
            modes,
            mixture,
            prior,
            energy_model
        )

        plt.figure(figsize=(8,4))

        w = particle_weights / (particle_weights.sum() + 1e-12)

        plt.hist(
            pts.numpy(),
            bins=80,
            weights=w.detach().cpu().numpy(),
            density=True,
            alpha=0.55,
            color="green",
            label="Weighted Samples",
        )
        # walker histogram
        plt.hist(
            pts.numpy(),
            bins=80,
            density=True,
            alpha=0.4,
            color="gray",
            label="Walkers"
        )
        plt.hist(
            true_samples.numpy(),
            bins=80,
            density=True,
            alpha=0.4,
            color="red",
            label="True Samples"
        )

        # density curve
        plt.plot(
            x_grid.cpu(),
            p_vals.cpu(),
            linewidth=2,
            color="blue",
            label="p(x,t)"
        )

        # mark modes
        tm = true_modes.detach().cpu().view(-1)
        plt.scatter(
            tm,
            torch.zeros_like(tm),
            color="red",
            marker="x",
            s=80,
            label="Modes"
        )

        plt.title(f"1D walkers – step {step}")
        plt.xlabel("x")
        plt.ylabel("density")
        plt.xlim([-size, size])
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
        # density at each true mode (direct eval)
        t_modes = torch.full((true_modes.shape[0],), float(t), device=device)
        p_at_modes = p_t(true_modes.to(device), t_modes, means, U_net, modes, mixture, prior, energy_model)  # [M]
        p_modes = p_at_modes / (p_at_modes.sum() + 1e-12)
        mode_w, _ = mode_weights_from_particles(x, true_modes, particle_weights=particle_weights)
        print("mode weights from walkers:", mode_w.detach().cpu().numpy())

        # optional: print / annotate
        # print("p(t) at modes:", p_modes.detach().cpu().numpy())

        # pts = x.detach().cpu().view(-1)
        # true_samples = true_samples.detach().cpu()

        # # plotting range
        # size = 20
        # x_grid = torch.linspace(-size, size, grid_res, device=device)

        # # compute density
        # t_tensor = torch.full((x_grid.shape[0],), float(t), device=device)
        # p_vals = p_t(
        #     x_grid.unsqueeze(-1),
        #     t_tensor,
        #     means,
        #     U_net,
        #     modes,
        #     mixture,
        #     prior,
        #     energy_model
        # )

        # w = particle_weights / (particle_weights.sum() + 1e-12)
        # w_cpu = w.detach().cpu()

        # fig, axes = plt.subplots(1, 2, figsize=(12,4))

        # # ================================
        # # Left subplot: samples + density
        # # ================================
        # axes[0].hist(
        #     pts.numpy(),
        #     bins=80,
        #     weights=w_cpu.numpy(),
        #     density=True,
        #     alpha=0.55,
        #     color="green",
        #     label="Weighted Samples",
        # )

        # axes[0].hist(
        #     pts.numpy(),
        #     bins=80,
        #     density=True,
        #     alpha=0.4,
        #     color="gray",
        #     label="Walkers"
        # )

        # axes[0].hist(
        #     true_samples.numpy(),
        #     bins=80,
        #     density=True,
        #     alpha=0.4,
        #     color="red",
        #     label="True Samples"
        # )

        # axes[0].plot(
        #     x_grid.cpu(),
        #     p_vals.cpu(),
        #     linewidth=2,
        #     color="blue",
        #     label="p(x,t)"
        # )

        # # mark modes
        # tm = true_modes.detach().cpu().view(-1)
        # axes[0].scatter(
        #     tm,
        #     torch.zeros_like(tm),
        #     color="red",
        #     marker="x",
        #     s=80,
        #     label="Modes"
        # )

        # axes[0].set_title(f"1D walkers – step {step}")
        # axes[0].set_xlabel("x")
        # axes[0].set_ylabel("density")
        # axes[0].set_xlim([-size, size])
        # axes[0].grid(True)
        # axes[0].legend()

        # # ==================================
        # # Right subplot: particle weights
        # # ==================================
        # axes[1].hist(
        #     torch.log(w_cpu + 1e-12).numpy(),
        #     bins=60,
        #     color="purple",
        #     alpha=0.7
        # )

        # axes[1].set_title("Particle weights")
        # axes[1].set_xlabel("weight")
        # axes[1].set_ylabel("count")
        # axes[1].grid(True)

        # plt.tight_layout()
        # plt.show()

        # mode_w, _ = mode_weights_from_particles(x, true_modes, particle_weights=particle_weights)
        # print("mode weights from walkers:", mode_w.detach().cpu().numpy())

    else:
        # --- Move walkers to CPU numpy once ---
        with torch.no_grad():
            pts = x.detach().cpu().numpy()

        # --- compute mode weights (on torch tensors) ---
        mode_w, _ = mode_weights_from_particles(x, true_modes, particle_weights=particle_weights)

        # --- size heuristic ---
        if len(mixture.component_distribution.mean) == 40:
            size = 45
        else:
            size = 20

        # --- Build grid for contour ---
        xlin = torch.linspace(-size, size, grid_res, device=device)
        ylin = torch.linspace(-size, size, grid_res, device=device)
        X, Y = torch.meshgrid(xlin, ylin, indexing="xy")
        grid = torch.stack([X.flatten(), Y.flatten()], dim=-1)

        p_vals = None
        if use_contour:
            t_tensor = torch.full((grid.shape[0],), float(t), device=device)
            p_vals = p_t(grid, t_tensor, means, U_net, modes, mixture, prior, energy_model).reshape(grid_res, grid_res)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        vmin, vmax = 0, 10

        # ===== Left: Walkers heatmap =====
        hb1 = axes[0].hexbin(
            pts[:, 0], pts[:, 1],
            gridsize=gridsize_hex,
            extent=[-size, size, -size, size],
            cmap=cmap_heat,
            mincnt=1,
            vmin=vmin, vmax=vmax
        )

        # True modes overlay
        tm = true_modes.detach().cpu()
        axes[0].scatter(tm[:, 0], tm[:, 1], marker="x", color="red", s=60, linewidths=2, zorder=3, alpha=0.5)

        # Annotate mode weights near each mode marker
        if annotate_modes:
            mw_cpu = mode_w.detach().cpu()
            # for k in range(tm.shape[0]):
            #     axes[0].text(
            #         tm[k, 0].item(), tm[k, 1].item(),
            #         f"  {mw_cpu[k].item():.3f}",
            #         color="red",
            #         fontsize=16,
            #         va="center",
            #         zorder=4,
            #     )

            # Also add a compact summary box
            # summary = "Mode weights:\n" + "\n".join([f"{k}: {mw_cpu[k].item():.3f}" for k in range(len(mw_cpu))])
            # axes[0].text(
            #     0.02, 0.98, summary,
            #     transform=axes[0].transAxes,
            #     va="top", ha="left",
            #     fontsize=10,
            #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            #     zorder=5,
            # )

        if use_contour and p_vals is not None:
            axes[0].contour(
                X.detach().cpu().numpy(),
                Y.detach().cpu().numpy(),
                p_vals.detach().cpu().numpy(),
                levels=levels,
                colors=cmap_contour,
                alpha=alpha_contour,
                zorder=2,
            )
            # density at each true mode (direct eval)
            t_modes = torch.full((true_modes.shape[0],), float(t), device=device)
            p_at_modes = p_t(true_modes.to(device), t_modes, means, U_net, modes, mixture, prior, energy_model)  # [M]
            p_modes = p_at_modes / (p_at_modes.sum() + 1e-12)

            # optional: print / annotate
            print("p(t) at modes:", p_modes.detach().cpu().numpy())

            # p_cpu = p_at_modes.detach().cpu()
            # for k in range(tm.shape[0]):
            #     axes[0].text(
            #         tm[k, 0].item(), 
            #         tm[k, 1].item(),
            #         f"\n p={p_cpu[k].item():.2e}",
            #         color="red", 
            #         fontsize=10,
            #          va="center",
            #           zorder=4
            #     )

        axes[0].set_title(f"Walkers – step {step}")
        axes[0].set_xlabel(r"$x_1$")
        axes[0].set_ylabel(r"$x_2$")

        # ===== Right: True samples heatmap =====
        hb2 = None
        if true_samples is not None:
            true_pts = true_samples.detach().cpu().numpy()
            hb2 = axes[1].hexbin(
                true_pts[:, 0], true_pts[:, 1],
                gridsize=gridsize_hex,
                extent=[-size, size, -size, size],
                cmap=cmap_heat,
                mincnt=1,
                vmin=vmin, vmax=vmax
            )
            if share_color_scale and hb2 is not None:
                hb2.set_clim(hb1.get_clim())

            axes[1].scatter(tm[:, 0], tm[:, 1], marker="x", color="red", s=60, linewidths=2, zorder=3)

        axes[1].set_title("True Samples" if true_samples is not None else "True Samples (none)")
        axes[1].set_xlabel(r"$x_1$")

        for ax in axes:
            ax.set_xlim([-size, size])
            ax.set_ylim([-size, size])
            ax.set_facecolor("#f0f0f0")
            ax.grid(True)

        fig.subplots_adjust(right=1.0)
        cbar_ax = fig.add_axes([0.97, 0.15, 0.02, 0.70])
        cbar = fig.colorbar(hb1, cax=cbar_ax)
        cbar.set_label("Density")

        plt.tight_layout()
        plt.show()
        mode_w, _ = mode_weights_from_particles(x, true_modes, particle_weights=particle_weights)
        print("mode weights from walkers:", mode_w.detach().cpu().numpy())


def plot_annealed_langevin(means, modes, n_steps, true_modes, mixture, device, load_dir):
    plot_dir = _resolve_plot_dir(load_dir)
    # 6. Set up plot and animation
    if len(mixture.component_distribution.mean) == 40:
        size = 45
    else:
        size = 20
        
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_aspect('equal')
    ax.set_title("Annealed Langevin Dynamics")
    ax.grid(True)

    # Create a 2D grid
    x = torch.linspace(-size, size, 500, device=device)
    y = torch.linspace(-size, size, 500, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    # Precompute contour levels once (fix levels for stability)
    levels = 10

    def update(frame):
        ax.clear()
        t_tensor = torch.full((grid.shape[0],), frame*dt, device=device)
        ax.scatter(true_modes[:, 0].detach().cpu(), true_modes[:, 1].detach().cpu(), marker='x', color='red', s=10)
        p_vals = p_t(grid, t_tensor, means, U_net, modes, mixture).reshape(500, 500)
        ax.contour(X.cpu().numpy(), Y.cpu().numpy(), p_vals.detach().cpu().numpy(), levels=levels, cmap='summer', alpha=0.5)
        ax.scatter(x_langevin[frame][:, 0].cpu(), x_langevin[frame][:, 1].cpu(),
                s=10, color='grey', alpha=0.3)
        ax.set_xlim(-size, size)
        ax.set_ylim(-size, size) 
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.set_title(f"Annealed Langevin Dynamics, time={t_tensor[0] + dt:.2f}")

    anim = FuncAnimation(fig, update, frames=n_steps, interval=100)

    # To save as GIF uncomment:
    gif_path = os.path.join(plot_dir, 'ALD_eps1_hard.gif')
    anim.save(gif_path, writer=PillowWriter(fps=60))

    plt.show()

def plot_nets(means, modes, n_steps, true_modes, mixture, device, load_dir):
    plot_dir = _resolve_plot_dir(load_dir)
    # Create a 2D grid
    if len(mixture.component_distribution.mean) == 40:
        size = 45
    else:
        size = 20
    dt = 1 / n_steps
    x = torch.linspace(-size, size, 500, device=device)
    y = torch.linspace(-size, size, 500, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    # 6. Set up plot and animation
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_aspect('equal')
    ax.set_title("NETS Dynamics")
    ax.grid(True)

    levels = 10

    def update(frame):
        ax.clear()
        t_tensor = torch.full((grid.shape[0],), frame * dt, device=grid.device)
        p_vals = p_t(grid, t_tensor, means, U_net, modes, mixture).reshape(500, 500)
        ax.scatter(true_modes[:, 0].detach().cpu(), true_modes[:, 1].detach().cpu(), marker='x', color='red', s=10)
        ax.contour(X.cpu(), Y.cpu(), p_vals.detach().cpu(), levels=levels, cmap='summer', alpha=0.5)
        ax.scatter(x_nets[frame][:, 0].cpu(), x_nets[frame][:, 1].cpu(),
                s=10, color='grey', alpha=0.3)
        ax.set_xlim(-size, size)
        ax.set_ylim(-size, size) 
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.set_title(f"NETS Dynamics, time={t_tensor[0]+dt:.2f}")


    anim = FuncAnimation(fig, update, frames=n_steps, interval=100)

    # To save as GIF uncomment:
    gif_path = os.path.join(plot_dir, 'nets_eps4.gif')
    anim.save(gif_path, writer=PillowWriter(fps=60))

    plt.show()

def plot_weighted_samples(generated_samples, true_samples, weights, labels, labels_true, is_ais=True):

    X_true = true_samples
    # -------- Convert model samples --------
    if isinstance(generated_samples, torch.Tensor):
        X_plot = generated_samples.detach().cpu().numpy()
    else:
        X_plot = generated_samples

    if isinstance(labels, torch.Tensor):
        labels_plot = labels.detach().cpu().numpy()
    else:
        labels_plot = labels

    if isinstance(weights, torch.Tensor):
        w = normalize_weights(weights).detach().cpu().numpy()
    else:
        w = normalize_weights(weights)

    # -------- Convert true samples --------
    if isinstance(X_true, torch.Tensor):
        X_true_plot = X_true.detach().cpu().numpy()
    else:
        X_true_plot = X_true

    if isinstance(labels_true, torch.Tensor):
        labels_true_plot = labels_true.detach().cpu().numpy()
    else:
        labels_true_plot = labels_true

    if X_plot.ndim == 1:
        X_plot = X_plot[:, None]
    if X_true_plot.ndim == 1:
        X_true_plot = X_true_plot[:, None]

    generated_is_1d = X_plot.shape[1] == 1
    true_is_1d = X_true_plot.shape[1] == 1

    # -------- Create figure --------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # ===============================
    # LEFT: Model Samples (weighted)
    # ===============================
    ax = axes[0]
    unique_modes = np.unique(labels_plot)

    for k in unique_modes:
        idx = labels_plot == k
        y_vals = np.zeros(np.sum(idx)) if generated_is_1d else X_plot[idx, 1]

        ax.scatter(
            X_plot[idx, 0],
            y_vals,
            s=20,
            alpha=0.7,
            label=f"Mode {k}"
        )

        mode_weight = w[idx].sum()

        center_x = _weighted_center(X_plot[idx, 0], w[idx])
        center_y = 0.0 if generated_is_1d else _weighted_center(X_plot[idx, 1], w[idx])

        ax.text(
            center_x,
            center_y,
            f"{mode_weight:.3f}",
            fontsize=13,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.8)
        )
    if is_ais:
        title = "AIS Samples (Weighted)"
    else:
        title = "NETS Samples (Weighted)"
    ax.set_title(title)
    ax.legend()

    # ===============================
    # RIGHT: True Samples (uniform)
    # ===============================
    ax = axes[1]
    unique_modes_true = np.unique(labels_true_plot)

    N_true = len(X_true_plot)
    uniform_w = np.ones(N_true) / N_true

    for k in unique_modes_true:
        idx = labels_true_plot == k
        y_vals = np.zeros(np.sum(idx)) if true_is_1d else X_true_plot[idx, 1]

        ax.scatter(
            X_true_plot[idx, 0],
            y_vals,
            s=20,
            alpha=0.7,
            label=f"Mode {k}"
        )

        mode_weight = uniform_w[idx].sum()

        center_x = X_true_plot[idx, 0].mean()
        center_y = 0.0 if true_is_1d else X_true_plot[idx, 1].mean()

        ax.text(
            center_x,
            center_y,
            f"{mode_weight:.3f}",
            fontsize=13,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.8)
        )

    ax.set_title("True Samples")
    ax.legend()

    for ax in axes:
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂" if not (generated_is_1d and true_is_1d) else "0")

    plt.tight_layout()
    plt.show()


def save_weighted_samples_plot(
    generated_samples,
    true_samples,
    weights,
    labels,
    labels_true,
    save_path,
    is_ais=True,
):
    X_true = true_samples

    if isinstance(generated_samples, torch.Tensor):
        X_plot = generated_samples.detach().cpu().numpy()
    else:
        X_plot = generated_samples

    if isinstance(labels, torch.Tensor):
        labels_plot = labels.detach().cpu().numpy()
    else:
        labels_plot = labels

    if isinstance(weights, torch.Tensor):
        w = normalize_weights(weights).detach().cpu().numpy()
    else:
        w = normalize_weights(weights)

    if isinstance(X_true, torch.Tensor):
        X_true_plot = X_true.detach().cpu().numpy()
    else:
        X_true_plot = X_true

    if isinstance(labels_true, torch.Tensor):
        labels_true_plot = labels_true.detach().cpu().numpy()
    else:
        labels_true_plot = labels_true

    if X_plot.ndim == 1:
        X_plot = X_plot[:, None]
    if X_true_plot.ndim == 1:
        X_true_plot = X_true_plot[:, None]

    generated_is_1d = X_plot.shape[1] == 1
    true_is_1d = X_true_plot.shape[1] == 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    ax = axes[0]
    unique_modes = np.unique(labels_plot)
    for k in unique_modes:
        idx = labels_plot == k
        y_vals = np.zeros(np.sum(idx)) if generated_is_1d else X_plot[idx, 1]
        ax.scatter(X_plot[idx, 0], y_vals, s=20, alpha=0.7, label=f"Mode {k}")

        mode_weight = w[idx].sum()
        center_x = _weighted_center(X_plot[idx, 0], w[idx])
        center_y = 0.0 if generated_is_1d else _weighted_center(X_plot[idx, 1], w[idx])
        ax.text(
            center_x,
            center_y,
            f"{mode_weight:.3f}",
            fontsize=13,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.8),
        )

    ax.set_title("AIS Samples (Weighted)" if is_ais else "NETS Samples (Weighted)")
    ax.legend()

    ax = axes[1]
    unique_modes_true = np.unique(labels_true_plot)
    n_true = len(X_true_plot)
    uniform_w = np.ones(n_true) / n_true
    for k in unique_modes_true:
        idx = labels_true_plot == k
        y_vals = np.zeros(np.sum(idx)) if true_is_1d else X_true_plot[idx, 1]
        ax.scatter(X_true_plot[idx, 0], y_vals, s=20, alpha=0.7, label=f"Mode {k}")

        mode_weight = uniform_w[idx].sum()
        center_x = X_true_plot[idx, 0].mean()
        center_y = 0.0 if true_is_1d else X_true_plot[idx, 1].mean()
        ax.text(
            center_x,
            center_y,
            f"{mode_weight:.3f}",
            fontsize=13,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.8),
        )

    ax.set_title("True Samples")
    ax.legend()

    for ax in axes:
        ax.set_xlabel("x1")
        ax.set_ylabel("x2" if not (generated_is_1d and true_is_1d) else "0")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

def plot_weights_hist(weights):
    w = weights.detach().cpu().numpy()
    plt.figure()
    plt.hist(w, bins=50)
    plt.xlabel("Weight value")
    plt.ylabel("Frequency")
    plt.title("Histogram of importance weights")
    plt.show()        
