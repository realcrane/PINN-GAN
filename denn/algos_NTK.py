import os
import numpy as np
import torch
import torch.nn as nn
import time
from denn.utils import plot_results, calc_gradient_penalty, handle_overwrite, plot_3D
from denn.config.config import write_config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
this_dir = os.path.dirname(os.path.abspath(__file__))


def flatten_grads(grad_list):
    chunks = []
    for g in grad_list:
        if g is None:
            continue
        chunks.append(g.reshape(-1))
    if len(chunks) == 0:
        return torch.tensor([])
    return torch.cat(chunks)


def empirical_ntk(module, outputs):
    """Return the exact empirical NTK for scalar outputs on a small probe set.

    The returned matrix is ``J J^T``, where each row of ``J`` is the gradient
    of one scalar output with respect to all trainable module parameters.
    This helper is intended for diagnostics on a small, fixed probe set; it is
    deliberately not used on the full collocation grid.
    """
    params = [p for p in module.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Cannot compute an NTK without trainable parameters.")

    output_vec = outputs.reshape(-1)
    jacobian_rows = []
    for output_idx in range(output_vec.numel()):
        grads = torch.autograd.grad(
            outputs=output_vec[output_idx],
            inputs=params,
            retain_graph=output_idx + 1 < output_vec.numel(),
            create_graph=False,
            allow_unused=True,
        )
        jacobian_rows.append(torch.cat([
            (
                grad if grad is not None else torch.zeros_like(param)
            ).reshape(-1)
            for grad, param in zip(grads, params)
        ]))

    jacobian = torch.stack(jacobian_rows)
    return (jacobian @ jacobian.T).detach()


def ntk_comparison(kernel, reference, eps=1e-30):
    """Compute scale-sensitive drift and scale-free alignment to a reference."""
    kernel_norm = torch.linalg.vector_norm(kernel)
    reference_norm = torch.linalg.vector_norm(reference)
    relative_drift = (
        torch.linalg.vector_norm(kernel - reference)
        / reference_norm.clamp_min(eps)
    )
    alignment = (
        torch.sum(kernel * reference)
        / (kernel_norm * reference_norm).clamp_min(eps)
    )
    mean_diagonal = torch.diagonal(kernel).mean()
    return {
        'relative_drift': relative_drift.item(),
        'alignment': alignment.item(),
        'trace': torch.trace(kernel).item(),
        'mean_diagonal': mean_diagonal.item(),
    }


def save_ntk_drift_plot(history, save_path):
    """Save a compact visualization of generator/discriminator NTK drift."""
    import matplotlib.pyplot as plt

    steps = np.asarray(history['steps'])
    generator_drift = np.asarray(history['generator_relative_drift'])
    discriminator_drift = np.asarray(
        history['discriminator_relative_drift']
    )
    generator_alignment = np.asarray(history['generator_alignment'])
    discriminator_alignment = np.asarray(
        history['discriminator_alignment']
    )
    generator_trace = np.asarray(history['generator_trace'])
    discriminator_trace = np.asarray(history['discriminator_trace'])

    generator_trace_ratio = generator_trace / max(
        abs(generator_trace[0]), 1e-30
    )
    discriminator_trace_ratio = discriminator_trace / max(
        abs(discriminator_trace[0]), 1e-30
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].semilogy(
        steps, np.maximum(generator_drift, 1e-16), label='Generator'
    )
    axes[0].semilogy(
        steps, np.maximum(discriminator_drift, 1e-16),
        label='Discriminator',
    )
    axes[0].set_title('Relative NTK drift')
    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel('Relative Frobenius change')
    axes[0].legend()

    axes[1].plot(steps, generator_alignment, label='Generator')
    axes[1].plot(steps, discriminator_alignment, label='Discriminator')
    axes[1].set_title('NTK alignment')
    axes[1].set_xlabel('Training step')
    axes[1].set_ylabel('Kernel alignment')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()

    axes[2].plot(steps, generator_trace_ratio, label='Generator')
    axes[2].plot(
        steps, discriminator_trace_ratio, label='Discriminator'
    )
    axes[2].axhline(1.0, color='black', linestyle='--', linewidth=1)
    axes[2].set_title('NTK trace / initial trace')
    axes[2].set_xlabel('Training step')
    axes[2].set_ylabel('Relative trace')
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def clone_module_state(module):
    """Clone parameters and buffers without moving them off device."""
    return {
        key: value.detach().clone()
        for key, value in module.state_dict().items()
    }


def update_and_restore_best_state(module, best_state, is_better):
    """Update a best-state snapshot and restore it without a CPU sync."""
    with torch.no_grad():
        for key, current_value in module.state_dict().items():
            best_value = best_state[key]
            best_value.copy_(
                torch.where(is_better, current_value, best_value)
            )
            current_value.copy_(best_value)


def update_best_state(module, best_state, is_better):
    """Update a best-state snapshot without changing the live module."""
    with torch.no_grad():
        for key, current_value in module.state_dict().items():
            best_value = best_state[key]
            best_value.copy_(
                torch.where(is_better, current_value, best_value)
            )


def compute_gamma(fake, D, wgan=True,lsgan=True,gan=True, eps=1e-14):
    """
    fake: residual vector, shape [N, 1] or [N]
    D: discriminator
    returns:
        gamma: shape same as fake
        D_fake: D(fake)
        dD_dfake: derivative of D wrt fake
    """
    fake_for_gamma = fake.detach().clone().requires_grad_(True)
    D_fake = D(fake_for_gamma)

    dD_dfake = torch.autograd.grad(
        outputs=D_fake.sum(),
        inputs=fake_for_gamma,
        create_graph=False,
        retain_graph=False
    )[0]

    N = fake_for_gamma.numel()

    if wgan:
        # Generator loss in your code:
        # g_loss = mean(D(fake))  (since real_labels = 1)
        # So gamma_i = -(1/N) * dD/dfake_i
        gamma = (1.0 / N) * dD_dfake

    if gan:
        gamma = (1.0 / N) * dD_dfake / (1.0 - D_fake.detach() + eps)

    if lsgan:
        gamma =- (1.0 / N) * dD_dfake * (D_fake.detach()-1)

    return gamma.detach(), D_fake.detach(), dD_dfake.detach()


def compute_rKr_gamma(
    G,
    residuals,
    gamma,
    cached_JTr=None,
    return_JTr=False,
):
    """
    Compute s = r^T K_rr gamma without explicitly building K_rr.

    residuals: current residual vector r, must still connect to G graph
    gamma: detached weight vector, same shape as residuals
    """
    params = [p for p in G.parameters() if p.requires_grad]

    r_vec = residuals.reshape(-1)
    gamma_vec = gamma.reshape(-1)

    # J^T r is invariant while G, the grid, and residuals are fixed.
    if cached_JTr is None:
        G.zero_grad(set_to_none=True)
        scalar_r = torch.dot(r_vec, r_vec.detach())
        grads_r = torch.autograd.grad(
            outputs=scalar_r,
            inputs=params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )
        JTr = flatten_grads(grads_r)
    else:
        JTr = cached_JTr

    # J^T gamma
    G.zero_grad(set_to_none=True)
    scalar_gamma = torch.dot(r_vec, gamma_vec)
    grads_gamma = torch.autograd.grad(
        outputs=scalar_gamma,
        inputs=params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True
    )
    JTgamma = flatten_grads(grads_gamma)

    s = torch.dot(JTr, JTgamma)
    if return_JTr:
        return s.detach(), JTr
    return s.detach()


def train_GAN_2D(
    G, D, problem, method='unsupervised', niters=100,
    g_lr=1e-3, g_betas=(0.0, 0.9), d_lr=1e-3, d_betas=(0.0, 0.9),
    lr_schedule=True, gamma=0.999, step_size=15, noise=False, G_iters=100, D_iters=100,omega = 1,
    search_G_steps=100, search_D_steps=100, tol_G=1e-12, tol_D=1e-10,
    wgan=True, gan=True, lsgan=True,gp=0.1, train_mse=True, log=True, log_every=1,
    plot=False, plot_1d_curves=False,
    save=False, dirname='train_GAN', config=None, save_for_animation=False, view=[35, -55],
    hard=True,ntk=True, energy=True, optimize_redundant_metrics=True,
    track_ntk_drift=False, ntk_probe_size=16, ntk_log_every=100,
    save_ntk_kernels=True, ntk_discriminator_min=-1.0,
    ntk_discriminator_max=1.0,**kwargs):

    assert method in ['supervised', 'semisupervised', 'unsupervised'], \
        f'Method {method} not understood!'
    if track_ntk_drift:
        if ntk_probe_size < 1:
            raise ValueError("ntk_probe_size must be at least 1.")
        if ntk_log_every < 1:
            raise ValueError("ntk_log_every must be at least 1.")
        if ntk_discriminator_min >= ntk_discriminator_max:
            raise ValueError(
                "ntk_discriminator_min must be smaller than "
                "ntk_discriminator_max."
            )

    print(torch.cuda.is_available())

    dirname = os.path.join(this_dir, '../experiments/runs', dirname)
    if plot and save:
        handle_overwrite(dirname)

    # --------------------------------------------------
    # validation: fixed grid/solution
    # --------------------------------------------------
    x, y = problem.get_grid()
    grid = torch.cat((x, y), 1)
    soln = problem.get_solution(x, y)

    # initialize sampling state
    grid_samp = grid
    residuals = torch.zeros_like(x)
    residuals_delta = torch.zeros_like(grid_samp)

    # plot grid
    if plot or save_for_animation:
        dims = problem.get_plot_dims()
        plot_x, plot_y = problem.get_plot_grid(grid_samp, residuals, residuals_delta)
        plot_grid = torch.cat((plot_x, plot_y), 1)
        plot_soln = problem.get_plot_solution(plot_x, plot_y)

    # labels
    real_label = 1
    fake_label = -1 if wgan else 0
    real_labels = torch.full(
        (len(grid),), real_label, dtype=torch.float, device=grid.device
    ).reshape(-1, 1)
    fake_labels = torch.full(
        (len(grid),), fake_label, dtype=torch.float, device=grid.device
    ).reshape(-1, 1)

    diff = 0.0

    # --------------------------------------------------
    # optimizers
    # --------------------------------------------------
    optiG = torch.optim.Adam(G.parameters(), lr=g_lr, betas=g_betas)
    optiD = torch.optim.Adam(D.parameters(), lr=d_lr, betas=d_betas)

    if lr_schedule:
        lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optiG, step_size=step_size, gamma=gamma)
        lr_scheduler_D = torch.optim.lr_scheduler.StepLR(optiD, step_size=step_size, gamma=gamma)

    # losses
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    wass = lambda y_true, y_pred: torch.mean(y_true * y_pred)
    criterion = wass if wgan else bce

    # --------------------------------------------------
    # history
    # --------------------------------------------------
    losses = {
        'G': [],
        'D': [],
        'rKr_gamma': [],
        'residual_energy': [],
        'actual_G_iters': [],
        'actual_D_iters': [],
        'best_G_step': [],
        'best_D_step': [],
    }
    mses = {'train': [], 'val': []} if train_mse else {'val': []}
    ntk_history = {
        'steps': [],
        'generator_relative_drift': [],
        'discriminator_relative_drift': [],
        'generator_alignment': [],
        'discriminator_alignment': [],
        'generator_trace': [],
        'discriminator_trace': [],
        'generator_mean_diagonal': [],
        'discriminator_mean_diagonal': [],
    }
    ntk_reference = {}
    ntk_kernels = {}

    preds = {'pred': [], 'soln': []}
    resids = {'resid': []}
    compare_time_reported = False

    def make_fixed_epoch_set():
        nonlocal grid_samp, residuals, residuals_delta
        xs_local, ys_local = problem.get_grid_sample(grid_samp, residuals, residuals_delta)
        grid_samp_local = torch.cat((xs_local, ys_local), 1)
        return xs_local, ys_local, grid_samp_local

    def evaluate_residual_state(
        G,
        grid_samp_local,
        xs_local,
        ys_local,
        compute_energy=True,
    ):
        """Compute the generator prediction, PDE residuals, and energy."""
        pred_local = G(grid_samp_local)

        if hard:
            residuals_local = problem.get_equation(
                pred_local, xs_local, ys_local
            )
        else:
            residuals_local = problem.get_equation_soft(
                pred_local, xs_local, ys_local
            )

        if compute_energy or not optimize_redundant_metrics:
            residual_vec_local = residuals_local.detach().reshape(-1)
            residual_energy_local = 0.5 * torch.sum(
                residual_vec_local ** 2
            )
        else:
            residual_energy_local = None
        return {
            'pred': pred_local,
            'residuals': residuals_local,
            'residual_energy': residual_energy_local,
        }

    def evaluate_state_from_residuals(
        residual_state,
        compute_ntk_metric=True,
        cached_JTr=None,
        return_JTr=False,
    ):

        residuals_local = residual_state['residuals']

        if noise:
            std = max(diff, 0.0)
            real_local = torch.zeros_like(residuals_local) + torch.normal(
                mean=0.0, std=std, size=residuals_local.shape, device=residuals_local.device
            )
            fake_local = residuals_local + torch.normal(
                mean=0.0, std=std, size=residuals_local.shape, device=residuals_local.device
            )
        else:
            real_local = torch.zeros_like(residuals_local)
            fake_local = residuals_local

        if compute_ntk_metric or not optimize_redundant_metrics:
            gamma_vec_local, _, _ = compute_gamma(
                fake_local, D, wgan=wgan, gan=gan, lsgan=lsgan
            )
            rKr_result = compute_rKr_gamma(
                G,
                residuals_local,
                gamma_vec_local,
                cached_JTr=cached_JTr,
                return_JTr=return_JTr,
            )
            if return_JTr:
                rKr_gamma_local, JTr_local = rKr_result
            else:
                rKr_gamma_local = rKr_result
                JTr_local = None
        else:
            # Match the discriminator forward performed inside compute_gamma
            # so spectral-normalization buffers evolve exactly as before.
            fake_for_gamma = fake_local.detach().clone().requires_grad_(True)
            with torch.no_grad():
                D(fake_for_gamma)
            gamma_vec_local = None
            rKr_gamma_local = None
            JTr_local = None

        return {
            **residual_state,
            'real': real_local,
            'fake': fake_local,
            'gamma_vec': gamma_vec_local,
            'rKr_gamma': rKr_gamma_local,
            'JTr': JTr_local,
        }

    if track_ntk_drift:
        probe_count = min(int(ntk_probe_size), len(grid))
        probe_indices = torch.linspace(
            0,
            len(grid) - 1,
            steps=probe_count,
            device=grid.device,
        ).round().long()
        ntk_probe_coordinates = grid.detach()[probe_indices].clone()
        ntk_discriminator_inputs = torch.linspace(
            ntk_discriminator_min,
            ntk_discriminator_max,
            steps=probe_count,
            device=grid.device,
            dtype=grid.dtype,
        ).reshape(-1, 1)

    def generator_probe_residuals():
        """Build residuals on fresh coordinate leaves for the G NTK."""
        probe_x = (
            ntk_probe_coordinates[:, 0:1].detach().clone()
            .requires_grad_(True)
        )
        probe_y = (
            ntk_probe_coordinates[:, 1:2].detach().clone()
            .requires_grad_(True)
        )
        probe_grid = torch.cat((probe_x, probe_y), dim=1)
        probe_pred = G(probe_grid)
        if hard:
            return problem.get_equation(probe_pred, probe_x, probe_y)
        return problem.get_equation_soft(probe_pred, probe_x, probe_y)

    def measure_ntk_drift(step):
        """Measure G residual NTK and D output NTK on fixed probe inputs."""
        generator_training = G.training
        discriminator_training = D.training
        generator_requires_grad = [p.requires_grad for p in G.parameters()]
        discriminator_requires_grad = [p.requires_grad for p in D.parameters()]

        try:
            G.eval()
            D.eval()
            for parameter in G.parameters():
                parameter.requires_grad_(True)
            for parameter in D.parameters():
                parameter.requires_grad_(True)

            probe_residuals = generator_probe_residuals()
            generator_kernel = empirical_ntk(G, probe_residuals)

            discriminator_outputs = D(ntk_discriminator_inputs)
            discriminator_kernel = empirical_ntk(D, discriminator_outputs)
        finally:
            for parameter, requires_grad in zip(
                G.parameters(), generator_requires_grad
            ):
                parameter.requires_grad_(requires_grad)
            for parameter, requires_grad in zip(
                D.parameters(), discriminator_requires_grad
            ):
                parameter.requires_grad_(requires_grad)
            G.train(generator_training)
            D.train(discriminator_training)

        if not ntk_reference:
            ntk_reference['generator'] = generator_kernel.clone()
            ntk_reference['discriminator'] = discriminator_kernel.clone()
            ntk_kernels['generator_initial'] = generator_kernel.cpu()
            ntk_kernels['discriminator_initial'] = discriminator_kernel.cpu()

        generator_stats = ntk_comparison(
            generator_kernel, ntk_reference['generator']
        )
        discriminator_stats = ntk_comparison(
            discriminator_kernel, ntk_reference['discriminator']
        )

        ntk_history['steps'].append(int(step))
        for metric in (
            'relative_drift', 'alignment', 'trace', 'mean_diagonal'
        ):
            ntk_history[f'generator_{metric}'].append(
                generator_stats[metric]
            )
            ntk_history[f'discriminator_{metric}'].append(
                discriminator_stats[metric]
            )

        ntk_kernels['generator_final'] = generator_kernel.cpu()
        ntk_kernels['discriminator_final'] = discriminator_kernel.cpu()

        if log:
            print(
                f"NTK step {step}: "
                f"G drift {generator_stats['relative_drift']:.4e} | "
                f"G alignment {generator_stats['alignment']:.6f} | "
                f"D drift {discriminator_stats['relative_drift']:.4e} | "
                f"D alignment {discriminator_stats['alignment']:.6f}"
            )

    time_start = time.time()
    if track_ntk_drift:
        measure_ntk_drift(step=0)

    for epoch in range(niters):
        xs_eval, ys_eval, eval_grid_samp = make_fixed_epoch_set()
        for p in D.parameters():
            p.requires_grad = True

        current_residual_state = evaluate_residual_state(
            G,
            eval_grid_samp,
            xs_eval,
            ys_eval,
            compute_energy=False,
        )
        eval_init_D = evaluate_state_from_residuals(
            current_residual_state,
            return_JTr=optimize_redundant_metrics,
        )
        cached_D_JTr = (
            eval_init_D['JTr'] if optimize_redundant_metrics else None
        )

        if ntk and optimize_redundant_metrics:
            best_D_score = eval_init_D['rKr_gamma'].detach().clone()
            best_D_step_device = torch.zeros(
                (), dtype=torch.long, device=grid.device
            )
            best_D_state = clone_module_state(D)
        else:
            best_D_score = eval_init_D['rKr_gamma'].item()
            best_D_step = 0
            if ntk:
                best_D_state = clone_module_state(D)

        actual_D_iters = min(D_iters, search_D_steps)
        d_loss = torch.tensor(0.0, device=grid.device)

        for i in range(actual_D_iters):
            current_residual_state = evaluate_residual_state(
                G,
                eval_grid_samp,
                xs_eval,
                ys_eval,
                compute_energy=False,
            )
            eval_for_D = evaluate_state_from_residuals(
                current_residual_state,
                compute_ntk_metric=False,
            )
            real = eval_for_D['real']
            fake = eval_for_D['fake']

            if wgan:
                norm_penalty = calc_gradient_penalty(D, real, fake.detach(), gp, cuda=False)
            else:
                norm_penalty = torch.zeros(1, device=fake.device)

            if wgan or gan:
                real_loss = criterion(D(real), real_labels)
                fake_loss = criterion(D(fake.detach()), fake_labels)
                #if wgan:
                    #print('wgan')
                #else:
                    #print('gan')
            else:
                real_loss = mse(D(real), real_labels)
                fake_loss = mse(D(fake.detach()), fake_labels)
                #print('lsgan')

            optiD.zero_grad()
            if wgan:
                d_loss = -(real_loss + fake_loss) / 2 + omega * norm_penalty
                #print('wgan')
            if gan:
                d_loss = (real_loss + fake_loss) / 2 + norm_penalty
                #print('gan')
            if lsgan:
                d_loss = (real_loss + fake_loss)/4 + norm_penalty
                #print('lsgan')
            d_loss.backward()
            optiD.step()

            current_residual_state = evaluate_residual_state(
                G,
                eval_grid_samp,
                xs_eval,
                ys_eval,
                compute_energy=False,
            )
            eval_after_D = evaluate_state_from_residuals(
                current_residual_state,
                cached_JTr=cached_D_JTr,
            )
            score_after_D = eval_after_D['rKr_gamma']

            if ntk and optimize_redundant_metrics:
                is_better_D = score_after_D < best_D_score - tol_D
                best_D_score = torch.where(
                    is_better_D, score_after_D, best_D_score
                )
                best_D_step_device = torch.where(
                    is_better_D,
                    best_D_step_device.new_tensor(i + 1),
                    best_D_step_device,
                )
                update_and_restore_best_state(
                    D, best_D_state, is_better_D
                )
            elif ntk:
                score_after_D_value = score_after_D.item()
                if score_after_D_value < best_D_score - tol_D:
                    best_D_score = score_after_D_value
                    best_D_step = i + 1
                    best_D_state = clone_module_state(D)
                else:
                    D.load_state_dict(best_D_state)
            else:
                score_after_D_value = score_after_D.item()
                if score_after_D_value < best_D_score - tol_D:
                    best_D_score = score_after_D_value
                    best_D_step = i + 1
        if ntk and optimize_redundant_metrics:
            best_D_step = best_D_step_device.item()

        for p in D.parameters():
            p.requires_grad = False

        current_residual_state = evaluate_residual_state(
            G, eval_grid_samp, xs_eval, ys_eval
        )
        eval_init_G = evaluate_state_from_residuals(
            current_residual_state,
            compute_ntk_metric=False,
        )

        if optimize_redundant_metrics:
            best_G_energy = eval_init_G['residual_energy'].detach().clone()
            best_G_step_device = torch.zeros(
                (), dtype=torch.long, device=grid.device
            )
            best_G_state = clone_module_state(G)
        else:
            best_G_energy = eval_init_G['residual_energy'].item()
            best_G_step = 0
            best_G_state = clone_module_state(G)

        actual_G_iters = min(G_iters, search_G_steps)
        g_loss = torch.tensor(0.0, device=grid.device)

        for i in range(actual_G_iters):
            current_residual_state = evaluate_residual_state(
                G,
                eval_grid_samp,
                xs_eval,
                ys_eval,
                compute_energy=False,
            )
            eval_for_G = evaluate_state_from_residuals(
                current_residual_state,
                compute_ntk_metric=False,
            )
            fake = eval_for_G['fake']

            optiG.zero_grad()
            if wgan:
                g_loss = -criterion(D(fake), real_labels)
                #print('wgan')

            if gan:
                g_loss = criterion(D(fake), real_labels)
                #print('gan')

            if lsgan:
                g_loss = mse(D(fake), real_labels)/2
                #print('lsgan')
            g_loss.backward()
            optiG.step()

            current_residual_state = evaluate_residual_state(
                G, eval_grid_samp, xs_eval, ys_eval
            )
            eval_after_G = evaluate_state_from_residuals(
                current_residual_state,
                compute_ntk_metric=False,
            )
            energy_after_G = eval_after_G['residual_energy']
            # 对 G：残差能量越小越好
            if optimize_redundant_metrics:
                is_better_G = energy_after_G < best_G_energy - tol_G
                best_G_energy = torch.where(
                    is_better_G, energy_after_G, best_G_energy
                )
                best_G_step_device = torch.where(
                    is_better_G,
                    best_G_step_device.new_tensor(i + 1),
                    best_G_step_device,
                )
                update_best_state(G, best_G_state, is_better_G)
            else:
                energy_after_G_value = energy_after_G.item()
                if energy_after_G_value < best_G_energy - tol_G:
                    best_G_energy = energy_after_G_value
                    best_G_step = i + 1
                    best_G_state = clone_module_state(G)
        if optimize_redundant_metrics:
            best_G_step = best_G_step_device.item()
        if energy:
            if best_G_step > 1:
                G.load_state_dict(best_G_state)

        current_residual_state = evaluate_residual_state(
            G, eval_grid_samp, xs_eval, ys_eval
        )
        eval_final = evaluate_state_from_residuals(current_residual_state)

        residuals = eval_final['residuals'].detach()
        final_rKr_gamma = eval_final['rKr_gamma']
        final_residual_energy = eval_final['residual_energy']

        grid_samp = eval_grid_samp.detach()
        residuals_delta = torch.zeros_like(grid_samp)

        g_loss_value = g_loss.item()
        d_loss_value = d_loss.item()
        final_rKr_gamma_value = final_rKr_gamma.item()
        final_residual_energy_value = final_residual_energy.item()

        if noise:
            diff = g_loss_value - d_loss_value

        losses['G'].append(g_loss_value)
        losses['D'].append(d_loss_value)
        losses['rKr_gamma'].append(final_rKr_gamma_value)
        losses['residual_energy'].append(final_residual_energy_value)
        losses['actual_G_iters'].append(actual_G_iters)
        losses['actual_D_iters'].append(actual_D_iters)
        losses['best_G_step'].append(best_G_step)
        losses['best_D_step'].append(best_D_step)

        if lr_schedule:
            lr_scheduler_G.step()
            lr_scheduler_D.step()

        if train_mse:
            with torch.no_grad():
                pred_train = G(eval_grid_samp)
                pred_train_adj = problem.adjust(pred_train, xs_eval, ys_eval)['pred']
                sol_train = problem.get_solution(xs_eval, ys_eval)
                sol_train = sol_train.to(pred_train_adj.device)
                train_mse_val = mse(pred_train_adj, sol_train).item()
            mses['train'].append(train_mse_val)

        with torch.no_grad():
            val_pred = G(grid)
            val_pred_adj = problem.adjust(val_pred, x, y)['pred']
            soln = soln.to(val_pred_adj.device)
            val_mse = mse(val_pred_adj, soln).item()
        mses['val'].append(val_mse)

        completed_steps = epoch + 1
        if (
            track_ntk_drift
            and (
                completed_steps % ntk_log_every == 0
                or completed_steps == niters
            )
        ):
            measure_ntk_drift(step=completed_steps)

        if save_for_animation:
            plot_pred = G(plot_grid)
            plot_pred_adj = problem.adjust(plot_pred, plot_x, plot_y)['pred']
            preds['pred'].append(plot_pred_adj.detach().cpu().numpy())
            preds['soln'].append(plot_soln.detach().cpu().numpy())
            resids['resid'].append(np.abs(residuals.cpu().numpy()))

        if log and (epoch + 1) % log_every == 0:
            if train_mse:
                print(
                    f"Step {epoch}: "
                    f"G Loss: {g_loss_value:.4e} | "
                    f"D Loss: {d_loss_value:.4e} | "
                    f"r^T K gamma: {final_rKr_gamma_value:.4e} | "
                    f"Residual E: {final_residual_energy_value:.4e} | "
                    f"best D step: {best_D_step} | "
                    f"best G step: {best_G_step} | "
                    f"Train MSE {train_mse_val:.4e} | "
                    f"Val MSE {val_mse:.4e}"
                )
            else:
                print(
                    f"Step {epoch}: "
                    f"G Loss: {g_loss_value:.4e} | "
                    f"D Loss: {d_loss_value:.4e} | "
                    f"r^T K gamma: {final_rKr_gamma_value:.4e} | "
                    f"Residual E: {final_residual_energy_value:.4e} | "
                    f"best D step: {best_D_step} | "
                    f"best G step: {best_G_step} | "
                    f"Val MSE {val_mse:.4e}"
                )
            if (
                train_mse
                and not compare_time_reported
                and train_mse_val < 3.16e-06
            ):
                time_end_compare = time.time()
                print('compare time is :', time_end_compare - time_start)
                compare_time_reported = True

    time_end = time.time()
    print('time is :', time_end - time_start)
    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    if plot:
        pred_dict, diff_dict = problem.get_plot_dicts(G(plot_grid), plot_x, plot_y, plot_soln)
        plot_results(
            mses, losses, plot_grid.detach(), pred_dict, diff_dict=diff_dict,
            save=save, dirname=dirname, logloss=False, alpha=0.7,
            dims=dims, plot_1d_curves=plot_1d_curves
        )
        plot_3D(plot_grid.detach(), pred_dict, view=view, dims=dims, save=save, dirname=dirname)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    if save:
        os.makedirs(dirname, exist_ok=True)
        write_config(config, os.path.join(dirname, 'config.yaml'))
        # save G
        checkpoint_path = os.path.join(dirname, "generator_final.pt")
        torch.save(
            {
                "generator_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in G.state_dict().items()
                },
                "config": config,
                "torch_version": str(torch.__version__),
            },
            checkpoint_path,
        )
        print(f"Final generator saved to: {checkpoint_path}")
        
        np.save(os.path.join(dirname, "loss_G.npy"), np.array(losses['G']))
        np.save(os.path.join(dirname, "loss_D.npy"), np.array(losses['D']))
        np.save(os.path.join(dirname, "loss_rKr_gamma.npy"), np.array(losses['rKr_gamma']))
        np.save(os.path.join(dirname, "loss_residual_energy.npy"), np.array(losses['residual_energy']))
        np.save(os.path.join(dirname, "best_G_step.npy"), np.array(losses['best_G_step']))
        np.save(os.path.join(dirname, "best_D_step.npy"), np.array(losses['best_D_step']))
        np.save(os.path.join(dirname, "actual_G_iters.npy"), np.array(losses['actual_G_iters']))
        np.save(os.path.join(dirname, "actual_D_iters.npy"), np.array(losses['actual_D_iters']))
        np.save(os.path.join(dirname, "mse_val.npy"), np.array(mses['val']))
        
        if train_mse:
            np.save(os.path.join(dirname, "mse_train.npy"), np.array(mses['train']))
        if track_ntk_drift:
            for key, values in ntk_history.items():
                np.save(
                    os.path.join(dirname, f"ntk_{key}.npy"),
                    np.asarray(values),
                )
            if save_ntk_kernels:
                for key, kernel in ntk_kernels.items():
                    np.save(
                        os.path.join(dirname, f"ntk_{key}.npy"),
                        kernel.numpy(),
                    )
            save_ntk_drift_plot(
                ntk_history,
                os.path.join(dirname, "ntk_drift.png"),
            )

    if save_for_animation:
        os.makedirs(dirname, exist_ok=True)
        anim_dir = os.path.join(dirname, "animation")
        os.makedirs(anim_dir, exist_ok=True)
        np.save(os.path.join(anim_dir, "grid.npy"), plot_grid.detach().cpu().numpy())
        for k, v in preds.items():
            if len(v) > 0:
                np.save(os.path.join(anim_dir, f"{k}_pred.npy"), np.array(v, dtype=object))
        for k, v in resids.items():
            if len(v) > 0:
                np.save(os.path.join(anim_dir, f"{k}_resid.npy"), np.array(v, dtype=object))

    return {
        'mses': mses,
        'model': G,
        'losses': losses,
        'ntk_history': ntk_history,
    }
