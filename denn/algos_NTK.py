import os
import copy
import numpy as np
import torch
import torch.nn as nn
import time
from denn.utils import (
    plot_results,
    calc_gradient_penalty,
    handle_overwrite,
    plot_3D,
    save_training_arrays,
)
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
        gamma = (1.0 / N) * dD_dfake

    if gan:
        gamma = (1.0 / N) * dD_dfake / (1.0 - D_fake.detach() + eps)

    if lsgan:
        gamma =- (1.0 / N) * dD_dfake * (D_fake.detach()-1)
        #gamma = (1.0 / N) * dD_dfake / (1.0 - D_fake.detach() + eps)

    return gamma.detach(), D_fake.detach(), dD_dfake.detach()


def compute_rKr_gamma(G, residuals, gamma):
    """
    Compute s = r^T K_rr gamma without explicitly building K_rr.

    residuals: current residual vector r, must still connect to G graph
    gamma: detached weight vector, same shape as residuals
    """
    params = [p for p in G.parameters() if p.requires_grad]

    r_vec = residuals.reshape(-1)
    gamma_vec = gamma.reshape(-1)

    # J^T r
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
    return s.detach()


def train_GAN_2D(
    G, D, problem, method='unsupervised', niters=100,
    g_lr=1e-3, g_betas=(0.0, 0.9), d_lr=1e-3, d_betas=(0.0, 0.9),
    lr_schedule=True, gamma=0.999, step_size=15, noise=False, G_iters=100, D_iters=100,omega = 1,
    search_G_steps=100, search_D_steps=100, tol_G=1e-12, tol_D=1e-10,
    wgan=True, gan=True, lsgan=True,gp=0.1, train_mse=True, log=True, plot=False, plot_1d_curves=False,
    save=False, dirname='train_GAN', config=None, save_for_animation=False, view=[35, -55],
    hard=True,ntk=True, energy=True,
    **kwargs
):

    assert method in ['supervised', 'semisupervised', 'unsupervised'], \
        f'Method {method} not understood!'

    print(torch.cuda.is_available())

    dirname = os.path.join(this_dir, '../experiments/runs', dirname)
    if plot and save:
        handle_overwrite(dirname)

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
    real_labels = torch.full((len(grid),), real_label, dtype=torch.float).reshape(-1, 1)
    fake_labels = torch.full((len(grid),), fake_label, dtype=torch.float).reshape(-1, 1)

    diff = 0.0

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

    losses = {
        'G': [],
        'D': [],
        'rKr_gamma': [],
        'residual_energy': [],
        'best_G_step': [],
        'best_D_step': [],
    }
    mses = {'train': [], 'val': []} if train_mse else {'val': []}

    preds = {'pred': [], 'soln': []}
    resids = {'resid': []}

    def make_fixed_epoch_set():

        nonlocal grid_samp, residuals, residuals_delta
        xs_local, ys_local = problem.get_grid_sample(grid_samp, residuals, residuals_delta)
        grid_samp_local = torch.cat((xs_local, ys_local), 1)
        return xs_local, ys_local, grid_samp_local

    def evaluate_state_on_fixed_set(G,grid_samp_local, xs_local, ys_local):

        pred_local = G(grid_samp_local)

        if hard:
            residuals_local = problem.get_equation(pred_local, xs_local, ys_local)
        else:
            residuals_local = problem.get_equation_soft(pred_local, xs_local, ys_local)

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
        #print(lsgan,wgan,gan)
        gamma_vec_local, _, _ = compute_gamma(fake_local, D, wgan=wgan,gan=gan,lsgan=lsgan)
        rKr_gamma_local = compute_rKr_gamma(G, residuals_local, gamma_vec_local)

        residual_vec_local = residuals_local.detach().reshape(-1)
        residual_energy_local = 0.5 * torch.sum(residual_vec_local ** 2)

        return {
            'pred': pred_local,
            'residuals': residuals_local,
            'real': real_local,
            'fake': fake_local,
            'gamma_vec': gamma_vec_local,
            'rKr_gamma': rKr_gamma_local,
            'residual_energy': residual_energy_local,
        }
    time_start = time.time()
    for epoch in range(niters):
        xs_eval, ys_eval, eval_grid_samp = make_fixed_epoch_set()
        for p in D.parameters():
            p.requires_grad = True

        eval_init_D = evaluate_state_on_fixed_set(G,eval_grid_samp, xs_eval, ys_eval)

        init_D_score = eval_init_D['rKr_gamma'].item()
        best_D_score = init_D_score
        best_D_step = 0
        best_D_state = copy.deepcopy(D.state_dict())

        actual_D_iters = min(D_iters, search_D_steps)
        d_loss = torch.tensor(0.0, device=grid.device)

        for i in range(actual_D_iters):
            eval_for_D = evaluate_state_on_fixed_set(G,eval_grid_samp, xs_eval, ys_eval)
            real = eval_for_D['real']
            fake = eval_for_D['fake']

            if wgan:
                norm_penalty = calc_gradient_penalty(D, real, fake.detach(), gp, cuda=False)
            else:
                norm_penalty = torch.zeros(1, device=fake.device)

            if wgan or gan:
                real_loss = criterion(D(real), real_labels)
                fake_loss = criterion(D(fake.detach()), fake_labels)

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

            eval_after_D = evaluate_state_on_fixed_set(G,eval_grid_samp, xs_eval, ys_eval)
            score_after_D = eval_after_D['rKr_gamma'].item()

            if ntk:
                if score_after_D < best_D_score - tol_D:
                    best_D_score = score_after_D
                    best_D_step = i + 1
                    best_D_state = copy.deepcopy(D.state_dict())

                D.load_state_dict(best_D_state)
            if ntk == False:
                if score_after_D < best_D_score - tol_D:
                    best_D_score = score_after_D
                    best_D_step = i + 1

        for p in D.parameters():
            p.requires_grad = False

        eval_init_G = evaluate_state_on_fixed_set(G,eval_grid_samp, xs_eval, ys_eval)

        init_G_energy = eval_init_G['residual_energy'].item()
        best_G_energy = init_G_energy
        best_G_step = 0
        best_G_state = copy.deepcopy(G.state_dict())

        actual_G_iters = min(G_iters, search_G_steps)
        g_loss = torch.tensor(0.0, device=grid.device)

        for i in range(actual_G_iters):
            eval_for_G = evaluate_state_on_fixed_set(G,eval_grid_samp, xs_eval, ys_eval)
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

            eval_after_G = evaluate_state_on_fixed_set(G,eval_grid_samp, xs_eval, ys_eval)
            energy_after_G = eval_after_G['residual_energy'].item()

            if energy_after_G < best_G_energy - tol_G:
                best_G_energy = energy_after_G
                best_G_step = i + 1
                best_G_state = copy.deepcopy(G.state_dict())
        if energy:
            if best_G_step > 1:
                G.load_state_dict(best_G_state)

        eval_final = evaluate_state_on_fixed_set(G,eval_grid_samp, xs_eval, ys_eval)

        residuals = eval_final['residuals'].detach()
        final_rKr_gamma = eval_final['rKr_gamma']
        final_residual_energy = eval_final['residual_energy']

        grid_samp = eval_grid_samp.detach()
        residuals_delta = torch.zeros_like(grid_samp)

        if noise:
            diff = g_loss.item() - d_loss.item()

        losses['G'].append(g_loss.item())
        losses['D'].append(d_loss.item())
        losses['rKr_gamma'].append(final_rKr_gamma.item())
        losses['residual_energy'].append(final_residual_energy.item())
        losses['best_G_step'].append(best_G_step)
        losses['best_D_step'].append(best_D_step)

        if lr_schedule:
            lr_scheduler_G.step()
            lr_scheduler_D.step()

        if train_mse:
            pred_train = G(eval_grid_samp)
            pred_train_adj = problem.adjust(pred_train, xs_eval, ys_eval)['pred']
            sol_train = problem.get_solution(xs_eval, ys_eval)
            train_mse_val = mse(pred_train_adj, sol_train).item()
            mses['train'].append(train_mse_val)

        val_pred = G(grid)
        val_pred_adj = problem.adjust(val_pred, x, y)['pred']
        val_mse = mse(val_pred_adj, soln).item()
        mses['val'].append(val_mse)

        if save_for_animation:
            plot_pred = G(plot_grid)
            plot_pred_adj = problem.adjust(plot_pred, plot_x, plot_y)['pred']
            preds['pred'].append(plot_pred_adj.detach().cpu().numpy())
            preds['soln'].append(plot_soln.detach().cpu().numpy())
            resids['resid'].append(np.abs(residuals.cpu().numpy()))


        if log:
            if train_mse:
                print(
                    f"Step {epoch}: "
                    f"G Loss: {g_loss.item():.4e} | "
                    f"D Loss: {d_loss.item():.4e} | "
                    f"r^T K gamma: {final_rKr_gamma.item():.4e} | "
                    f"Residual E: {final_residual_energy.item():.4e} | "
                    f"best D step: {best_D_step} | "
                    f"best G step: {best_G_step} | "
                    f"Train MSE {train_mse_val:.4e} | "
                    f"Val MSE {val_mse:.4e}"
                )
            else:
                print(
                    f"Step {epoch}: "
                    f"G Loss: {g_loss.item():.4e} | "
                    f"D Loss: {d_loss.item():.4e} | "
                    f"r^T K gamma: {final_rKr_gamma.item():.4e} | "
                    f"Residual E: {final_residual_energy.item():.4e} | "
                    f"best D step: {best_D_step} | "
                    f"best G step: {best_G_step} | "
                    f"Val MSE {val_mse:.4e}"
                )
            '''if train_mse_val < 3.16e-06:
                time_end_compare = time.time()
                print('compare time is :', time_end_compare - time_start)'''

    time_end = time.time()
    print('time is :', time_end - time_start)

    if plot:
        pred_dict, diff_dict = problem.get_plot_dicts(G(plot_grid), plot_x, plot_y, plot_soln)
        plot_results(
            mses, losses, plot_grid.detach(), pred_dict, diff_dict=diff_dict,
            save=save, dirname=dirname, logloss=False, alpha=0.7,
            dims=dims, plot_1d_curves=plot_1d_curves, save_arrays=False
        )
        plot_3D(plot_grid.detach(), pred_dict, view=view, dims=dims, save=save, dirname=dirname)

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
        save_training_arrays(mses, losses, dirname)

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
        'losses': losses
    }
