import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from denn.utils import plot_results, calc_gradient_penalty, handle_overwrite, plot_3D
from denn.config.config import write_config

try:
    from ray import tune
except:
    print("Ray not loaded.")
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


def compute_gamma(fake, D, wgan=True,gan=True,lsgan=True, eps=1e-14):
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


def compute_rKr_gamma(G, residuals, gamma):
    """
    Compute s = r^T K_rr gamma without explicitly building K_rr.

    residuals: current residual vector r, must still connect to G graph
    gamma: detached weight vector, same shape as residuals

    returns:
        s: scalar tensor
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


def train_GAN_2D(G, D, problem, method='unsupervised', niters=100,
    g_lr=1e-3, g_betas=(0.0, 0.9), d_lr=1e-3, d_betas=(0.0, 0.9),
    lr_schedule=True, gamma=0.999, momentum=0.95, noise=False, 
    step_size=15, obs_every=1, d1=1., d2=1., G_iters=1, D_iters=1, 
    wgan=True,lsgan=True,gan=True, gp=0.1, conditional=True, train_mse=True, log=True,
    plot=True, plot_1d_curves=False, save=False, dirname='train_GAN', 
    config=None, save_for_animation=True, view=[35, -55], grad_NTK =False,
    hard = True,plot_del = False, **kwargs):

    print(torch.cuda.is_available())

    pkey = dirname.split('_')[0].lower()
    dirname = os.path.join(this_dir, '../experiments/runs', dirname)
    if plot and save:
        handle_overwrite(dirname)

    # validation: fixed grid/solution
    x, y = problem.get_grid()
    grid = torch.cat((x, y), 1)
    soln = problem.get_solution(x, y)

    # initialize residuals and grid for active sampling
    grid_samp = grid
    residuals = torch.zeros_like(x)
    residuals_delta = torch.zeros_like(grid_samp)

    # grid for plotting and animation
    if plot or save_for_animation:
        dims = problem.get_plot_dims()
        plot_x, plot_y = problem.get_plot_grid(grid_samp, residuals, residuals_delta)
        plot_grid = torch.cat((plot_x, plot_y), 1)
        plot_soln = problem.get_plot_solution(plot_x, plot_y)

    # observer mask and masked grid/solution (t_obs/y_obs)
    observers = torch.arange(0, len(grid), obs_every, device=grid.device)

    # labels
    real_label = 1
    fake_label = -1 if wgan else 0
    real_labels = torch.full((len(grid),), real_label, dtype=torch.float, device=grid.device).reshape(-1,1) #len(grid)+195 for no reparam
    fake_labels = torch.full((len(grid),), fake_label, dtype=torch.float, device=grid.device).reshape(-1,1) #len(grid)+195 for no reparam
    # masked label vectors
    real_labels_obs = real_labels[observers, :]
    fake_labels_obs = fake_labels[observers, :]

    # initialize difference parameter for noise
    if noise:
        diff = 0

    # optimization
    optiG = torch.optim.Adam(G.parameters(), lr=g_lr, betas=g_betas)
    optiD = torch.optim.Adam(D.parameters(), lr=d_lr, betas=d_betas)
    
    if lr_schedule:
        lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer=optiG, step_size=step_size, gamma=gamma)
        lr_scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer=optiD, step_size=step_size, gamma=gamma)

    # losses
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    wass = lambda y_true, y_pred: torch.mean(y_true * y_pred)
    criterion = wass if wgan else bce

    # history
    losses = {'G': [], 'D': []} #, 'LHS': []}
    mses = {'train': [], 'val': []} if train_mse else {'val': []}
    preds = {'pred': [], 'soln': []}
    #resids = {'grid': [], 'resid': []}
    resids = { 'resid': []}
    fake_0 = {'fake_0': []}
    fake_1 = {'fake_1': []}
    metrics = {
    'rKr_gamma': [],          # scalar: r^T K_rr gamma
    'residual_energy': [],    # scalar: 0.5 * ||r||^2
    'gamma_norm': [],         # scalar: ||gamma||
    'pred_delta_E': [],       # scalar: -eta_G * r^T K_rr gamma
    'gamma': [],              # vector: gamma at each step
    'residual_vec': [],       # vector: residual vector at each step
}
    K = 1000
    D1 = {'D': []}
    D2 = {'D_x': []}

    for epoch in range(niters):
        # Train Generator

        if epoch < niters:
            ntk = torch.linspace(0.0,1.0,K,device=next(D.parameters()).device).unsqueeze(1).requires_grad_(True)
            D_x = D(ntk)
            D1['D'].append(D_x.detach())
            grad_x = torch.autograd.grad(
                outputs=D_x.sum(),
                inputs=ntk,
                create_graph=False
            )[0]
            D2['D_x'].append(grad_x.detach())

        for p in D.parameters():
            p.requires_grad = False # turn off computation for D

        metric_rKr_gamma = None

        for i in range(G_iters):
            xs, ys = problem.get_grid_sample(grid_samp, residuals, residuals_delta)
            grid_samp = torch.cat((xs, ys), 1)
            pred = G(grid_samp)
            if hard == True:
                residuals = problem.get_equation(pred, xs, ys)
            else:
                residuals = problem.get_equation_soft(pred, xs, ys)

            if noise:
                diff = diff if diff >= 0 else 0
                real = torch.zeros_like(residuals) + torch.normal(mean=0,std=diff,size=residuals.shape,device=residuals.device)
                fake = residuals + torch.normal(0, diff, size=residuals.shape,device=residuals.device)
            else:
                real = torch.zeros_like(residuals)
                fake = residuals
            if i == 0:
                gamma_vec, D_fake_val, dD_dfake_val = compute_gamma(fake, D, wgan=wgan,gan=gan,lsgan=lsgan)
                metric_rKr_gamma = compute_rKr_gamma(G, residuals, gamma_vec)
                residual_vec = residuals.detach().reshape(-1)
                gamma_flat = gamma_vec.detach().reshape(-1)

                residual_energy = 0.5 * torch.sum(residual_vec ** 2)
                gamma_norm = torch.norm(gamma_flat)
                pred_delta_E = -g_lr * metric_rKr_gamma  # 或者用当前 generator lr

                metrics['rKr_gamma'].append(metric_rKr_gamma.item())
                metrics['residual_energy'].append(residual_energy.item())
                metrics['gamma_norm'].append(gamma_norm.item())
                metrics['pred_delta_E'].append(pred_delta_E.item())

                metrics['gamma'].append(gamma_flat.cpu().numpy())
                metrics['residual_vec'].append(residual_vec.cpu().numpy())

            optiG.zero_grad()
            if wgan:
                g_loss = -criterion(D(fake), real_labels)
            else:
                g_loss = criterion(D(fake), real_labels)
            g_loss.backward()
            optiG.step()

            pred_1 = G(grid_samp)
            fake_zero = problem.get_equation(pred_1, xs, ys)
            fake_grad_1 = fake_zero[0]
            fake_00 = D(fake_grad_1)
            fake_0['fake_0'].append(fake_00.item())
            if grad_NTK == True:
                fake_grad_1.requires_grad_(True)  # ensure fake has grad
                D_fake_zero = D(fake_grad_1)
                # Compute gradient of D(fake) w.r.t. fake
                D_fake_grad = torch.autograd.grad(
                    outputs=D_fake_zero,
                    inputs=fake_grad_1,
                    grad_outputs=torch.ones_like(D_fake_zero),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                fake_1['fake_1'].append(D_fake_grad.item())


        # Train Discriminator
        for p in D.parameters():
            p.requires_grad = True # turn on computation for D

        for i in range(D_iters):
            if wgan:
                norm_penalty = calc_gradient_penalty(D, real, fake, gp, cuda=False)
            else:
                norm_penalty = torch.zeros(1, device=real.device)

            real_loss = criterion(D(real), real_labels)
            fake_loss = criterion(D(fake.detach()), fake_labels)

            optiD.zero_grad()
            if wgan:
                d_loss = -(real_loss + fake_loss) / 2 - norm_penalty
            else:
                d_loss = (real_loss + fake_loss) / 2 + norm_penalty
            d_loss.backward()
            optiD.step()

        if noise:
            diff = g_loss.item() - d_loss.item()

        losses['D'].append(d_loss.item())
        losses['G'].append(g_loss.item())

        if lr_schedule:
          lr_scheduler_G.step()
          lr_scheduler_D.step()

        if train_mse:
            pred = G(grid_samp)
            pred_adj = problem.adjust(pred, xs, ys)['pred']
            sol_samp = problem.get_solution(xs, ys)
            train_mse = mse(pred_adj, sol_samp).item()
            mses['train'].append(train_mse)

        # val MSE: fixed grid vs true soln
        val_pred = G(grid)
        val_pred_adj = problem.adjust(val_pred, x, y)['pred']
        val_mse = mse(val_pred_adj, soln).item()
        mses['val'].append(val_mse)

        # save preds and residuals for animation
        if save_for_animation:
            plot_pred = G(plot_grid)
            plot_pred_adj = problem.adjust(plot_pred, plot_x, plot_y)['pred']
            preds['pred'].append(plot_pred_adj.detach())
            preds['soln'].append(plot_soln.detach())
            resids['resid'].append(np.abs(residuals.detach()))

        try:
            if (epoch+1) % 10 == 0:
                tune.report(mean_squared_error=np.mean(mses['val'][-10:]))

        except Exception as e:
            pass
        
        if 'rKr_gamma' not in losses:
            losses['rKr_gamma'] = []
        losses['rKr_gamma'].append(metric_rKr_gamma.item())

        if log:
            if train_mse:
                print(f'Step {epoch}: G Loss: {g_loss.item():.4e} | D Loss: {d_loss.item():.4e} | r^T K gamma: {metric_rKr_gamma.item():.4e} | Val MSE {val_mse:.4e}')            
            else:
                print(f'Step {epoch}: G Loss: {g_loss.item():.4e} | D Loss: {d_loss.item():.4e} | Val MSE {val_mse:.4e}')
    
    if plot:
        pred_dict, diff_dict = problem.get_plot_dicts(G(plot_grid), plot_x, plot_y, plot_soln)
        plot_results(mses, losses, plot_grid.detach(), pred_dict, diff_dict=diff_dict,
            save=save, dirname=dirname, logloss=False, alpha=0.7, dims=dims,
            plot_1d_curves=plot_1d_curves)
        plot_3D(plot_grid.detach(), pred_dict, view=view, dims=dims, save=save, dirname=dirname)

    if save:
        write_config(config, os.path.join(dirname, 'config.yaml'))
        np.save(os.path.join(dirname, "pred_dict"), pred_dict)
        np.save(os.path.join(dirname, "D"), D1)
        np.save(os.path.join(dirname, "D_x"), D2)

        # save scalar metrics
        np.save(os.path.join(dirname, "metrics_rKr_gamma.npy"), np.array(metrics['rKr_gamma']))
        np.save(os.path.join(dirname, "metrics_residual_energy.npy"), np.array(metrics['residual_energy']))
        np.save(os.path.join(dirname, "metrics_gamma_norm.npy"), np.array(metrics['gamma_norm']))
        np.save(os.path.join(dirname, "metrics_pred_delta_E.npy"), np.array(metrics['pred_delta_E']))

        # save vector metrics
        np.save(os.path.join(dirname, "metrics_gamma.npy"), np.array(metrics['gamma'], dtype=object))
        np.save(os.path.join(dirname, "metrics_residual_vec.npy"), np.array(metrics['residual_vec'], dtype=object))

    if save_for_animation:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        anim_dir = os.path.join(dirname, "animation")
        print(f'Saving animation traces to {anim_dir}')
        if not os.path.exists(anim_dir):
            os.mkdir(anim_dir)
        np.save(os.path.join(anim_dir, "grid"), plot_grid.detach())
        for k, v in preds.items():
            v = np.hstack(v)
            np.save(os.path.join(anim_dir, f"{k}_pred"), v)
        for k, v in resids.items():
            v = np.hstack(v)
            np.save(os.path.join(anim_dir, f"{k}_resid"), v)

    if plot_del:
        def plot_dict_curves(D_dict, title="Curves", ylabel="y", figsize=(8, 5), dpi=150, show_legend=True,
                             save_path=None):
            
            plt.figure(figsize=figsize, dpi=dpi)

            colors = plt.cm.tab20(np.linspace(0, 1, 20))

            for key_idx, (key, val_list) in enumerate(D_dict.items()):
                for idx, val in enumerate(val_list):
                    if isinstance(val, torch.Tensor):
                        y = val.detach().cpu().numpy().flatten()
                    else:
                        y = np.array(val).flatten()
                    y = y
                    x = np.linspace(0.0, 1, K)

                    color = colors[(idx + key_idx * len(val_list)) % len(colors)]

                    plt.plot(x, y, label=f"{key}_{idx}", color=color, linewidth=1.5)

            plt.xlabel("x", fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.title(title, fontsize=14)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.5)

            if show_legend:
                plt.legend(fontsize=8, ncol=2)
            plt.tight_layout()

            if save_path is not None:
                plt.savefig(save_path, dpi=dpi)

            plt.show()

        plot_dict_curves(D1, title="Output of discriminator", ylabel="D(x)")
        plot_dict_curves(D2, title="Derivative of the discriminator", ylabel="dD(x)/dx")
        D1_mod = [1 / (1 - x) for x in D1['D']]
        result = [a * b for a, b in zip(D1_mod, D2['D_x'])]
        result_dict = {'D_x / (1 - D)': result}
        plot_dict_curves(result_dict, title="NTK of the discriminator", ylabel="1/(1-D)*dD(x)/dx")

    if grad_NTK == True:
        linestyles = ['solid', 'dashed', 'dashdot', 'dotted'] * 3  
        linewidth = 2
        alpha = 0.8
        alphas = [alpha] * 10 
        colors = ['red', 'blue', 'skyblue', 'limegreen',
                  'aquamarine', 'violet', 'black', 'brown', 'pink', 'gold']
        for i, (k, v) in enumerate(fake_0.items()):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(np.arange(len(v)), v, label=k,
                           alpha=alphas[i], linewidth=linewidth, color=colors[i],
                           linestyle=linestyles[i])
            ax.legend()  
            ax.set_xlabel('Step') 
            ax.set_ylabel('Value')  
            ax.set_title('Curve') 
            plt.tight_layout() 
            plt.show()  # 显示图像

        for i, (k, v) in enumerate(fake_1.items()):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(np.arange(len(v)), v, label=k,
                           alpha=alphas[i], linewidth=linewidth, color=colors[i],
                           linestyle=linestyles[i])
            ax.legend()  
            ax.set_xlabel('Step') 
            ax.set_ylabel('Value')  
            ax.set_title('Curve') 
            plt.tight_layout() 
            plt.show()  # 显示图像

        diff = [torch.tensor(f1) / torch.norm(torch.tensor(1.0) - torch.tensor(f0)).item() for f0, f1 in
                zip(fake_0['fake_0'], fake_1['fake_1'])]
        diff_array = torch.tensor(diff).cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(diff_array, label='My Data', color='blue', linewidth=2)
        ax.legend([f'Curve {i}'])  
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title(f'Curve {i}')
        plt.tight_layout()
        plt.show()

    return {'mses': mses, 'model': G, 'losses': losses}
