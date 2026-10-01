import numpy as np
import torch
import torch.nn as nn
import os
import time
import matplotlib.pyplot as plt
from denn.utils import plot_results, plot_multihead, calc_gradient_penalty, handle_overwrite, plot_3D
from denn.config.config import write_config

try:
    from ray import tune
except:
    print("Ray not loaded.")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

this_dir = os.path.dirname(os.path.abspath(__file__))

def train_GAN_2D_save(G, D, problem, method='unsupervised', niters=100,
    g_lr=1e-3, g_betas=(0.0, 0.9), d_lr=1e-3, d_betas=(0.0, 0.9),
    lr_schedule=True, gamma=0.999, momentum=0.95, noise=False, 
    step_size=15, obs_every=1, d1=1., d2=1., G_iters=1, D_iters=1, 
    wgan=True, gp=0.1, conditional=True, train_mse=True, log=True, 
    plot=True, plot_1d_curves=False, save=False, dirname='train_GAN', 
    config=None, save_for_animation=True, view=[35, -55], grad_NTK =False,
    hard = True,plot_del = False, **kwargs):
    """
    Train/test GAN method: supervised/semisupervised/unsupervised
    """
    assert method in ['supervised', 'semisupervised', 'unsupervised'], f'Method {method} not understood!'

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
    # grid_obs = grid[observers, :]
    # soln_obs = soln[observers, :]

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
    #optiG = torch.optim.SGD(G.parameters(), lr=g_lr, momentum=momentum, nesterov=True)
    #optiD = torch.optim.SGD(D.parameters(), lr=g_lr, momentum=momentum, nesterov=True)
    if lr_schedule:
        #lr_scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer=optiG, gamma=gamma)
        #lr_scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer=optiD, gamma=gamma)
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
    K = 100000
    D1 = {'D': []}
    D2 = {'D_x': []}
    #record_steps = [1, 10, 50, 100, 1000, 2000, 2999, 4000, 6000, 8000, 10000]

    for epoch in range(niters):
        # Train Generator

        if epoch < niters:
            # 输入网格
            ntk = torch.linspace(0.0, 1.0, K, device=next(D.parameters()).device).unsqueeze(1)
            ntk.requires_grad_(True)

            D_x = D(ntk)
            D1['D'].append(D_x.detach())
            # 梯度 dD/dx
            grad_x = torch.autograd.grad(
                outputs=D_x.sum(),
                inputs=ntk,
                create_graph=False
            )[0]
            D2['D_x'].append(grad_x.detach())

        for p in D.parameters():
            p.requires_grad = False # turn off computation for D

        for i in range(G_iters):
            xs, ys = problem.get_grid_sample(grid_samp, residuals, residuals_delta)
            grid_samp = torch.cat((xs, ys), 1)
            # grid_samp_deltax = torch.cat((xs+1e-5, ys), 1)
            # grid_samp_deltay = torch.cat((xs, ys+1e-5), 1)
            pred = G(grid_samp)
            # pred_deltax = G(grid_samp_deltax)
            # pred_deltay = G(grid_samp_deltay)
            if hard == True:
                residuals = problem.get_equation(pred, xs, ys)
            else:
                residuals = problem.get_equation_soft(pred, xs, ys)

            # residuals_deltax = problem.get_equation(pred_deltax, xs+1e-5, ys)
            # residuals_deltay = problem.get_equation(pred_deltay, xs, ys+1e-5)
            # residuals_delta = torch.cat((residuals_deltax, residuals_deltay), 1)
            # if epoch == 1000:
            #     problem.sampling = 'gradient'
            #     fig, ax = plt.subplots(figsize=(12,12))
            #     xx, yy = xs.detach().numpy().reshape((65,65)), ys.detach().numpy().reshape((65,65))
            #     resid = residuals.detach().numpy().reshape((65,65))
            #     ax.contourf(xx, yy, resid, cmap='Reds')
            #     plt.show()
            #     assert False
            # idea: add noise to relax from dirac delta at 0 to distb'n
            # + torch.normal(0, .1/(i+1), size=residuals.shape)
            if noise:
                diff = diff if diff >= 0 else 0
                #print(residuals.device, torch.normal(0, diff, size=residuals.shape).device)
                real = torch.zeros_like(residuals) + torch.normal(mean=0,std=diff,size=residuals.shape,device=residuals.device)
                fake = residuals + torch.normal(0, diff, size=residuals.shape,device=residuals.device)


            else:
                real = torch.zeros_like(residuals)
                fake = residuals
            # fake = residuals

            optiG.zero_grad()
            g_loss = criterion(D(fake), real_labels)
            # g_loss = criterion(D(fake), torch.ones_like(fake))
            g_loss.backward()
            optiG.step()

            #2025年4月20日 检验鉴别器对生成数据的零阶信息和一阶信息
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

            # print(real.shape, fake.shape)
            real_loss = criterion(D(real), real_labels)
            # real_loss = criterion(D(real), torch.ones_like(real))
            fake_loss = criterion(D(fake.detach()), fake_labels)
            # fake_loss = criterion(D(fake), torch.zeros_like(fake))

            optiD.zero_grad()
            d_loss = (real_loss + fake_loss)/2 + norm_penalty
            d_loss.backward()
            optiD.step()


        # update the difference between losses
        if noise:
            diff = g_loss.item() - d_loss.item()

        losses['D'].append(d_loss.item())
        losses['G'].append(g_loss.item())
        # losses['LHS'].append(torch.mean(torch.abs(fake)).item())

        if lr_schedule:
          lr_scheduler_G.step()
          lr_scheduler_D.step()

        # train MSE: grid sample vs true soln
        # grid_samp, sort_ids = torch.sort(grid_samp, axis=0)
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
            #resids['grid'].append(grid_samp.detach())
            resids['resid'].append(np.abs(residuals.detach()))

        try:
            if (epoch+1) % 10 == 0:
                # mean of val mses for last 10 steps
                #track.log(lhs=np.mean(losses['LHS'][-10:])) # mean LHS for last 10 steps
                tune.report(mean_squared_error=np.mean(mses['val'][-10:]))

        except Exception as e:
            # print(f'Caught exception {e}')
            pass

        if log:
            if train_mse:
                print(f'Step {epoch}: G Loss: {g_loss.item():.4e} | D Loss: {d_loss.item():.4e} | Train MSE {train_mse:.4e} | Val MSE {val_mse:.4e}')
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
            # TODO: for systems (i.e. multi-dim preds),
            # hstack flattens preds, need to use dstack
            # v = np.dstack(v)
            np.save(os.path.join(anim_dir, f"{k}_pred"), v)
        for k, v in resids.items():
            v = np.hstack(v)
            np.save(os.path.join(anim_dir, f"{k}_resid"), v)

    if plot_del:
        def plot_dict_curves(D_dict, title="Curves", ylabel="y", figsize=(8, 5), dpi=150, show_legend=True,
                             save_path=None):
            """
            绘制字典里的曲线
            D_dict: dict, key -> list of tensors
            title: 图标题
            ylabel: y轴标签
            figsize: 图像大小
            dpi: 分辨率
            show_legend: 是否显示图例
            save_path: 如果不为None，则保存图片到指定路径
            """
            plt.figure(figsize=figsize, dpi=dpi)

            # 设置颜色和线型循环，避免曲线颜色重复
            colors = plt.cm.tab20(np.linspace(0, 1, 20))

            for key_idx, (key, val_list) in enumerate(D_dict.items()):
                for idx, val in enumerate(val_list):
                    # 将 tensor 转为 numpy 并 flatten
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
                plt.legend(fontsize=8, ncol=2)  # 图例两列，防止重叠

            plt.tight_layout()

            if save_path is not None:
                plt.savefig(save_path, dpi=dpi)

            plt.show()

        # --- 绘制 D1 ---
        plot_dict_curves(D1, title="Output of discriminator", ylabel="D(x)")

        # --- 绘制 D2 ---
        plot_dict_curves(D2, title="Derivative of the discriminator", ylabel="dD(x)/dx")
        D1_mod = [1 / (1 - x) for x in D1['D']]
        result = [a * b for a, b in zip(D1_mod, D2['D_x'])]
        result_dict = {'D_x / (1 - D)': result}
        plot_dict_curves(result_dict, title="NTK of the discriminator", ylabel="1/(1-D)*dD(x)/dx")

    if grad_NTK == True:
        linestyles = ['solid', 'dashed', 'dashdot', 'dotted'] * 3  # 确保在一个图中有很多组合可以区分不同线条
        linewidth = 2
        alpha = 0.8
        alphas = [alpha] * 10  # 通过这种方式，可以为多条曲线指定相同的透明度。
        colors = ['red', 'blue', 'skyblue', 'limegreen',
                  'aquamarine', 'violet', 'black', 'brown', 'pink', 'gold']
        for i, (k, v) in enumerate(fake_0.items()):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(np.arange(len(v)), v, label=k,
                           alpha=alphas[i], linewidth=linewidth, color=colors[i],
                           linestyle=linestyles[i])
            ax.legend()  # 显示图例
            ax.set_xlabel('Step')  # 设置x轴标签
            ax.set_ylabel('Value')  # 设置y轴标签
            ax.set_title('Curve')  # 图标题（可选）
            plt.tight_layout()  # 自动布局
            plt.show()  # 显示图像

        for i, (k, v) in enumerate(fake_1.items()):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(np.arange(len(v)), v, label=k,
                           alpha=alphas[i], linewidth=linewidth, color=colors[i],
                           linestyle=linestyles[i])
            ax.legend()  # 显示图例
            ax.set_xlabel('Step')  # 设置x轴标签
            ax.set_ylabel('Value')  # 设置y轴标签
            ax.set_title('Curve')  # 图标题（可选）
            plt.tight_layout()  # 自动布局
            plt.show()  # 显示图像

        diff = [torch.tensor(f1) / torch.norm(torch.tensor(1.0) - torch.tensor(f0)).item() for f0, f1 in
                zip(fake_0['fake_0'], fake_1['fake_1'])]
        diff_array = torch.tensor(diff).cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(diff_array, label='My Data', color='blue', linewidth=2)
        ax.legend([f'Curve {i}'])  # 设置图例
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title(f'Curve {i}')
        plt.tight_layout()
        plt.show()

    return {'mses': mses, 'model': G, 'losses': losses}


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
    """
    Train/test GAN method: supervised/semisupervised/unsupervised
    """
    assert method in ['supervised', 'semisupervised', 'unsupervised'], f'Method {method} not understood!'

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
    # grid_obs = grid[observers, :]
    # soln_obs = soln[observers, :]

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
    #optiG = torch.optim.SGD(G.parameters(), lr=g_lr, momentum=momentum, nesterov=True)
    #optiD = torch.optim.SGD(D.parameters(), lr=g_lr, momentum=momentum, nesterov=True)
    if lr_schedule:
        #lr_scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer=optiG, gamma=gamma)
        #lr_scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer=optiD, gamma=gamma)
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
    #record_steps = [1, 10, 50, 100, 1000, 2000, 2999, 4000, 6000, 8000, 10000]

    for epoch in range(niters):
        # Train Generator

        if epoch < niters:
            # 输入网格
            #ntk = torch.linspace(0.0, 1.0, K).unsqueeze(1)
            #ntk.requires_grad_(True)
            ntk = torch.linspace(0.0,1.0,K,device=next(D.parameters()).device).unsqueeze(1).requires_grad_(True)
            D_x = D(ntk)
            D1['D'].append(D_x.detach())
            # 梯度 dD/dx
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
            # grid_samp_deltax = torch.cat((xs+1e-5, ys), 1)
            # grid_samp_deltay = torch.cat((xs, ys+1e-5), 1)
            pred = G(grid_samp)
            # pred_deltax = G(grid_samp_deltax)
            # pred_deltay = G(grid_samp_deltay)
            if hard == True:
                residuals = problem.get_equation(pred, xs, ys)
            else:
                residuals = problem.get_equation_soft(pred, xs, ys)

            # residuals_deltax = problem.get_equation(pred_deltax, xs+1e-5, ys)
            # residuals_deltay = problem.get_equation(pred_deltay, xs, ys+1e-5)
            # residuals_delta = torch.cat((residuals_deltax, residuals_deltay), 1)
            # if epoch == 1000:
            #     problem.sampling = 'gradient'
            #     fig, ax = plt.subplots(figsize=(12,12))
            #     xx, yy = xs.detach().numpy().reshape((65,65)), ys.detach().numpy().reshape((65,65))
            #     resid = residuals.detach().numpy().reshape((65,65))
            #     ax.contourf(xx, yy, resid, cmap='Reds')
            #     plt.show()
            #     assert False
            # idea: add noise to relax from dirac delta at 0 to distb'n
            # + torch.normal(0, .1/(i+1), size=residuals.shape)
            if noise:
                diff = diff if diff >= 0 else 0
                #print(residuals.device, torch.normal(0, diff, size=residuals.shape).device)
                real = torch.zeros_like(residuals) + torch.normal(mean=0,std=diff,size=residuals.shape,device=residuals.device)
                fake = residuals + torch.normal(0, diff, size=residuals.shape,device=residuals.device)
            else:
                real = torch.zeros_like(residuals)
                fake = residuals
            # fake = residuals
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

                # 若想保存每一步的完整向量
                metrics['gamma'].append(gamma_flat.cpu().numpy())
                metrics['residual_vec'].append(residual_vec.cpu().numpy())

            optiG.zero_grad()
            #g_loss = criterion(D(fake), real_labels)
            if wgan:
                g_loss = -criterion(D(fake), real_labels)
            else:
                g_loss = criterion(D(fake), real_labels)
            # g_loss = criterion(D(fake), torch.ones_like(fake))
            g_loss.backward()
            optiG.step()

            #2025年4月20日 检验鉴别器对生成数据的零阶信息和一阶信息
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

            # print(real.shape, fake.shape)
            real_loss = criterion(D(real), real_labels)
            # real_loss = criterion(D(real), torch.ones_like(real))
            fake_loss = criterion(D(fake.detach()), fake_labels)
            # fake_loss = criterion(D(fake), torch.zeros_like(fake))

            optiD.zero_grad()
            if wgan:
                d_loss = -(real_loss + fake_loss) / 2 - norm_penalty
            else:
                d_loss = (real_loss + fake_loss) / 2 + norm_penalty
            #d_loss = (real_loss + fake_loss)/2 + norm_penalty
            d_loss.backward()
            optiD.step()


        # update the difference between losses
        if noise:
            diff = g_loss.item() - d_loss.item()

        losses['D'].append(d_loss.item())
        losses['G'].append(g_loss.item())
        # losses['LHS'].append(torch.mean(torch.abs(fake)).item())

        if lr_schedule:
          lr_scheduler_G.step()
          lr_scheduler_D.step()

        # train MSE: grid sample vs true soln
        # grid_samp, sort_ids = torch.sort(grid_samp, axis=0)
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
            #resids['grid'].append(grid_samp.detach())
            resids['resid'].append(np.abs(residuals.detach()))

        try:
            if (epoch+1) % 10 == 0:
                # mean of val mses for last 10 steps
                #track.log(lhs=np.mean(losses['LHS'][-10:])) # mean LHS for last 10 steps
                tune.report(mean_squared_error=np.mean(mses['val'][-10:]))

        except Exception as e:
            # print(f'Caught exception {e}')
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
            # TODO: for systems (i.e. multi-dim preds),
            # hstack flattens preds, need to use dstack
            # v = np.dstack(v)
            np.save(os.path.join(anim_dir, f"{k}_pred"), v)
        for k, v in resids.items():
            v = np.hstack(v)
            np.save(os.path.join(anim_dir, f"{k}_resid"), v)

    if plot_del:
        def plot_dict_curves(D_dict, title="Curves", ylabel="y", figsize=(8, 5), dpi=150, show_legend=True,
                             save_path=None):
            """
            绘制字典里的曲线
            D_dict: dict, key -> list of tensors
            title: 图标题
            ylabel: y轴标签
            figsize: 图像大小
            dpi: 分辨率
            show_legend: 是否显示图例
            save_path: 如果不为None，则保存图片到指定路径
            """
            plt.figure(figsize=figsize, dpi=dpi)

            # 设置颜色和线型循环，避免曲线颜色重复
            colors = plt.cm.tab20(np.linspace(0, 1, 20))

            for key_idx, (key, val_list) in enumerate(D_dict.items()):
                for idx, val in enumerate(val_list):
                    # 将 tensor 转为 numpy 并 flatten
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
                plt.legend(fontsize=8, ncol=2)  # 图例两列，防止重叠

            plt.tight_layout()

            if save_path is not None:
                plt.savefig(save_path, dpi=dpi)

            plt.show()

        # --- 绘制 D1 ---
        plot_dict_curves(D1, title="Output of discriminator", ylabel="D(x)")

        # --- 绘制 D2 ---
        plot_dict_curves(D2, title="Derivative of the discriminator", ylabel="dD(x)/dx")
        D1_mod = [1 / (1 - x) for x in D1['D']]
        result = [a * b for a, b in zip(D1_mod, D2['D_x'])]
        result_dict = {'D_x / (1 - D)': result}
        plot_dict_curves(result_dict, title="NTK of the discriminator", ylabel="1/(1-D)*dD(x)/dx")

    if grad_NTK == True:
        linestyles = ['solid', 'dashed', 'dashdot', 'dotted'] * 3  # 确保在一个图中有很多组合可以区分不同线条
        linewidth = 2
        alpha = 0.8
        alphas = [alpha] * 10  # 通过这种方式，可以为多条曲线指定相同的透明度。
        colors = ['red', 'blue', 'skyblue', 'limegreen',
                  'aquamarine', 'violet', 'black', 'brown', 'pink', 'gold']
        for i, (k, v) in enumerate(fake_0.items()):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(np.arange(len(v)), v, label=k,
                           alpha=alphas[i], linewidth=linewidth, color=colors[i],
                           linestyle=linestyles[i])
            ax.legend()  # 显示图例
            ax.set_xlabel('Step')  # 设置x轴标签
            ax.set_ylabel('Value')  # 设置y轴标签
            ax.set_title('Curve')  # 图标题（可选）
            plt.tight_layout()  # 自动布局
            plt.show()  # 显示图像

        for i, (k, v) in enumerate(fake_1.items()):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(np.arange(len(v)), v, label=k,
                           alpha=alphas[i], linewidth=linewidth, color=colors[i],
                           linestyle=linestyles[i])
            ax.legend()  # 显示图例
            ax.set_xlabel('Step')  # 设置x轴标签
            ax.set_ylabel('Value')  # 设置y轴标签
            ax.set_title('Curve')  # 图标题（可选）
            plt.tight_layout()  # 自动布局
            plt.show()  # 显示图像

        diff = [torch.tensor(f1) / torch.norm(torch.tensor(1.0) - torch.tensor(f0)).item() for f0, f1 in
                zip(fake_0['fake_0'], fake_1['fake_1'])]
        diff_array = torch.tensor(diff).cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(diff_array, label='My Data', color='blue', linewidth=2)
        ax.legend([f'Curve {i}'])  # 设置图例
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title(f'Curve {i}')
        plt.tight_layout()
        plt.show()

    return {'mses': mses, 'model': G, 'losses': losses}

def train_GAN_soft_2D(G, D, problem, method='unsupervised', niters=100,
                 g_lr=1e-3, g_betas=(0.0, 0.9), d_lr=1e-3, d_betas=(0.0, 0.9),
                 lr_schedule=True, gamma=0.999, momentum=0.95, noise=False,
                 step_size=15, obs_every=1, d1=1., d2=1., G_iters=1, D_iters=1,
                 wgan=True, gp=0.1, conditional=True, train_mse=True, log=True,
                 plot=True, plot_1d_curves=False, save=False, dirname='train_GAN',
                 config=None, save_for_animation=True, view=[35, -55], grad_NTK=True,
                 hard = False,   **kwargs):
    """
    Train/test GAN method: supervised/semisupervised/unsupervised
    """
    assert method in ['supervised', 'semisupervised', 'unsupervised'], f'Method {method} not understood!'

    pkey = dirname.split('_')[0].lower()
    dirname = os.path.join(this_dir, '../experiments/runs', dirname)
    if plot and save:
        handle_overwrite(dirname)

    # validation: fixed grid/solution
    x, y = problem.get_grid()
    grid = torch.cat((x, y), 1)
    soln = problem.get_solution(x, y)
    #print('grid', grid.shape)
    bound = problem.get_grid_sample_ub_all()
    #print('bound', bound.shape)

    soft_all = torch.cat([grid, bound, bound, bound, bound], dim=0)
    #print('soft_all', soft_all.shape)
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
    # grid_obs = grid[observers, :]
    # soln_obs = soln[observers, :]

    # labels
    real_label = 1
    fake_label = -1 if wgan else 0
    if hard == True:
        real_labels = torch.full((len(grid),), real_label, dtype=torch.float, device=grid.device).reshape(-1, 1)  # len(grid)+195 for no reparam
        fake_labels = torch.full((len(grid),), fake_label, dtype=torch.float, device=grid.device).reshape(-1, 1)  # len(grid)+195 for no reparam
    else:
        real_labels = torch.full((len(soft_all), ), real_label, dtype=torch.float, device=soft_all.device).reshape(-1, 1)  # len(grid)+195 for no reparam                                                                               1)  # len(grid)+195 for no reparam
        fake_labels = torch.full((len(soft_all), ), fake_label, dtype=torch.float, device=soft_all.device).reshape(-1, 1)  # len(grid)+195 for no reparam
        #print('real_labels',real_labels.shape)
    # masked label vectors
    real_labels_obs = real_labels[observers, :]
    fake_labels_obs = fake_labels[observers, :]

    # initialize difference parameter for noise
    if noise:
        diff = 0

    # optimization
    optiG = torch.optim.Adam(G.parameters(), lr=g_lr, betas=g_betas)
    optiD = torch.optim.Adam(D.parameters(), lr=d_lr, betas=d_betas)
    # optiG = torch.optim.SGD(G.parameters(), lr=g_lr, momentum=momentum, nesterov=True)
    # optiD = torch.optim.SGD(D.parameters(), lr=g_lr, momentum=momentum, nesterov=True)
    if lr_schedule:
        # lr_scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer=optiG, gamma=gamma)
        # lr_scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer=optiD, gamma=gamma)
        lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer=optiG, step_size=step_size, gamma=gamma)
        lr_scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer=optiD, step_size=step_size, gamma=gamma)

    # losses
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    wass = lambda y_true, y_pred: torch.mean(y_true * y_pred)
    criterion = wass if wgan else bce

    # history
    losses = {'G': [], 'D': []}  # , 'LHS': []}
    mses = {'train': [], 'val': []} if train_mse else {'val': []}
    preds = {'pred': [], 'soln': []}
    # resids = {'grid': [], 'resid': []}
    resids = {'resid': []}
    fake_0 = {'fake_0': []}
    fake_1 = {'fake_1': []}
    for epoch in range(niters):
        # Train Generator
        for p in D.parameters():
            p.requires_grad = False  # turn off computation for D

        for i in range(G_iters):
            xs, ys = problem.get_grid_sample(grid_samp, residuals, residuals_delta)
            grid_samp = torch.cat((xs, ys), 1)
            # grid_samp_deltax = torch.cat((xs+1e-5, ys), 1)
            # grid_samp_deltay = torch.cat((xs, ys+1e-5), 1)
            pred = G(grid_samp)
            # pred_deltax = G(grid_samp_deltax)
            # pred_deltay = G(grid_samp_deltay)
            residuals = problem.get_equation_soft(pred, xs, ys, G)

            # residuals_deltax = problem.get_equation(pred_deltax, xs+1e-5, ys)
            # residuals_deltay = problem.get_equation(pred_deltay, xs, ys+1e-5)
            # residuals_delta = torch.cat((residuals_deltax, residuals_deltay), 1)
            # if epoch == 1000:
            #     problem.sampling = 'gradient'
            #     fig, ax = plt.subplots(figsize=(12,12))
            #     xx, yy = xs.detach().numpy().reshape((65,65)), ys.detach().numpy().reshape((65,65))
            #     resid = residuals.detach().numpy().reshape((65,65))
            #     ax.contourf(xx, yy, resid, cmap='Reds')
            #     plt.show()
            #     assert False
            # idea: add noise to relax from dirac delta at 0 to distb'n
            # + torch.normal(0, .1/(i+1), size=residuals.shape)
            if noise:
                diff = diff if diff >= 0 else 0
                real = torch.zeros_like(residuals) + torch.normal(0, diff, size=residuals.shape, device=residuals.device)
                fake = residuals + torch.normal(0, diff, size=residuals.shape, device=residuals.device)
            else:
                real = torch.zeros_like(residuals)
                fake = residuals
            # fake = residuals

            optiG.zero_grad()
            g_loss = criterion(D(fake), real_labels)
            # g_loss = criterion(D(fake), torch.ones_like(fake))
            g_loss.backward()
            optiG.step()

            # 2025年4月20日 检验鉴别器对生成数据的零阶信息和一阶信息
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
            p.requires_grad = True  # turn on computation for D

        for i in range(D_iters):
            if wgan:
                norm_penalty = calc_gradient_penalty(D, real, fake, gp, cuda=False)
            else:
                norm_penalty = torch.zeros(1, device=real.device)

            # print(real.shape, fake.shape)
            real_loss = criterion(D(real), real_labels)
            # real_loss = criterion(D(real), torch.ones_like(real))
            fake_loss = criterion(D(fake.detach()), fake_labels)
            # fake_loss = criterion(D(fake), torch.zeros_like(fake))

            optiD.zero_grad()
            d_loss = (real_loss + fake_loss) / 2 + norm_penalty
            d_loss.backward()
            optiD.step()

        # update the difference between losses
        if noise:
            diff = g_loss.item() - d_loss.item()

        losses['D'].append(d_loss.item())
        losses['G'].append(g_loss.item())
        # losses['LHS'].append(torch.mean(torch.abs(fake)).item())

        if lr_schedule:
            lr_scheduler_G.step()
            lr_scheduler_D.step()

        # train MSE: grid sample vs true soln
        # grid_samp, sort_ids = torch.sort(grid_samp, axis=0)
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
            # resids['grid'].append(grid_samp.detach())
            resids['resid'].append(np.abs(residuals.detach()))

        try:
            if (epoch + 1) % 10 == 0:
                # mean of val mses for last 10 steps
                # track.log(lhs=np.mean(losses['LHS'][-10:])) # mean LHS for last 10 steps
                tune.report(mean_squared_error=np.mean(mses['val'][-10:]))

        except Exception as e:
            # print(f'Caught exception {e}')
            pass

        if log:
            if train_mse:
                print(
                    f'Step {epoch}: G Loss: {g_loss.item():.4e} | D Loss: {d_loss.item():.4e} | Train MSE {train_mse:.4e} | Val MSE {val_mse:.4e}')
            else:
                print(
                    f'Step {epoch}: G Loss: {g_loss.item():.4e} | D Loss: {d_loss.item():.4e} | Val MSE {val_mse:.4e}')

    if plot:
        if hard == True:
            pred_dict, diff_dict = problem.get_plot_dicts(G(plot_grid), plot_x, plot_y, plot_soln)
        else:
            pred_dict, diff_dict = problem.get_plot_dicts_soft(G(plot_grid), plot_x, plot_y, plot_soln)

        plot_results(mses, losses, plot_grid.detach(), pred_dict, diff_dict=diff_dict,
                     save=save, dirname=dirname, logloss=False, alpha=0.7, dims=dims,
                     plot_1d_curves=plot_1d_curves)
        plot_3D(plot_grid.detach(), pred_dict, view=view, dims=dims, save=save, dirname=dirname)

    if save:
        write_config(config, os.path.join(dirname, 'config.yaml'))
        np.save(os.path.join(dirname, "pred_dict"), pred_dict)

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
            # TODO: for systems (i.e. multi-dim preds),
            # hstack flattens preds, need to use dstack
            # v = np.dstack(v)
            np.save(os.path.join(anim_dir, f"{k}_pred"), v)
        for k, v in resids.items():
            v = np.hstack(v)
            np.save(os.path.join(anim_dir, f"{k}_resid"), v)

    if grad_NTK == True:
        linestyles = ['solid', 'dashed', 'dashdot', 'dotted'] * 3  # 确保在一个图中有很多组合可以区分不同线条
        linewidth = 2
        alpha = 0.8
        alphas = [alpha] * 10  # 通过这种方式，可以为多条曲线指定相同的透明度。
        colors = ['red', 'blue', 'skyblue', 'limegreen',
                  'aquamarine', 'violet', 'black', 'brown', 'pink', 'gold']
        for i, (k, v) in enumerate(fake_0.items()):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(np.arange(len(v)), v, label=k,
                    alpha=alphas[i], linewidth=linewidth, color=colors[i],
                    linestyle=linestyles[i])
            ax.legend()  # 显示图例
            ax.set_xlabel('Step')  # 设置x轴标签
            ax.set_ylabel('Value')  # 设置y轴标签
            ax.set_title('Curve')  # 图标题（可选）
            plt.tight_layout()  # 自动布局
            plt.show()  # 显示图像

        for i, (k, v) in enumerate(fake_1.items()):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(np.arange(len(v)), v, label=k,
                    alpha=alphas[i], linewidth=linewidth, color=colors[i],
                    linestyle=linestyles[i])
            ax.legend()  # 显示图例
            ax.set_xlabel('Step')  # 设置x轴标签
            ax.set_ylabel('Value')  # 设置y轴标签
            ax.set_title('Curve')  # 图标题（可选）
            plt.tight_layout()  # 自动布局
            plt.show()  # 显示图像

        diff = [torch.tensor(f1) / torch.norm(torch.tensor(1.0) - torch.tensor(f0)).item() for f0, f1 in
                zip(fake_0['fake_0'], fake_1['fake_1'])]
        diff_array = torch.tensor(diff).cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(diff_array, label='My Data', color='blue', linewidth=2)
        ax.legend([f'Curve {i}'])  # 设置图例
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title(f'Curve {i}')
        plt.tight_layout()
        plt.show()

    return {'mses': mses, 'model': G, 'losses': losses}

def train_L2_2D(model, problem, method='unsupervised', niters=100,
    lr=1e-3, betas=(0, 0.9), lr_schedule=True, gamma=0.999, step_size=15,
    obs_every=1, d1=1, d2=1, log=True, plot=True, save=False,
    dirname='train_L2', config=None, loss_fn=None, save_for_animation=False,
    **kwargs):
    """
    Train/test Lagaris method: supervised/semisupervised/unsupervised
    """
    assert method in ['supervised', 'semisupervised', 'unsupervised'], f'Method {method} not understood!'

    dirname = os.path.join(this_dir, '../experiments/runs', dirname)
    if plot and save:
        handle_overwrite(dirname)

    # validation: fixed grid/solution
    x, y = problem.get_grid()
    grid = torch.cat((x, y), 1)
    sol = problem.get_solution(x, y)

    # optimizers & loss functions
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    if loss_fn:
        mse = eval(f"torch.nn.{loss_fn}()")
    else:
        mse = torch.nn.MSELoss()
    # lr scheduler
    if lr_schedule:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=step_size, gamma=gamma)

    loss_trace = []
    mses = {'train': [], 'val': []}
    preds = {'pred': [], 'soln': []}

    for i in range(niters):
        xs, ys = problem.get_grid_sample(None, None, None) #TODO: remove these unnecessary args
        grid_samp = torch.cat((xs, ys), 1)
        pred = model(grid_samp)
        residuals = problem.get_equation(pred, xs, ys)
        loss = mse(residuals, torch.zeros_like(residuals))
        loss_trace.append(loss.item())

        # train MSE: grid sample vs true soln
        # grid_samp, sort_ids = torch.sort(grid_samp, axis=0)
        pred = model(grid_samp)
        try:
            pred_adj = problem.adjust(pred, xs, ys)['pred']
            sol_samp = problem.get_solution(xs, ys)
            train_mse = mse(pred_adj, sol_samp).item()
        except Exception as e:
            print(f'Exception: {e}')
        mses['train'].append(train_mse)

        # val MSE: fixed grid vs true soln
        val_pred = model(grid)
        val_pred_adj = problem.adjust(val_pred, x, y)['pred']
        val_mse = mse(val_pred_adj, sol).item()
        mses['val'].append(val_mse)

        # store preds for animation
        preds['pred'].append(val_pred_adj.detach())
        preds['soln'].append(sol.detach())

        try:
            if (i+1) % 10 == 0:
                # mean of val mses for last 10 steps
                tune.report(mean_squared_error=np.mean(mses['val'][-10:]))
        except Exception as e:
            # print(f'Caught exception {e}')
            pass

        if log:
            print(f'Step {i}: Loss {loss.item():.4e} | Train MSE {train_mse:.4e} | Val MSE {val_mse:.4e}')

        opt.zero_grad()
        loss.backward(retain_graph=True)
        opt.step()
        if lr_schedule:
            lr_scheduler.step()

    if plot:
        loss_dict = {}
        if method == 'supervised':
            loss_dict['$L_S$'] = loss_trace
        elif method == 'semisupervised':
            loss_dict['$L_S$'] = [l[0] for l in loss_trace]
            loss_dict['$L_U$'] = [l[1] for l in loss_trace]
        else:
            loss_dict['$L_U$'] = loss_trace
        dims = problem.get_plot_dims()

        save_to = os.path.join(this_dir, '../experiments/runs', dirname)

        pred_dict, diff_dict = problem.get_plot_dicts(model(grid), x, y, sol)
        plot_results(mses, loss_dict, grid.detach(), pred_dict, diff_dict=diff_dict,
            save=save, dirname=dirname, logloss=True, alpha=0.7, dims=dims)

    if save:
        write_config(config, os.path.join(dirname, 'config.yaml'))

    if save_for_animation:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        anim_dir = os.path.join(dirname, "animation")
        print(f'Saving animation traces to {anim_dir}')
        if not os.path.exists(anim_dir):
            os.mkdir(anim_dir)
        np.save(os.path.join(anim_dir, "grid"), grid.detach())
        for k, v in preds.items():
            v = np.hstack(v)
            # TODO: for systems (i.e. multi-dim preds),
            # hstack flattens preds, need to use dstack
            # v = np.dstack(v)
            np.save(os.path.join(anim_dir, f"{k}_pred"), v)

    return {'mses': mses, 'model': model, 'losses': loss_trace}
