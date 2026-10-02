import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w=50):
    return np.convolve(x, np.ones(w)/w, mode='valid')
# 加载数据文件
mse1 = np.load('Lap1_del_1_1_gkr_new/train_mse.npy', allow_pickle=True)
val1 = np.load('Lap1_del_1_1_gkr_new/val_mse.npy', allow_pickle=True)
gkr1 = np.load('Lap1_del_1_1_gkr_new/rKr_gamma_loss.npy', allow_pickle=True)
energy1 = np.load('Lap1_del_1_1_gkr_new/metrics_residual_energy.npy', allow_pickle=True)

mse_potential_gan = np.load('Lap1_del_potential_gan/train_mse.npy', allow_pickle=True)
val_potential_gan = np.load('Lap1_del_potential_gan/val_mse.npy', allow_pickle=True)
gkr_potential_gan = np.load('Lap1_del_potential_gan/rKr_gamma_loss.npy', allow_pickle=True)
energy_potential_gan = np.load('Lap1_del_potential_gan/metrics_residual_energy.npy', allow_pickle=True)

val_NTK_gan_potential_rb = np.load('Lap1_NTK_potential_gan/val_mse.npy', allow_pickle=True)
mse_NTK_gan_potential_rb = np.load('Lap1_NTK_potential_gan/train_mse.npy', allow_pickle=True)
energy_NTK_potential_gan_rb = np.load('Lap1_NTK_potential_gan/residual_energy_loss.npy', allow_pickle=True)
gkr_NTK_gan_potential_rb = np.load('Lap1_NTK_potential_gan/rKr_gamma_loss.npy', allow_pickle=True)

mse_NTK_gan_rb = np.load('Lap1_NTK_gan_rb/train_mse.npy', allow_pickle=True)
val_NTK_gan_rb = np.load('Lap1_NTK_gan_rb/val_mse.npy', allow_pickle=True)
gkr_NTK_gan_rb = np.load('Lap1_NTK_gan_rb/rKr_gamma_loss.npy', allow_pickle=True)
energy_NTK_gan_rb = np.load('Lap1_NTK_gan_rb/residual_energy_loss.npy', allow_pickle=True)

gkr1 = moving_average(gkr1, w=300)
gkr_NTK_gan_rb = moving_average(gkr_NTK_gan_rb, w=300)
gkr_potential_gan = moving_average(gkr_potential_gan, w=300)
gkr_NTK_gan_potential_rb = moving_average(gkr_NTK_gan_potential_rb, w=300)

linestyles = ['solid', 'dashed'] #确保在一个图中有很多组合可以区分不同线条
linewidth = 2
colors = ['blue', 'orange', 'red', 'green', 'black', 'purple', 'yellow']

plt.figure(figsize=(8, 6))
plt.plot(energy_potential_gan, label='DEQGAN G:D=20:20', linewidth=linewidth, color=colors[0],
                linestyle=linestyles[0])
plt.plot(energy1, label='DEQGAN G:D=1:1', linewidth=linewidth, color=colors[1],
                linestyle=linestyles[0])
plt.plot(energy_NTK_potential_gan_rb, label='DEQGAN-RB', linewidth=linewidth, color=colors[2],
                linestyle=linestyles[0])
plt.plot(energy_NTK_gan_rb, label='GAN-RB', linewidth=linewidth, color=colors[3],
                linestyle=linestyles[0])
plt.xlabel('Iteration')
plt.ylabel('Residual Energy Loss')
plt.title(r'Laplace1: Rollback Strategy Outperforms under DEQGAN Hyperparameters')
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(val_potential_gan, label='DEQGAN G:D=20:20', linewidth=linewidth, color=colors[0],
                linestyle=linestyles[0])
plt.plot(val1, label='DEQGAN G:D=1:1', linewidth=linewidth, color=colors[1],
                linestyle=linestyles[0])
plt.plot(val_NTK_gan_potential_rb, label='DEQGAN-RB', linewidth=linewidth, color=colors[2],
                linestyle=linestyles[0])
plt.plot(val_NTK_gan_rb, label='GAN-RB', linewidth=linewidth, color=colors[3],
                linestyle=linestyles[0])
plt.xlabel('Iteration')
plt.ylabel('VAL')
plt.title(r'Laplace1: Rollback Strategy Outperforms under DEQGAN Hyperparameters')
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(mse_potential_gan, label='DEQGAN G:D=20:20', linewidth=linewidth, color=colors[0],
                linestyle=linestyles[0])
plt.plot(mse1, label='DEQGAN G:D=1:1', linewidth=linewidth, color=colors[1],
                linestyle=linestyles[0])
plt.plot(mse_NTK_gan_potential_rb, label='DEQGAN-RB', linewidth=linewidth, color=colors[2],
                linestyle=linestyles[0])
plt.plot(mse_NTK_gan_rb, label='GAN-RB', linewidth=linewidth, color=colors[3],
                linestyle=linestyles[0])

plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title(r'Laplace1: Rollback Strategy Outperforms under DEQGAN Hyperparameters')
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(gkr_NTK_gan_rb, label='GAN-RB', linewidth=linewidth, color=colors[3],
                linestyle=linestyles[0])
plt.plot(gkr_potential_gan, label='DEQGAN G:D=20:20', linewidth=linewidth, color=colors[0],
                linestyle=linestyles[0])
plt.plot(gkr1, label='DEQGAN G:D=1:1', linewidth=linewidth, color=colors[1],
                linestyle=linestyles[0])
plt.plot(gkr_NTK_gan_potential_rb, label='DEQGAN-RB', linewidth=linewidth, color=colors[2],
                linestyle=linestyles[0])
plt.xlabel('Iteration')
plt.ylabel(r'$rK\gamma$')
plt.title(r'Ablation Study on G:D Update Ratio and Rollback Strategy: $rK\gamma$')
plt.legend()
plt.yscale('symlog')
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,2,3,0]  
plt.legend([handles[i] for i in order],
           [labels[i] for i in order])
plt.show()


gkr = np.load('Lap1_NTK_gan_rb/rKr_gamma_loss.npy', allow_pickle=True)
step_G = np.load('Lap1_NTK_gan_rb/best_G_step.npy', allow_pickle=True)
step_D = np.load('Lap1_NTK_gan_rb/best_D_step.npy', allow_pickle=True)
gkr_smooth = moving_average(gkr, w=300)
G_smooth = moving_average(step_G, w=300)
D_smooth = moving_average(step_D, w=300)
x = np.arange(len(gkr_smooth))
fig, ax1 = plt.subplots(figsize=(8, 6))
l1 = ax1.plot(x, gkr_smooth, label=r'$rK\gamma$',color='orange')
ax1.set_xlabel('Iteration')
ax1.set_ylabel(r'$rK\gamma$')
ax1.set_yscale('symlog')
ax1.set_ylim(-1e2, 1e2)
ax2 = ax1.twinx()
l2 = ax2.plot(x, G_smooth, label=r'$G$ steps',color='red')
ax2.set_ylabel(r'$G$ and $D$ steps')
ax2.set_ylim(-0.5, 21)
ax3 = ax1.twinx()
l3 = ax3.plot(x, D_smooth, label=r'$D$ steps',color='blue')
ax3.set_ylim(-0.5, 21)
plt.title(r'Laplace1: Training Steps of $D$, $G$ and $rK\gamma$ (GAN-RB)')
lines = l2 +l3 +l1
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='best')
plt.show()



gkr = np.load('Lap1_NTK_potential_gan/rKr_gamma_loss.npy', allow_pickle=True)
step_G = np.load('Lap1_NTK_potential_gan/best_G_step.npy', allow_pickle=True)
step_D = np.load('Lap1_NTK_potential_gan/best_D_step.npy', allow_pickle=True)
gkr_smooth = moving_average(gkr, w=300)
G_smooth = moving_average(step_G, w=300)
D_smooth = moving_average(step_D, w=300)
x = np.arange(len(gkr_smooth))
fig, ax1 = plt.subplots(figsize=(8, 6))
l1 = ax1.plot(x, gkr_smooth, label=r'$rK\gamma$',color='orange')
ax1.set_xlabel('Iteration')
ax1.set_ylabel(r'$rK\gamma$')
ax1.set_yscale('symlog')
ax2 = ax1.twinx()
l2 = ax2.plot(x, G_smooth, label=r'$G$ steps',color='red')
ax2.set_ylabel(r'$G$ and $D$ steps')
ax2.set_ylim(-0.5, 21)
ax3 = ax1.twinx()
l3 = ax3.plot(x, D_smooth, label=r'$D$ steps',color='blue')
ax3.set_ylim(-0.5, 21)
plt.title(r'Laplace1: Training Steps of $D$, $G$ and $rK\gamma$ (DEQGAN-RB)')
lines = l2 +l3 +l1
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='best')
plt.show()
