import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w=50):
    return np.convolve(x, np.ones(w)/w, mode='valid')

val_NTK_gan_rb = np.load('Kle_NTK_gan_rb/val_mse.npy', allow_pickle=True)
mse_NTK_gan_rb = np.load('Kle_NTK_gan_rb/train_mse.npy', allow_pickle=True)
energy_NTK_gan_rb = np.load('Kle_NTK_gan_rb/residual_energy_loss.npy', allow_pickle=True)

val_NTK_1_1_none_gan = np.load('Kle_NTK_none_1_1_gan/val_mse.npy', allow_pickle=True)
mse_NTK_1_1_none_gan = np.load('Kle_NTK_none_1_1_gan/train_mse.npy', allow_pickle=True)
energy_NTK_1_1_none_gan = np.load('Kle_NTK_none_1_1_gan/residual_energy_loss.npy', allow_pickle=True)

val_NTK_gan_norb = np.load('Kle_NTK_gan_norb/val_mse.npy', allow_pickle=True)
mse_NTK_gan_norb = np.load('Kle_NTK_gan_norb/train_mse.npy', allow_pickle=True)
energy_NTK_gan_norb = np.load('Kle_NTK_gan_norb/residual_energy_loss.npy', allow_pickle=True)


linestyles = ['solid', 'dashed']
linewidth = 2
colors = ['blue', 'orange', 'red', 'green', 'black', 'purple', 'yellow']


plt.figure(figsize=(8, 6))
plt.plot(energy_NTK_1_1_none_gan, label='G:D=1:1 noRB', linewidth=linewidth, color=colors[1],
                linestyle=linestyles[0])
plt.plot(energy_NTK_gan_norb, label='G:D=20:20 noRB', linewidth=linewidth, color=colors[2],
                linestyle=linestyles[0])
plt.plot(energy_NTK_gan_rb, label='G:D=20:20 RB', linewidth=linewidth, color=colors[0],
                linestyle=linestyles[0])
plt.xlabel('Iteration')
plt.ylabel('Residual Energy Loss')
plt.title(r'Ablation Study on G:D Update Ratio and Rollback Strategy: Residual Energy (GAN)')
plt.legend()
plt.yscale('log')
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1,2]
plt.legend([handles[i] for i in order],
           [labels[i] for i in order])
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.85))
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(val_NTK_1_1_none_gan, label='G:D=1:1 noRB', linewidth=linewidth, color=colors[1],
                linestyle=linestyles[0])
plt.plot(val_NTK_gan_norb, label='G:D=20:20 noRB', linewidth=linewidth, color=colors[2],
                linestyle=linestyles[0])
plt.plot(val_NTK_gan_rb, label='G:D=20:20 RB', linewidth=linewidth, color=colors[0],
                linestyle=linestyles[0])
plt.xlabel('Iteration')
plt.ylabel('VAL')
plt.title(r'Ablation Study on G:D Update Ratio and Rollback Strategy: VAL (GAN)')
plt.legend()
plt.yscale('log')
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1,2]
plt.legend([handles[i] for i in order],
           [labels[i] for i in order])
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.85))
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(mse_NTK_1_1_none_gan, label='G:D=1:1 noRB', linewidth=linewidth, color=colors[1],
                linestyle=linestyles[0])
plt.plot(mse_NTK_gan_norb, label='G:D=20:20 noRB', linewidth=linewidth, color=colors[2],
                linestyle=linestyles[0])
plt.plot(mse_NTK_gan_rb, label='G:D=20:20 RB', linewidth=linewidth, color=colors[0],
                linestyle=linestyles[0])
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title(r'Ablation Study on G:D Update Ratio and Rollback Strategy: MSE (GAN)')
plt.legend()
plt.yscale('log')
order = [0,1,2]
plt.legend([handles[i] for i in order],
           [labels[i] for i in order])
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.85))
plt.tight_layout()
plt.show()


gkr = np.load('Kle_NTK_gan_norb/rKr_gamma_loss.npy', allow_pickle=True)
step_G = np.load('Kle_NTK_gan_norb/best_G_step.npy', allow_pickle=True)
step_D = np.load('Kle_NTK_gan_norb/best_D_step.npy', allow_pickle=True)
gkr_smooth = moving_average(gkr, w=300)
G_smooth = moving_average(step_G, w=300)
D_smooth = moving_average(step_D+1, w=300)
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
plt.title(r'Klein–Gordon: Training Steps of $D$, $G$ and $rK\gamma$ (G:D = 20:20 noRB)')
lines = l2 +l3 +l1
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 0.85))
plt.show()

gkr = np.load('Kle_NTK_gan_rb/rKr_gamma_loss.npy', allow_pickle=True)
step_G = np.load('Kle_NTK_gan_rb/best_G_step.npy', allow_pickle=True)
step_D = np.load('Kle_NTK_gan_rb/best_D_step.npy', allow_pickle=True)
gkr_smooth = moving_average(gkr, w=300)
G_smooth = moving_average(step_G, w=300)
D_smooth = moving_average(step_D+1, w=300)
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
plt.title(r'Klein–Gordon: Training Steps of $D$, $G$ and $rK\gamma$ (G:D = 20:20 RB)')
lines = l2 +l3 +l1
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 0.85))
plt.show()

gkr1 = np.load('Kle_NTK_gan_norb/rKr_gamma_loss.npy', allow_pickle=True)
gkr2 = np.load('Kle_NTK_gan_rb/rKr_gamma_loss.npy', allow_pickle=True)
gkr3 = np.load('Kle_NTK_none_1_1_gan/rKr_gamma_loss.npy', allow_pickle=True)
gkr_smooth1 = moving_average(gkr1, w=300)
gkr_smooth2 = moving_average(gkr2, w=300)
gkr_smooth3 = moving_average(gkr3, w=300)
plt.plot(gkr_smooth1, label='G:D=20:20 noRB', linewidth=linewidth, color=colors[2],
                linestyle=linestyles[0])
plt.plot(gkr_smooth2, label='G:D=20:20 RB', linewidth=linewidth, color=colors[0],
                linestyle=linestyles[0])
plt.plot(gkr_smooth3, label='G:D=1:1 noRB', linewidth=linewidth, color=colors[1],
                linestyle=linestyles[0])
plt.xlabel('Iteration')
plt.ylabel('$rK\gamma$')
plt.title(r'Ablation Study on G:D Update Ratio and Rollback Strategy: $rK\gamma$')
plt.legend()
plt.yscale('symlog')
plt.tight_layout()
plt.show()