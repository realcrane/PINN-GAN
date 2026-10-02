import numpy as np
import matplotlib.pyplot as plt
# 加载数据文件
def moving_average(x, w=50):
    return np.convolve(x, np.ones(w)/w, mode='valid')

mse1 = np.load('Lap2_del_1_1_gkr/train_mse.npy', allow_pickle=True)
mse_NTK_1_1_none_gan = np.load('Lap2_NTK_none_1_1_gan/train_mse.npy', allow_pickle=True)
mse_NTK_1_1_none_wgan = np.load('Lap2_NTK_none_1_1_wgan/train_mse.npy', allow_pickle=True)
mse_NTK_1_1_none_lsgan = np.load('Lap2_NTK_none_1_1_lsgan/train_mse.npy', allow_pickle=True)
mse_NTK_wgan_rb = np.load('Lap2_NTK_wgan_rb/train_mse.npy', allow_pickle=True)
mse_NTK_lsgan_rb = np.load('Lap2_NTK_lsgan_rb/train_mse.npy', allow_pickle=True)
mse_NTK_gan_rb = np.load('Lap2_NTK_gan_rb/train_mse.npy', allow_pickle=True)

val1 = np.load('Lap2_del_1_1_gkr/val_mse.npy', allow_pickle=True)
val_NTK_1_1_none_gan = np.load('Lap2_NTK_none_1_1_gan/val_mse.npy', allow_pickle=True)
val_NTK_1_1_none_wgan = np.load('Lap2_NTK_none_1_1_wgan/val_mse.npy', allow_pickle=True)
val_NTK_1_1_none_lsgan = np.load('Lap2_NTK_none_1_1_lsgan/val_mse.npy', allow_pickle=True)
val_NTK_wgan_rb = np.load('Lap2_NTK_wgan_rb/val_mse.npy', allow_pickle=True)
val_NTK_lsgan_rb = np.load('Lap2_NTK_lsgan_rb/val_mse.npy', allow_pickle=True)
val_NTK_gan_rb = np.load('Lap2_NTK_gan_rb/val_mse.npy', allow_pickle=True)

print('DEQGAN',val1[-1])
print('GAN',val_NTK_1_1_none_gan[-1])
print('WGAN-GP',val_NTK_1_1_none_wgan[-1])
print('LSGAN',val_NTK_1_1_none_lsgan[-1])
print('WGAN-GP-RB',val_NTK_wgan_rb[-1])
print('LSGAN-RB',val_NTK_lsgan_rb[-1])
print('GAN-RB',val_NTK_gan_rb[-1])

print('DEQGAN',mse1[-1])
print('GAN',mse_NTK_1_1_none_gan[-1])
print('WGAN-GP',mse_NTK_1_1_none_wgan[-1])
print('LSGAN',mse_NTK_1_1_none_lsgan[-1])
print('WGAN-GP-RB',mse_NTK_wgan_rb[-1])
print('LSGAN-RB',mse_NTK_lsgan_rb[-1])
print('GAN-RB',mse_NTK_gan_rb[-1])

gkr1 = np.load('Lap2_del_1_1_gkr/rKr_gamma_loss.npy', allow_pickle=True)
gkr_NTK_1_1_none_gan = np.load('Lap2_NTK_none_1_1_gan/rKr_gamma_loss.npy', allow_pickle=True)
gkr_NTK_1_1_none_wgan = np.load('Lap2_NTK_none_1_1_wgan/rKr_gamma_loss.npy', allow_pickle=True)
gkr_NTK_1_1_none_lsgan = np.load('Lap2_NTK_none_1_1_lsgan/rKr_gamma_loss.npy', allow_pickle=True)
gkr_NTK_wgan_rb = np.load('Lap2_NTK_wgan_rb/rKr_gamma_loss.npy', allow_pickle=True)
gkr_NTK_lsgan_rb = np.load('Lap2_NTK_lsgan_rb/rKr_gamma_loss.npy', allow_pickle=True)
gkr_NTK_gan_rb = np.load('Lap2_NTK_gan_rb/rKr_gamma_loss.npy', allow_pickle=True)

energy1 = np.load('Lap2_del_1_1_gkr/metrics_residual_energy.npy', allow_pickle=True)
energy_NTK_1_1_none_gan = np.load('Lap2_NTK_none_1_1_gan/residual_energy_loss.npy', allow_pickle=True)
energy_NTK_1_1_none_wgan = np.load('Lap2_NTK_none_1_1_wgan/residual_energy_loss.npy', allow_pickle=True)
energy_NTK_1_1_none_lsgan = np.load('Lap2_NTK_none_1_1_lsgan/residual_energy_loss.npy', allow_pickle=True)
energy_NTK_wgan_rb = np.load('Lap2_NTK_wgan_rb/residual_energy_loss.npy', allow_pickle=True)
energy_NTK_lsgan_rb = np.load('Lap2_NTK_lsgan_rb/residual_energy_loss.npy', allow_pickle=True)
energy_NTK_gan_rb = np.load('Lap2_NTK_gan_rb/residual_energy_loss.npy', allow_pickle=True)

gkr1 = moving_average(gkr1, w=300)
gkr_NTK_1_1_none_gan = moving_average(gkr_NTK_1_1_none_gan, w=300)
gkr_NTK_1_1_none_wgan = moving_average(gkr_NTK_1_1_none_wgan, w=300)
gkr_NTK_1_1_none_lsgan = moving_average(gkr_NTK_1_1_none_lsgan, w=300)
gkr_NTK_wgan_rb = moving_average(gkr_NTK_wgan_rb, w=300)
gkr_NTK_lsgan_rb = moving_average(gkr_NTK_lsgan_rb, w=300)
gkr_NTK_gan_rb = moving_average(gkr_NTK_gan_rb, w=300)



linestyles = ['solid', 'dashed'] 
linewidth = 2
colors = ['blue', 'orange', 'red', 'green', 'black', 'purple', 'yellow']


plt.figure(figsize=(8, 6))
plt.plot(energy1, label='DEQGAN', linewidth=linewidth, color=colors[1],
                linestyle=linestyles[0])
plt.plot(energy_NTK_1_1_none_gan, label='GAN', linewidth=linewidth, color=colors[3],
                linestyle=linestyles[0])
plt.plot(energy_NTK_1_1_none_lsgan, label='LSGAN', linewidth=linewidth, color=colors[4],
                linestyle=linestyles[0])
plt.plot(energy_NTK_1_1_none_wgan, label='WGAN-GP', linewidth=linewidth, color=colors[6],
                linestyle=linestyles[0])
plt.plot(energy_NTK_gan_rb, label='GAN-RB (ours)', linewidth=linewidth, color=colors[0],
                linestyle=linestyles[0])
plt.plot(energy_NTK_wgan_rb, label='WGAN-GP-RB (ours)', linewidth=linewidth, color=colors[5],
                linestyle=linestyles[0])
plt.plot(energy_NTK_lsgan_rb, label='LSGAN-RB (ours)', linewidth=linewidth, color=colors[2],
                linestyle=linestyles[0])
plt.xlabel('Iteration')
plt.ylabel('Residual Energy Loss')
plt.title(r'Laplace2: Comparison of GAN-based and NTK-guided Methods')
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(val1, label='DEQGAN', linewidth=linewidth, color=colors[1],
                linestyle=linestyles[0])
plt.plot(val_NTK_1_1_none_gan, label='GAN', linewidth=linewidth, color=colors[3],
                linestyle=linestyles[0])
plt.plot(val_NTK_1_1_none_lsgan, label='LSGAN', linewidth=linewidth, color=colors[4],
                linestyle=linestyles[0])
plt.plot(val_NTK_1_1_none_wgan, label='WGAN-GP', linewidth=linewidth, color=colors[6],
                linestyle=linestyles[0])
plt.plot(val_NTK_gan_rb, label='GAN-RB (ours)', linewidth=linewidth, color=colors[0],
                linestyle=linestyles[0])
plt.plot(val_NTK_wgan_rb, label='WGAN-GP-RB (ours)', linewidth=linewidth, color=colors[5],
                linestyle=linestyles[0])
plt.plot(val_NTK_lsgan_rb, label='LSGAN-RB (ours)', linewidth=linewidth, color=colors[2],
                linestyle=linestyles[0])
plt.xlabel('Iteration')
plt.ylabel('VAL')
plt.title(r'Laplace2: Comparison of GAN-based and NTK-guided Methods')
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(mse1, label='DEQGAN', linewidth=linewidth, color=colors[1],
                linestyle=linestyles[0])
plt.plot(mse_NTK_1_1_none_gan, label='GAN', linewidth=linewidth, color=colors[3],
                linestyle=linestyles[0])
plt.plot(mse_NTK_1_1_none_lsgan, label='LSGAN', linewidth=linewidth, color=colors[4],
                linestyle=linestyles[0])
plt.plot(mse_NTK_1_1_none_wgan, label='WGAN-GP', linewidth=linewidth, color=colors[6],
                linestyle=linestyles[0])
plt.plot(mse_NTK_gan_rb, label='GAN-RB (ours)', linewidth=linewidth, color=colors[0],
                linestyle=linestyles[0])
plt.plot(mse_NTK_wgan_rb, label='WGAN-GP-RB (ours)', linewidth=linewidth, color=colors[5],
                linestyle=linestyles[0])
plt.plot(mse_NTK_lsgan_rb, label='LSGAN-RB (ours)', linewidth=linewidth, color=colors[2],
                linestyle=linestyles[0])
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title(r'Laplace2: Comparison of GAN-based and NTK-guided Methods')
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.show()


gkr = np.load('Lap2_NTK_gan_rb/rKr_gamma_loss.npy', allow_pickle=True)
step_G = np.load('Lap2_NTK_gan_rb/best_G_step.npy', allow_pickle=True)
step_D = np.load('Lap2_NTK_gan_rb/best_D_step.npy', allow_pickle=True)

gkr_smooth = moving_average(gkr, w=300)
G_smooth = moving_average(step_G, w=300)
D_smooth = moving_average(step_D, w=300)
x = np.arange(len(gkr_smooth))
print(len(gkr_smooth))
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
plt.title(r'Laplace2: Training Steps of $D$, $G$ and $rK\gamma$ (GAN-RB)')

lines = l2 +l3 +l1
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='best')
plt.show()



gkr = np.load('Lap2_NTK_lsgan_rb/rKr_gamma_loss.npy', allow_pickle=True)
step_G = np.load('Lap2_NTK_lsgan_rb/best_G_step.npy', allow_pickle=True)
step_D = np.load('Lap2_NTK_lsgan_rb/best_D_step.npy', allow_pickle=True)

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
plt.title(r'Laplace2: Training Steps of $D$, $G$ and $rK\gamma$ (LSGAN-RB)')

lines = l2 +l3 +l1
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='best')
plt.show()

gkr = np.load('Lap2_NTK_wgan_rb/rKr_gamma_loss.npy', allow_pickle=True)
step_G = np.load('Lap2_NTK_wgan_rb/best_G_step.npy', allow_pickle=True)
step_D = np.load('Lap2_NTK_wgan_rb/best_D_step.npy', allow_pickle=True)

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
plt.title(r'Laplace2: Training Steps of $D$, $G$ and $rK\gamma$ (WGAN-RB)')

lines = l2 +l3 +l1
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='best')
plt.show()