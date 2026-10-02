import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def moving_average(x, w=50):
    return np.convolve(x, np.ones(w)/w, mode='valid')

gkr_1000_1 = np.load('Lap1_del_G_1000_gkr_new/metrics_rKr_gamma.npy', allow_pickle=True)
gkr_100_1 = np.load('Lap1_del_G_100_gkr_new/metrics_rKr_gamma.npy', allow_pickle=True)
gkr_10_1 = np.load('Lap1_del_G_10_gkr_new/metrics_rKr_gamma.npy', allow_pickle=True)
gkr_1_1 = np.load('Lap1_del_1_1_gkr_new/metrics_rKr_gamma.npy', allow_pickle=True)
gkr_1_10 = np.load('Lap1_del_D_10_gkr_new/metrics_rKr_gamma.npy', allow_pickle=True)
gkr_1_100 = np.load('Lap1_del_D_100_gkr_new/metrics_rKr_gamma.npy', allow_pickle=True)
gkr_1_1000 = np.load('Lap1_del_D_1000_gkr_new/metrics_rKr_gamma.npy', allow_pickle=True)

mse01 = np.load('Lap1_del_G_10_gkr_new/train_mse.npy', allow_pickle=True)
mse001 = np.load('Lap1_del_G_100_gkr_new/train_mse.npy', allow_pickle=True)
mse0001 = np.load('Lap1_del_G_1000_gkr_new/train_mse.npy', allow_pickle=True)
mse1 = np.load('Lap1_del_1_1_gkr_new/train_mse.npy', allow_pickle=True)
mse10 = np.load('Lap1_del_D_10_gkr_new/train_mse.npy', allow_pickle=True)
mse100 = np.load('Lap1_del_D_100_gkr_new/train_mse.npy', allow_pickle=True)
mse1000 = np.load('Lap1_del_D_1000_gkr_new/train_mse.npy', allow_pickle=True)

val01 = np.load('Lap1_del_G_10_gkr_new/val_mse.npy', allow_pickle=True)
val001 = np.load('Lap1_del_G_100_gkr_new/val_mse.npy', allow_pickle=True)
val0001 = np.load('Lap1_del_G_1000_gkr_new/val_mse.npy', allow_pickle=True)
val1 = np.load('Lap1_del_1_1_gkr_new/val_mse.npy', allow_pickle=True)
val10 = np.load('Lap1_del_D_10_gkr_new/val_mse.npy', allow_pickle=True)
val100 = np.load('Lap1_del_D_100_gkr_new/val_mse.npy', allow_pickle=True)
val1000 = np.load('Lap1_del_D_1000_gkr_new/val_mse.npy', allow_pickle=True)

mse_NTK_1_1_none_gan = np.load('Lap1_NTK_none_1_1_gan/train_mse.npy', allow_pickle=True)
mse_NTK_1_1_none_wgan = np.load('Lap1_NTK_none_1_1_wgan/train_mse.npy', allow_pickle=True)
mse_NTK_1_1_none_lsgan = np.load('Lap1_NTK_none_1_1_lsgan/train_mse.npy', allow_pickle=True)
mse_NTK_wgan_rb = np.load('Lap1_NTK_wgan_rb/train_mse.npy', allow_pickle=True)
mse_NTK_lsgan_rb = np.load('Lap1_NTK_lsgan_rb/train_mse.npy', allow_pickle=True)
mse_NTK_gan_rb = np.load('Lap1_NTK_gan_rb/train_mse.npy', allow_pickle=True)

val_NTK_1_1_none_gan = np.load('Lap1_NTK_none_1_1_gan/val_mse.npy', allow_pickle=True)
val_NTK_1_1_none_wgan = np.load('Lap1_NTK_none_1_1_wgan/val_mse.npy', allow_pickle=True)
val_NTK_1_1_none_lsgan = np.load('Lap1_NTK_none_1_1_lsgan/val_mse.npy', allow_pickle=True)
val_NTK_wgan_rb = np.load('Lap1_NTK_wgan_rb/val_mse.npy', allow_pickle=True)
val_NTK_lsgan_rb = np.load('Lap1_NTK_lsgan_rb/val_mse.npy', allow_pickle=True)
val_NTK_gan_rb = np.load('Lap1_NTK_gan_rb/val_mse.npy', allow_pickle=True)

'''print('DEQGAN',val1[-1])
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
print('GAN-RB',mse_NTK_gan_rb[-1])'''

gkr1 = np.load('Lap1_del_1_1_gkr_new/rKr_gamma_loss.npy', allow_pickle=True)
gkr_NTK_1_1_none_gan = np.load('Lap1_NTK_none_1_1_gan/rKr_gamma_loss.npy', allow_pickle=True)
gkr_NTK_1_1_none_wgan = np.load('Lap1_NTK_none_1_1_wgan/rKr_gamma_loss.npy', allow_pickle=True)
gkr_NTK_1_1_none_lsgan = np.load('Lap1_NTK_none_1_1_lsgan/rKr_gamma_loss.npy', allow_pickle=True)
gkr_NTK_wgan_rb = np.load('Lap1_NTK_wgan_rb/rKr_gamma_loss.npy', allow_pickle=True)
gkr_NTK_lsgan_rb = np.load('Lap1_NTK_lsgan_rb/rKr_gamma_loss.npy', allow_pickle=True)
gkr_NTK_gan_rb = np.load('Lap1_NTK_gan_rb/rKr_gamma_loss.npy', allow_pickle=True)

energy1 = np.load('Lap1_del_1_1_gkr_new/metrics_residual_energy.npy', allow_pickle=True)
energy_NTK_1_1_none_gan = np.load('Lap1_NTK_none_1_1_gan/residual_energy_loss.npy', allow_pickle=True)
energy_NTK_1_1_none_wgan = np.load('Lap1_NTK_none_1_1_wgan/residual_energy_loss.npy', allow_pickle=True)
energy_NTK_1_1_none_lsgan = np.load('Lap1_NTK_none_1_1_lsgan/residual_energy_loss.npy', allow_pickle=True)
energy_NTK_wgan_rb = np.load('Lap1_NTK_wgan_rb/residual_energy_loss.npy', allow_pickle=True)
energy_NTK_lsgan_rb = np.load('Lap1_NTK_lsgan_rb/residual_energy_loss.npy', allow_pickle=True)
energy_NTK_gan_rb = np.load('Lap1_NTK_gan_rb/residual_energy_loss.npy', allow_pickle=True)



gkr1 = moving_average(gkr1, w=300)
gkr_NTK_1_1_none_gan = moving_average(gkr_NTK_1_1_none_gan, w=300)
gkr_NTK_1_1_none_wgan = moving_average(gkr_NTK_1_1_none_wgan, w=300)
gkr_NTK_1_1_none_lsgan = moving_average(gkr_NTK_1_1_none_lsgan, w=300)
gkr_NTK_wgan_rb = moving_average(gkr_NTK_wgan_rb, w=300)
gkr_NTK_lsgan_rb = moving_average(gkr_NTK_lsgan_rb, w=300)
gkr_NTK_gan_rb = moving_average(gkr_NTK_gan_rb, w=300)


linestyles = ['solid', 'dashed'] #确保在一个图中有很多组合可以区分不同线条
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
plt.title(r'Laplace1: Comparison of GAN-based and NTK-guided Methods')
plt.legend(loc = 'upper right')
plt.yscale('log')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(gkr_1_100, label='G:D = 1:100', linewidth=linewidth, color=colors[5],linestyle=linestyles[0])
plt.plot(gkr_1_10, label='G:D = 1:10', linewidth=linewidth, color=colors[1],linestyle=linestyles[0])
plt.plot(gkr_1_1000, label='G:D = 1:1000', linewidth=linewidth, color=colors[3],linestyle=linestyles[0])
plt.plot(gkr_1000_1, label='G:D = 1000:1', linewidth=linewidth, color=colors[6],linestyle=linestyles[0])
plt.xlabel('Iteration')
plt.ylabel(r'$rK\gamma$')
plt.title(r'Laplace: $rK\gamma$ Loss (Failure)')
plt.legend()
#plt.yscale('log')
plt.yscale('symlog')
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,0,2,3]
plt.legend([handles[i] for i in order],
           [labels[i] for i in order])
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(gkr_1_1, label='G:D = 1:1', linewidth=linewidth, color=colors[0],linestyle=linestyles[0])
plt.plot(gkr_100_1, label='G:D = 100:1', linewidth=linewidth, color=colors[4],linestyle=linestyles[0])
plt.plot(gkr_10_1, label='G:D = 10:1', linewidth=linewidth, color=colors[2],linestyle=linestyles[0])
plt.xlabel('Iteration')
plt.ylabel(r'$rK\gamma$')
plt.title(r'Laplace: $rK\gamma$ Loss (Success)')
plt.legend()
#plt.yscale('log')
plt.yscale('symlog')
ax = plt.gca()
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,2,1]
plt.legend([handles[i] for i in order],
           [labels[i] for i in order])
plt.show()

# X-axis labels
labels = ["1000:1", "100:1", "10:1", "1:1", "1:10", "1:100", "1:1000"]

# Example y-values for demonstration
mse = [mse0001[-1], mse001[-1], mse01[-1], mse1[-1], mse10[-1], mse100[-1], mse1000[-1]]
val = [val0001[-1], val001[-1], val01[-1], val1[-1], val10[-1], val100[-1], val1000[-1]]
plt.figure(figsize=(8, 6))
plt.plot(labels, val, marker='x', color = 'blue', label = 'VAL', linestyle='--')
plt.plot(labels, mse, marker='o',color = 'orange', label = 'MSE', linestyle='-')
plt.xlabel("G : D ratio")
plt.ylabel("Loss of MSE/VAL")
plt.legend()
plt.yscale('log')
plt.title("Laplace: MSE/VAL curve under different G:D ratios")
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
plt.title(r'Laplace1: Comparison of GAN-based and NTK-guided Methods')
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
plt.title(r'Laplace1: Comparison of GAN-based and NTK-guided Methods')
plt.legend()
plt.yscale('log')
plt.tight_layout()
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



gkr = np.load('Lap1_NTK_lsgan_rb/rKr_gamma_loss.npy', allow_pickle=True)
step_G = np.load('Lap1_NTK_lsgan_rb/best_G_step.npy', allow_pickle=True)
step_D = np.load('Lap1_NTK_lsgan_rb/best_D_step.npy', allow_pickle=True)

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
plt.title(r'Laplace1: Training Steps of $D$, $G$ and $rK\gamma$ (LSGAN-RB)')
lines = l2 +l3 +l1
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='best')
plt.show()



gkr = np.load('Lap1_NTK_wgan_rb/rKr_gamma_loss.npy', allow_pickle=True)
step_G = np.load('Lap1_NTK_wgan_rb/best_G_step.npy', allow_pickle=True)
step_D = np.load('Lap1_NTK_wgan_rb/best_D_step.npy', allow_pickle=True)

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
plt.title(r'Laplace1: Training Steps of $D$, $G$ and $rK\gamma$ (WGAN-RB)')
lines = l2 +l3 +l1
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='best')
plt.show()