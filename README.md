# GAN-NTK-PINN code
This is the official code repository for our NeurIPS 2026 paper, **“When and Why Adversarial Training Improves PINNs: A Neural Tangent Kernel Perspective.”**

This repository is a minimal, reproducible release of two experiment families
for six partial differential equations. It contains only the training,
configuration, reference-solution, plotting, and result-summary code required
by these experiments.

1. **DEQGAN:** six experiments using `denn.algos_del` and the base YAML files (`pos`, `lap1`, `lap2`, `kle`, `rea`, and `burv`).
2. **NTK-GAN-RB:** eighteen experiments using `denn.algos_NTK`, with GAN, LSGAN, and WGAN configurations for each of the six PDEs.
3. **NTK-GAN-noRB** eighteen experiments using `denn.algos_NTK`, with GAN, LSGAN, and WGAN configurations for each of the six PDEs. For specific adjustments, refer to Understanding the YAML configurations.
4. **Result Plots:** All experimental results and the corresponding plotting scripts can be found in the `experiments_plot` directory.

The original `denn/experiments.py` is retained as research source code. Use
`run_experiment.py` to choose a trainer without editing that file between runs.

## Included problems and configurations

| PDE | Base DEQGAN key | NTK keys |
|---|---|---:|
| Poisson | `pos` | `pos_ntk_{gan,lsgan,wgan}` |
| Laplace 1 | `lap1` | `lap1_ntk_{gan,lsgan,wgan}` |
| Laplace 2 | `lap2` | `lap2_ntk_{gan,lsgan,wgan}` |
| klein-gordon | `kle` | `kle_ntk_{gan,lsgan,wgan}` |
| reaction-difussion | `rea` | `rea_ntk_{gan,lsgan,wgan}` | 
| Viscous Burgers | `burv` | `burv_ntk_{gan,lsgan,wgan}` |

## Installation

The reported runs were produced with Python 3.8.18 and PyTorch 2.1.2
(`torch 2.1.2+cu121` on the tested CUDA system).

### Conda installation

```bash
conda env create -f environment.yml
conda activate pde-gan-ntk
python -m pip install -e .
python -m pip check
```

Verify the GPU before running the full experiments:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### `venv`/pip installation

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CUDA 12.1 build used for the reported experiments:
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
```

For a CPU-only installation, install the appropriate PyTorch 2.1.2 wheel for
your platform before installing `requirements.txt`.

## Running DEQGAN (6 experiments)

Run one configuration:

```bash
python -u run_experiment.py \
  --trainer deqgan \
  --pkey rea \
  --device cuda:0
```

Run all six base configurations sequentially:

```bash
bash scripts/run_deqgan_6.sh cuda:0
```

The six keys executed by the script are:

```text
pos lap1 lap2 kle rea burv
```

The full hyperparameters, random seeds, epoch counts, model widths, and output
directory names are read from `denn/config/<pkey>.yaml`.

## Running NTK-GAN/LSGAN/WGAN (18 experiments)

Run one configuration:

```bash
python -u run_experiment.py \
  --trainer ntk \
  --pkey rea_ntk_gan \
  --device cuda:0
```

Switch the method by changing only the key:

```bash
python -u run_experiment.py --trainer ntk --pkey rea_ntk_lsgan --device cuda:0
python -u run_experiment.py --trainer ntk --pkey rea_ntk_wgan  --device cuda:0
```

Run all eighteen configurations sequentially:

```bash
bash scripts/run_ntk_18.sh cuda:0
```

Run all eighteen GAN-without-rollback baselines without editing or
overwriting the rollback configurations:

```bash
bash scripts/run_ntk_18_norb.sh cuda:0
```


## Understanding the YAML configurations

Configuration files are stored in `denn/config/`. The `--pkey` argument selects
the YAML filename without the `.yaml` extension, while `--trainer` selects the
training implementation.

| Configuration example | Purpose | Trainer |
|---|---|---|
| `lap1.yaml` | Original DEQGAN hyperparameters for Laplace 1 | `deqgan` |
| `lap1_ntk_gan.yaml` | GAN with the proposed rollback method | `ntk` |
| `lap1_ntk_gan_deqgan.yaml` | Rollback with the original DEQGAN hyperparameters, as in Appendix H.3.2 | `ntk` |

The same naming convention applies to the other PDEs. Base files such as
`lap1.yaml` contain the original DEQGAN hyperparameters. Files such as
`lap1_ntk_gan.yaml` contain the hyperparameters for GAN with rollback; no
special learning-rate adjustment is required to use the rollback method.

### GAN with and without rollback

To run GAN with rollback, use the corresponding `*_ntk_gan.yaml` file with
`--trainer ntk`. The relevant settings in its `training` section are:

```yaml
training:
  G_iters: 20
  D_iters: 20
  ntk: true
  energy: true
```

To run GAN without rollback, edit these entries in the same configuration:

```yaml
training:
  G_iters: 1
  D_iters: 1
  ntk: false
  energy: false
```

These are excerpts: replace the corresponding entries in the existing
`training` section rather than adding a second `training` section.
Keep the remaining hyperparameters, including the learning rates and GAN
loss selection, unchanged for this comparison. Continue using `--trainer ntk`:
disabling rollback does not require switching to the DEQGAN trainer.

| Parameter | Meaning | GAN with rollback | GAN without rollback |
|---|---|---:|---:|
| `G_iters` | Number of generator update steps per training cycle | `20` | `1` |
| `D_iters` | Number of discriminator update steps per training cycle | `20` | `1` |
| `ntk` | Enable the NTK-based indicator | `true` | `false` |
| `energy` | Enable residual-energy monitoring | `true` | `false` |

### Appendix H.3.2: Laplace 1 ablation

To apply rollback to DEQGAN, use the original DEQGAN hyperparameters from
`lap1.yaml` in the rollback configuration, following
`lap1_ntk_gan_deqgan.yaml`. Preserve the rollback controls and monitoring
settings required by that experiment. This tests rollback under the original
DEQGAN hyperparameter setting; it does not require a new learning-rate search.

```bash
python -u run_experiment.py \
  --trainer ntk \
  --pkey lap1_ntk_gan_deqgan \
  --device cuda:0
```

### Appendix H.3.3: Klein–Gordon and Laplace 2 ablations

Start from the corresponding configurations (`kle_ntk_gan.yaml` and
`lap2_ntk_gan.yaml`) and adjust the update counts and rollback switches to
match the ablation being reproduced. Set `G_iters` and `D_iters` to the required
update counts, and adjust the two Boolean options (
`ntk`, and `energy`) according to the experiment. For the
complete GAN-without-rollback baseline, use all six values in the
"GAN without rollback" column above. Keep the other hyperparameters unchanged
unless the particular ablation explicitly varies them.

## Citation

If this code is used in academic work, cite our paper and the original DEQGAN work.
