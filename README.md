# GAN-NTK-PINN

Official code for the NeurIPS 2026 paper **“When and Why Adversarial
Training Improves PINNs: A Neural Tangent Kernel Perspective.”**

The repository contains seven trainers for six PDEs:

- **DEQGAN:** 6 base experiments using `denn.algos_del`.
- **NTK-GAN:** GAN, LSGAN, and WGAN for each PDE, with or without rollback
  (18 configurations per variant).
- **Appendix H.3.2:** one Laplace 1 rollback ablation using the original
  DEQGAN hyperparameters.

Use `run_experiment.py` as the normal entry point. It selects the trainer,
configuration, and CPU/GPU device without requiring edits to the research
source in `denn/experiments.py`.

## Installation

The reported environment uses Python 3.8.18 and PyTorch 2.1.2
(`torch 2.1.2+cu121` on the tested CUDA system).

The recommended setup is Conda:

```bash
conda env create -f environment.yml
conda activate pde-gan-ntk
python -m pip install -e .
python -m pip check
```

For a `venv` installation, create a Python 3.8 environment, install the
appropriate PyTorch 2.1.2 build for the machine, and then run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
```

Confirm that PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Run experiments

Run commands from the repository root. Device choices are `auto`, `cpu`,
`cuda`, or `cuda:N`. `auto` selects `cuda:0` when CUDA is available and
otherwise uses CPU. An explicit unavailable CUDA device produces an error
instead of silently falling back to CPU.

### One DEQGAN experiment

```bash
python -u run_experiment.py \
  --trainer deqgan \
  --pkey rea \
  --device cuda:0
```

### One NTK-GAN experiment

```bash
python -u run_experiment.py \
  --trainer ntk \
  --pkey rea_ntk_gan \
  --device cuda:0
```

Change the suffix to select the adversarial loss:

```bash
python -u run_experiment.py --trainer ntk --pkey rea_ntk_lsgan --device cuda:0
python -u run_experiment.py --trainer ntk --pkey rea_ntk_wgan  --device cuda:0
```

Add `--disable-rollback` for the corresponding no-rollback baseline. This
sets the no-rollback controls at runtime and writes to a separate directory;
the YAML file does not need to be edited.

```bash
python -u run_experiment.py \
  --trainer ntk \
  --pkey rea_ntk_gan \
  --disable-rollback \
  --device cuda:0
```

### Batch commands

```bash
# Six DEQGAN experiments
bash scripts/run_deqgan_6.sh cuda:0

# Eighteen NTK-GAN/LSGAN/WGAN experiments with rollback
bash scripts/run_ntk_18.sh cuda:0

# Eighteen NTK-GAN/LSGAN/WGAN experiments without rollback
bash scripts/run_ntk_18_norb.sh cuda:0
```

The YAML files in `denn/config/` provide the full epoch counts, seeds, model
sizes, learning rates, and output directory names. Use `--niters 3` for a
short launchability check without changing the YAML file.

The formal outer-epoch counts are:

| PDE | DEQGAN | NTK-GAN / LSGAN / WGAN |
|---|---:|---:|
| Poisson | 3000 | 3000 |
| Laplace 1 | 3000 | 3000 |
| Laplace 2 | 4000 | 4000 |
| Klein–Gordon | 4000 | 4000 |
| Reaction–diffusion | 4500 | 4500 |
| Viscous Burgers | 3300 | 3300 |

All published YAML configurations use `seed: 0` and `modelseed: 0`. The
standard NTK configurations use `g_lr: 0.001`, `d_lr: 0.001`, Adam betas
`[0.9, 0.999]`, and the rollback controls shown below. A `--niters` override
changes only the outer-iteration count; the configured network, PDE grid,
optimizer, loss, and inner generator/discriminator update counts are retained.

## Problems and configuration keys

| PDE | DEQGAN key | NTK keys |
|---|---|---|
| Poisson | `pos` | `pos_ntk_{gan,lsgan,wgan}` |
| Laplace 1 | `lap1` | `lap1_ntk_{gan,lsgan,wgan}` |
| Laplace 2 | `lap2` | `lap2_ntk_{gan,lsgan,wgan}` |
| Klein–Gordon | `kle` | `kle_ntk_{gan,lsgan,wgan}` |
| Reaction–diffusion | `rea` | `rea_ntk_{gan,lsgan,wgan}` |
| Viscous Burgers | `burv` | `burv_ntk_{gan,lsgan,wgan}` |

The `--pkey` value is the YAML filename without `.yaml`. Base YAML files use
the DEQGAN trainer; `*_ntk_*` files use the NTK trainer.

Rollback and no-rollback use the following controls:

| Setting | Rollback | No rollback |
|---|---:|---:|
| `G_iters` | `20` | `1` |
| `D_iters` | `20` | `1` |
| `ntk` | `true` | `false` |
| `energy` | `true` | `false` |

For the Appendix H.3.2 Laplace 1 ablation:

```bash
python -u run_experiment.py \
  --trainer ntk \
  --pkey lap1_ntk_gan_deqgan \
  --device cuda:0
```

For the Appendix H.3.3 Klein–Gordon and Laplace 2 ablations, start from
`kle_ntk_gan.yaml` or `lap2_ntk_gan.yaml` and change only the update counts
and rollback switches required by the ablation.

## Direct use of `denn/experiments.py`

Direct execution and `run_experiment.py` use the same device resolver. The
device is selected before models and PDE grids are constructed, and an
explicit unavailable CUDA device raises an error rather than falling back to
CPU. `run_experiment.py` passes its selected trainer explicitly, so it does not
replace the direct entry point's module-level NTK trainer. Run the module from
the repository root:

```bash
python -u -m denn.experiments --pkey pos_ntk_gan --device cuda:0
```

For an isolated direct-entry smoke check, override the iteration count and use
an unused output directory so a formal result is not overwritten:

```bash
python -u -m denn.experiments \
  --pkey pos_ntk_gan \
  --device cuda:0 \
  --niters 1 \
  --dirname /tmp/pos_ntk_gan_direct_smoke
```

The legacy module still imports the NTK trainer by default, so use
`run_experiment.py --trainer deqgan` when selecting DEQGAN. A runtime check
prints the devices of the generator, discriminator, and both PDE grid tensors;
all four should report the selected `cuda:N` value.

## GPU launchability validation

The current source was validated on 2026-10-14 with Python 3.8.18,
PyTorch 2.1.2+cu121, and two NVIDIA GeForce RTX 4090 GPUs. Each README
configuration completed one real outer training iteration using its configured
PDE, model, optimizer, loss, and inner update counts:

| Family | Passed | GPU assignment |
|---|---:|---|
| DEQGAN | 6/6 | `cuda:0`, `cuda:1` |
| NTK with rollback | 18/18 | `cuda:0`, `cuda:1` |
| NTK without rollback | 18/18 | `cuda:0`, `cuda:1` |
| Appendix H.3.2 | 1/1 | `cuda:1` |
| **Total** | **43/43** | **21 on `cuda:0`, 22 on `cuda:1`** |

For all 43 runs, the generator, discriminator, and both PDE grid tensors
reported the requested GPU. All runs exited with status zero and produced a
saved configuration, finite one-step metric arrays, and both result plots; all
37 NTK runs also produced a loadable finite generator checkpoint. The status
files and detailed audit are in
`logs/readme_validation_smoke_20261014_shared_device_recheck4_cuda_smoke/` and
`results/readme_gpu_validation_20261014.md`.

This is a launchability/GPU-routing check, not evidence that all formal
3000–4500-epoch experiments have converged or reproduced the paper metrics.
For a full-run claim, run the batch commands without `--niters`, require every
process to exit successfully, and audit the configured-length finite metric
arrays, plots, saved configuration, and NTK checkpoint.

## Results

Training outputs are written under `experiments/runs/`. Existing experiment
results and plotting scripts are under `experiments_plot/`.

## Citation

If this code is used in academic work, cite this paper and the original
DEQGAN work.
