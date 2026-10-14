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

Directly running `denn/experiments.py` bypasses the trainer and device setup
in `run_experiment.py`. In particular, merely checking
`torch.cuda.is_available()` does **not** move the models or PDE grids to the
GPU. The legacy file also imports the NTK trainer by default, so it should not
be used as a replacement for the unified entry point when selecting DEQGAN.

If direct execution is required, set the default device before
`gan_experiment()` creates the models and problem. For example, add the
following inside the `if __name__ == '__main__':` block, before the call to
`gan_experiment(...)`:

```python
device = torch.device("cuda:0")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but is unavailable")

torch.cuda.set_device(device)
torch.set_default_device(device)
```

Then run the module from the repository root:

```bash
python -u -m denn.experiments --pkey pos_ntk_gan
```

The device must be set before model and PDE construction. A reliable runtime
check prints the devices of the generator, discriminator, and both PDE grid
tensors; all four should report the same `cuda:N` value.

## Results

Training outputs are written under `experiments/runs/`. Existing experiment
results and plotting scripts are under `experiments_plot/`.

## Citation

If this code is used in academic work, cite this paper and the original
DEQGAN work.
