# NTK-GAN Experiments for Six PDEs

This repository is a minimal, reproducible release of two experiment families
for six partial differential equations. It contains only the training,
configuration, reference-solution, plotting, and result-summary code required
by these experiments.

1. **DEQGAN:** six experiments using `denn.algos_del` and the base YAML files
   (`pos`, `lap1`, `lap2`, `wav`, `hea`, and `burv`).
2. **NTK-GAN:** eighteen experiments using `denn.algos_NTK`, with GAN, LSGAN,
   and WGAN configurations for each of the six PDEs.

The original `denn/experiments.py` is retained as research source code. Use
`run_experiment.py` to choose a trainer without editing that file between runs.

## Included problems and configurations

| PDE | Base DEQGAN key | NTK keys | Full epochs |
|---|---|---|---:|
| Poisson | `pos` | `pos_ntk_{gan,lsgan,wgan}` | 3000 |
| Laplace 1 | `lap1` | `lap1_ntk_{gan,lsgan,wgan}` | 3000 |
| Laplace 2 | `lap2` | `lap2_ntk_{gan,lsgan,wgan}` | 4000 |
| klein-gordon | `wav` | `wav_ntk_{gan,lsgan,wgan}` | 4000 |
| reaction-difussion | `hea` | `hea_ntk_{gan,lsgan,wgan}` | 4500 |
| Viscous Burgers | `burv` | `burv_ntk_{gan,lsgan,wgan}` | 3300 |

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
  --pkey hea \
  --device cuda:0
```

Run all six base configurations sequentially:

```bash
bash scripts/run_deqgan_6.sh cuda:0
```

The six keys executed by the script are:

```text
pos lap1 lap2 wav hea burv
```

The full hyperparameters, random seeds, epoch counts, model widths, and output
directory names are read from `denn/config/<pkey>.yaml`.

## Running NTK-GAN/LSGAN/WGAN (18 experiments)

Run one configuration:

```bash
python -u run_experiment.py \
  --trainer ntk \
  --pkey hea_ntk_gan \
  --device cuda:0
```

Switch the method by changing only the key:

```bash
python -u run_experiment.py --trainer ntk --pkey hea_ntk_lsgan --device cuda:0
python -u run_experiment.py --trainer ntk --pkey hea_ntk_wgan  --device cuda:0
```

Run all eighteen configurations sequentially:

```bash
bash scripts/run_ntk_18.sh cuda:0
```

## Citation

If this code is used in academic work, cite our paper and the original DEQGAN work.
