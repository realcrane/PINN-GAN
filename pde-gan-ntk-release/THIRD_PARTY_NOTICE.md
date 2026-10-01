# Third-party source and provenance notice

This research package was prepared from a working tree derived from:

- **Project:** `dylanrandle/denn`
- **Repository:** https://github.com/dylanrandle/denn
- **Upstream revision inspected:** `70d3d7e6e8715524d2181dddd6d228c7ec27bc4f`
- **Original authors named by the upstream package:** Dylan L. Randle and
  Blake Bullwinkel
- **Associated DEQGAN authors:** Blake Bullwinkel, Dylan Randle, Pavlos
  Protopapas, and David Sondak

The upstream revision above, inspected on 2026-10-01, contained no `LICENSE`, `COPYING`, or
`NOTICE` file, and the GitHub repository page displayed no recognized license.
Consequently, this package does not assign a new license to inherited or
derived upstream source files. Anyone planning public redistribution should
first obtain permission or a license clarification from the upstream authors.

## Provenance of retained research source

The following local files have upstream counterparts and must be treated as
inherited or derived work:

- `denn/experiments.py`
- `denn/models.py`
- `denn/pde_problems.py`
- `denn/problem.py`
- `denn/utils.py`
- `denn/config/config.py`
- `denn/burgers/fft_burgers.py`
- `denn/allen_cahn/fft_allen_cahn.py`
- the six base PDE YAML configurations where upstream counterparts exist

At the inspected revisions, `denn/problem.py` and `denn/config/config.py` were
byte-identical to the upstream files. Other listed files contained local
changes relative to the inspected upstream `master` branch.

The following experiment files were not present under the same names in the
inspected upstream tree. Their detailed authorship and any third-party sources
should be confirmed by the repository maintainer before publication:

- `denn/algos_del.py`
- `denn/algos_NTK.py`
- the `lap1` and `lap2` base configurations
- the eighteen `*_ntk_{gan,lsgan,wgan}.yaml` configurations

The following release-support files were newly added to make the selected
experiments installable and reproducible without altering the retained
training modules:

- `run_experiment.py`
- `scripts/run_deqgan_6.sh`
- `scripts/run_ntk_18.sh`
- `scripts/summarize_results.py`
- `requirements.txt`, `environment.yml`, and the packaging metadata
- the compact reference result tables

## Excluded material

The release intentionally excludes unrelated ODE, RANS, finite-difference,
Ray Tune, pretrained-model, notebook, SLURM, legacy numerical, test-cache, and
large generated-artifact trees. Exclusion indicates that those files are not
required by the two documented experiment families; it is not a claim of
authorship over the retained files.
