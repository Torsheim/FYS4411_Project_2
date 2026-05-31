# FYS4411 Project 2: Neural Quantum State VMC

This repository contains a Python implementation of a Gaussian-binary restricted Boltzmann machine (RBM) used as a neural quantum state (NQS) in variational Monte Carlo (VMC).

The code is organized around the project parts:

- Part b: brute-force Metropolis sampling without interaction.
- Part c: importance sampling.
- Part d: blocking analysis.
- Part e: interacting two-particle, two-dimensional oscillator.
- Part f: optional neural-Jastrow placeholder.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run tests

```bash
pytest -q
```

## Example runs

```bash
python scripts/run_experiment.py --config configs/part_b_1p1d_2h_bruteforce.yaml
python scripts/run_experiment.py --config configs/part_c_1p1d_2h_importance.yaml
python scripts/run_experiment.py --config configs/part_e_2p2d_interacting.yaml
```

## Output

- Histories are written to `results/selected/`.
- Local-energy samples for blocking are written to `results/processed/`.
- Figures are written to `results/figures/`.
- Tables are written to `results/tables/`.
