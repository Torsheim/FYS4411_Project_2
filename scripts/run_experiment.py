#!/usr/bin/env python3
"""Run one VMC experiment from a YAML config file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from nqs_vmc.config import experiment_config_from_dict
from nqs_vmc.io.checkpointing import save_model
from nqs_vmc.io.config_loader import load_config
from nqs_vmc.io.result_writer import write_history_csv, write_json, write_local_energies_csv
from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM
from nqs_vmc.optimization.optimizers import create_optimizer
from nqs_vmc.optimization.trainer import VMCTrainer
from nqs_vmc.physics.hamiltonian import HarmonicOscillatorHamiltonian
from nqs_vmc.sampling.importance_sampling import ImportanceSampler
from nqs_vmc.sampling.metropolis import MetropolisSampler


def build_objects(config_path: str | Path):
    raw = load_config(config_path)
    config = experiment_config_from_dict(raw)

    hamiltonian = HarmonicOscillatorHamiltonian(
        num_particles=config.system.num_particles,
        dimensions=config.system.dimensions,
        omega=config.system.omega,
        include_interaction=config.system.include_interaction,
        coulomb_epsilon=config.system.coulomb_epsilon,
    )

    model = GaussianBinaryRBM.random_initialization(
        num_visible=hamiltonian.num_visible,
        num_hidden=config.model.num_hidden,
        sigma=config.model.sigma,
        scale=config.model.parameter_scale,
        seed=config.seed,
    )

    if config.sampler.type.lower() in {"metropolis", "bruteforce", "brute_force"}:
        sampler = MetropolisSampler(
            model=model,
            hamiltonian=hamiltonian,
            step_size=config.sampler.step_size,
            seed=config.seed + 1000,
        )
    elif config.sampler.type.lower() in {"importance", "importance_sampling", "langevin"}:
        sampler = ImportanceSampler(
            model=model,
            hamiltonian=hamiltonian,
            time_step=config.sampler.time_step,
            diffusion=config.sampler.diffusion,
            seed=config.seed + 1000,
        )
    else:
        raise ValueError(f"Unknown sampler type: {config.sampler.type}")

    optimizer = create_optimizer(
        name=config.optimization.optimizer,
        learning_rate=config.optimization.learning_rate,
        beta1=config.optimization.beta1,
        beta2=config.optimization.beta2,
        epsilon=config.optimization.epsilon,
    )

    return raw, config, hamiltonian, model, sampler, optimizer


def run(config_path: str | Path, verbose: bool = False) -> dict[str, float]:
    raw, config, hamiltonian, model, sampler, optimizer = build_objects(config_path)
    trainer = VMCTrainer(model=model, sampler=sampler, optimizer=optimizer)

    history, final_position, final_energies = trainer.train(
        n_iterations=config.optimization.n_iterations,
        n_samples=config.sampler.n_samples,
        n_burn_in=config.sampler.n_burn_in,
        thin=config.sampler.thin,
        verbose=verbose,
    )

    write_history_csv(history, config.output.history_path)
    write_local_energies_csv(final_energies, config.output.samples_path)
    save_model(model, config.output.checkpoint_path)

    summary = {
        "config_path": str(config_path),
        "history_path": config.output.history_path,
        "samples_path": config.output.samples_path,
        "checkpoint_path": config.output.checkpoint_path,
        "final_energy": float(history[-1]["energy_mean"]),
        "final_error": float(history[-1]["energy_error"]),
        "acceptance_rate": float(history[-1]["acceptance_rate"]),
        "num_particles": float(config.system.num_particles),
        "dimensions": float(config.system.dimensions),
        "num_hidden": float(config.model.num_hidden),
        "learning_rate": float(config.optimization.learning_rate),
        "sampler": config.sampler.type,
        "include_interaction": bool(config.system.include_interaction),
        "exact_non_interacting_energy": hamiltonian.exact_non_interacting_energy(),
        "exact_interacting_2p2d_energy": hamiltonian.exact_interacting_2p2d_energy(),
    }
    write_json(summary, Path(config.output.history_path).with_suffix(".json"))

    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress printing.")
    args = parser.parse_args()
    run(args.config, verbose=not args.quiet)


if __name__ == "__main__":
    main()
