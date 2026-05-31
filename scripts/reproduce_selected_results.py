#!/usr/bin/env python3
"""Run the selected Project 2 pipeline.

This runs baseline checks, parameter sweeps, the longer interacting run,
blocking analyses, and then regenerates figures and tables.
"""

from __future__ import annotations

import subprocess
import sys

COMMANDS = [
    [sys.executable, "-m", "pytest", "-q"],

    [sys.executable, "scripts/run_experiment.py", "--config", "configs/part_b_1p1d_2h_bruteforce.yaml"],
    [sys.executable, "scripts/run_experiment.py", "--config", "configs/part_b_2p2d_no_interaction_bruteforce.yaml"],

    [sys.executable, "scripts/run_experiment.py", "--config", "configs/part_c_1p1d_2h_importance.yaml"],
    [sys.executable, "scripts/run_experiment.py", "--config", "configs/part_c_2p2d_no_interaction_importance.yaml"],

    [sys.executable, "scripts/run_blocking.py", "--config", "configs/part_d_blocking_importance.yaml"],

    [sys.executable, "scripts/run_sweep.py", "--config", "configs/sweep_learning_rate.yaml"],
    [sys.executable, "scripts/run_sweep.py", "--config", "configs/sweep_hidden_units.yaml"],

    [sys.executable, "scripts/run_experiment.py", "--config", "configs/part_e_2p2d_interacting.yaml"],
    [sys.executable, "scripts/run_experiment.py", "--config", "configs/part_e_2p2d_interacting_long.yaml"],
    [sys.executable, "scripts/run_blocking.py", "--config", "configs/part_e_blocking_interacting_long.yaml"],

    [sys.executable, "scripts/run_sweep.py", "--config", "configs/sweep_part_e_learning_rate.yaml"],
    [sys.executable, "scripts/run_sweep.py", "--config", "configs/sweep_part_e_hidden_units.yaml"],
    [sys.executable, "scripts/run_sweep.py", "--config", "configs/sweep_part_e_time_step.yaml"],

    [sys.executable, "scripts/make_figures.py"],
    [sys.executable, "scripts/make_tables.py"],
]


def main() -> None:
    for command in COMMANDS:
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
