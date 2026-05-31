"""Configuration dataclasses and defaults.

The scripts use YAML files, but this module gives one central place for the
schema and default values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SystemConfig:
    num_particles: int = 1
    dimensions: int = 1
    omega: float = 1.0
    include_interaction: bool = False
    coulomb_epsilon: float = 1.0e-12

    @property
    def num_visible(self) -> int:
        return self.num_particles * self.dimensions


@dataclass(frozen=True)
class ModelConfig:
    num_hidden: int = 2
    sigma: float = 1.0
    parameter_scale: float = 0.01


@dataclass(frozen=True)
class SamplerConfig:
    type: str = "metropolis"
    step_size: float = 1.0
    time_step: float = 0.05
    diffusion: float = 0.5
    n_samples: int = 5000
    n_burn_in: int = 1000
    thin: int = 1


@dataclass(frozen=True)
class OptimizationConfig:
    optimizer: str = "adam"
    learning_rate: float = 0.03
    n_iterations: int = 100
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8


@dataclass(frozen=True)
class OutputConfig:
    history_path: str = "results/selected/history.csv"
    samples_path: str = "results/processed/local_energies.csv"
    checkpoint_path: str = "results/checkpoints/model.npz"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 2026
    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _section(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Config section '{name}' must be a mapping, got {type(value)!r}.")
    return dict(value)


def experiment_config_from_dict(data: Mapping[str, Any]) -> ExperimentConfig:
    """Create an :class:`ExperimentConfig` from a nested dictionary."""
    return ExperimentConfig(
        seed=int(data.get("seed", 2026)),
        system=SystemConfig(**_section(data, "system")),
        model=ModelConfig(**_section(data, "model")),
        sampler=SamplerConfig(**_section(data, "sampler")),
        optimization=OptimizationConfig(**_section(data, "optimization")),
        output=OutputConfig(**_section(data, "output")),
    )
