"""Container returned by samplers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass
class SamplerResult:
    positions: Array
    local_energies: Array
    accepted_moves: int
    total_moves: int
    final_position: Array

    @property
    def acceptance_rate(self) -> float:
        if self.total_moves == 0:
            return 0.0
        return self.accepted_moves / self.total_moves

    @property
    def energy_mean(self) -> float:
        return float(np.mean(self.local_energies))

    @property
    def energy_std(self) -> float:
        if self.local_energies.size < 2:
            return 0.0
        return float(np.std(self.local_energies, ddof=1))

    @property
    def energy_error(self) -> float:
        if self.local_energies.size < 2:
            return float("nan")
        return float(self.energy_std / np.sqrt(self.local_energies.size))
