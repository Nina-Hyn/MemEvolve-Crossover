#!/usr/bin/env python
# coding=utf-8

"""
MemEvolveK2 - K=2 dual-parent memory evolution (MVP).

Wraps MemEvolve with fixed crossover parents: agent_kb + voyager.
Does not modify the original MemEvolve package.
"""

from .core.memory_evolver_k2 import MemoryEvolverK2
from .core.auto_evolver_k2 import AutoEvolverK2
from .phases.phase_generator_k2 import PhaseGeneratorK2
from .config import DEFAULT_PARENT_PROVIDERS, PRIMARY_PARENT, SECONDARY_PARENT

__all__ = [
    "MemoryEvolverK2",
    "AutoEvolverK2",
    "PhaseGeneratorK2",
    "DEFAULT_PARENT_PROVIDERS",
    "PRIMARY_PARENT",
    "SECONDARY_PARENT",
]
