#!/usr/bin/env python
# coding=utf-8

"""
K=2 AutoEvolver: patches MemoryEvolver during evolution to use dual-parent crossover.

Fixes:
- Analyze uses current round base provider (from state), not static default_provider
- Generate uses agent_kb + voyager as fixed crossover parents
"""

from typing import Any, Dict, List, Optional

from MemEvolve.core.auto_evolver import AutoEvolver
from MemEvolve.config import ANALYSIS_MAX_STEPS

from ..config import DEFAULT_PARENT_PROVIDERS
from .memory_evolver_k2 import MemoryEvolverK2


class AutoEvolverK2(AutoEvolver):
    """Auto-evolver with K=2 dual-parent crossover (agent_kb + voyager)."""

    def __init__(
        self,
        parent_providers: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.parent_providers = parent_providers or DEFAULT_PARENT_PROVIDERS
        print(f"[AutoEvolverK2] Crossover parents: {self.parent_providers}")

    def _run_memory_evolver(
        self, round_dir, checkpoint: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Delegate to base implementation but swap in MemoryEvolverK2 with:
        - analyze() -> current round base provider
        - generate() -> dual-parent crossover templates
        """
        import MemEvolve.core.auto_evolver as auto_mod

        original_cls = auto_mod.MemoryEvolver
        base_provider = self.state.get("best_provider", self.default_provider)
        parent_providers = self.parent_providers

        class _K2EvolverAdapter(MemoryEvolverK2):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("parent_providers", parent_providers)
                super().__init__(*args, **kwargs)

            def analyze(
                self,
                task_logs_dir,
                default_provider=None,
                parent_providers=None,
                max_steps=ANALYSIS_MAX_STEPS,
            ):
                return super().analyze(
                    task_logs_dir=task_logs_dir,
                    default_provider=base_provider,
                    parent_providers=parent_providers or self.parent_providers,
                    max_steps=max_steps,
                )

        auto_mod.MemoryEvolver = _K2EvolverAdapter
        try:
            return super()._run_memory_evolver(round_dir, checkpoint)
        finally:
            auto_mod.MemoryEvolver = original_cls
