#!/usr/bin/env python
# coding=utf-8

"""
K=2 MemoryEvolver: analyze with current base provider, generate with dual-parent crossover.
"""

from datetime import datetime
from typing import Dict, List, Optional
import json

from MemEvolve.core.memory_evolver import MemoryEvolver
from MemEvolve.config import ANALYSIS_MAX_STEPS, CREATIVITY_INDEX

from ..config import DEFAULT_PARENT_PROVIDERS
from ..phases.phase_generator_k2 import PhaseGeneratorK2


class MemoryEvolverK2(MemoryEvolver):
    """Memory evolver with K=2 dual-parent crossover generation."""

    def __init__(
        self,
        work_dir: str,
        analysis_model_id: Optional[str] = None,
        gen_model_id: Optional[str] = None,
        parent_providers: Optional[List[str]] = None,
    ):
        super().__init__(
            work_dir=work_dir,
            analysis_model_id=analysis_model_id,
            gen_model_id=gen_model_id,
        )
        self.parent_providers = parent_providers or DEFAULT_PARENT_PROVIDERS

    def analyze(
        self,
        task_logs_dir: str,
        default_provider: Optional[str] = "agent_kb",
        parent_providers: Optional[List[str]] = None,
        max_steps: int = ANALYSIS_MAX_STEPS,
    ) -> Dict:
        """
        Analyze trajectories from the current base provider run.

        Uses default_provider (typically the round's base provider, e.g. agent_kb)
        for analysis context — NOT hardcoded to primary parent only.
        """
        if not default_provider or default_provider.strip() == "":
            raise ValueError("default_provider is required and cannot be empty")

        parents = parent_providers or self.parent_providers
        print(f"[Analyze-K2] Starting trajectory analysis")
        print(f"  Task logs: {task_logs_dir}")
        print(f"  Base provider (analysis): {default_provider}")
        print(f"  Crossover parents (generation): {parents}")

        result = super().analyze(
            task_logs_dir=task_logs_dir,
            default_provider=default_provider,
            max_steps=max_steps,
        )

        self.state["phases"]["analyze"]["default_provider"] = default_provider
        self.state["phases"]["analyze"]["parent_providers"] = parents
        self.state["phases"]["analyze"]["crossover_mode"] = "k2"
        self._save_state()

        return result

    def generate(self, creativity_index: float = CREATIVITY_INDEX) -> Dict:
        """Generate a memory system using dual-parent crossover templates."""
        creativity_index = max(0.0, min(1.0, creativity_index))

        if not self.state["phases"]["analyze"]["completed"]:
            raise ValueError("Analysis phase not completed. Run analyze() first.")

        parent_providers = self.state["phases"]["analyze"].get(
            "parent_providers", self.parent_providers
        )
        default_provider = self.state["phases"]["analyze"].get("default_provider", "agent_kb")

        print(f"[Generate-K2] Dual-parent crossover generation")
        print(f"  Creativity index: {creativity_index:.2f}")
        print(f"  Analysis base provider: {default_provider}")
        print(f"  Crossover parents: {parent_providers}")

        report_path = self.state["phases"]["analyze"]["output"]
        with open(report_path, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)

        generator = PhaseGeneratorK2(
            openai_client=self.openai_client,
            model_id=self.gen_model_id,
            work_dir=self.work_dir,
            default_provider=default_provider,
            parent_providers=parent_providers,
        )

        gen_result = generator.run_generation(
            analysis_data=analysis_data,
            creativity_index=creativity_index,
        )

        if not gen_result.get("success") or not gen_result.get("config"):
            return {"success": False, "error": "Generation failed"}

        config = gen_result["config"]
        config_updates = config.get("config_updates", {})
        if config_updates:
            print(f"\n[Validation] Validating generated configuration...")
            from MemEvolve.phases.memory_creator import MemorySystemCreator

            validation_result = MemorySystemCreator._validate_config_updates(config_updates)
            if validation_result["warnings"]:
                print(f"[WARNING] Configuration Warnings:")
                for warning in validation_result["warnings"]:
                    print(f"  - {warning}")
            if not validation_result["success"]:
                print(f"[WARNING] Configuration Validation Issues Detected:")
                for error in validation_result["errors"]:
                    print(f"  - {error}")
            else:
                print(f"[OK] Configuration validation passed!")

        config_path = self._get_next_generated_system_path()
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        self.state["phases"]["generate"] = {
            "completed": True,
            "output": str(config_path),
            "creativity_index": creativity_index,
            "parent_providers": parent_providers,
            "crossover_mode": "k2",
            "timestamp": datetime.now().isoformat(),
        }
        self._save_state()

        print(f"\n[Generate-K2] Complete. Config saved to: {config_path}")

        return {
            "success": True,
            "config_path": str(config_path),
            "config": config,
        }
