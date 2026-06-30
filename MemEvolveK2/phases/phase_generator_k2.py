#!/usr/bin/env python
# coding=utf-8

"""
K=2 Phase Generator: dual-parent crossover prompt with agent_kb + voyager templates.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from MemEvolve.phases.phase_generator import PhaseGenerator
from MemEvolve.config import CREATIVITY_INDEX

from ..config import DEFAULT_PARENT_PROVIDERS


class PhaseGeneratorK2(PhaseGenerator):
    """
    Generates memory systems using two parent provider templates (crossover).
    """

    def __init__(
        self,
        openai_client,
        model_id: str,
        work_dir: Path,
        default_provider: Optional[str] = "agent_kb",
        parent_providers: Optional[List[str]] = None,
    ):
        self.parent_providers = parent_providers or DEFAULT_PARENT_PROVIDERS
        if len(self.parent_providers) < 2:
            raise ValueError("parent_providers must contain at least 2 providers for K=2 crossover")

        primary = self.parent_providers[0]
        super().__init__(
            openai_client=openai_client,
            model_id=model_id,
            work_dir=work_dir,
            default_provider=primary if not default_provider else default_provider,
        )
        self.primary_provider = self.parent_providers[0]
        self.secondary_provider = self.parent_providers[1]

    def _load_prompt_template(self) -> str:
        prompt_file = Path(__file__).parent.parent / "prompts" / "generation_prompt_k2.yaml"
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data.get("prompt_template", "")
        except Exception as e:
            print(f"  Error: Failed to load K2 prompt template from {prompt_file}: {e}")
            raise

    def _load_provider_template(self, provider_name: str) -> str:
        try:
            from EvolveLab.memory_types import MemoryType, PROVIDER_MAPPING

            memory_type = MemoryType(provider_name)
            if memory_type in PROVIDER_MAPPING:
                _, module_name = PROVIDER_MAPPING[memory_type]
                provider_path = f"EvolveLab/providers/{module_name}.py"
            else:
                provider_path = f"EvolveLab/providers/{provider_name}_provider.py"

            if os.path.exists(provider_path):
                with open(provider_path, "r", encoding="utf-8") as f:
                    return f"""```python
{f.read()}
```"""
            print(f"  Warning: Provider file not found: {provider_path}")
            return ""
        except Exception as e:
            print(f"  Warning: Failed to load provider template for {provider_name}: {e}")
            return ""

    def _build_generation_prompt(
        self, analysis_text: str, memory_types_ref: str, creativity_index: float
    ) -> str:
        existing_systems = self._extract_existing_systems()
        existing_systems_section = ""
        if existing_systems:
            existing_systems_section = f"""
### CRITICAL: Existing System Names (DO NOT USE THESE)
The following system names ALREADY EXIST. You MUST create a completely different, unique name:
{', '.join(existing_systems)}

Your new system name must be different from ALL of the above names.
"""

        if analysis_text.strip():
            analysis_section = f"""### Analysis Insights

{analysis_text}"""
        else:
            analysis_section = "No analysis report available. Design based on parent crossover guidelines."

        primary_template = self._load_provider_template(self.primary_provider)
        secondary_template = self._load_provider_template(self.secondary_provider)

        if primary_template:
            print(f"  [OK] Loaded primary parent template: {self.primary_provider}")
        if secondary_template:
            print(f"  [OK] Loaded secondary parent template: {self.secondary_provider}")

        memory_types_definition = f"""```python
{memory_types_ref}
```"""

        template = self._load_prompt_template()
        if not template:
            raise ValueError("Failed to load generation_prompt_k2.yaml")

        prompt = template.format(
            primary_provider=self.primary_provider,
            secondary_provider=self.secondary_provider,
            primary_provider_template=primary_template or "(template unavailable)",
            secondary_provider_template=secondary_template or "(template unavailable)",
            analysis_section=analysis_section,
            memory_types_definition=memory_types_definition,
        )

        if existing_systems_section:
            prompt = existing_systems_section + "\n" + prompt

        return prompt

    @staticmethod
    def _extract_metadata_field(response_text: str, field_name: str) -> Optional[str]:
        """Extract metadata from colon or markdown-table formats (with optional backticks)."""
        escaped = re.escape(field_name)
        patterns = [
            rf"\*\*{escaped}\*\*:\s*`?([\w]+)`?",
            rf"\|\s*\*\*{escaped}\*\*\s*\|\s*`?([\w]+)`?\s*\|",
        ]
        alias_map = {
            "Enum Name": ["Memory Type Enum"],
            "Enum Value": ["Memory Type Value"],
        }
        for alias in alias_map.get(field_name, []):
            alias_escaped = re.escape(alias)
            patterns.extend(
                [
                    rf"\*\*{alias_escaped}\*\*:\s*`?([\w]+)`?",
                    rf"\|\s*\*\*{alias_escaped}\*\*\s*\|\s*`?([\w]+)`?\s*\|",
                ]
            )
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _parse_system_config(self, response_text: str) -> Optional[Dict]:
        """
        Parse LLM response with K2-tolerant format handling.

        Supports both canonical `**Class Name**: value` lines and markdown tables
        like `| **Class Name** | `value` |`, plus JSON configuration blocks.
        """
        try:
            class_name = self._extract_metadata_field(response_text, "Class Name")
            module_name = self._extract_metadata_field(response_text, "Module Name")
            if not class_name or not module_name:
                print("  Parse error: Could not find Class Name or Module Name")
                return None

            code_match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
            if not code_match:
                print("  Parse error: Could not find Python code block")
                return None
            code = code_match.group(1).strip()

            enum_name = self._extract_metadata_field(response_text, "Enum Name")
            enum_value = self._extract_metadata_field(response_text, "Enum Value")
            if not enum_name or not enum_value:
                print("  Parse error: Could not find Enum Name or Enum Value")
                return None

            metadata_keys = {
                "Class Name",
                "Module Name",
                "Enum Name",
                "Enum Value",
                "Memory Type Enum",
                "Memory Type Value",
            }
            config_updates: Dict[str, str] = {}
            config_pattern = r"\*\*([^*]+)\*\*:\s*([^\n]+)"
            for match in re.finditer(config_pattern, response_text):
                key, value = match.groups()
                key = key.strip()
                value = value.strip().strip("`")
                if key in metadata_keys or not value:
                    continue
                config_updates[key] = value

            if not config_updates:
                json_match = re.search(
                    r"(?:##\s*)?Configuration\s*\n+```json\s*\n(.*?)\n```",
                    response_text,
                    re.DOTALL | re.IGNORECASE,
                )
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(1))
                        if isinstance(parsed, dict):
                            config_updates = parsed
                    except json.JSONDecodeError:
                        pass
            else:
                json_match = re.search(
                    r"(?:##\s*)?Configuration\s*\n+```json\s*\n(.*?)\n```",
                    response_text,
                    re.DOTALL | re.IGNORECASE,
                )
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(1))
                        if isinstance(parsed, dict):
                            config_updates.update(parsed)
                    except json.JSONDecodeError:
                        pass

            print(f"  Successfully parsed: {len(config_updates)} config parameters")
            return {
                "provider_code": {
                    "class_name": class_name,
                    "module_name": module_name,
                    "code": code,
                },
                "config_updates": config_updates,
                "memory_type_info": {
                    "enum_name": enum_name,
                    "enum_value": enum_value,
                },
            }
        except Exception as e:
            print(f"  Parse error: {e}")
            import traceback

            traceback.print_exc()
            return None
