# Experiment Results (GAIA validation tasks 1–20)

Hold-out evaluation subset. Judge: `deepseek-v4-flash`. Max steps: 40.

## Summary

| Memory System | Category | Perf. | Level 1 · 2 · 3 | Token Cost | #Steps (action) |
|---------------|----------|-------|-----------------|------------|-------------------|
| Agent-KB | Base | 16/20 | 5/6 · 11/12 · 0/2 | 158,853 | 7.0 |
| Cerebra | Base | 14/20 | 4/6 · 10/12 · 0/2 | 141,785 | 7.5 |
| Voyager | Base | 16/20 | 5/6 · 11/12 · 0/2 | 173,217 | 7.3 |
| pathfinder | Evolved | 12/20 | 3/6 · 9/12 · 0/2 | 179,943 | 7.0 |
| adaptive_trajectory_knowledge | Evolved | 15/20 | 4/6 · 11/12 · 0/2 | 157,235 | 6.8 |
| **Echo-Base** | **Evolved** | **16/20** | **5/6 · 10/12 · 1/2** | **110,724** | **6.3** |

- **#Steps**: average agent action steps (tool-calling rounds) per task.
- **Token Cost**: average total tokens per task.

## Result directories

| System | Path |
|--------|------|
| Agent-KB baseline | `gaia_output/agent_kb_results_runs/agent_kb_20260627_195855/` |
| Voyager baseline | `gaia_output/voyager_results_runs/voyager_20260627_214726/` |
| Cerebra baseline | `gaia_output/cerebra_fusion_memory_results_runs/cerebra_fusion_memory_20260627_200731/` |
| pathfinder (evolved) | `gaia_output/holdout_k2_evolved_20260630_083152/pathfinder/` |
| adaptive_trajectory_knowledge (evolved) | `gaia_output/holdout_k2_evolved_20260630_083152/adaptive_trajectory_knowledge/` |
| Echo-Base (evolved) | `gaia_output/holdout_k2_evolved_20260630_083152/echo_base/` |

Each directory contains `1.json`–`20.json` (per-task trajectories), `report.txt`, and `results.jsonl`.

## Evolution setup

- **Parents**: Agent-KB + Voyager (K=2 crossover)
- **Evolution pool**: GAIA validation tasks 21–40
- **Hold-out**: tasks 1–20 (this table)
