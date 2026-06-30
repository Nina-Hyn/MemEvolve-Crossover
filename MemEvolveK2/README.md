# MemEvolveK2 — K=2 Dual-Parent Crossover (MVP)

在**不修改**原有 `MemEvolve/` 与 `evolve_cli.py` 的前提下，本目录提供 **K=2 双 parent 杂交进化** 的最小实现。

## 固定 Parent（默认）

| 角色 | Provider | 说明 |
|------|----------|------|
| Primary | `agent_kb` | 结构化 KB 检索、workflow 注入 |
| Secondary | `voyager` | 轨迹/skill 嵌入检索、渐进积累 |

Generation 阶段会同时加载两个 parent 的 provider 源码作为模板，要求 LLM 做 **crossover 融合**；Analyze 阶段仍基于**当前 round 的 base provider**（默认 `agent_kb`）的任务日志。

## 目录结构

```
MemEvolveK2/
├── config.py                 # DEFAULT_PARENT_PROVIDERS = ["agent_kb", "voyager"]
├── core/
│   ├── auto_evolver_k2.py    # 修复 analyze 用 current base provider
│   └── memory_evolver_k2.py  # K=2 analyze/generate 入口
├── phases/
│   └── phase_generator_k2.py # 双模板 crossover prompt
└── prompts/
    └── generation_prompt_k2.yaml
```

入口 CLI：`../evolve_k2_cli.py`

## 环境准备

```bash
conda activate memevolve
cd Flash-Searcher-main

# HPC 出网（Mac 反向代理示例）
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897

# .env 中需配置（与 GAIA 评测一致）
# OPENAI_API_KEY=...
# OPENAI_API_BASE=https://api.deepseek.com
# DEFAULT_MODEL=deepseek-v4-flash
# SERPER_API_KEY=...
```

## 复现：GAIA 20 题 baseline 评测（已有结果，无需 K2）

```bash
# Agent-KB
python run_flash_searcher_mm_gaia.py \
  --infile ./data/gaia/validation/metadata.jsonl \
  --outfile ./gaia_output/agent_kb_results.jsonl \
  --memory_provider agent_kb \
  --sample_num 20 --max_steps 40 \
  --judge_model deepseek-v4-flash --concurrency 1 \
  2>&1 | tee gaia_output/agent_kb_run.log

# Voyager
python run_flash_searcher_mm_gaia.py \
  --infile ./data/gaia/validation/metadata.jsonl \
  --outfile ./gaia_output/voyager_results.jsonl \
  --memory_provider voyager \
  --sample_num 20 --max_steps 40 \
  --judge_model deepseek-v4-flash --concurrency 1 \
  2>&1 | tee gaia_output/voyager_run.log
```

已有 run 目录（供 lab report / 进化输入）：

- `gaia_output/agent_kb_results_runs/agent_kb_20260627_195855/`
- `gaia_output/voyager_results_runs/voyager_20260627_214726/`

## 复现：K=2 自动进化（推荐）

Round 1 会在 20 个 GAIA task 上跑 base provider（默认 `agent_kb`），再基于日志 + **agent_kb/voyager 双模板** 生成新 memory system。

```bash
cd Flash-Searcher-main

python evolve_k2_cli.py auto-evolve gaia \
  --work-dir ./memevolve_k2_work \
  --provider agent_kb \
  --parent-primary agent_kb \
  --parent-secondary voyager \
  --num-rounds 1 \
  --num-systems 3 \
  --task-batch-x 20 \
  --top-t 2 \
  --extra-sample-y 5 \
  --creativity 0.5 \
  -y
```

输出：`./memevolve_k2_work/round_1/`（含 `base_logs/`、`analysis_report.json`、`generated_system_*.json` 等）。

## 复现：手动分步（使用已有 Agent-KB 日志）

若已有 `base_logs`（例如从 agent_kb GAIA run 复制），可跳过 Step 1：

```bash
# 1. 分析（base provider = agent_kb，generation parents = agent_kb + voyager）
python evolve_k2_cli.py analyze \
  ./memevolve_k2_work/round_1/base_logs \
  --work-dir ./memevolve_k2_work/round_1 \
  --provider agent_kb \
  --parent-primary agent_kb \
  --parent-secondary voyager

# 2. 双 parent crossover 生成
python evolve_k2_cli.py generate \
  --work-dir ./memevolve_k2_work/round_1 \
  --creativity 0.5

# 3. 创建 provider 文件
python evolve_k2_cli.py create --work-dir ./memevolve_k2_work/round_1

# 4. 冒烟验证
python evolve_k2_cli.py validate --work-dir ./memevolve_k2_work/round_1 --dataset gaia

# 5. 查看状态
python evolve_k2_cli.py status --work-dir ./memevolve_k2_work/round_1
```

## 评测进化出的新 system

创建成功后，新 provider 会注册到 `EvolveLab/memory_types.py`。用与 baseline 相同命令评测（替换 `--memory_provider`）：

```bash
python run_flash_searcher_mm_gaia.py \
  --infile ./data/gaia/validation/metadata.jsonl \
  --outfile ./gaia_output/<new_provider>_results.jsonl \
  --memory_provider <new_provider_enum_value> \
  --sample_num 20 --max_steps 40 \
  --judge_model deepseek-v4-flash --concurrency 1
```

## 与原版 MemEvolve 的差异

| 项目 | 原版 `evolve_cli.py` | `evolve_k2_cli.py` |
|------|----------------------|---------------------|
| Generation 模板 | 单 parent (`default_provider`) | **双 parent crossover** |
| Analyze 上下文 | 固定 `self.default_provider` | **当前 round base provider** |
| Parent 指定 | 无 | `--parent-primary` / `--parent-secondary` |

## GitHub 上传建议

上传以下内容即可复现：

1. 本目录 `MemEvolveK2/` + `evolve_k2_cli.py`
2. 原有 `MemEvolve/`、`EvolveLab/`、评测脚本（未改动）
3. GAIA 20 题 baseline 结果目录（agent_kb / voyager）
4. `.env.example`（不含真实 key）
