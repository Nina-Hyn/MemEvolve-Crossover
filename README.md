# MemEvolve-Crossover

This repository contains the complete reproducible code for the lab report Memory Systems that Evolve.
Built upon the open-source MemEvolve framework, this work extends the original single-parent memory meta-evolution pipeline to support multi-parent crossover (K=2).
I hybridize the Agent-KB and Voyager to generate three novel hybrid memory variants, with Echo as the optimal evolved architecture.
All memory systems are fully evaluated on GAIA subset and the result is in [RESULTS.md](RESULTS.md).

**Best evolved system (Echo)**: 80% accuracy, ~30% lower token cost, sole Level-3 success on Task 11.


## Repository layout

```
MemEvolve-Crossover/
├── evolve_k2_cli.py              # K=2 evolution entry
├── run_flash_searcher_mm_gaia.py # GAIA evaluation with memory providers
├── MemEvolveK2/                   # Dual-parent crossover implementation
├── MemEvolve/                     # Original evolution pipeline (inherited)
├── EvolveLab/                     # Memory providers (baselines + evolved)
├── FlashOAgents/                  # Agent framework
├── scripts/run_holdout_eval_k2.sh
├── data/gaia/validation/          # GAIA validation split + attachments
├── storage/                       # Embedding model + memory DB seeds
├── gaia_output/                   # Pre-computed experiment results
├── RESULTS.md                     # Results summary table
├── requirements.txt
└── .env.example
```


## 1. Environment setup

```bash
conda create -n memevolve python=3.10 -y
conda activate memevolve
cd MemEvolve-Crossover
pip install -r requirements.txt
```

Copy environment template and fill in your API keys:

```bash
cp .env.example .env
# Edit .env: OPENAI_API_KEY, SERPER_API_KEY, etc.
```

Required keys:
- `OPENAI_API_KEY` + `OPENAI_API_BASE` (I used DeepSeek: `https://api.deepseek.com`)
- `SERPER_API_KEY` ([serper.dev](https://serper.dev/))
- `WEB_ACCESS_PROVIDER=crawl4ai`**or** `JINA_API_KEY`


## 2. View pre-computed results 

```bash
cat RESULTS.md
cat gaia_output/holdout_k2_evolved_20260630_083152/echo_base/report.txt
ls gaia_output/holdout_k2_evolved_20260630_083152/echo_base/{1..20}.json
```


## 3. Re-run baseline evaluation

```bash
# Agent-KB
python run_flash_searcher_mm_gaia.py \
  --infile ./data/gaia/validation/metadata.jsonl \
  --outfile ./gaia_output/agent_kb_results.jsonl \
  --task_indices 1-20 \
  --memory_provider agent_kb \
  --max_steps 40 \
  --judge_model deepseek-v4-flash \
  --concurrency 1

# Voyager
python run_flash_searcher_mm_gaia.py \
  --infile ./data/gaia/validation/metadata.jsonl \
  --outfile ./gaia_output/voyager_results.jsonl \
  --task_indices 1-20 \
  --memory_provider voyager \
  --max_steps 40 \
  --judge_model deepseek-v4-flash \
  --concurrency 1

# Cerebra
python run_flash_searcher_mm_gaia.py \
  --infile ./data/gaia/validation/metadata.jsonl \
  --outfile ./gaia_output/cerebra_results.jsonl \
  --task_indices 1-20 \
  --memory_provider cerebra_fusion_memory \
  --max_steps 40 \
  --judge_model deepseek-v4-flash \
  --concurrency 1
```


## 4. Re-run K=2 evolution

```bash
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

This generates three hybrid providers under `EvolveLab/providers/` and registers them in `EvolveLab/memory_types.py`.

Manual step-by-step (if you prefer):

```bash
python evolve_k2_cli.py analyze ./memevolve_k2_work/round_00/base_logs \
  --work-dir ./memevolve_k2_work/round_00 --provider agent_kb \
  --parent-primary agent_kb --parent-secondary voyager

python evolve_k2_cli.py generate --work-dir ./memevolve_k2_work/round_00 --creativity 0.5
python evolve_k2_cli.py create --work-dir ./memevolve_k2_work/round_00
python evolve_k2_cli.py validate --work-dir ./memevolve_k2_work/round_00 --dataset gaia
```


## 5. Re-run evaluation of evolved systems

```bash
bash scripts/run_holdout_eval_k2.sh
```

Or evaluate a single evolved system:

```bash
python run_flash_searcher_mm_gaia.py \
  --infile ./data/gaia/validation/metadata.jsonl \
  --outfile ./gaia_output/echo_base_results.jsonl \
  --task_indices 1-20 \
  --memory_provider echo_base \
  --max_steps 40 \
  --judge_model deepseek-v4-flash \
  --concurrency 1 \
  --direct_output_dir ./gaia_output/echo_base_rerun
```

Available evolved providers: `pathfinder`, `adaptive_trajectory_knowledge`, `echo_base`.


## 6. Dataset split

| Split | Task indices | Purpose |
|-------|--------------|---------|
| Hold-out | 1–20 | Final evaluation (baselines + evolved systems) |
| Evolution pool | 21–40 | Base trajectories + K=2 crossover generation |


