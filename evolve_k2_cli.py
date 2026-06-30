#!/usr/bin/env python
# coding=utf-8

"""
MemEvolveK2 Command-Line Interface

K=2 dual-parent crossover evolution (agent_kb + voyager).
Does not modify the original evolve_cli.py or MemEvolve package.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)


def cmd_analyze(args):
    from MemEvolveK2 import MemoryEvolverK2, DEFAULT_PARENT_PROVIDERS

    parent_providers = _parse_parent_providers(args)
    analysis_model_id = args.model or os.getenv(
        "ANALYSIS_MODEL", os.getenv("DEFAULT_MODEL", "deepseek-v4-flash")
    )
    generation_model_id = os.getenv(
        "GENERATION_MODEL", os.getenv("DEFAULT_MODEL", "deepseek-v4-flash")
    )

    print("=== K2 Phase 1: Analyze ===")
    print(f"Task logs: {args.task_logs_dir}")
    print(f"Work directory: {args.work_dir}")
    print(f"Base provider: {args.provider}")
    print(f"Crossover parents: {parent_providers}")
    print(f"Analysis Model: {analysis_model_id}")

    evolver = MemoryEvolverK2(
        work_dir=args.work_dir,
        analysis_model_id=analysis_model_id,
        gen_model_id=generation_model_id,
        parent_providers=parent_providers,
    )

    result = evolver.analyze(
        task_logs_dir=args.task_logs_dir,
        default_provider=args.provider,
        parent_providers=parent_providers,
    )

    if result["success"]:
        print(f"\nAnalysis complete!")
        print(f"Report: {result['report_path']}")
    else:
        print("\nAnalysis failed")
        sys.exit(1)


def cmd_generate(args):
    from MemEvolveK2 import MemoryEvolverK2

    generation_model_id = args.model or os.getenv(
        "GENERATION_MODEL", os.getenv("DEFAULT_MODEL", "deepseek-v4-flash")
    )

    print("=== K2 Phase 2: Generate (dual-parent crossover) ===")
    print(f"Work directory: {args.work_dir}")
    print(f"Creativity index: {args.creativity}")
    print(f"Generation Model: {generation_model_id}")

    evolver = MemoryEvolverK2(
        work_dir=args.work_dir,
        gen_model_id=generation_model_id,
    )

    if args.provider and evolver.state["phases"]["analyze"]["completed"]:
        evolver.state["phases"]["analyze"]["default_provider"] = args.provider
        evolver._save_state()

    result = evolver.generate(creativity_index=args.creativity)

    if result["success"]:
        print(f"\nGeneration complete!")
        print(f"Config saved: {result['config_path']}")
    else:
        print(f"\nGeneration failed: {result.get('error', 'unknown')}")
        sys.exit(1)


def cmd_create(args):
    from MemEvolveK2 import MemoryEvolverK2

    print("=== K2 Phase 3: Create ===")
    evolver = MemoryEvolverK2(work_dir=args.work_dir)
    result = evolver.create()
    if not result.get("success"):
        sys.exit(1)
    print(f"Created: {result.get('created', [])}")


def cmd_validate(args):
    from MemEvolveK2 import MemoryEvolverK2
    from MemEvolve.config import DEFAULT_DATASETS

    print("=== K2 Phase 4: Validate ===")
    evolver = MemoryEvolverK2(work_dir=args.work_dir)
    result = evolver.validate(
        dataset_name=args.dataset,
        datasets_config=DEFAULT_DATASETS,
    )
    if not result.get("success"):
        sys.exit(1)
    print(f"Validated: {result.get('validated', [])}")


def cmd_status(args):
    work_dir = Path(args.work_dir)
    state_file = work_dir / "state.json"
    if not state_file.exists():
        print(f"No evolution state found in {work_dir}")
        sys.exit(1)

    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)

    print(f"=== K2 Evolution Status ===")
    print(f"Work directory: {work_dir}")
    for phase, info in state.get("phases", {}).items():
        done = info.get("completed", False)
        print(f"  {phase}: {'done' if done else 'pending'}")
        if phase == "analyze" and info.get("parent_providers"):
            print(f"    crossover parents: {info['parent_providers']}")


def _ensure_dataset_cursor(work_dir: str, dataset_name: str, cursor: int, provider: str):
    """Initialize evolve_state.json with dataset_cursor for hold-out splits."""
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    state_path = work_path / "evolve_state.json"
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        current = state.get("dataset_cursor", 0)
        print(f"[State] Existing evolve_state.json: dataset_cursor={current}")
        return
    state = {
        "round": 0,
        "dataset_name": dataset_name,
        "dataset_cursor": cursor,
        "best_provider": provider,
        "history": [],
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    # 1-based GAIA task numbers for default x=20 round: cursor..cursor+19 and cursor+20..cursor+39
    t_start = cursor + 1
    t_end = cursor + 40
    print(f"[State] Created {state_path} with dataset_cursor={cursor}")
    print(f"[State] Evolution will use GAIA tasks ~{t_start}-{t_end} (hold-out: 1-{cursor})")


def _make_run_provider_k2(judge_model: str, max_steps: int):
    from MemEvolveK2.utils.run_provider_k2 import run_provider_k2

    def _run(dataset_name, provider_name, task_indices, dataset_file, output_dir):
        return run_provider_k2(
            dataset_name,
            provider_name,
            task_indices,
            dataset_file,
            output_dir,
            judge_model=judge_model,
            max_steps=max_steps,
        )

    return _run


def cmd_auto_evolve(args):
    from MemEvolveK2 import AutoEvolverK2, DEFAULT_PARENT_PROVIDERS
    from MemEvolveK2.config import DEFAULT_EVAL_MAX_STEPS, DEFAULT_JUDGE_MODEL
    from MemEvolve.config import DEFAULT_DATASETS

    _ensure_dataset_cursor(
        args.work_dir, args.dataset, args.dataset_offset, args.provider
    )

    parent_providers = _parse_parent_providers(args)
    analysis_model_id = args.model or os.getenv(
        "ANALYSIS_MODEL", os.getenv("DEFAULT_MODEL", "deepseek-v4-flash")
    )
    generation_model_id = os.getenv(
        "GENERATION_MODEL", os.getenv("DEFAULT_MODEL", "deepseek-v4-flash")
    )
    judge_model = getattr(args, "judge_model", None) or os.getenv(
        "DEFAULT_JUDGE_MODEL", DEFAULT_JUDGE_MODEL
    )
    eval_max_steps = getattr(args, "max_steps", DEFAULT_EVAL_MAX_STEPS)

    print("=== K2 Auto Evolution (Dual-Parent Crossover) ===")
    print(f"Dataset: {args.dataset}")
    print(f"Rounds: {args.num_rounds}")
    print(f"Work directory: {args.work_dir}")
    print(f"Initial base provider: {args.provider}")
    print(f"Crossover parents: {parent_providers}")
    print(f"Systems per round: {args.num_systems}")
    print(f"Task batch (x): {args.task_batch_x}")
    print(f"Dataset offset (cursor): {args.dataset_offset}")
    print(f"Analysis Model: {analysis_model_id}")
    print(f"Generation Model: {generation_model_id}")
    print(f"Judge Model: {judge_model}")
    print(f"Eval max_steps: {eval_max_steps}")

    auto_evolver = AutoEvolverK2(
        analysis_model_id=analysis_model_id,
        gen_model_id=generation_model_id,
        work_root=args.work_dir,
        dataset_name=args.dataset,
        run_provider=_make_run_provider_k2(judge_model, eval_max_steps),
        default_provider=args.provider,
        num_systems=args.num_systems,
        creativity_index=args.creativity,
        task_batch_x=args.task_batch_x,
        top_t=args.top_t,
        extra_sample_y=args.extra_sample_y,
        datasets_config=DEFAULT_DATASETS,
        use_pareto_selection=args.use_pareto_selection,
        clear_storage_per_round=args.clear_storage_per_round,
        parent_providers=parent_providers,
    )

    if not args.yes:
        print("\n" + "=" * 50)
        print(f"This will run {args.num_rounds} round(s) of K=2 crossover evolution.")
        print(f"Crossover parents (fixed for generation): {parent_providers}")
        print(f"Each round:")
        print(f"  1. Run base provider on {args.task_batch_x} tasks")
        print(f"  2. Analyze trajectories (base provider context)")
        print(f"  3. Generate {args.num_systems} systems from BOTH parent templates")
        print(f"  4. Evaluate, select top {args.top_t}, run finals")
        print("=" * 50)
        confirm = input("Continue? (yes/no): ")
        if confirm.lower() not in ["yes", "y"]:
            print("Cancelled.")
            return

    result = auto_evolver.run(num_rounds=args.num_rounds)

    print("\n=== K2 Auto Evolution Complete ===")
    print(f"Total rounds: {result['rounds']}")
    print(f"History: {result['history_path']}")
    for round_info in result["history"]:
        print(f"  Round {round_info['round']}: winner = {round_info['winner']}")


def _parse_parent_providers(args):
    if getattr(args, "parent_primary", None) and getattr(args, "parent_secondary", None):
        return [args.parent_primary, args.parent_secondary]
    from MemEvolveK2.config import DEFAULT_PARENT_PROVIDERS

    return DEFAULT_PARENT_PROVIDERS


def _add_parent_args(parser):
    parser.add_argument(
        "--parent-primary",
        default="agent_kb",
        help="Primary crossover parent (default: agent_kb)",
    )
    parser.add_argument(
        "--parent-secondary",
        default="voyager",
        help="Secondary crossover parent (default: voyager)",
    )


def main():
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument(
        "--work-dir",
        default="./memevolve_k2_work",
        help="Working directory (default: ./memevolve_k2_work)",
    )
    global_parser.add_argument(
        "--model",
        default=None,
        help="Model ID (default: ANALYSIS_MODEL / GENERATION_MODEL env vars)",
    )

    parser = argparse.ArgumentParser(
        prog="memevolve-k2",
        description="K=2 dual-parent memory evolution (agent_kb + voyager crossover)",
        parents=[global_parser],
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    p_analyze = subparsers.add_parser(
        "analyze", help="Analyze task trajectories", parents=[global_parser]
    )
    p_analyze.add_argument("task_logs_dir", help="Directory with task logs")
    p_analyze.add_argument(
        "--provider", default="agent_kb", help="Base provider for analysis"
    )
    _add_parent_args(p_analyze)
    p_analyze.set_defaults(func=cmd_analyze)

    p_generate = subparsers.add_parser(
        "generate",
        help="Generate system via dual-parent crossover",
        parents=[global_parser],
    )
    p_generate.add_argument(
        "--creativity", type=float, default=0.5, help="Creativity 0-1"
    )
    p_generate.add_argument("--provider", default=None, help="Override base provider")
    p_generate.set_defaults(func=cmd_generate)

    p_create = subparsers.add_parser(
        "create", help="Create memory system files", parents=[global_parser]
    )
    p_create.set_defaults(func=cmd_create)

    p_validate = subparsers.add_parser(
        "validate", help="Validate created systems", parents=[global_parser]
    )
    p_validate.add_argument(
        "--dataset",
        default="gaia",
        choices=["gaia", "webwalkerqa", "xbench", "taskcraft"],
    )
    p_validate.set_defaults(func=cmd_validate)

    p_status = subparsers.add_parser(
        "status", help="Show evolution status", parents=[global_parser]
    )
    p_status.set_defaults(func=cmd_status)

    p_auto = subparsers.add_parser(
        "auto-evolve",
        help="Run multi-round K=2 crossover evolution",
        parents=[global_parser],
    )
    p_auto.add_argument(
        "dataset",
        choices=["gaia", "webwalkerqa", "xbench", "taskcraft"],
    )
    p_auto.add_argument("--num-rounds", type=int, default=1)
    p_auto.add_argument(
        "--dataset-offset",
        type=int,
        default=20,
        help="Skip first N GAIA tasks (default: 20, hold-out tasks 1-20 for final eval)",
    )
    p_auto.add_argument("--provider", default="agent_kb", help="Initial base provider")
    p_auto.add_argument("--num-systems", type=int, default=3)
    p_auto.add_argument("--task-batch-x", type=int, default=20)
    p_auto.add_argument("--top-t", type=int, default=2)
    p_auto.add_argument("--extra-sample-y", type=int, default=5)
    p_auto.add_argument("--creativity", type=float, default=0.5)
    p_auto.add_argument(
        "--judge-model",
        default=None,
        help="Judge model for GAIA eval (default: DEFAULT_JUDGE_MODEL env or deepseek-v4-flash)",
    )
    p_auto.add_argument(
        "--max-steps",
        type=int,
        default=40,
        help="Max agent steps during GAIA eval runs",
    )
    p_auto.add_argument("--use-pareto-selection", action="store_true", default=False)
    p_auto.add_argument("--clear-storage-per-round", action="store_true", default=True)
    p_auto.add_argument(
        "--no-clear-storage",
        dest="clear_storage_per_round",
        action="store_false",
    )
    _add_parent_args(p_auto)
    p_auto.add_argument("-y", "--yes", action="store_true")
    p_auto.set_defaults(func=cmd_auto_evolve)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
