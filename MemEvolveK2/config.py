#!/usr/bin/env python
# coding=utf-8

"""
K=2 dual-parent crossover configuration.

Default parents: agent_kb (primary) + voyager (secondary).
"""

# Fixed parent providers for crossover generation (MVP)
DEFAULT_PARENT_PROVIDERS = ["agent_kb", "voyager"]

PRIMARY_PARENT = DEFAULT_PARENT_PROVIDERS[0]
SECONDARY_PARENT = DEFAULT_PARENT_PROVIDERS[1]

# GAIA judgement during auto-evolve eval (override via DEFAULT_JUDGE_MODEL in .env)
DEFAULT_JUDGE_MODEL = "deepseek-v4-flash"
DEFAULT_EVAL_MAX_STEPS = 40
