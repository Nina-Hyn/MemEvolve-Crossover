import json
import os
import re
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from ..base_memory import BaseMemoryProvider
from ..memory_types import (
    MemoryRequest,
    MemoryResponse,
    TrajectoryData,
    MemoryType,
    MemoryItem,
    MemoryStatus,
    MemoryItemType
)


def load_embedding_model(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                         cache_dir: str = './storage/models') -> SentenceTransformer:
    os.makedirs(cache_dir, exist_ok=True)
    local_model_path = os.path.join(cache_dir, model_name.replace('/', '_'))

    try:
        if os.path.exists(local_model_path) and os.listdir(local_model_path):
            model = SentenceTransformer(local_model_path)
            return model
    except Exception as e:
        print(f"Local model load failed: {e}", file=sys.stderr)

    try:
        model = SentenceTransformer(model_name)
        model.save(local_model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model {model_name}: {e}")


@dataclass
class EchoMemoryEntry:
    memory_id: str
    query: str
    high_level_strategy: str          # 1-2 sentences
    planning_steps: str               # bullet points
    experience_details: str           # 3-5 sentences
    trajectory_summary: str           # from voyager-style summarization
    tool_patterns: List[Dict]         # list of extracted tool patterns
    metadata: Dict[str, Any]
    query_embedding: Optional[np.ndarray] = None
    planning_embedding: Optional[np.ndarray] = None
    experience_embedding: Optional[np.ndarray] = None
    retrieval_count: int = 0
    success_contribution: float = 0.0
    last_accessed: Optional[str] = None


class EchoBaseProvider(BaseMemoryProvider):
    """
    Self-evolving, adaptive memory system that crossbreeds agent_kb (structured KB,
    hybrid retrieval, quality filtering) and voyager (trajectory summarization,
    skill accumulation, embedding-only retrieval).
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(memory_type=MemoryType.ECHO_BASE, config=config)

        self.model = self.config.get('model', None)

        self.db_path = self.config.get('db_path', './storage/echo_base/echo_memory.json')
        self.model_name = self.config.get('embedding_model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.model_cache_dir = self.config.get('embedding_model_cache', './storage/models')

        self.top_k_begin = self.config.get('top_k_begin', 3)
        self.top_k_in = self.config.get('top_k_in', 2)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.3)  # adaptive filtering
        self.redundancy_threshold = self.config.get('redundancy_threshold', 0.85)
        self.weights_begin = self.config.get('weights_begin', {'query': 0.6, 'planning': 0.3, 'experience': 0.1})
        self.weights_in = self.config.get('weights_in', {'query': 0.1, 'planning': 0.2, 'experience': 0.7})

        self.embedding_model: Optional[SentenceTransformer] = None
        self.embedding_dim: int = 384
        self.memories: List[EchoMemoryEntry] = []
        self.tool_memories: List[Dict] = []  # reusable tool patterns

        # Cached embeddings for fast retrieval
        self.query_embeddings: Optional[np.ndarray] = None
        self.planning_embeddings: Optional[np.ndarray] = None
        self.experience_embeddings: Optional[np.ndarray] = None

    def initialize(self) -> bool:
        try:
            self.embedding_model = load_embedding_model(
                model_name=self.model_name,
                cache_dir=self.model_cache_dir
            )
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self._load_memories_from_json()
            self._rebuild_embeddings_cache()
            print(f"EchoBaseProvider initialized. Storing memories in {self.db_path}")
            return True
        except Exception as e:
            print(f"Error initializing EchoBaseProvider: {e}", file=sys.stderr)
            return False

    def _load_memories_from_json(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for mem in data.get('memories', []):
                        entry = EchoMemoryEntry(
                            memory_id=mem.get('memory_id', str(uuid.uuid4())),
                            query=mem.get('query', ''),
                            high_level_strategy=mem.get('high_level_strategy', ''),
                            planning_steps=mem.get('planning_steps', ''),
                            experience_details=mem.get('experience_details', ''),
                            trajectory_summary=mem.get('trajectory_summary', ''),
                            tool_patterns=mem.get('tool_patterns', []),
                            metadata=mem.get('metadata', {}),
                            retrieval_count=mem.get('retrieval_count', 0),
                            success_contribution=mem.get('success_contribution', 0.0),
                            last_accessed=mem.get('last_accessed', None)
                        )
                        # Embeddings loaded separately
                        self.memories.append(entry)
                    self.tool_memories = data.get('tool_memories', [])
                    print(f"Loaded {len(self.memories)} memories and {len(self.tool_memories)} tool patterns.")
            except Exception as e:
                print(f"Error loading memories: {e}. Starting fresh.", file=sys.stderr)
                self.memories = []
                self.tool_memories = []
        else:
            print("No memory file found. Starting empty.")

    def _rebuild_embeddings_cache(self):
        """Rebuild embedding arrays from stored embeddings or compute from text."""
        if not self.memories:
            self.query_embeddings = np.empty((0, self.embedding_dim))
            self.planning_embeddings = np.empty((0, self.embedding_dim))
            self.experience_embeddings = np.empty((0, self.embedding_dim))
            return

        q_embs, p_embs, e_embs = [], [], []
        for mem in self.memories:
            # If embeddings not stored, compute now (for backward compatibility)
            if mem.query_embedding is None:
                mem.query_embedding = self.embedding_model.encode([mem.query])[0]
            if mem.planning_embedding is None:
                mem.planning_embedding = self.embedding_model.encode([mem.planning_steps])[0] if mem.planning_steps else np.zeros(self.embedding_dim)
            if mem.experience_embedding is None:
                mem.experience_embedding = self.embedding_model.encode([mem.experience_details])[0] if mem.experience_details else np.zeros(self.embedding_dim)
            q_embs.append(mem.query_embedding)
            p_embs.append(mem.planning_embedding)
            e_embs.append(mem.experience_embedding)

        self.query_embeddings = np.vstack(q_embs)
        self.planning_embeddings = np.vstack(p_embs)
        self.experience_embeddings = np.vstack(e_embs)

    def _save_memories_to_json(self):
        try:
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

            serializable_memories = []
            for mem in self.memories:
                serializable_memories.append({
                    'memory_id': mem.memory_id,
                    'query': mem.query,
                    'high_level_strategy': mem.high_level_strategy,
                    'planning_steps': mem.planning_steps,
                    'experience_details': mem.experience_details,
                    'trajectory_summary': mem.trajectory_summary,
                    'tool_patterns': mem.tool_patterns,
                    'metadata': mem.metadata,
                    'query_embedding': mem.query_embedding.tolist() if mem.query_embedding is not None else None,
                    'planning_embedding': mem.planning_embedding.tolist() if mem.planning_embedding is not None else None,
                    'experience_embedding': mem.experience_embedding.tolist() if mem.experience_embedding is not None else None,
                    'retrieval_count': mem.retrieval_count,
                    'success_contribution': mem.success_contribution,
                    'last_accessed': mem.last_accessed
                })

            data = {
                'memories': serializable_memories,
                'tool_memories': self.tool_memories
            }
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving memories: {e}", file=sys.stderr)

    def _reconstruct_trajectory_string(self, trajectory_data: TrajectoryData) -> str:
        if not trajectory_data.trajectory:
            return "No execution trajectory available"
        parts = [f"Task: {trajectory_data.query}", ""]
        for i, step in enumerate(trajectory_data.trajectory, 1):
            step_type = step.get('type', 'step')
            content = step.get('content', '')
            parts.append(f"Step {i} ({step_type}): {content}")
        if trajectory_data.result:
            parts.append("")
            parts.append(f"Final Result: {trajectory_data.result}")
        return "\n".join(parts)

    # --------------------------------------------------------------------------
    # PROVIDE
    # --------------------------------------------------------------------------
    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        if self.embedding_model is None:
            raise Exception("Memory provider not initialized.")

        if not self.memories:
            return MemoryResponse(
                memories=[],
                memory_type=self.memory_type,
                total_count=0,
                request_id=str(uuid.uuid4())
            )

        request_id = str(uuid.uuid4())

        try:
            if request.status == MemoryStatus.BEGIN:
                return self._provide_begin(request, request_id)
            elif request.status == MemoryStatus.IN:
                return self._provide_in(request, request_id)
            else:
                return MemoryResponse(
                    memories=[],
                    memory_type=self.memory_type,
                    total_count=0,
                    request_id=request_id
                )
        except Exception as e:
            print(f"Error in provide_memory: {e}", file=sys.stderr)
            return MemoryResponse(
                memories=[],
                memory_type=self.memory_type,
                total_count=0,
                request_id=request_id
            )

    def _retrieve_similar_memories(self, query: str, weights: Dict[str, float], top_k: int) -> List[Tuple[int, float, EchoMemoryEntry]]:
        """Hybrid retrieval using weighted cosine similarity on query, planning, and experience embeddings."""
        if not self.memories:
            return []

        query_emb = self.embedding_model.encode([query])[0]

        # Compute similarities for each field
        sim_query = cosine_similarity([query_emb], self.query_embeddings)[0] if self.query_embeddings.size > 0 else np.array([])
        sim_planning = cosine_similarity([query_emb], self.planning_embeddings)[0] if self.planning_embeddings.size > 0 else np.array([])
        sim_experience = cosine_similarity([query_emb], self.experience_embeddings)[0] if self.experience_embeddings.size > 0 else np.array([])

        # Weighted combination
        combined = (weights.get('query', 0.0) * sim_query +
                    weights.get('planning', 0.0) * sim_planning +
                    weights.get('experience', 0.0) * sim_experience)

        # Adaptive filtering: skip if top score below threshold
        if len(combined) == 0:
            return []

        top_indices = np.argsort(combined)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            score = float(combined[idx])
            if score < self.similarity_threshold:
                continue
            results.append((idx, score, self.memories[idx]))

        # Update retrieval counts and last accessed
        for idx, _, _ in results:
            self.memories[idx].retrieval_count += 1
            self.memories[idx].last_accessed = datetime.now().isoformat()

        return results

    def _provide_begin(self, request: MemoryRequest, request_id: str) -> MemoryResponse:
        results = self._retrieve_similar_memories(
            query=request.query,
            weights=self.weights_begin,
            top_k=self.top_k_begin
        )
        if not results:
            return MemoryResponse(
                memories=[],
                memory_type=self.memory_type,
                total_count=0,
                request_id=request_id
            )

        # Synthesize high-level guidance
        synthesized_text = self._synthesize_begin_guidance(results, request)

        mem_item = MemoryItem(
            id=str(uuid.uuid4()),
            content=synthesized_text,
            metadata={
                'source_ids': [r[2].memory_id for r in results],
                'avg_score': sum(r[1] for r in results) / len(results),
                'phase': 'begin'
            },
            score=results[0][1],
            type=MemoryItemType.TEXT
        )
        return MemoryResponse(
            memories=[mem_item],
            memory_type=self.memory_type,
            total_count=1,
            request_id=request_id
        )

    def _synthesize_begin_guidance(self, results: List[Tuple[int, float, EchoMemoryEntry]], request: MemoryRequest) -> str:
        if not self.model:
            # Fallback: concatenate high-level strategies
            strategies = [r[2].high_level_strategy for r in results if r[2].high_level_strategy]
            return "\n".join(strategies) if strategies else results[0][2].query

        # Build context from top memories
        context_parts = []
        for i, (_, score, mem) in enumerate(results[:3], 1):
            context_parts.append(f"Memory {i} (relevance: {score:.2f}):")
            if mem.high_level_strategy:
                context_parts.append(f"Strategy: {mem.high_level_strategy}")
            if mem.planning_steps:
                context_parts.append(f"Planning Steps: {mem.planning_steps}")
        context = "\n".join(context_parts)

        prompt = f"""You are an adaptive memory synthesizer. Based on the following similar past experiences, generate concise, actionable guidance for the current task.

Current Task: {request.query}

Retrieved experiences:
{context}

Generate a single paragraph of guidance that:
- Focuses on the most relevant strategy and key planning steps.
- Provides specific, actionable advice.
- Uses natural, suggestive language (not imperative).
- Is no more than 150 words.
"""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            response = self.model(messages)
            guidance = getattr(response, "content", str(response)).strip()
            return guidance if guidance else context
        except Exception as e:
            print(f"LLM synthesis failed: {e}", file=sys.stderr)
            return context

    def _provide_in(self, request: MemoryRequest, request_id: str) -> MemoryResponse:
        # Use the context (current state) as query to retrieve detailed experience
        query = request.context if request.context else request.query
        results = self._retrieve_similar_memories(
            query=query,
            weights=self.weights_in,
            top_k=self.top_k_in
        )
        if not results:
            return MemoryResponse(
                memories=[],
                memory_type=self.memory_type,
                total_count=0,
                request_id=request_id
            )

        # Synthesize operational tip
        synthesized = self._synthesize_in_guidance(results, request)

        # Also retrieve tool memories if relevant
        tool_items = self._retrieve_tool_memories(query, top_k=2)

        memories = []
        if synthesized:
            mem_item = MemoryItem(
                id=str(uuid.uuid4()),
                content=synthesized,
                metadata={'phase': 'in', 'source_ids': [r[2].memory_id for r in results]},
                score=results[0][1],
                type=MemoryItemType.TEXT
            )
            memories.append(mem_item)
        for tm in tool_items:
            memories.append(tm)

        return MemoryResponse(
            memories=memories,
            memory_type=self.memory_type,
            total_count=len(memories),
            request_id=request_id
        )

    def _synthesize_in_guidance(self, results: List[Tuple[int, float, EchoMemoryEntry]], request: MemoryRequest) -> str:
        if not self.model:
            # Fallback: most relevant experience details
            return results[0][2].experience_details if results else ""

        context_parts = []
        for _, score, mem in results[:2]:
            context_parts.append(f"Experience snippet (relevance {score:.2f}): {mem.experience_details[:300]}")
        context = "\n".join(context_parts)

        prompt = f"""Provide a short, actionable tip (2-3 sentences) for the current task execution, based on stored operational experience.

Current context: {request.context if request.context else request.query}

Similar past experience:
{context}

Output only the tip, no extra text.
"""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            response = self.model(messages)
            tip = getattr(response, "content", str(response)).strip()
            return tip if tip else context
        except Exception as e:
            print(f"LLM IN synthesis failed: {e}", file=sys.stderr)
            return context

    def _retrieve_tool_memories(self, query: str, top_k: int) -> List[MemoryItem]:
        if not self.tool_memories:
            return []
        # Simple keyword embedding matching on tool description
        tool_texts = [tm.get('description', '') for tm in self.tool_memories]
        if not any(tool_texts):
            return []
        try:
            query_emb = self.embedding_model.encode([query])[0]
            tool_embs = self.embedding_model.encode(tool_texts)
            sims = cosine_similarity([query_emb], tool_embs)[0]
            top_indices = np.argsort(sims)[-top_k:][::-1]
            items = []
            for idx in top_indices:
                if sims[idx] < 0.4:
                    continue
                tm = self.tool_memories[idx]
                items.append(MemoryItem(
                    id=str(uuid.uuid4()),
                    content=tm,
                    metadata={'tool_name': tm.get('tool_name', 'unknown')},
                    score=float(sims[idx]),
                    type=MemoryItemType.API
                ))
            return items
        except Exception as e:
            print(f"Tool memory retrieval error: {e}", file=sys.stderr)
            return []

    # --------------------------------------------------------------------------
    # TAKE-IN
    # --------------------------------------------------------------------------
    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        if self.embedding_model is None:
            return (False, "Provider not initialized.")

        if not self.model:
            return (False, "No LLM model available for summarization.")

        # Only store successful tasks
        if not self._is_task_successful(trajectory_data):
            return (False, "Skipping: task not successful.")

        try:
            # 1. Summarize trajectory into multi-level memory
            memory_summary = self._summarize_trajectory(trajectory_data)
            if not memory_summary:
                return (False, "Summarization failed.")

            # 2. Extract tool patterns
            tool_patterns = self._extract_tool_patterns(trajectory_data)

            # 3. Check redundancy
            new_entry = EchoMemoryEntry(
                memory_id=str(uuid.uuid4()),
                query=trajectory_data.query,
                high_level_strategy=memory_summary.get('high_level_strategy', ''),
                planning_steps=memory_summary.get('planning_steps', ''),
                experience_details=memory_summary.get('experience_details', ''),
                trajectory_summary=memory_summary.get('trajectory_summary', ''),
                tool_patterns=tool_patterns,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'task_id': trajectory_data.metadata.get('task_id', ''),
                    'status': 'success',
                    'is_correct': True
                },
                retrieval_count=0,
                success_contribution=0.0,
                last_accessed=None
            )

            # Compute embeddings
            try:
                new_entry.query_embedding = self.embedding_model.encode([new_entry.query])[0]
                if new_entry.planning_steps:
                    new_entry.planning_embedding = self.embedding_model.encode([new_entry.planning_steps])[0]
                else:
                    new_entry.planning_embedding = np.zeros(self.embedding_dim)
                if new_entry.experience_details:
                    new_entry.experience_embedding = self.embedding_model.encode([new_entry.experience_details])[0]
                else:
                    new_entry.experience_embedding = np.zeros(self.embedding_dim)
            except Exception as e:
                print(f"Embedding computation failed: {e}", file=sys.stderr)
                return (False, "Embedding error.")

            # 4. Redundancy detection: compare with existing memories (based on planning and experience)
            merged = self._check_redundancy_and_merge(new_entry)
            if merged:
                self._save_memories_to_json()
                return (True, f"Merged with existing memory {merged.memory_id}")
            else:
                # Append new memory
                self.memories.append(new_entry)
                self.query_embeddings = np.vstack([self.query_embeddings, new_entry.query_embedding]) if self.query_embeddings.size > 0 else np.array([new_entry.query_embedding])
                self.planning_embeddings = np.vstack([self.planning_embeddings, new_entry.planning_embedding]) if self.planning_embeddings.size > 0 else np.array([new_entry.planning_embedding])
                self.experience_embeddings = np.vstack([self.experience_embeddings, new_entry.experience_embedding]) if self.experience_embeddings.size > 0 else np.array([new_entry.experience_embedding])

                # Merge tool patterns into global list (avoid duplicates)
                self._merge_tool_patterns(tool_patterns)

                self._save_memories_to_json()
                return (True, f"Stored memory {new_entry.memory_id}")
        except Exception as e:
            print(f"Error in take_in_memory: {e}", file=sys.stderr)
            return (False, f"Exception: {e}")

    def _is_task_successful(self, trajectory_data: TrajectoryData) -> bool:
        metadata = trajectory_data.metadata or {}
        return metadata.get('is_correct', False) or metadata.get('success', False) or metadata.get('task_success', False)

    def _summarize_trajectory(self, trajectory_data: TrajectoryData) -> Optional[Dict[str, str]]:
        """Use LLM to extract multi-level memory components."""
        trajectory_text = self._reconstruct_trajectory_string(trajectory_data)

        prompt = f"""You are an expert memory extractor. Analyze the following successful task execution and produce a structured memory entry.

Task: {trajectory_data.query}
Trajectory:
{trajectory_text}

Extract the following memory components in JSON format:
{{
    "high_level_strategy": "One-line abstract strategy that captures the core approach (e.g., 'Use multi-source validation with cross-referencing')",
    "planning_steps": "Numbered bullet points of the main planning steps (2-4 steps)",
    "experience_details": "Detailed operational tips, pitfalls avoided, and techniques used (3-5 sentences)",
    "trajectory_summary": "Ultra-concise summary of the entire trajectory (max 50 words)"
}}

Ensure each field is substantive and actionable. Return ONLY the JSON.
"""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            response = self.model(messages)
            resp_text = getattr(response, "content", str(response)).strip()
            # Extract JSON
            json_match = re.search(r'\{.*\}', resp_text, re.DOTALL)
            if json_match:
                summary = json.loads(json_match.group(0))
                # Validate
                required = ['high_level_strategy', 'planning_steps', 'experience_details', 'trajectory_summary']
                if all(field in summary and len(summary[field].strip()) >= 20 for field in required):
                    return summary
            print("Failed to extract valid summary from LLM", file=sys.stderr)
            return None
        except Exception as e:
            print(f"LLM summarization error: {e}", file=sys.stderr)
            return None

    def _extract_tool_patterns(self, trajectory_data: TrajectoryData) -> List[Dict]:
        """Analyze trajectory steps and extract reusable tool call patterns."""
        if not trajectory_data.trajectory:
            return []
        patterns = []
        tool_calls = []
        for step in trajectory_data.trajectory:
            if step.get('type') == 'tool_call':
                content = step.get('content', '{}')
                try:
                    call = json.loads(content) if isinstance(content, str) else content
                    tool_name = call.get('tool_name', call.get('name', 'unknown'))
                    args = call.get('arguments', call.get('parameters', {}))
                    tool_calls.append({'tool_name': tool_name, 'args': str(args)[:200]})
                except:
                    pass
        # Cluster similar tool call sequences (simplified: just group by tool name)
        from collections import Counter
        if tool_calls:
            # Keep unique tool names with description
            seen = set()
            for tc in tool_calls:
                key = tc['tool_name']
                if key not in seen:
                    seen.add(key)
                    patterns.append({
                        'tool_name': key,
                        'description': f"Call {key} with parameters like {tc['args']}",
                        'usage_count': 1
                    })
        return patterns

    def _check_redundancy_and_merge(self, new_entry: EchoMemoryEntry) -> Optional[EchoMemoryEntry]:
        """Check if very similar memory exists; if so, merge and return merged entry."""
        if not self.memories:
            return None
        # Compute similarity with all existing memories
        # Use planning and experience embeddings
        if self.planning_embeddings.size == 0:
            return None
        # Reduce dimensionality: concatenate planning and experience embeddings
        new_combined = np.concatenate([new_entry.planning_embedding, new_entry.experience_embedding])
        combined_existing = []
        for mem in self.memories:
            comb = np.concatenate([mem.planning_embedding, mem.experience_embedding])
            combined_existing.append(comb)
        combined_existing = np.array(combined_existing)

        sims = cosine_similarity([new_combined], combined_existing)[0]
        best_idx = np.argmax(sims)
        if sims[best_idx] >= self.redundancy_threshold:
            existing = self.memories[best_idx]
            # Merge: keep longer, more detailed fields
            if len(new_entry.high_level_strategy) > len(existing.high_level_strategy):
                existing.high_level_strategy = new_entry.high_level_strategy
            if len(new_entry.planning_steps) > len(existing.planning_steps):
                existing.planning_steps = new_entry.planning_steps
            if len(new_entry.experience_details) > len(existing.experience_details):
                existing.experience_details = new_entry.experience_details
            # Update timestamp
            existing.metadata['timestamp'] = datetime.now().isoformat()
            # Increment usage count
            existing.retrieval_count += 1
            # Average embedding (optional)
            existing.query_embedding = (existing.query_embedding + new_entry.query_embedding) / 2
            existing.planning_embedding = (existing.planning_embedding + new_entry.planning_embedding) / 2
            existing.experience_embedding = (existing.experience_embedding + new_entry.experience_embedding) / 2
            # Merge tool patterns
            existing.tool_patterns.extend(new_entry.tool_patterns)
            return existing
        return None

    def _merge_tool_patterns(self, new_patterns: List[Dict]):
        """Add new tool patterns to global list, merging duplicates."""
        existing_names = {tm['tool_name']: idx for idx, tm in enumerate(self.tool_memories)}
        for pat in new_patterns:
            if pat['tool_name'] in existing_names:
                idx = existing_names[pat['tool_name']]
                self.tool_memories[idx]['usage_count'] += 1
                # Optionally update description
            else:
                self.tool_memories.append(pat)

    # Maintenance: prune low-utility memories
    def _prune_memories(self, max_age_days=30, min_access=2):
        """Remove memories that are old and rarely used."""
        now = datetime.now()
        to_remove = []
        for i, mem in enumerate(self.memories):
            if mem.last_accessed:
                last = datetime.fromisoformat(mem.last_accessed)
                age_days = (now - last).days
                if age_days > max_age_days and mem.retrieval_count < min_access:
                    to_remove.append(i)
        if to_remove:
            print(f"Pruning {len(to_remove)} low-utility memories.")
            for idx in reversed(to_remove):
                del self.memories[idx]
            self._rebuild_embeddings_cache()
            self._save_memories_to_json()