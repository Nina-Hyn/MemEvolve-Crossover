import json
import os
import uuid
import sys
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
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
class MemoryEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    plan: str = ""
    experience: str = ""
    detailed_plan: str = ""
    detailed_experience: str = ""
    trajectory_summary: str = ""
    failure_reason: str = ""
    is_failure: bool = False
    utility: float = 1.0
    timestamp: str = ""
    combined_embedding: Optional[np.ndarray] = None


class PathfinderMemoryProvider(BaseMemoryProvider):
    """
    Hybrid memory provider combining agent_kb's structured retrieval with voyager's trajectory indexing.
    Features:
    - Phase-aware retrieval (BEGIN strategic, IN execution tips)
    - Failure pattern learning
    - Variable abstraction levels (concise/detailed)
    - Incremental embedding updates
    - Semantic deduplication with merging
    - Utility scoring for relevance promotion
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(memory_type=MemoryType.PATHFINDER, config=config)
        self.model = self.config.get('model', None)
        self.db_path = self.config.get('db_path', './storage/pathfinder/pathfinder_memory.json')
        self.top_k_begin = self.config.get('top_k_begin', 3)
        self.top_k_in = self.config.get('top_k_in', 1)
        self.failure_threshold = self.config.get('failure_threshold', 0.75)
        self.merge_threshold = self.config.get('merge_threshold', 0.95)

        self.embedding_model_name = self.config.get('embedding_model',
                                                    'sentence-transformers/all-MiniLM-L6-v2')
        self.model_cache_dir = self.config.get('model_cache_dir', './storage/models')

        self.embedding_model: Optional[SentenceTransformer] = None
        self.memories: List[MemoryEntry] = []
        self.embeddings_cache: Optional[np.ndarray] = None

    def initialize(self) -> bool:
        try:
            self.embedding_model = load_embedding_model(
                model_name=self.embedding_model_name,
                cache_dir=self.model_cache_dir
            )
            self._load_memories()
            print(f"PathfinderMemoryProvider initialized. {len(self.memories)} memories loaded.")
            return True
        except Exception as e:
            print(f"Error initializing PathfinderMemoryProvider: {e}", file=sys.stderr)
            return False

    def _load_memories(self):
        if not os.path.exists(self.db_path):
            self.memories = []
            self.embeddings_cache = np.empty((0, self.embedding_model.get_sentence_embedding_dimension()))
            return
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.memories = []
            for entry in data:
                mem = MemoryEntry(**{k: v for k, v in entry.items() if k != 'combined_embedding'})
                if 'combined_embedding' in entry and entry['combined_embedding'] is not None:
                    mem.combined_embedding = np.array(entry['combined_embedding'])
                self.memories.append(mem)
            # rebuild embeddings array
            embeddings_list = [m.combined_embedding for m in self.memories if m.combined_embedding is not None]
            if embeddings_list:
                self.embeddings_cache = np.array(embeddings_list)
            else:
                self.embeddings_cache = np.empty((0, self.embedding_model.get_sentence_embedding_dimension()))
        except Exception as e:
            print(f"Error loading memories: {e}", file=sys.stderr)
            self.memories = []
            self.embeddings_cache = np.empty((0, self.embedding_model.get_sentence_embedding_dimension()))

    def _save_memories(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = []
        for mem in self.memories:
            d = mem.__dict__.copy()
            if d['combined_embedding'] is not None:
                d['combined_embedding'] = d['combined_embedding'].tolist()
            data.append(d)
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _compute_combined_text(self, mem: MemoryEntry) -> str:
        if mem.is_failure:
            return f"Query: {mem.query}\nFailure: {mem.failure_reason}"
        else:
            return f"Query: {mem.query}\nPlan: {mem.plan}\nExperience: {mem.experience}"

    def _embed_text(self, text: str) -> np.ndarray:
        return self.embedding_model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]

    def _add_memory(self, mem: MemoryEntry):
        # deduplication check
        if len(self.memories) > 0:
            combined_text = self._compute_combined_text(mem)
            new_emb = self._embed_text(combined_text)
            sims = cosine_similarity([new_emb], self.embeddings_cache)[0]
            max_idx = np.argmax(sims)
            if sims[max_idx] > self.merge_threshold:
                existing = self.memories[max_idx]
                # merge: update fields, keep higher utility, combine experiences
                if not mem.is_failure and not existing.is_failure:
                    existing.plan = mem.plan if len(mem.plan) > len(existing.plan) else existing.plan
                    existing.experience = existing.experience + "\n" + mem.experience if mem.experience else existing.experience
                    existing.detailed_plan = mem.detailed_plan if len(mem.detailed_plan) > len(existing.detailed_plan) else existing.detailed_plan
                    existing.detailed_experience = existing.detailed_experience + "\n" + mem.detailed_experience if mem.detailed_experience else existing.detailed_experience
                    existing.utility = max(existing.utility, mem.utility)
                elif mem.is_failure and not existing.is_failure:
                    # keep existing as success, but add failure reason as metadata? maybe store separately
                    pass
                # update embedding (recompute from merged text)
                combined_text = self._compute_combined_text(existing)
                new_emb = self._embed_text(combined_text)
                existing.combined_embedding = new_emb
                self.embeddings_cache[max_idx] = new_emb
                print(f"Merged memory {mem.id} into existing {existing.id}")
                return

        # add new
        combined_text = self._compute_combined_text(mem)
        new_emb = self._embed_text(combined_text)
        mem.combined_embedding = new_emb
        self.memories.append(mem)
        self.embeddings_cache = np.vstack([self.embeddings_cache, new_emb]) if self.embeddings_cache.size > 0 else np.array([new_emb])

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        if self.embedding_model is None or len(self.memories) == 0:
            return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0,
                                  request_id=str(uuid.uuid4()))

        if request.status == MemoryStatus.BEGIN:
            return self._provide_begin(request)
        elif request.status == MemoryStatus.IN:
            return self._provide_in(request)
        else:
            return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0,
                                  request_id=str(uuid.uuid4()))

    def _provide_begin(self, request: MemoryRequest) -> MemoryResponse:
        try:
            query = request.query
            query_emb = self._embed_text(query)
            similarities = cosine_similarity([query_emb], self.embeddings_cache)[0]

            # separate success and failure indices
            success_indices = [i for i, m in enumerate(self.memories) if not m.is_failure]
            failure_indices = [i for i, m in enumerate(self.memories) if m.is_failure]

            # get top success memories
            success_scores = [(i, similarities[i]) for i in success_indices]
            success_scores.sort(key=lambda x: x[1], reverse=True)
            top_success = success_scores[:self.top_k_begin]

            # get top failure (only if above threshold)
            failure_scores = [(i, similarities[i]) for i in failure_indices if similarities[i] >= self.failure_threshold]
            failure_scores.sort(key=lambda x: x[1], reverse=True)
            top_failure = failure_scores[:1]  # at most 1 failure warning

            retrieved_content = ""

            # synthesize student guidance from plans of top success memories
            if top_success and self.model:
                plans = []
                for idx, score in top_success:
                    mem = self.memories[idx]
                    if mem.plan:
                        plans.append(f"Similar task: {mem.query}\nPlan: {mem.plan}")
                if plans:
                    plan_context = "\n\n".join(plans)
                    prompt = f"""You are assisting an agent by providing planning guidance. Based on past successful plans for similar tasks, create concise, actionable suggestions (2-3 items) for the current task.

Current Task: {query}

Past Plans:
{plan_context}

Provide suggestions only, numbered, without headings."""
                    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                    response = self.model(messages)
                    student_guidance = getattr(response, "content", str(response)).strip()
                    if student_guidance:
                        retrieved_content += f"## Planning Guidance\n{student_guidance}\n\n"

            # synthesize teacher guidance from experiences of top success memories
            if top_success and self.model:
                experiences = []
                for idx, score in top_success:
                    mem = self.memories[idx]
                    if mem.experience:
                        experiences.append(f"Query: {mem.query}\nExperience: {mem.experience}")
                if experiences:
                    exp_context = "\n\n".join(experiences)
                    prompt = f"""You are an experienced teacher guiding an agent. Synthesize the following past experiences into unified operational advice for the current task.

Current Task: {query}

Past Experiences:
{exp_context}

Provide 2-3 sentences of actionable advice."""
                    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                    response = self.model(messages)
                    teacher_guidance = getattr(response, "content", str(response)).strip()
                    if teacher_guidance:
                        retrieved_content += f"## Experience Guidance\n{teacher_guidance}\n\n"

            # include failure warnings
            if top_failure and self.model:
                for idx, score in top_failure:
                    mem = self.memories[idx]
                    if mem.failure_reason:
                        retrieved_content += f"## Failure Warning\nSimilar task previously failed due to: {mem.failure_reason}\n"

            if not retrieved_content.strip():
                # fallback: return top success queries
                for idx, score in top_success[:1]:
                    mem = self.memories[idx]
                    retrieved_content = f"Past similar task: {mem.query}"

            memory_item = MemoryItem(
                id=str(uuid.uuid4()),
                content=retrieved_content.strip(),
                metadata={"phase": "begin", "num_sources": len(top_success)},
                score=float(top_success[0][1]) if top_success else 0.0,
                type=MemoryItemType.TEXT
            )
            return MemoryResponse(
                memories=[memory_item],
                memory_type=self.memory_type,
                total_count=1,
                request_id=str(uuid.uuid4())
            )
        except Exception as e:
            print(f"Error in provide_begin: {e}", file=sys.stderr)
            return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0,
                                  request_id=str(uuid.uuid4()))

    def _provide_in(self, request: MemoryRequest) -> MemoryResponse:
        try:
            # use request.context (current state) or query as context
            context = request.context if request.context else request.query
            query_emb = self._embed_text(context)
            similarities = cosine_similarity([query_emb], self.embeddings_cache)[0]

            # retrieve top success memory for execution tips
            success_indices = [i for i, m in enumerate(self.memories) if not m.is_failure]
            if not success_indices:
                return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0,
                                      request_id=str(uuid.uuid4()))
            success_scores = [(i, similarities[i]) for i in success_indices]
            success_scores.sort(key=lambda x: x[1], reverse=True)
            top_in = success_scores[:self.top_k_in]

            # also retrieve failure warning if above threshold
            failure_indices = [i for i, m in enumerate(self.memories) if m.is_failure and similarities[i] >= self.failure_threshold]
            failure_scores = [(i, similarities[i]) for i in failure_indices]
            failure_scores.sort(key=lambda x: x[1], reverse=True)
            top_failure = failure_scores[:1]

            content_parts = []
            for idx, score in top_in:
                mem = self.memories[idx]
                # use concise experience for execution tips
                if mem.experience:
                    content_parts.append(f"Tip from similar execution:\n{mem.experience}")
                elif mem.trajectory_summary:
                    content_parts.append(f"Trajectory hint: {mem.trajectory_summary}")
            for idx, score in top_failure:
                mem = self.memories[idx]
                if mem.failure_reason:
                    content_parts.append(f"Warning: similar situation previously failed due to: {mem.failure_reason}")

            if not content_parts:
                return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0,
                                      request_id=str(uuid.uuid4()))

            memory_item = MemoryItem(
                id=str(uuid.uuid4()),
                content="\n\n".join(content_parts),
                metadata={"phase": "in"},
                score=float(top_in[0][1]) if top_in else 0.0,
                type=MemoryItemType.TEXT
            )
            return MemoryResponse(
                memories=[memory_item],
                memory_type=self.memory_type,
                total_count=1,
                request_id=str(uuid.uuid4())
            )
        except Exception as e:
            print(f"Error in provide_in: {e}", file=sys.stderr)
            return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0,
                                  request_id=str(uuid.uuid4()))

    def take_in_memory(self, trajectory_data: TrajectoryData) -> Tuple[bool, str]:
        if self.embedding_model is None:
            return False, "Memory provider not initialized."

        try:
            # Determine success/failure
            is_success = trajectory_data.metadata.get("is_correct", False) or trajectory_data.metadata.get("success", False)

            # Format trajectory
            trajectory_str = self._format_trajectory(trajectory_data)

            if self.model is None:
                return False, "No LLM model for summarization."

            mem = MemoryEntry(
                id=str(uuid.uuid4()),
                query=trajectory_data.query,
                timestamp=datetime.now().isoformat()
            )

            if is_success:
                # Generate concise plan and experience using LLM
                prompt_plan = f"""Given the following successful task execution, extract a concise strategic plan (2-3 steps) and actionable experience (key lessons). Keep each under 100 words.

Task: {trajectory_data.query}
Trajectory: {trajectory_str}
Final Result: {trajectory_data.result}

Output exactly in this format:
Plan: <concise plan>
Experience: <concise experience>"""
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt_plan}]}]
                response = self.model(messages)
                result_text = getattr(response, "content", str(response)).strip()
                if result_text:
                    lines = result_text.split('\n')
                    for line in lines:
                        if line.lower().startswith("plan:"):
                            mem.plan = line[5:].strip()
                        elif line.lower().startswith("experience:"):
                            mem.experience = line[11:].strip()

                # Generate detailed plan and experience
                prompt_detailed = f"""Extract a detailed plan (full step-by-step) and detailed experience (best practices, pitfalls avoided) from the successful trajectory.

Task: {trajectory_data.query}
Trajectory: {trajectory_str}
Final Result: {trajectory_data.result}

Output:
Detailed Plan: <full plan>
Detailed Experience: <full experience>"""
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt_detailed}]}]
                response = self.model(messages)
                detailed_text = getattr(response, "content", str(response)).strip()
                if detailed_text:
                    lines = detailed_text.split('\n')
                    for line in lines:
                        if line.lower().startswith("detailed plan:"):
                            mem.detailed_plan = line[14:].strip()
                        elif line.lower().startswith("detailed experience:"):
                            mem.detailed_experience = line[20:].strip()

                # Generate a short trajectory summary (like voyager)
                prompt_summary = f"""Summarize the following trajectory in one sentence.

Task: {trajectory_data.query}
Trajectory: {trajectory_str}

Summary:"""
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt_summary}]}]
                response = self.model(messages)
                summary_text = getattr(response, "content", str(response)).strip()
                if summary_text:
                    mem.trajectory_summary = summary_text

                mem.is_failure = False
                mem.utility = 1.0
            else:
                # Extract failure reason
                prompt_failure = f"""Why did the following task fail? Provide a single reason (1 sentence).

Task: {trajectory_data.query}
Trajectory: {trajectory_str}
Final Result: {trajectory_data.result}

Failure Reason:"""
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt_failure}]}]
                response = self.model(messages)
                failure_reason = getattr(response, "content", str(response)).strip()
                if failure_reason:
                    mem.failure_reason = failure_reason
                mem.is_failure = True
                mem.utility = 0.5  # lower initial utility

            # Add to memory (with dedup)
            self._add_memory(mem)
            self._save_memories()

            summary = f"Stored memory for {trajectory_data.query} ({'success' if is_success else 'failure'})"
            return True, summary
        except Exception as e:
            print(f"Error in take_in_memory: {e}", file=sys.stderr)
            return False, str(e)

    def _format_trajectory(self, trajectory_data: TrajectoryData) -> str:
        if not trajectory_data.trajectory:
            return "No trajectory."
        parts = []
        parts.append(f"Task: {trajectory_data.query}")
        for i, step in enumerate(trajectory_data.trajectory, 1):
            parts.append(f"Step {i}: {step.get('type', '')} - {step.get('content', '')}")
        if trajectory_data.result:
            parts.append(f"Result: {trajectory_data.result}")
        return "\n".join(parts)

    def update_memory_utility(self, memory_id: str, increment: float) -> bool:
        """Optional: adjust utility based on external feedback."""
        for mem in self.memories:
            if mem.id == memory_id:
                mem.utility = max(0.0, min(2.0, mem.utility + increment))
                self._save_memories()
                return True
        return False


# Alias for PROVIDER_MAPPING / generated configs that use PathfinderProvider
PathfinderProvider = PathfinderMemoryProvider