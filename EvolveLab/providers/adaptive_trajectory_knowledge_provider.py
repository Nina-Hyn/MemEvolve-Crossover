import json
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from ..base_memory import BaseMemoryProvider
from ..memory_types import (
    MemoryRequest, MemoryResponse, TrajectoryData,
    MemoryType, MemoryItem, MemoryStatus, MemoryItemType
)

def load_embedding_model(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                         cache_dir: str = './storage/models') -> SentenceTransformer:
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, model_name.replace('/', '_'))
    try:
        if os.path.exists(local_path) and os.listdir(local_path):
            return SentenceTransformer(local_path)
    except Exception:
        pass
    try:
        model = SentenceTransformer(model_name)
        model.save(local_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model {model_name}: {e}")

class AdaptiveTrajectoryKnowledgeProvider(BaseMemoryProvider):

    def __init__(self, config: Optional[dict] = None):
        super().__init__(memory_type=MemoryType.ADAPTIVE_TRAJECTORY_KNOWLEDGE, config=config)
        self.model = self.config.get('model', None)
        self.db_path = self.config.get('db_path', './storage/adaptive_trajectory_knowledge/memory_store.json')
        self.embedding_model_name = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.model_cache_dir = self.config.get('embedding_cache', './storage/models')
        self.top_k = self.config.get('top_k', 3)
        self.max_memories = self.config.get('max_memories', 500)
        self.field_weights = self.config.get('field_weights', {
            'query': 0.5, 'plan': 0.3, 'experience': 0.2
        })
        self.similarity_threshold = self.config.get('dedup_threshold', 0.95)
        self.recency_weight = self.config.get('recency_weight', 0.2)
        self.utility_weight = self.config.get('utility_weight', 0.3)

        self.embedding_model: Optional[SentenceTransformer] = None
        self.memories: List[dict] = []
        self.field_embeddings: Dict[str, np.ndarray] = {
            'query': np.empty((0, 384)),
            'plan': np.empty((0, 384)),
            'experience': np.empty((0, 384)),
            'failure': np.empty((0, 384))
        }
        self.next_id = 0

    def initialize(self) -> bool:
        try:
            self.embedding_model = load_embedding_model(self.embedding_model_name, self.model_cache_dir)
            self._load_memories()
            return True
        except Exception as e:
            print(f"Error initializing AdaptiveTrajectoryKnowledgeProvider: {e}", file=sys.stderr)
            return False

    def _load_memories(self):
        if not os.path.exists(self.db_path):
            self.memories = []
            return
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.memories = data.get('memories', [])
            for field in ['query', 'plan', 'experience', 'failure']:
                emb_list = data.get(f'{field}_embeddings', [])
                if emb_list:
                    self.field_embeddings[field] = np.array(emb_list)
                else:
                    self.field_embeddings[field] = np.empty((0, self.embedding_model.get_sentence_embedding_dimension()))
            self.next_id = max((m.get('id', 0) for m in self.memories), default=0) + 1
        except Exception as e:
            print(f"Error loading memory store: {e}", file=sys.stderr)
            self.memories = []

    def _save_memories(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        data = {
            'memories': self.memories,
            'query_embeddings': self.field_embeddings['query'].tolist(),
            'plan_embeddings': self.field_embeddings['plan'].tolist(),
            'experience_embeddings': self.field_embeddings['experience'].tolist(),
            'failure_embeddings': self.field_embeddings['failure'].tolist()
        }
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _compute_utility_score(self, memory: dict) -> float:
        base = 1.0 if memory.get('is_success', True) else 0.4
        now = datetime.now()
        age_days = (now - datetime.fromisoformat(memory.get('timestamp', now.isoformat()))).days
        recency = max(0, 1 - age_days / 365.0)
        return base * 0.7 + recency * 0.3

    def _embed_fields(self, memory: dict) -> Dict[str, Optional[np.ndarray]]:
        result = {}
        texts = {
            'query': memory.get('query', ''),
            'plan': (memory.get('agent_planning', '') + ' ' + memory.get('search_agent_planning', '')).strip(),
            'experience': (memory.get('agent_experience', '') + ' ' + memory.get('search_agent_experience', '')).strip(),
            'failure': memory.get('failure_analysis', '')
        }
        for field, text in texts.items():
            if text:
                result[field] = self.embedding_model.encode([text], convert_to_numpy=True)[0]
            else:
                result[field] = None
        return result

    def _retrieve(self, query: str, fields: List[str], top_k: int, phase: MemoryStatus) -> List[Tuple[dict, float]]:
        if not self.memories:
            return []

        query_emb = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        scores = []
        for idx, mem in enumerate(self.memories):
            field_score = 0.0
            total_weight = 0.0
            for field in fields:
                weight = self.field_weights.get(field, 0.25)
                if idx < self.field_embeddings[field].shape[0]:
                    mem_emb = self.field_embeddings[field][idx]
                    if np.any(mem_emb):
                        sim = cosine_similarity([query_emb], [mem_emb])[0][0]
                        field_score += weight * sim
                        total_weight += weight
            if total_weight > 0:
                field_score /= total_weight
            else:
                field_score = 0.0

            utility = self._compute_utility_score(mem)
            recency_factor = 1.0  # already integrated in utility
            final_score = field_score * (1 - self.recency_weight - self.utility_weight) \
                          + self.recency_weight * 0.5 + self.utility_weight * utility
            scores.append((mem, final_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        if not self.embedding_model or not self.memories:
            return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0,
                                  request_id=str(uuid.uuid4()))

        if request.status == MemoryStatus.BEGIN:
            refined_query = self._refine_query(request.query) if self.model else request.query
            fields = ['query', 'plan', 'experience']
            top_k = self.top_k
        elif request.status == MemoryStatus.IN:
            refined_query = request.query
            fields = ['experience']
            top_k = min(2, self.top_k)
        else:
            return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0,
                                  request_id=str(uuid.uuid4()))

        retrieved = self._retrieve(refined_query, fields, top_k, request.status)
        if not retrieved:
            return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0,
                                  request_id=str(uuid.uuid4()))

        if request.status == MemoryStatus.BEGIN:
            guidance = self._synthesize_begin_guidance(retrieved, request.query)
        else:
            guidance = self._synthesize_in_guidance(retrieved, request.context)

        memory_item = MemoryItem(
            id=str(uuid.uuid4()),
            content=guidance,
            metadata={
                'num_sources': len(retrieved),
                'status': request.status.value,
                'original_query': request.query
            },
            score=retrieved[0][1],
            type=MemoryItemType.TEXT
        )
        return MemoryResponse(
            memories=[memory_item],
            memory_type=self.memory_type,
            total_count=1,
            request_id=str(uuid.uuid4())
        )

    def _refine_query(self, query: str) -> str:
        prompt = f"""Extract core concepts and search terms from the following user query to retrieve relevant past experiences.

User query: {query}

Return only a short, keyword-rich phrase (max 20 words) that captures the essential task components."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            resp = self.model(messages)
            return getattr(resp, "content", str(resp)).strip() or query
        except Exception:
            return query

    def _synthesize_begin_guidance(self, retrieved: List[Tuple[dict, float]], query: str) -> str:
        if not self.model:
            return self._simple_concatenate(retrieved, 'plan')
        sources = []
        for mem, score in retrieved[:3]:
            s = f"Similar query: {mem.get('query', '')}\nPlanning: {mem.get('agent_planning', '')}\nExperience: {mem.get('agent_experience', '')}"
            sources.append(s)
        context = "\n\n".join(sources)
        prompt = f"""You are an expert assistant providing guidance for a task. Based on the following similar past experiences, produce **concise, actionable** advice (2-3 specific points) for the current task.

Current task: {query}

Past experiences:
{context}

Output only the advice, no extra text."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            resp = self.model(messages)
            return getattr(resp, "content", str(resp)).strip()
        except Exception:
            return self._simple_concatenate(retrieved, 'plan')

    def _synthesize_in_guidance(self, retrieved: List[Tuple[dict, float]], context: str) -> str:
        if not self.model:
            return self._simple_concatenate(retrieved, 'experience')
        snippets = []
        for mem, score in retrieved[:2]:
            exp = mem.get('agent_experience', '') or mem.get('search_agent_experience', '')
            if exp:
                snippets.append(exp[:200])
        combined = "\n".join(snippets)
        prompt = f"""Given the current execution context, extract a short tactical hint (max 2 sentences) from the following experience fragments.

Experience fragments:
{combined}

Current context: {context[:300]}

Hint:"""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            resp = self.model(messages)
            hint = getattr(resp, "content", str(resp)).strip()
            return hint if hint else self._simple_concatenate(retrieved, 'experience')
        except Exception:
            return self._simple_concatenate(retrieved, 'experience')

    def _simple_concatenate(self, retrieved: List[Tuple[dict, float]], field: str) -> str:
        parts = []
        for mem, _ in retrieved:
            val = mem.get(field, '') or ''
            if val:
                parts.append(val)
        return ' '.join(parts) if parts else ''

    def take_in_memory(self, trajectory_data: TrajectoryData) -> Tuple[bool, str]:
        if not self.embedding_model:
            return False, "Provider not initialized"

        is_success = trajectory_data.metadata.get('is_correct', False) if trajectory_data.metadata else False
        try:
            distilled = self._distill_trajectory(trajectory_data)
            if not distilled:
                return False, "Distillation failed"

            new_memory = {
                'id': self.next_id,
                'query': trajectory_data.query,
                'agent_planning': distilled.get('agent_planning', ''),
                'search_agent_planning': distilled.get('search_agent_planning', ''),
                'agent_experience': distilled.get('agent_experience', ''),
                'search_agent_experience': distilled.get('search_agent_experience', ''),
                'failure_analysis': distilled.get('failure_analysis', '') if not is_success else '',
                'is_success': is_success,
                'timestamp': datetime.now().isoformat(),
                'utility_score': 1.0 if is_success else 0.4
            }

            # Deduplication
            dup_idx = self._find_duplicate(new_memory)
            if dup_idx is not None:
                self._merge_memory(dup_idx, new_memory)
                self._save_memories()
                return True, "Updated existing memory (merged)"

            # Add new memory
            embeddings = self._embed_fields(new_memory)
            for field, emb in embeddings.items():
                if emb is not None:
                    self.field_embeddings[field] = np.vstack([self.field_embeddings[field], emb])
                else:
                    self.field_embeddings[field] = np.vstack([self.field_embeddings[field], np.zeros(self.embedding_model.get_sentence_embedding_dimension())])
            self.memories.append(new_memory)
            self.next_id += 1

            # Prune if exceed max
            if len(self.memories) > self.max_memories:
                self._prune()

            self._save_memories()
            return True, "Memory ingested successfully"

        except Exception as e:
            return False, f"Error ingesting memory: {e}"

    def _distill_trajectory(self, trajectory_data: TrajectoryData) -> Optional[Dict[str, str]]:
        if not self.model:
            return None
        is_success = trajectory_data.metadata.get('is_correct', False) if trajectory_data.metadata else False
        traj_str = self._format_trajectory(trajectory_data)
        success_flag = "successful" if is_success else "failed"
        prompt = f"""You are an AI agent trainer. Analyze the following task execution trajectory and extract structured memory.

Task: {trajectory_data.query}
Result: {trajectory_data.result if trajectory_data.result else "N/A"}
This task was {success_flag}.

Trajectory:
{traj_str}

Extract the following fields as a JSON object:
- "agent_planning": Strategic reasoning steps used (concise, numbered).
- "search_agent_planning": Search or tool usage strategy.
- "agent_experience": Lessons learned, best practices, pitfalls (for successful tasks) OR what went wrong (for failed tasks).
- "search_agent_experience": Search-specific insights.
- "failure_analysis" (only if the task failed): Root cause analysis.

Return ONLY valid JSON, no extra text."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            resp = self.model(messages)
            text = getattr(resp, "content", str(resp)).strip()
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_match:
                return None
            parsed = json.loads(json_match.group(0))
            mandatory = ['agent_planning', 'search_agent_planning', 'agent_experience', 'search_agent_experience']
            for field in mandatory:
                if field not in parsed or not parsed[field].strip():
                    return None
            if not is_success and 'failure_analysis' not in parsed:
                parsed['failure_analysis'] = 'No failure analysis extracted'
            return parsed
        except Exception as e:
            print(f"Distillation error: {e}", file=sys.stderr)
            return None

    def _format_trajectory(self, trajectory_data: TrajectoryData) -> str:
        if not trajectory_data.trajectory:
            return "No trajectory"
        parts = [f"Query: {trajectory_data.query}"]
        for i, step in enumerate(trajectory_data.trajectory, 1):
            step_type = step.get('type', 'step')
            content = step.get('content', '')
            parts.append(f"Step {i} ({step_type}): {content[:200]}")
        if trajectory_data.result:
            parts.append(f"Result: {trajectory_data.result}")
        return "\n".join(parts)

    def _find_duplicate(self, new_mem: dict) -> Optional[int]:
        if not self.memories:
            return None
        new_emb = self._embed_fields(new_mem)
        # Compare using query embedding only (fast)
        if new_emb.get('query') is None:
            return None
        existing_queries = self.field_embeddings['query']
        if existing_queries.shape[0] == 0:
            return None
        sims = cosine_similarity([new_emb['query']], existing_queries)[0]
        max_idx = np.argmax(sims)
        if sims[max_idx] >= self.similarity_threshold:
            return max_idx
        return None

    def _merge_memory(self, idx: int, new_mem: dict):
        existing = self.memories[idx]
        # Update timestamp
        existing['timestamp'] = datetime.now().isoformat()
        # Combine text fields (append)
        for field in ['agent_planning', 'search_agent_planning', 'agent_experience', 'search_agent_experience', 'failure_analysis']:
            new_val = new_mem.get(field, '')
            if new_val and new_val not in existing.get(field, ''):
                existing[field] = (existing.get(field, '') + '\n' + new_val).strip()
        # Update success flag (keep true if either is true)
        existing['is_success'] = existing.get('is_success', True) or new_mem.get('is_success', True)
        existing['utility_score'] = max(existing.get('utility_score', 0.5), new_mem.get('utility_score', 0.5))

    def _prune(self):
        # Compute current utility for all entries
        utilities = [self._compute_utility_score(m) for m in self.memories]
        # Sort by utility ascending, prune bottom 20%
        sorted_indices = np.argsort(utilities)
        keep_count = int(self.max_memories * 0.8)
        remove_indices = sorted_indices[:-keep_count] if keep_count > 0 else sorted_indices
        # Rebuild in order
        self.memories = [self.memories[i] for i in sorted_indices[-keep_count:]]
        for field in self.field_embeddings:
            self.field_embeddings[field] = self.field_embeddings[field][sorted_indices[-keep_count:]]