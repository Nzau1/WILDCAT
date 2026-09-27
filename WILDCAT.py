#!/usr/bin/env python3
"""
================================================================================
 WildCat — An AGI-Inspired Engineering Intelligence (full Python core)
================================================================================
 Built from: "AI / AGI Engineering Algorithms Reference"

 Architecture (Section 2):
     USER INPUT -> PERCEPTION -> WORKING MEMORY -> REASONING -> PLANNING
         -> TOOLS -> OBSERVATION -> VERIFICATION -> LONG-TERM MEMORY

 Section coverage:
   §4  Math foundations    : vector math, Bayes, statistics, gradient updates
   §5  Classical ML        : k-means (unsupervised), statistical clustering
   §6  Neural AI           : attention-style weighting, embeddings
   §6.6 Transformer        : pluggable LLM backend (OpenAI/Anthropic/Gemini/Ollama)
   §7  Memory / Retrieval  : embeddings, cosine search, RAG, knowledge graph
   §8  Reasoning           : decomposition, ReAct, hypotheses
   §9  Planning            : A*-style plan ordering, hierarchical plans
   §10 Reinforcement       : Q-learning over response strategies + replay
   §11 Agents              : tool/function calling, observe-think-act loop
   §12 Verification        : critic loop + rule-based checks
   §13 Adaptation          : experience-based + continual (online) learning
   §15 Minimal control alg : the main respond() loop
   §19 Caveat              : an AGI-inspired platform, not human-level AGI

 Requirements:
     pip install numpy
     pip install openai                # optional
     pip install anthropic             # optional
     pip install google-generativeai   # optional
     pip install requests              # for Ollama
================================================================================
"""

from __future__ import annotations

import os
import re
import sys
import json
import math
import pickle
import random
import hashlib
from datetime import datetime
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Iterator

import numpy as np


# ==============================================================================
# .env LOADER (zero-dependency)
# ==============================================================================
def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key, val)
    except Exception as e:
        print(f"[wildcat] .env load failed: {e}", file=sys.stderr)


load_dotenv()


# ==============================================================================
# OPTIONAL LLM SDK IMPORTS
# ==============================================================================
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import anthropic
except Exception:
    anthropic = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    import requests
except Exception:
    requests = None


# ==============================================================================
# GLOBAL CONFIG
# ==============================================================================
STATE_FILE          = os.environ.get("WILDCAT_STATE_FILE", "wildcat_state.pkl")
EMBED_DIM           = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS  = 800
LLM_TIMEOUT         = 60.0

DEFAULT_MODELS = {
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "gemini":    "gemini-1.5-flash",
    "ollama":    os.environ.get("OLLAMA_MODEL", "llama3.1"),
}


# ==============================================================================
# §4  MATH FOUNDATIONS
# ==============================================================================
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def bayes_prior_update(priors: Dict[str, float],
                       likelihoods: Dict[str, float]) -> Dict[str, float]:
    post = {h: priors[h] * likelihoods.get(h, 1e-6) for h in priors}
    z = sum(post.values()) + 1e-12
    return {h: v / z for h, v in post.items()}


# ==============================================================================
# §7.1  VECTOR EMBEDDINGS
# ==============================================================================
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def embed(text: str) -> np.ndarray:
    v = np.zeros(EMBED_DIM, dtype=np.float32)
    toks = tokenize(text)
    for i, t in enumerate(toks):
        grams = [t]
        if i + 1 < len(toks):
            grams.append(f"{t}_{toks[i+1]}")
        for gram in grams:
            h = int(hashlib.md5(gram.encode()).hexdigest(), 16) % EMBED_DIM
            v[h] += 1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else v


# ==============================================================================
# §6.6  LLM BACKEND
# ==============================================================================
class LLMBackend:
    def __init__(self,
                 provider: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 timeout: float = LLM_TIMEOUT,
                 verbose: bool = True):
        self.provider = (provider or os.environ.get("WILDCAT_LLM_PROVIDER", "auto")).lower()
        self.temperature = float(temperature if temperature is not None
                                 else os.environ.get("WILDCAT_LLM_TEMPERATURE", DEFAULT_TEMPERATURE))
        self.max_tokens = int(max_tokens if max_tokens is not None
                              else os.environ.get("WILDCAT_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))
        self.timeout = timeout
        self.verbose = verbose
        self._client = None
        self._model_override = model or os.environ.get("WILDCAT_LLM_MODEL") or None
        self._resolve_provider()

        if verbose and self.provider != "none":
            print(f"[wildcat-llm] provider={self.provider} model={self.model}")
        elif verbose:
            print("[wildcat-llm] no provider detected — using rule-based fallback")

    def _resolve_provider(self) -> None:
        if self.provider == "auto":
            if OpenAI and os.environ.get("OPENAI_API_KEY"):
                self.provider = "openai"
            elif anthropic and os.environ.get("ANTHROPIC_API_KEY"):
                self.provider = "anthropic"
            elif genai and (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
                self.provider = "gemini"
            elif requests and self._ollama_alive():
                self.provider = "ollama"
            else:
                self.provider = "none"

        if self.provider == "openai" and not (OpenAI and os.environ.get("OPENAI_API_KEY")):
            self.provider = "none"
        if self.provider == "anthropic" and not (anthropic and os.environ.get("ANTHROPIC_API_KEY")):
            self.provider = "none"
        if self.provider == "gemini" and not genai:
            self.provider = "none"

        self.model = self._model_override or DEFAULT_MODELS.get(self.provider, "unknown")

        try:
            if self.provider == "openai":
                self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            elif self.provider == "anthropic":
                self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            elif self.provider == "gemini":
                genai.configure(api_key=os.environ.get("GOOGLE_API_KEY")
                                       or os.environ.get("GEMINI_API_KEY"))
                self._client = genai.GenerativeModel(self.model)
            elif self.provider == "ollama":
                self._client = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        except Exception as e:
            if self.verbose:
                print(f"[wildcat-llm] client init failed: {e}", file=sys.stderr)
            self.provider = "none"

    def _ollama_alive(self) -> bool:
        try:
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            r = requests.get(host + "/api/tags", timeout=1.5)
            return r.status_code == 200
        except Exception:
            return False

    def available(self) -> bool:
        return self.provider != "none" and self._client is not None

    def chat(self, system: str, messages: List[Dict[str, str]]) -> str:
        if not self.available():
            raise RuntimeError("No LLM provider available")

        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, *messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            return resp.choices[0].message.content or ""

        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                system=system,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return "".join(b.text for b in resp.content
                           if getattr(b, "type", "") == "text")

        if self.provider == "gemini":
            history = []
            for m in messages:
                role = "user" if m["role"] == "user" else "model"
                history.append({"role": role, "parts": [m["content"]]})
            chat = self._client.start_chat(history=history[:-1] if history else [])
            prompt = (system + "\n\n" + messages[-1]["content"]) if messages else system
            resp = chat.send_message(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )
            return getattr(resp, "text", "") or ""

        if self.provider == "ollama":
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, *messages],
                "stream": False,
                "options": {"temperature": self.temperature,
                            "num_predict": self.max_tokens},
            }
            r = requests.post(self._client + "/api/chat", json=payload,
                              timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "")

        raise RuntimeError(f"Unsupported provider {self.provider}")

    def chat_stream(self, system: str, messages: List[Dict[str, str]]) -> Iterator[str]:
        if not self.available():
            raise RuntimeError("No LLM provider available")

        if self.provider == "openai":
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, *messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                timeout=self.timeout,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
            return

        if self.provider == "anthropic":
            with self._client.messages.stream(
                model=self.model, system=system, messages=messages,
                temperature=self.temperature, max_tokens=self.max_tokens,
            ) as s:
                for text in s.text_stream:
                    yield text
            return

        if self.provider == "gemini":
            history = []
            for m in messages:
                role = "user" if m["role"] == "user" else "model"
                history.append({"role": role, "parts": [m["content"]]})
            chat = self._client.start_chat(history=history[:-1] if history else [])
            prompt = (system + "\n\n" + messages[-1]["content"]) if messages else system
            for piece in chat.send_message(prompt, stream=True):
                if getattr(piece, "text", ""):
                    yield piece.text
            return

        if self.provider == "ollama":
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, *messages],
                "stream": True,
                "options": {"temperature": self.temperature,
                            "num_predict": self.max_tokens},
            }
            with requests.post(self._client + "/api/chat", json=payload,
                               stream=True, timeout=self.timeout) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line.decode("utf-8"))
                        tok = obj.get("message", {}).get("content", "")
                        if tok:
                            yield tok
                        if obj.get("done"):
                            break
                    except Exception:
                        continue
            return

        raise RuntimeError(f"Unsupported provider {self.provider}")


# ==============================================================================
# §2  PERCEPTION LAYER
# ==============================================================================
POS_WORDS = {
    "good", "great", "nice", "love", "thanks", "thank", "awesome", "perfect",
    "yes", "correct", "right", "excellent", "helpful", "cool", "happy", "👍", "❤️",
}
NEG_WORDS = {
    "bad", "wrong", "hate", "no", "nope", "terrible", "awful", "useless",
    "stupid", "annoying", "angry", "sad", "👎", "broken", "fail", "failed",
}
INTENT_RULES = [
    ("greeting",  r"^\s*(hi|hello|hey|yo|hola|good (morning|evening|afternoon))"),
    ("farewell",  r"\b(bye|goodbye|see you|later|exit|quit)\b"),
    ("question",  r"\?|\b(what|why|how|when|where|who|which|can you|could you)\b"),
    ("command",   r"^\s*(please\s+)?(do|run|make|create|build|calculate|compute|find|search|remember|forget)\b"),
    ("feedback",  r"\b(good|bad|wrong|right|correct|incorrect|thanks|thank you)\b"),
    ("statement", r"."),
]


class PerceptionLayer:
    def perceive(self, text: str) -> Dict[str, Any]:
        low = text.lower()
        toks = tokenize(text)

        intent = "statement"
        for name, pat in INTENT_RULES:
            if re.search(pat, low):
                intent = name
                break

        pos = sum(1 for t in toks if t in POS_WORDS)
        neg = sum(1 for t in toks if t in NEG_WORDS)
        if text.count("!") >= 2 or any(e in text for e in ("👍", "❤️", "😊", "🎉")):
            pos += 1
        if text.count("?") >= 3 or any(e in text for e in ("😡", "😠", "👎", "💩")):
            neg += 1

        sentiment = 0
        if pos > neg:
            sentiment = 1
        elif neg > pos:
            sentiment = -1

        return {
            "raw":       text,
            "tokens":    toks,
            "intent":    intent,
            "sentiment": sentiment,
            "pos_hits":  pos,
            "neg_hits":  neg,
            "embedding": embed(text),
            "entities":  self._extract_entities(text),
            "triples":   self._extract_triples(text),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _extract_entities(self, text: str) -> List[str]:
        ents = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
        ents += re.findall(r"\b\d+(?:\.\d+)?\s?(?:nm|km|m|mm|dB|dBm|G\.\d+|°C|%)\b", text)
        return list(dict.fromkeys(ents))

    def _extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        pats = [
            (r"([A-Za-z][\w\s]{1,30}?)\s+is\s+(?:a|an|the)?\s*([\w\s]{1,30})", "is"),
            (r"([A-Za-z][\w\s]{1,30}?)\s+has\s+([\w\s]{1,30})",                "has"),
            (r"([A-Za-z][\w\s]{1,30}?)\s+causes?\s+([\w\s]{1,30})",            "causes"),
            (r"([A-Za-z][\w\s]{1,30}?)\s+measures?\s+([\w\s]{1,30})",          "measures"),
        ]
        out = []
        for pat, rel in pats:
            for m in re.finditer(pat, text):
                s = m.group(1).strip()
                o = m.group(2).strip()
                out.append((s, rel, o))
        return out


# ==============================================================================
# §2  WORKING MEMORY
# ==============================================================================
class WorkingMemory:
    def __init__(self, capacity: int = 8):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, obs: Dict[str, Any]) -> None:
        self.buffer.append(obs)

    def snapshot(self) -> List[Dict[str, Any]]:
        return list(self.buffer)

    def recent_sentiment(self) -> int:
        s = [o["sentiment"] for o in self.buffer if o.get("sentiment") is not None]
        return int(np.sign(sum(s))) if s else 0

    def clear(self) -> None:
        self.buffer.clear()


# ==============================================================================
# §7.2 / §7.3 / §13.1  LONG-TERM MEMORY
# ==============================================================================
@dataclass
class Experience:
    query: str
    response: str
    intent: str
    strategy: str
    reward: float
    timestamp: str


@dataclass
class Case:
    problem: str
    solution: str
    outcome: float
    tags: List[str] = field(default_factory=list)


class LongTermMemory:
    def __init__(self):
        self.vectors: List[np.ndarray] = []
        self.texts: List[str] = []
        self.meta: List[Dict[str, Any]] = []
        self.kg: Dict[Tuple[str, str], set] = defaultdict(set)
        self.experiences: List[Experience] = []
        self.cases: List[Case] = []
        self.trained: int = 0

    def store(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.vectors.append(embed(text))
        self.texts.append(text)
        self.meta.append(meta or {})

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not self.vectors:
            return []
        q = embed(query)
        sims = [cosine(q, v) for v in self.vectors]
        idx = np.argsort(sims)[::-1][:k]
        return [(self.texts[i], float(sims[i]), self.meta[i])
                for i in idx if sims[i] > 0.05]

    def search_training(self, query: str, k: int = 2) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not self.vectors:
            return []
        q = embed(query)
        scored = []
        for i, v in enumerate(self.vectors):
            if self.meta[i].get("type") == "training":
                sim = cosine(q, v)
                if sim > 0.15:
                    scored.append((self.texts[i], sim, self.meta[i]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def add_triples(self, triples: List[Tuple[str, str, str]]) -> None:
        for s, r, o in triples:
            self.kg[(s, r)].add(o)

    def kg_query(self, subject: Optional[str] = None,
                 relation: Optional[str] = None) -> List[Tuple[str, str, str]]:
        out = []
        for (s, r), objs in self.kg.items():
            if subject and s != subject:
                continue
            if relation and r != relation:
                continue
            for o in objs:
                out.append((s, r, o))
        return out

    def add_experience(self, exp: Experience) -> None:
        self.experiences.append(exp)
        self.store(f"Q:{exp.query} A:{exp.response}",
                   {"type": "experience", "reward": exp.reward})

    def similar_experiences(self, query: str, k: int = 3) -> List[Experience]:
        if not self.experiences:
            return []
        q = embed(query)
        scored = [(cosine(q, embed(e.query)), e) for e in self.experiences]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for s, e in scored[:k] if s > 0.1]

    def add_case(self, c: Case) -> None:
        self.cases.append(c)
        self.store(f"CASE:{c.problem} -> {c.solution}",
                   {"type": "case", "outcome": c.outcome})

    def save(self, path: str = STATE_FILE) -> None:
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "vectors":     self.vectors,
                    "texts":       self.texts,
                    "meta":        self.meta,
                    "kg":          dict(self.kg),
                    "experiences": self.experiences,
                    "cases":       self.cases,
                    "trained":     self.trained,
                }, f)
        except Exception as e:
            print(f"[wildcat] save failed: {e}", file=sys.stderr)

    @classmethod
    def load(cls, path: str = STATE_FILE) -> "LongTermMemory":
        m = cls()
        if not os.path.exists(path):
            return m
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
            m.vectors = d.get("vectors", [])
            m.texts = d.get("texts", [])
            m.meta = d.get("meta", [])
            m.kg = defaultdict(set, d.get("kg", {}))
            m.experiences = d.get("experiences", [])
            m.cases = d.get("cases", [])
            m.trained = d.get("trained", 0)
        except Exception as e:
            print(f"[wildcat] load failed, starting fresh: {e}", file=sys.stderr)
        return m

    def clear(self) -> None:
        self.vectors.clear()
        self.texts.clear()
        self.meta.clear()
        self.kg.clear()
        self.experiences.clear()
        self.cases.clear()
        self.trained = 0


# ==============================================================================
# §10.2  Q-LEARNING
# ==============================================================================
class QLearner:
    def __init__(self, alpha: float = 0.3, gamma: float = 0.8):
        self.Q: Dict[Tuple[str, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha
        self.gamma = gamma
        self.pending: Optional[Tuple[Tuple[str, int], str]] = None
        self.actions = ["direct", "clarify", "empathize", "retrieve", "calculate", "suggest"]

    def choose(self, state: Tuple[str, int], explore: float = 0.15) -> str:
        if random.random() < explore:
            return random.choice(self.actions)
        q = self.Q[state]
        return max(self.actions, key=lambda a: q.get(a, 0.0))

    def note(self, state: Tuple[str, int], action: str) -> None:
        self.pending = (state, action)

    def reward(self, r: float, next_state: Tuple[str, int]) -> None:
        if not self.pending:
            return
        s, a = self.pending
        best_next = max(self.Q[next_state].values(), default=0.0)
        old = self.Q[s][a]
        self.Q[s][a] = old + self.alpha * (r + self.gamma * best_next - old)
        self.pending = None


# ==============================================================================
# §8  REASONING ENGINE
# ==============================================================================
class ReasoningEngine:
    def decompose(self, goal: str) -> List[str]:
        g = goal.lower()
        if any(w in g for w in ("diagnose", "debug", "why", "fail", "error", "fault")):
            return ["collect observations", "list candidate causes",
                    "score candidates with evidence", "pick top hypothesis",
                    "propose verification test", "report + store case"]
        if any(w in g for w in ("calculate", "compute", "how much", "how many")):
            return ["extract numbers", "select formula", "compute", "verify units"]
        if any(w in g for w in ("plan", "schedule", "roadmap", "how do i")):
            return ["clarify objective", "list subgoals", "order by dependency",
                    "estimate effort", "propose next action"]
        return ["parse intent", "retrieve relevant memory",
                "form hypotheses", "compose answer", "verify answer"]

    def hypotheses(self, problem: str, context: List[str]) -> Dict[str, float]:
        text = (problem + " " + " ".join(context)).lower()
        priors = {
            "connector issue":       0.25,
            "splice loss":           0.25,
            "macro-bend":            0.25,
            "fiber defect":          0.25,
            "user intent: question": 0.5,
            "user intent: request":  0.5,
        }
        likelihoods = {}
        for h in priors:
            lk = 1.0
            for kw in h.replace(":", " ").split():
                if kw in text:
                    lk *= 2.0
            likelihoods[h] = lk
        return bayes_prior_update(priors, likelihoods)


# ==============================================================================
# §9  PLANNING ENGINE
# ==============================================================================
class PlanningEngine:
    def make_plan(self, goal: str, steps: List[str]) -> List[Dict[str, Any]]:
        plan = []
        for i, s in enumerate(steps):
            plan.append({
                "step":    i + 1,
                "subtask": s,
                "action":  self._action_for(s),
                "cost":    self._estimate_cost(s),
            })
        plan.sort(key=lambda p: p["cost"])
        return plan

    def _action_for(self, s: str) -> str:
        s = s.lower()
        if "retrieve" in s:
            return "search_memory"
        if "compute" in s or "calculate" in s:
            return "python_eval"
        if "verify" in s:
            return "critic_check"
        if "report" in s or "compose" in s:
            return "generate_language"
        if "store" in s:
            return "store_memory"
        return "reason"

    def _estimate_cost(self, s: str) -> float:
        table = {
            "python_eval":       0.2,
            "search_memory":     0.3,
            "reason":            0.5,
            "critic_check":      0.6,
            "generate_language": 0.7,
            "store_memory":      0.8,
        }
        return table.get(self._action_for(s), 1.0)


# ==============================================================================
# §11  TOOL REGISTRY
# ==============================================================================
class ToolRegistry:
    def __init__(self, memory: LongTermMemory):
        self.memory = memory

    def python_eval(self, expr: str) -> str:
        try:
            safe = {k: getattr(math, k)
                    for k in ("sqrt", "sin", "cos", "tan", "log", "log10",
                              "exp", "pi", "e", "pow")}
            if not re.fullmatch(r"[\d\.\+\-\*/\(\)%\s a-zA-Z_]+", expr):
                return "[calc rejected: illegal characters]"
            return str(eval(expr, {"__builtins__": {}}, safe))
        except Exception as e:
            return f"[calc error: {e}]"

    def search_memory(self, query: str) -> List[Tuple[str, float, Dict[str, Any]]]:
        return self.memory.search(query, k=3)

    def run(self, action: str, payload: Any) -> Any:
        if action == "python_eval" and isinstance(payload, str):
            return self.python_eval(payload)
        if action == "search_memory":
            return self.search_memory(str(payload))
        return None


# ==============================================================================
# §12  CRITIC / VERIFICATION
# ==============================================================================
class Critic:
    def verify(self, response: str, state: Dict[str, Any]) -> Tuple[bool, str]:
        if not response or len(response.strip()) < 2:
            return False, "empty response"
        if response.count("(") != response.count(")"):
            return False, "unbalanced parentheses"
        if " nan " in response.lower() and "nanny" not in response.lower():
            return False, "NaN detected"
        for m in re.finditer(r"-?\d+\.?\d*", response):
            try:
                v = float(m.group())
                if abs(v) > 1e12:
                    return False, f"unreasonable magnitude {v}"
            except Exception:
                pass
        return True, "ok"


# ==============================================================================
# §13.2  CONTINUAL LEARNING
# ==============================================================================
class ContinualLearner:
    def __init__(self, k: int = 5):
        self.k = k
        self.centroids: Optional[np.ndarray] = None
        self.counts: Counter = Counter()
        self.topic_history: List[str] = []

    def update(self, obs: Dict[str, Any]) -> str:
        for t in obs["tokens"]:
            self.counts[t] += 1

        v = obs["embedding"].reshape(1, -1).astype(np.float32)

        if self.centroids is None:
            self.centroids = v.copy()
            self.topic_history.append("topic_0")
            return "topic_0"

        dists = [float(np.linalg.norm(v - c)) for c in self.centroids]
        i = int(np.argmin(dists))
        lr = 0.1
        self.centroids[i] = (1 - lr) * self.centroids[i] + lr * v.flatten()
        label = f"topic_{i}"
        self.topic_history.append(label)
        return label


# ==============================================================================
# §15  WILDCAT
# ==============================================================================
class WildCat:
    def __init__(self,
                 name: str = "WildCat",
                 use_llm: bool = True,
                 stream: bool = False,
                 verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.perception = PerceptionLayer()
        self.wm = WorkingMemory(capacity=8)
        self.memory = LongTermMemory.load()
        self.reasoning = ReasoningEngine()
        self.planner = PlanningEngine()
        self.tools = ToolRegistry(self.memory)
        self.critic = Critic()
        self.q = QLearner()
        self.learner = ContinualLearner()
        self.turn = 0
        self.last_state: Optional[Tuple[str, int]] = None
        self.last_response: str = ""
        self.stream = stream

        self.llm: Optional[LLMBackend] = LLMBackend(verbose=verbose) if use_llm else None
        if self.llm is not None and not self.llm.available():
            self.llm = None

    # ------------------------------------------------------------------
    # §15  Minimal AGI-Inspired Control Algorithm — mode-aware
    # ------------------------------------------------------------------
    def respond(self, user_input: str, mode: str = "active") -> str:
        obs = self.perception.perceive(user_input)
        self.wm.push(obs)

        # ---- TRAINING MODE ----
        if mode == "training":
            return self._train(obs)

        # ---- Harvest reward from previous turn (testing + active) ----
        if self.turn > 0:
            self._harvest_feedback(obs)

        # ---- Retrieve (§7 RAG) ----
        retrieved = self.memory.search(user_input, k=3)
        context = [t for t, _, _ in retrieved]
        if obs["triples"]:
            self.memory.add_triples(obs["triples"])

        # ---- Reasoning (§8) ----
        steps = self.reasoning.decompose(user_input)
        hyps = self.reasoning.hypotheses(user_input, context)
        plan = self.planner.make_plan(user_input, steps)

        # ---- Q-strategy (§10.2) ----
        state = (obs["intent"], obs["sentiment"])
        strategy = self.q.choose(state)

        # ---- Generate ----
        solution = self._execute_plan(plan, obs, context, hyps, strategy)

        # ---- Critic (§12) ----
        ok, why = self.critic.verify(solution, obs)
        if not ok:
            solution = f"[revised] {solution}  (critic: {why})"

        # ---- Store ONLY in active mode ----
        if mode == "active":
            self.memory.store(user_input, {"type": "user", "turn": self.turn})
            self.learner.update(obs)
            self.memory.add_experience(Experience(
                query=user_input,
                response=solution,
                intent=obs["intent"],
                strategy=strategy,
                reward=0.0,
                timestamp=obs["timestamp"],
            ))
            self.q.note(state, strategy)
            self.last_state = state

        self.last_response = solution
        self.turn += 1
        return solution

    # ------------------------------------------------------------------
    def _train(self, obs: Dict[str, Any]) -> str:
        raw = obs["raw"]
        qm = re.search(r"^\s*Q:\s*(.+)$", raw, re.MULTILINE | re.IGNORECASE)
        am = re.search(r"^\s*A:\s*([\s\S]+)$", raw, re.MULTILINE | re.IGNORECASE)
        if qm and am:
            q = qm.group(1).strip()
            a = am.group(1).strip()
            self.memory.store(f"TRAIN_Q:{q}", {"type": "training", "answer": a})
            self.memory.trained += 1
            return (f"✓ Learned: \"{q[:60]}\" → \"{a[:80]}\"\n\n"
                    f"({self.memory.trained} training pairs stored)")
        self.memory.store(f"TRAIN:{raw}", {"type": "training", "raw": raw})
        if obs["triples"]:
            self.memory.add_triples(obs["triples"])
        self.memory.trained += 1
        return (f"✓ Noted ({self.memory.trained} items stored).\n"
                "Tip: use 'Q: ...\\nA: ...' to teach an explicit response.")

    # ------------------------------------------------------------------
    def _harvest_feedback(self, obs: Dict[str, Any]) -> None:
        r = 0.0
        if obs["intent"] == "feedback" or obs["sentiment"] != 0:
            r = float(obs["sentiment"])
        else:
            low = obs["raw"].lower()
            if any(w in low for w in ("no,", "not ", "wrong", "actually")):
                r = -0.5
            elif len(obs["tokens"]) > 3:
                r = 0.1

        if r != 0.0 and self.last_state is not None:
            self.q.reward(r, (obs["intent"], obs["sentiment"]))
            if self.memory.experiences:
                self.memory.experiences[-1].reward = r

    # ------------------------------------------------------------------
    def _execute_plan(self,
                      plan: List[Dict[str, Any]],
                      obs: Dict[str, Any],
                      context: List[str],
                      hyps: Dict[str, float],
                      strategy: str) -> str:
        pieces: List[str] = []
        for step in plan:
            a = step["action"]
            if a == "search_memory":
                hits = self.tools.run("search_memory", obs["raw"])
                if hits:
                    pieces.append("Related memory: " +
                                  " | ".join(h[0][:80] for h in hits[:2]))
            elif a == "python_eval":
                expr = self._extract_expr(obs["raw"])
                if expr:
                    pieces.append(f"calc({expr}) = " +
                                  self.tools.run("python_eval", expr))
            elif a == "generate_language":
                pieces.append(self._generate_language(obs, context, hyps, strategy))

        if not pieces:
            pieces.append(self._generate_language(obs, context, hyps, strategy))
        return " ".join(pieces)

    def _extract_expr(self, text: str) -> Optional[str]:
        m = re.search(r"(-?\d[\d\.\s\+\-\*/\(\)%]{2,})", text)
        if not m:
            return None
        expr = m.group(1).strip()
        if not re.search(r"[\+\-\*/]", expr):
            return None
        return re.sub(r"[^0-9\.\+\-\*/\(\) ]", "", expr).strip()

    # ------------------------------------------------------------------
    def _generate_language(self,
                           obs: Dict[str, Any],
                           context: List[str],
                           hyps: Dict[str, float],
                           strategy: str) -> str:
        # 1. Training memory has priority
        hits = self.memory.search_training(obs["raw"], k=2)
        if hits and hits[0][1] > 0.35 and hits[0][2].get("answer"):
            return hits[0][2]["answer"] + "\n\n(matched from training)"

        # 2. Fall back to LLM / rule-based
        if self.llm is None:
            return self._fallback_language(obs, context, hyps, strategy)

        system = self._build_system_prompt(obs, context, hyps, strategy)

        if self.stream:
            buf: List[str] = []
            try:
                for tok in self.llm.chat_stream(system,
                                                [{"role": "user",
                                                  "content": obs["raw"]}]):
                    print(tok, end="", flush=True)
                    buf.append(tok)
                print()
                reply = "".join(buf)
            except Exception as e:
                reply = f"[llm error: {e}]"
            return self._postprocess_tools(reply, system)

        try:
            reply = self.llm.chat(system,
                                  [{"role": "user", "content": obs["raw"]}])
        except Exception as e:
            reply = f"[llm error: {e}]"
        return self._postprocess_tools(reply, system)

    # ------------------------------------------------------------------
    def _build_system_prompt(self,
                             obs: Dict[str, Any],
                             context: List[str],
                             hyps: Dict[str, float],
                             strategy: str) -> str:
        top_hyp = max(hyps.items(), key=lambda x: x[1])[0] if hyps else "unknown"
        plan_steps = self.reasoning.decompose(obs["raw"])[:5]
        sim = self.memory.similar_experiences(obs["raw"], k=2)
        past = "\n".join(
            f"  - Q: {e.query[:60]}... | strategy={e.strategy} | reward={e.reward:+.2f}"
            for e in sim
        ) or "  (none yet)"

        mem_block = "\n".join(f"- {t[:160]}" for t in context) if context else "- (empty)"
        hyp_block = "\n".join(
            f"- {h}: {p:.2f}"
            for h, p in sorted(hyps.items(), key=lambda kv: -kv[1])[:4]
        ) or "- (none)"
        plan_block = "\n".join(f"  {i+1}. {s}"
                               for i, s in enumerate(plan_steps)) or "  (empty)"

        return (
            f"You are {self.name}, an AGI-inspired engineering intelligence.\n"
            "You are pragmatic, honest, and concise. You never fabricate measurements.\n"
            "\n"
            "## Session context\n"
            f"- user intent     : {obs['intent']}\n"
            f"- user sentiment  : {obs['sentiment']:+d}\n"
            f"- chosen strategy : {strategy}\n"
            f"- entities        : {obs.get('entities', []) or 'n/a'}\n"
            "\n"
            "## Retrieved long-term memory (§7 RAG)\n"
            f"{mem_block}\n"
            "\n"
            "## Bayesian hypotheses (§8.4)\n"
            f"{hyp_block}\n"
            f"  leading -> {top_hyp}\n"
            "\n"
            "## Current plan (§9)\n"
            f"{plan_block}\n"
            "\n"
            "## Similar past exchanges (with reward signal, §10.4)\n"
            f"{past}\n"
            "\n"
            "## Rules\n"
            "1. Match the user's language and register.\n"
            "2. Keep answers short unless the user asks for depth.\n"
            "3. If a calculation is needed, emit exactly: [[CALC: <arithmetic expr>]]\n"
            "   on its own line and STOP. WildCat will run it and re-prompt you.\n"
            "4. Never claim to be human. Never invent test data.\n"
            "5. When the user is frustrated, acknowledge briefly first, then help.\n"
        )

    # ------------------------------------------------------------------
    def _postprocess_tools(self,
                           reply: str,
                           system: str,
                           max_iters: int = 3) -> str:
        for _ in range(max_iters):
            calcs = re.findall(r"\[\[CALC:\s*([^\]\n]+?)\s*\]\]", reply)
            if not calcs:
                return reply

            results = []
            for expr in calcs:
                val = self.tools.python_eval(expr)
                results.append(f"{expr} = {val}")
                self.memory.store(f"CALC {expr} = {val}", {"type": "calc"})

            followup = (
                "Tool results:\n" + "\n".join(results) +
                "\n\nNow give the final answer. Do not emit [[CALC: ...]] again."
            )
            try:
                reply = self.llm.chat(system,
                                      [{"role": "user", "content": followup}])
            except Exception as e:
                return reply + f"\n[tool error: {e}]"
        return reply

    # ------------------------------------------------------------------
    def _fallback_language(self,
                           obs: Dict[str, Any],
                           context: List[str],
                           hyps: Dict[str, float],
                           strategy: str) -> str:
        intent = obs["intent"]
        sent = obs["sentiment"]
        user = obs["raw"].strip()
        top_hyp = max(hyps.items(), key=lambda x: x[1])[0] if hyps else "unknown"

        sim = self.memory.similar_experiences(user, k=1)
        past_hint = ""
        if sim:
            e = sim[0]
            if e.reward > 0:
                past_hint = " (a similar past exchange went well)"
            elif e.reward < 0:
                past_hint = " (a similar past exchange went poorly; trying differently)"

        if intent == "greeting":
            return f"Hi — I'm {self.name}. What are we working on today?"
        if intent == "farewell":
            return "Goodbye. I'll remember this session."
        if intent == "feedback":
            if sent > 0:
                return "Glad that helped — I'll reinforce that pattern."
            if sent < 0:
                return "Noted. I'll adjust my strategy for next time."
            return "Thanks for the feedback."
        if intent == "command":
            plan = self.reasoning.decompose(user)[:3]
            return f"On it. Plan: {', '.join(plan)}."
        if intent == "question":
            ctx = f" From memory: {context[0][:100]}" if context else ""
            conf = hyps.get(top_hyp, 0.0)
            return (f"Good question. Working hypothesis: {top_hyp} "
                    f"(confidence {conf:.2f}).{ctx}{past_hint}")

        if strategy == "empathize" and sent < 0:
            return "I hear you — let's break this down together. What's the sticking point?"
        if strategy == "clarify":
            return f"Tell me a bit more — is this about {top_hyp}?"
        if strategy == "suggest":
            return f"Here's a thought: start with {self.reasoning.decompose(user)[0]}."
        if strategy == "retrieve" and context:
            return f"Here's what I recall: {context[0][:140]}"
        return f"Understood. I'll treat this as '{top_hyp}'.{past_hint}"

    # ------------------------------------------------------------------
    def save(self) -> None:
        self.memory.save()

    def reset(self) -> None:
        self.memory.clear()
        self.q = QLearner()
        self.turn = 0
        self.last_state = None
        self.last_response = ""
        self.save()

    def stats(self) -> Dict[str, Any]:
        return {
            "turns":        self.turn,
            "memories":     len(self.memory.texts),
            "experiences":  len(self.memory.experiences),
            "cases":        len(self.memory.cases),
            "kg_triples":   sum(len(v) for v in self.memory.kg.values()),
            "q_states":     len(self.q.Q),
            "topics_seen":  len(self.learner.topic_history),
            "trained":      self.memory.trained,
            "llm_provider": self.llm.provider if self.llm else "none",
            "llm_model":    self.llm.model if self.llm else "n/a",
        }


# ==============================================================================
# CLI (for local use — server.py uses the same WildCat class)
# ==============================================================================
BANNER = r"""
 __      ___ _     _      _        _
 \ \    / (_) |   | |    | |      | |
  \ \  / / _| | __| | ___| |_ __ _| |_
   \ \/ / | | |/ _` |/ __| __/ _` | __|
    \  /  | | | (_| | (__| || (_| | |_
     \/   |_|_|\__,_|\___|\__\__,_|\__|
        AGI-Inspired Engineering Intelligence
"""


def main() -> None:
    print(BANNER)
    print("Commands: 'stats' | 'save' | 'reset' | 'exit'")
    print("-" * 64)

    use_llm = os.environ.get("WILDCAT_USE_LLM", "1") not in ("0", "false", "no")
    stream = os.environ.get("WILDCAT_STREAM", "1") not in ("0", "false", "no")

    cat = WildCat(use_llm=use_llm, stream=stream, verbose=True)
    try:
        while True:
            try:
                user = input("\nyou > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user:
                continue
            low = user.lower()
            if low in ("exit", "quit", "bye"):
                cat.respond(user, mode="active")
                break
            if low == "stats":
                print(json.dumps(cat.stats(), indent=2))
                continue
            if low == "save":
                cat.save()
                print("[saved]")
                continue
            if low == "reset":
                cat.reset()
                print("[reset]")
                continue

            if cat.stream and cat.llm is not None:
                print(f"\n{cat.name} > ", end="", flush=True)
                cat.respond(user, mode="active")
            else:
                print(f"\n{cat.name} > {cat.respond(user, mode='active')}")
    finally:
        cat.save()
        print(f"\n[{cat.name}] session saved.")
        print(json.dumps(cat.stats(), indent=2))


if __name__ == "__main__":
    main()