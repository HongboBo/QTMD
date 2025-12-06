import requests
import difflib
import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer


# ---- 配置 ----
OLLAMA_URL_GEN = "http://localhost:11434/api/generate"
OLLAMA_URL_EMB = "http://localhost:11434/api/embeddings"
JUDGE_MODEL = "llama3:latest"       # 裁判模型，可换 qwen2:7b / mistral:7b 等
# EMBED_MODEL = "nomic-embed-text"  # 可换 all-minilm / mxbai-embed-large
_sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

TEMPERATURE = 0.1

def _ollama_generate(prompt: str, model: str = JUDGE_MODEL, temperature: float = 0.1) -> str:
    data = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": temperature},
        "stream": False   # 🚩一次性返回 JSON，而不是流式
    }
    r = requests.post(OLLAMA_URL_GEN, json=data, timeout=120)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

# def _ollama_embed(texts):
#     if isinstance(texts, str):
#         texts = [texts]
#     r = requests.post(OLLAMA_URL_EMB, json={"model": EMBED_MODEL, "input": texts}, timeout=120)
#     r.raise_for_status()
#     embs = r.json().get("embeddings", [])
#     return [np.array(e["embedding"], dtype=float) for e in embs]

def _ollama_embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    embs = _sbert.encode(texts, normalize_embeddings=True)
    return [np.array(e, dtype=float) for e in embs]

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))

# ============= 1) 回应性（是否在回应上一句） =============
def is_responsive(response: str,
                  prev_text: str,
                  use_embedding: bool = False,
                  emb_thr: float = 0.62) -> int:
    """
    返回 0/1。默认用 LLM裁判；若 use_embedding=True，再用语义相似OR增强。
    """
    if not prev_text.strip():
        return 0

    # LLM裁判（Yes/No）
    p = f"Previous: {prev_text}\nCurrent: {response}\nIs CURRENT responding to PREVIOUS? Answer only Yes or No."
    ans = _ollama_generate(p).lower()
    judge = ans.startswith("y")

    if not use_embedding:
        return 1 if judge else 0

    # 语义相似增强（OR）
    try:
        ea, eb = _ollama_embed([response, prev_text])
        sim = _cos_sim(ea, eb)
        return 1 if (judge or sim >= emb_thr) else 0
    except Exception:
        return 1 if judge else 0

# ============= 2) 反驳（是否反对上一句） =============
def is_rebuttal(response: str, prev_text: str) -> int:
    """
    返回 0/1。
    """
    if not prev_text.strip():
        return 0
    p = (
        "Classify the stance of CURRENT toward PREVIOUS as one of: support, oppose, neutral.\n"
        "Return only one word.\n"
        f"PREVIOUS: {prev_text}\nCURRENT: {response}\nLabel:"
    )
    lab = _ollama_generate(p).lower()
    return 1 if lab.startswith("oppose") else 0

# ============= 3) 非重复度（与自己上一轮的相似度的反向） =============
def non_repetition(response: str,
                   prev_self: str,
                   use_embedding: bool = False) -> float:
    """
    返回 0~1，越大越不重复。
    默认用文本相似度；use_embedding=True 时，取(文本/语义)相似度的max再取反。
    """
    if not prev_self.strip():
        return 1.0

    # 文本相似
    text_sim = difflib.SequenceMatcher(None, response, prev_self).ratio()

    if not use_embedding:
        return max(0.0, 1.0 - text_sim)

    try:
        ea, eb = _ollama_embed([response, prev_self])
        sem_sim = _cos_sim(ea, eb)
        sim = max(text_sim, sem_sim)  # 更严格：两者取高
        # 惩罚相同句式（可选）
        boring_starts = ("Firstly", "Secondly", "In summary", "I think", "need to point out", "cannot be ignored")
        if response.strip().startswith(boring_starts) and prev_self.strip().startswith(boring_starts):
            sim = max(sim, 0.92)
        return max(0.0, 1.0 - sim)
    except Exception:
        return max(0.0, 1.0 - text_sim)


def evidence_usage(response: str, rag_sentences, min_match: int = 1) -> int:
    """
    检查 response 是否包含 RAG 检索到的关键片段.
    rag_sentences: list of strings from retrieval
    返回 1 表示至少匹配 min_match 个片段，否则 0
    """
    count = sum(1 for sent in rag_sentences if sent and sent.strip() and sent[:10] in response)
    return 1 if count >= min_match else 0

# ============= 5) Stance shift =============
def stance_shift(response: str, persona: str) -> float:
    """
    计算 response 与 persona 描述之间的余弦相似度 (0~1).
    值越高说明越贴近 persona，越低说明有偏离/变化.
    """
    try:
        ea, eb = _ollama_embed([response, persona])
        return _cos_sim(ea, eb)
    except Exception:
        return 0.0