# agent_community.py
from rag_llm import rag_search
from langchain_community.llms import Ollama
from qtmd_engine import build_qtmd_prompt, QTMDConfig, enhanced_build_qtmd_prompt

llm = Ollama(model="llama3:latest", base_url="http://localhost:11434",
             stop=["<|im_end|>"], temperature=0.5)

MY_TASK = (
    "You are a land justice and community rights advocate. Speak directly and assertively. Keep your answers short, no more than 3 sentences. You emphasize public interest, fair land use, and ordinary people's voices. Answer in English. /nothinking"

)

def invoke(
    history: str,
    round_num: int,
    query: str,
    # —— 实验参数（可不传，默认全开等权）——
    use_T: bool = True, use_M: bool = True, use_D: bool = True,
    wT: float = 1.0, wM: float = 1.0, wD: float = 1.0,
    use_R: bool = False, rule_mode: str = "light", max_sentences: int = 3,
) -> str:
    # Q/T/M/D
    Q = query
    T = MY_TASK
    M = history
    retrieved = rag_search(history, agent="CommunityAgent")
    D = "\n\n".join([r["content"] for r in retrieved])

    if round_num == 0:
        cfg = QTMDConfig(use_T=use_T, use_M=False, use_D=use_D,
                         use_R=False, wT=wT, wM=wM, wD=wD, max_sentences=max_sentences)
    else:
        cfg = QTMDConfig(use_T=use_T, use_M=use_M, use_D=use_D,
                         use_R=use_R, rule_mode=rule_mode,
                         wT=wT, wM=wM, wD=wD, max_sentences=max_sentences)
    prompt = build_qtmd_prompt(Q=Q, T=T, M=M, D=D, cfg=cfg)
    return llm.invoke(prompt).strip()
