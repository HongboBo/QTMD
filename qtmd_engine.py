# qtmd_engine.py
from dataclasses import dataclass
from textwrap import dedent


def weight_to_tier(w: float) -> str:
    """把连续权重映射成离散档位：low/mid/high"""
    if w is None:
        return "mid"
    if w < 0.85:
        return "low"
    elif w <= 1.25:
        return "mid"
    else:
        return "high"

SNIPPETS = {
    "T": {
        "high": (
            "Identity & stance (MANDATORY): Speak explicitly from the [T]  perspective. "
            "First state your stance and role-specific priority, then justify with reasons."
        ),
        "mid": (
            "Identity & stance (Preferred): Reflect the [T] perspective and state your position when appropriate."
        ),
        "low": (
            "Identity (Background): You may keep the [T]  perspective implicit; focus on arguments."
        ),
    },
    "M": {
        "high": (
            "Memory use (MANDATORY): Begin with a 1–2 sentence summary of the recent turns in [M] and address unresolved points directly."
        ),
        "mid": (
            "Memory use (Preferred): Consider the recent discussion [M] and avoid repeating earlier content."
        ),
        "low": (
            "Memory (Light): You may respond without summarising [M]; avoid verbatim repetition."
        ),
    },
    "D": {
        "high": (
            "Evidence (MANDATORY): Before concluding, list at least 2 concrete evidence points from the retrieved snippets [D]; "
            "quote/paraphrase and tie them to your claims."
        ),
        "mid": (
            "Evidence (Preferred): Use relevant retrieved snippets [D] to support key claims when available."
        ),
        "low": (
            "Evidence (Optional): You may proceed without citing retrieved snippets [D] if not essential."
        ),
    },
}




@dataclass
class QTMDConfig:
    use_T: bool = True
    use_M: bool = True
    use_D: bool = True
    use_R: bool = False              # 是否启用规则（可当实验变量R）
    rule_mode: str = "light"         # "light" | "struct"
    max_sentences: int = 3
    wT: float = 1.0                  # 仅写进提示，不做重复
    wM: float = 1.0
    wD: float = 1.0

LIGHT_RULE = "Answer [Q] directly first, then provide 1-2 pieces of evidence from [D]. Respond to [M] if necessary. A maximum of {N} sentences."
STRUCT_RULE = (
    "First, extract four categories of key points in order (≤ 3 items per category):"
    "1) Arguments supporting the goal 2) Arguments threatening the goal 3) Points of conflict to be resolved 4) Potential opportunities for cooperation;"
    "Then generate a response of no more than {N} sentences based on these points, prioritizing references to [D]. Must be in English，only output the response."
)

def build_qtmd_prompt(Q: str, T: str = "", M: str = "", D: str = "", cfg: QTMDConfig = QTMDConfig()) -> str:
    '''add'''
    tier_T = weight_to_tier(getattr(cfg, "wT", 1.0))
    tier_M = weight_to_tier(getattr(cfg, "wM", 1.0))
    tier_D = weight_to_tier(getattr(cfg, "wD", 1.0))

    # Weight instruction: only in the rule section and block headers, not in the main text
    weight_line = (
        f"Weight instruction: please relatively attend to "
        f"[T]={cfg.wT}, [M]={cfg.wM}, [D]={cfg.wD} "
        f"(1 = baseline, >1 = stronger, <1 = weaker)."
    )

    parts = [f"[Q]\n{Q.strip()}"]
    if cfg.use_T and T:
        parts.append(f"[T (weight={cfg.wT})]\n{T.strip()}")
    if cfg.use_M and M:
        parts.append(f"[M (weight={cfg.wM})]\n{M.strip()}")
    if cfg.use_D and D:
        parts.append(f"[D (weight={cfg.wD})]\n{D.strip()}")

    '''addd'''
    if tier_T != "mid":
        parts.append("T-guidance: " + SNIPPETS["T"][tier_T].format(persona=T.strip()[:120] if T else "given persona"))
    if tier_M != "mid":
        parts.append("M-guidance: " + SNIPPETS["M"][tier_M])
    if tier_D != "mid":
        parts.append("D-guidance: " + SNIPPETS["D"][tier_D])

    if cfg.use_R:
        rule = (STRUCT_RULE if cfg.rule_mode == "struct" else LIGHT_RULE).format(N=cfg.max_sentences)
        parts.append(f"[R]\n{weight_line}\n{rule}")
    else:
        parts.append(f"[R]\n{weight_line}")

    parts.append(f"# Please answer directly (≤{cfg.max_sentences} sentences):")
    return dedent("\n\n".join(parts)).strip()

def enhanced_build_qtmd_prompt(Q: str, T: str = "", M: str = "", D: str = "", cfg: QTMDConfig = QTMDConfig()) -> str:
    tier_T = weight_to_tier(getattr(cfg, "wT", 1.0))
    tier_M = weight_to_tier(getattr(cfg, "wM", 1.0))
    tier_D = weight_to_tier(getattr(cfg, "wD", 1.0))

    # --- 2) 档位片段（把 persona 注入到 T 档位片段）---
    t_block = SNIPPETS["T"][tier_T].format(persona=T.strip()[:120] if T else "given persona")
    m_block = SNIPPETS["M"][tier_M]
    d_block = SNIPPETS["D"][tier_D]

    # --- 3) 必做步骤汇总 ---
    steps = []
    if "MANDATORY" in t_block:
        steps.append("- State your role/stance explicitly.")
    if "MANDATORY" in m_block:
        steps.append("- Start with a brief summary of recent turns and address unresolved points.")
    if "MANDATORY" in d_block:
        steps.append("- Provide at least 2 evidence items from the retrieved snippets before concluding.")
    mandatory_block = ""
    if steps:
        mandatory_block = "Follow these mandatory steps:\n" + "\n".join(steps)

    # --- 4) 系统头 ---
    system_head = (
        "You are a helpful, debate-oriented agent in a multi-agent discussion.\n"
        "Be concise, factual, and avoid repetition."
    )

    # --- 5) 权重说明（可选，调试用）---
    weight_line = (
        f"Weight tiers: T={tier_T} (wT={cfg.wT}), "
        f"M={tier_M} (wM={cfg.wM}), "
        f"D={tier_D} (wD={cfg.wD})."
    )

    # --- 6) 拼装 parts ---
    parts = [system_head, weight_line]

    if mandatory_block:
        parts.append(mandatory_block)

    parts.append(t_block)
    parts.append(m_block)
    parts.append(d_block)

    if cfg.use_T and T:
        parts.append(f"[T]\n{T.strip()}")
    if cfg.use_M and M:
        parts.append(f"[M]\n{M.strip()}")
    if cfg.use_D and D:
        parts.append(f"[D]\n{D.strip()}")

    if cfg.use_R:
        rule_tpl = (STRUCT_RULE if cfg.rule_mode == "struct" else LIGHT_RULE)
        try:
            rule_text = rule_tpl.format(N=cfg.max_sentences)
        except Exception:
            rule_text = rule_tpl
        parts.append(f"[R]\n{rule_text}")

    parts.append(f"# Please answer directly (≤{cfg.max_sentences} sentences):")
    parts.append(f"[Q]\n{Q.strip()}")

    return dedent("\n\n".join(parts)).strip()
