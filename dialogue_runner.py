import agent_conservation
import agent_farmer
import agent_community
from adaptive_weight import scheduler
from rag_llm import rag_search
from evaluator import is_responsive, is_rebuttal, non_repetition, evidence_usage, stance_shift


def run_dialogue(query: str, use_R: bool, rule_mode: str, adaptive_weight: bool, rounds: int):
    agents = [
        {"name": "Conservation 🌲", "module": agent_conservation, "rag_name": "ConservationAgent"},
        {"name": "Farmer 🚜", "module": agent_farmer, "rag_name": "FarmerAgent"},
        {"name": "Community 🏘", "module": agent_community, "rag_name": "CommunityAgent"}
    ]

    print(f"\n--- Running dialogue ---")
    dialogue_history = [f"Initial topic:: {query}"]

    w_pool = {a["name"]: [1.0, 1.0, 1.0] for a in agents}


    last_self = {a["name"]: "" for a in agents}
    utter_history = []

    num_agents = len(agents)

    results = []

    for r in range(rounds):
        print(f"\n========== round {r}  ==========")
        for agent_info in agents:
            name = agent_info["name"]
            module = agent_info["module"]
            rag_name = agent_info["rag_name"]

            print(f"\nNow it's {name} speaking...")

            if r == 0:
                history_text = query
                use_R_this = False
                prev_round_text = ""
            else:
                history_text = "\n".join(dialogue_history[-3:])
                use_R_this = use_R
                idx_prev_round_last = r * num_agents - 1
                prev_round_text = utter_history[idx_prev_round_last] if idx_prev_round_last < len(utter_history) else ""

            wT, wM, wD = w_pool[name]
            prev_self = last_self[name]

            # Generate agent response
            response = module.invoke(
                history=history_text,
                round_num=r,
                query=query,
                use_T=True, use_M=True, use_D=True,
                wT=wT, wM=wM, wD=wD,
                use_R=use_R_this,
                rule_mode=rule_mode,
                max_sentences=3
            )

            print(f"{name}: {response}")

            # Update dialogue history
            dialogue_history.append(f"{name}: {response}")
            utter_history.append(response)
            last_self[name] = response

            # Retrieve RAG data
            retrieved = rag_search(history_text, agent=rag_name)
            rag_sents = [robj["content"] for robj in retrieved]

            # Calculate metrics
            resp_score = is_responsive(response, prev_round_text)
            reb_score = is_rebuttal(response, prev_round_text)
            nr_score = non_repetition(response, prev_self)
            evi_score = evidence_usage(response, rag_sents)
            stance_score = stance_shift(response, module.MY_TASK)

            print(
                f"[metrics] responsive={resp_score}, rebuttal={reb_score}, "
                f"non_repetition={nr_score:.2f}, evidence={evi_score}, stance_score={stance_score:.2f}")

            # Update weights if adaptive
            if adaptive_weight:
                new_wT, new_wM, new_wD = scheduler(
                    round_num=r,
                    last_response=response,
                    responsive=resp_score,
                    rag_sentences=rag_sents,
                    wT=wT,
                    wM=wM,
                    wD=wD,
                    alpha=0.2
                )
                w_pool[name] = [new_wT, new_wM, new_wD]

            # Yield result immediately for real-time UI updates
            result = {
                "round": r,
                "agent": name,
                "response": response,
                "metrics": {
                    "responsive": resp_score,
                    "rebuttal": reb_score,
                    "non_repetition": nr_score,
                    "evidence_usage": evi_score,
                    "stance_shift": stance_score,
                },
                "weights": {"wT": wT, "wM": wM, "wD": wD},
            }

            yield result  # 👈 关键改动：实时返回每条对话结果