# dialogue_runner_with_human.py
"""
Extended dialogue runner that supports human participation in the QTMD framework.
Human inputs are integrated into the dialogue history and influence agent responses.
"""

import agent_conservation
import agent_farmer
import agent_community
from adaptive_weight import scheduler
from rag_llm import rag_search
from evaluator import is_responsive, is_rebuttal, non_repetition, evidence_usage, stance_shift
from typing import Generator, Dict, List, Optional, Callable


def run_dialogue_with_human(
        query: str,
        use_R: bool,
        rule_mode: str,
        adaptive_weight: bool,
        rounds: int,
        human_frequency: str = "every_round",  # "every_round", "every_other_round", "on_demand"
        human_position: str = "after_all",  # "after_all", "after_first", "before_all"
        human_input_callback: Optional[Callable[[], str]] = None,
) -> Generator[Dict, None, None]:
    """
    Run a multi-agent dialogue with optional human participation.

    Args:
        query: The initial debate question
        use_R: Whether to use explicit reasoning rules
        rule_mode: "light" or "struct"
        adaptive_weight: Whether to dynamically adjust weights
        rounds: Number of dialogue rounds
        human_frequency: How often human participates
        human_position: When in each round human speaks
        human_input_callback: Function to get human input (for non-UI usage)

    Yields:
        Dict with round info, agent name, response, metrics, and weights
    """

    agents = [
        {"name": "Conservation 🌲", "module": agent_conservation, "rag_name": "ConservationAgent"},
        {"name": "Farmer 🚜", "module": agent_farmer, "rag_name": "FarmerAgent"},
        {"name": "Community 🏘", "module": agent_community, "rag_name": "CommunityAgent"}
    ]

    print(f"\n--- Running dialogue with human participation ---")
    print(f"Human frequency: {human_frequency}, Position: {human_position}")

    dialogue_history = [f"Initial topic: {query}"]
    w_pool = {a["name"]: [1.0, 1.0, 1.0] for a in agents}
    last_self = {a["name"]: "" for a in agents}
    utter_history = []
    num_agents = len(agents)

    for r in range(rounds):
        print(f"\n========== Round {r} ==========")

        # Determine if human participates this round
        human_participates = (
                human_frequency == "every_round" or
                (human_frequency == "every_other_round" and r % 2 == 0)
        )

        # Handle "before_all" human position
        if human_participates and human_position == "before_all" and human_input_callback:
            human_text = human_input_callback()
            if human_text and human_text.strip():
                dialogue_history.append(f"Human 👤: {human_text}")
                utter_history.append(human_text)
                yield {
                    "round": r,
                    "agent": "Human 👤",
                    "response": human_text,
                    "metrics": None,
                    "weights": {"wT": 1.0, "wM": 1.0, "wD": 1.0},
                    "is_human": True
                }

        # Generate agent responses
        for idx, agent_info in enumerate(agents):
            name = agent_info["name"]
            module = agent_info["module"]
            rag_name = agent_info["rag_name"]

            print(f"\nNow it's {name} speaking...")

            # Build history context
            if r == 0 and idx == 0:
                history_text = query
                use_R_this = False
                prev_round_text = ""
            else:
                # Include recent dialogue including human inputs
                history_text = "\n".join(dialogue_history[-4:])
                use_R_this = use_R
                prev_round_text = utter_history[-1] if utter_history else ""

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

            # Retrieve RAG data and calculate metrics
            retrieved = rag_search(history_text, agent=rag_name)
            rag_sents = [robj["content"] for robj in retrieved]

            resp_score = is_responsive(response, prev_round_text)
            reb_score = is_rebuttal(response, prev_round_text)
            nr_score = non_repetition(response, prev_self)
            evi_score = evidence_usage(response, rag_sents)
            stance_score = stance_shift(response, module.MY_TASK)

            print(
                f"[metrics] responsive={resp_score}, rebuttal={reb_score}, "
                f"non_repetition={nr_score:.2f}, evidence={evi_score}, stance_score={stance_score:.2f}"
            )

            # Update weights if adaptive
            if adaptive_weight:
                new_wT, new_wM, new_wD = scheduler(
                    round_num=r,
                    last_response=response,
                    responsive=resp_score,
                    rag_sentences=rag_sents,
                    wT=wT, wM=wM, wD=wD,
                    alpha=0.2
                )
                w_pool[name] = [new_wT, new_wM, new_wD]

            # Yield agent result
            yield {
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
                "is_human": False
            }

            # Handle "after_first" human position
            if (human_participates and human_position == "after_first" and
                    idx == 0 and human_input_callback):
                human_text = human_input_callback()
                if human_text and human_text.strip():
                    dialogue_history.append(f"Human 👤: {human_text}")
                    utter_history.append(human_text)
                    yield {
                        "round": r,
                        "agent": "Human 👤",
                        "response": human_text,
                        "metrics": None,
                        "weights": {"wT": 1.0, "wM": 1.0, "wD": 1.0},
                        "is_human": True
                    }

        # Handle "after_all" human position
        if human_participates and human_position == "after_all" and human_input_callback:
            human_text = human_input_callback()
            if human_text and human_text.strip():
                dialogue_history.append(f"Human 👤: {human_text}")
                utter_history.append(human_text)
                yield {
                    "round": r,
                    "agent": "Human 👤",
                    "response": human_text,
                    "metrics": None,
                    "weights": {"wT": 1.0, "wM": 1.0, "wD": 1.0},
                    "is_human": True
                }


def process_human_input(
        human_text: str,
        dialogue_history: List[str],
        current_round: int
) -> Dict:
    """
    Process human input and format it for the dialogue system.

    Args:
        human_text: The human's message
        dialogue_history: Current dialogue history list
        current_round: Current round number

    Returns:
        Formatted result dict for the human input
    """
    # Add to history
    formatted_entry = f"Human 👤: {human_text}"
    dialogue_history.append(formatted_entry)

    return {
        "round": current_round,
        "agent": "Human 👤",
        "response": human_text,
        "metrics": None,  # Humans don't have automated metrics
        "weights": {"wT": 1.0, "wM": 1.0, "wD": 1.0},
        "is_human": True
    }


class InteractiveDialogueSession:
    """
    Manages an interactive dialogue session with human participation.
    Designed for use with Streamlit or other UI frameworks.
    """

    def __init__(
            self,
            query: str,
            use_R: bool = True,
            rule_mode: str = "light",
            adaptive_weight: bool = True,
            rounds: int = 5,
            human_frequency: str = "every_round",
            human_position: str = "after_all"
    ):
        self.query = query
        self.use_R = use_R
        self.rule_mode = rule_mode
        self.adaptive_weight = adaptive_weight
        self.total_rounds = rounds
        self.human_frequency = human_frequency
        self.human_position = human_position

        # State
        self.current_round = 0
        self.dialogue_history = [f"Initial topic: {query}"]
        self.results = []
        self.is_complete = False
        self.waiting_for_human = False

        # Agent state
        self.agents = [
            {"name": "Conservation 🌲", "module": agent_conservation, "rag_name": "ConservationAgent"},
            {"name": "Farmer 🚜", "module": agent_farmer, "rag_name": "FarmerAgent"},
            {"name": "Community 🏘", "module": agent_community, "rag_name": "CommunityAgent"}
        ]
        self.w_pool = {a["name"]: [1.0, 1.0, 1.0] for a in self.agents}
        self.last_self = {a["name"]: "" for a in self.agents}
        self.utter_history = []
        self.current_agent_idx = 0

    def should_human_participate(self) -> bool:
        """Check if human should participate in current round."""
        if self.human_frequency == "every_round":
            return True
        elif self.human_frequency == "every_other_round":
            return self.current_round % 2 == 0
        return False

    def generate_next_agent_response(self) -> Optional[Dict]:
        """Generate the next agent's response."""
        if self.current_agent_idx >= len(self.agents):
            return None

        agent_info = self.agents[self.current_agent_idx]
        name = agent_info["name"]
        module = agent_info["module"]
        rag_name = agent_info["rag_name"]

        # Build history context
        if self.current_round == 0 and self.current_agent_idx == 0:
            history_text = self.query
            use_R_this = False
            prev_round_text = ""
        else:
            history_text = "\n".join(self.dialogue_history[-4:])
            use_R_this = self.use_R
            prev_round_text = self.utter_history[-1] if self.utter_history else ""

        wT, wM, wD = self.w_pool[name]
        prev_self = self.last_self[name]

        # Generate response
        response = module.invoke(
            history=history_text,
            round_num=self.current_round,
            query=self.query,
            use_T=True, use_M=True, use_D=True,
            wT=wT, wM=wM, wD=wD,
            use_R=use_R_this,
            rule_mode=self.rule_mode,
            max_sentences=3
        )

        # Update state
        self.dialogue_history.append(f"{name}: {response}")
        self.utter_history.append(response)
        self.last_self[name] = response

        # Calculate metrics
        retrieved = rag_search(history_text, agent=rag_name)
        rag_sents = [robj["content"] for robj in retrieved]

        resp_score = is_responsive(response, prev_round_text)
        reb_score = is_rebuttal(response, prev_round_text)
        nr_score = non_repetition(response, prev_self)
        evi_score = evidence_usage(response, rag_sents)
        stance_score = stance_shift(response, module.MY_TASK)

        # Update weights if adaptive
        if self.adaptive_weight:
            new_wT, new_wM, new_wD = scheduler(
                round_num=self.current_round,
                last_response=response,
                responsive=resp_score,
                rag_sentences=rag_sents,
                wT=wT, wM=wM, wD=wD,
                alpha=0.2
            )
            self.w_pool[name] = [new_wT, new_wM, new_wD]

        result = {
            "round": self.current_round,
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
            "is_human": False
        }

        self.results.append(result)
        self.current_agent_idx += 1

        return result

    def add_human_input(self, human_text: str) -> Dict:
        """Add human input to the dialogue."""
        self.dialogue_history.append(f"Human 👤: {human_text}")
        self.utter_history.append(human_text)

        result = {
            "round": self.current_round,
            "agent": "Human 👤",
            "response": human_text,
            "metrics": None,
            "weights": {"wT": 1.0, "wM": 1.0, "wD": 1.0},
            "is_human": True
        }

        self.results.append(result)
        self.waiting_for_human = False

        return result

    def advance_round(self):
        """Move to the next round."""
        self.current_round += 1
        self.current_agent_idx = 0

        if self.current_round >= self.total_rounds:
            self.is_complete = True

    def run_round_agents(self) -> List[Dict]:
        """Run all agents for the current round."""
        results = []
        while self.current_agent_idx < len(self.agents):
            result = self.generate_next_agent_response()
            if result:
                results.append(result)
        return results

    def get_state(self) -> Dict:
        """Get current session state."""
        return {
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "is_complete": self.is_complete,
            "waiting_for_human": self.waiting_for_human,
            "dialogue_history": self.dialogue_history,
            "results": self.results,
            "num_results": len(self.results)
        }