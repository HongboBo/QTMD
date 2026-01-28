## Repository layout

- `qtmd_engine.py` — Prompt construction, weighting tiers, and QTMD config.
- `dialogue_runner.py` — Multi-agent debate loop and metrics collection.
- `dialogue_runner_with_human.py` — Dialogue runner with human participation support.
- `agent_*.py` — Individual agent wrappers that call local Ollama models.
- `rag_llm.py` — Static multi-agent RAG (default KBs).
- `rag_llm_dynamic.py` — Dynamic RAG with uploaded file support (used by the UI).
- `adaptive_weight.py` — Adaptive scheduler for T/M/D weights.
- `UI.py` / `UI_enhanced.py` — Streamlit interfaces.
- `RAG/` — Knowledge base text files grouped by version.

## Requirements

The project relies on local Ollama models and common Python libraries. You will need:

- Python 3.9+
- Ollama running locally at `http://localhost:11434`
- Python packages such as:
  - `langchain-community`
  - `faiss-cpu`
  - `sentence-transformers`
  - `streamlit`, `pandas`, `plotly`

> Note: The agent modules currently expect the following local models to be available in Ollama:
> `qwen3:latest`, `mistral:latest`, and `llama3:latest`.
