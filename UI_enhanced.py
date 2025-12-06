import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dialogue_runner import run_dialogue
from dialogue_runner_with_human import run_dialogue_with_human, process_human_input
import time
import sys

from rag_llm_dynamic import get_rag_instance, read_file_content

# Page config
st.set_page_config(
    page_title="QTMD Multi-Agent Dialogue System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-size: 12px;
    }
    h1 {
        font-size: 1.5rem !important;
        margin-bottom: 0.3rem !important;
    }
    h2 {
        font-size: 1.2rem !important;
        margin-bottom: 0.3rem !important;
    }
    h3 {
        font-size: 1rem !important;
        margin-bottom: 0.3rem !important;
    }
    .stMarkdown, .stText {
        font-size: 0.8rem;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }
    .stButton button {
        font-size: 0.75rem;
        padding: 0.3rem 0.8rem;
    }
    .stSelectbox, .stTextInput, .stTextArea, .stNumberInput {
        font-size: 0.75rem;
    }
    .stMetric {
        font-size: 0.75rem;
    }
    .stMetric label {
        font-size: 0.65rem !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1rem !important;
    }
    .stTab {
        font-size: 0.75rem;
        padding: 0.3rem 0.8rem !important;
    }
    .stAlert {
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.75rem;
    }
    .agent-conservation, .agent-farmer, .agent-community, .agent-human {
        padding: 0.4rem 0.6rem;
        margin-bottom: 0.3rem;
        border-radius: 0.25rem;
    }
    .agent-conservation {
        border-left: 3px solid #10b981;
        background-color: #f0fdf4;
    }
    .agent-farmer {
        border-left: 3px solid #f59e0b;
        background-color: #fffbeb;
    }
    .agent-community {
        border-left: 3px solid #3b82f6;
        background-color: #eff6ff;
    }
    .agent-human {
        border-left: 3px solid #8b5cf6;
        background-color: #f5f3ff;
    }
    .agent-conservation strong, .agent-farmer strong, .agent-community strong, .agent-human strong {
        font-size: 0.75rem;
    }
    .agent-conservation p, .agent-farmer p, .agent-community p, .agent-human p {
        font-size: 0.7rem;
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
        line-height: 1.3;
    }
    .agent-conservation small, .agent-farmer small, .agent-community small, .agent-human small {
        font-size: 0.6rem;
    }
    [data-testid="stSidebar"] {
        font-size: 0.75rem;
    }
    [data-testid="stSidebar"] h1 {
        font-size: 1.1rem !important;
    }
    [data-testid="stSidebar"] h2 {
        font-size: 0.95rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.3rem !important;
    }
    [data-testid="stSidebar"] h3 {
        font-size: 0.85rem !important;
    }
    .stInfo, .stSuccess, .stWarning, .stError {
        font-size: 0.75rem;
        padding: 0.5rem;
    }
    .dataframe {
        font-size: 0.7rem;
    }
    hr {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    [data-testid="column"] {
        padding: 0.3rem !important;
    }
    .kb-status {
        font-size: 0.7rem;
        padding: 0.3rem 0.5rem;
        border-radius: 0.2rem;
        margin-bottom: 0.3rem;
    }
    .kb-default {
        background-color: #e5e7eb;
        color: #374151;
    }
    .kb-custom {
        background-color: #dbeafe;
        color: #1e40af;
    }
    .human-input-box {
        border: 2px solid #8b5cf6;
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: #faf5ff;
        margin: 1rem 0;
    }
    .waiting-indicator {
        animation: pulse 2s infinite;
        color: #8b5cf6;
        font-weight: bold;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'dialogue_results' not in st.session_state:
    st.session_state.dialogue_results = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'current_message_index' not in st.session_state:
    st.session_state.current_message_index = 0
if 'config' not in st.session_state:
    st.session_state.config = {
        'query': "Should the public be given more freedom to roam?",
        'rounds': 5,
        'use_R': True,
        'rule_mode': 'light',
        'adaptive_weight': True,
        'wT': 1.0,
        'wM': 1.0,
        'wD': 1.5,
        'human_participation': False,
        'human_frequency': 'every_round',
        'human_position': 'after_all'
    }
if 'rag_instance' not in st.session_state:
    st.session_state.rag_instance = get_rag_instance()
if 'kb_info' not in st.session_state:
    st.session_state.kb_info = {}

# Human participation state
if 'waiting_for_human' not in st.session_state:
    st.session_state.waiting_for_human = False
if 'human_input' not in st.session_state:
    st.session_state.human_input = ""
if 'current_round' not in st.session_state:
    st.session_state.current_round = 0
if 'dialogue_history_text' not in st.session_state:
    st.session_state.dialogue_history_text = []
if 'interactive_mode_active' not in st.session_state:
    st.session_state.interactive_mode_active = False
if 'pending_agent_responses' not in st.session_state:
    st.session_state.pending_agent_responses = []
if 'round_agents_done' not in st.session_state:
    st.session_state.round_agents_done = False  # 标记当前轮的agent是否已完成

# Agent configurations
AGENTS = {
    "Conservation 🌲": {
        "rag_name": "ConservationAgent",
        "description": "Environmental conservation advocate",
        "color": "#10b981"
    },
    "Farmer 🚜": {
        "rag_name": "FarmerAgent",
        "description": "UK farmer representative",
        "color": "#f59e0b"
    },
    "Community 🏘": {
        "rag_name": "CommunityAgent",
        "description": "Land justice and community rights advocate",
        "color": "#3b82f6"
    },
    "Human 👤": {
        "rag_name": None,
        "description": "Human participant in the dialogue",
        "color": "#8b5cf6"
    }
}

# Header
st.title("🧠 QTMD Multi-Agent Dialogue System")
st.markdown(
    "<p style='font-size: 0.9rem; color: #6b7280;'><strong>Query-Task-Memory-Data Framework</strong> for Multi-Agent Debate with Human Participation</p>",
    unsafe_allow_html=True)
st.divider()

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Query input
    query = st.text_area(
        "Debate Query",
        value=st.session_state.config['query'],
        height=100,
        help="The main question or topic for the debate"
    )

    st.markdown("---")

    # 👤 Human Participation Section
    st.subheader("👤 Human Participation")

    human_participation = st.checkbox(
        "Enable Human Participation",
        value=st.session_state.config['human_participation'],
        help="Allow human to join the multi-agent dialogue"
    )

    if human_participation:
        st.markdown("**Participation Settings:**")

        human_frequency = st.selectbox(
            "When to participate",
            options=['every_round', 'every_other_round', 'on_demand'],
            index=['every_round', 'every_other_round', 'on_demand'].index(
                st.session_state.config.get('human_frequency', 'every_round')
            ),
            help="How often the human can contribute"
        )

        human_position = st.selectbox(
            "Position in round",
            options=['after_all', 'after_first', 'before_all'],
            index=['after_all', 'after_first', 'before_all'].index(
                st.session_state.config.get('human_position', 'after_all')
            ),
            help="When in each round the human speaks"
        )

        st.info("💡 Your input will be integrated into the dialogue history and influence agent responses.")

    st.markdown("---")

    # 📚 Knowledge Base Upload Section
    st.subheader("📚 Knowledge Base")

    with st.expander("📤 Upload Custom Knowledge Base", expanded=False):
        st.markdown("**Upload files for each agent's knowledge base**")
        st.markdown("Supported formats: `.txt`, `.md`, `.pdf`, `.docx`")

        for agent_name, agent_config in list(AGENTS.items())[:3]:  # Only for AI agents
            st.markdown(f"**{agent_name}**")
            rag_name = agent_config["rag_name"]

            kb_info = st.session_state.rag_instance.get_kb_info(rag_name)
            if kb_info['loaded']:
                kb_type = "📘 Default KB" if kb_info['using_default'] else "📗 Custom KB"
                st.markdown(
                    f'<div class="kb-status {"kb-default" if kb_info["using_default"] else "kb-custom"}">'
                    f'{kb_type} | {kb_info["num_chunks"]} chunks'
                    f'</div>',
                    unsafe_allow_html=True
                )

            uploaded_files = st.file_uploader(
                f"Upload for {agent_name}",
                type=['txt', 'md', 'pdf', 'docx'],
                accept_multiple_files=True,
                key=f"upload_{rag_name}",
                label_visibility="collapsed"
            )

            if uploaded_files:
                if st.button(f"✅ Load Files for {agent_name}", key=f"load_{rag_name}"):
                    with st.spinner(f"Processing files for {agent_name}..."):
                        file_contents = []
                        for uploaded_file in uploaded_files:
                            content = read_file_content(
                                uploaded_file.read(),
                                uploaded_file.name
                            )
                            file_contents.append((content, uploaded_file.name))

                        success = st.session_state.rag_instance.load_from_multiple_files(
                            rag_name,
                            file_contents
                        )

                        if success:
                            st.success(f"✅ Loaded {len(uploaded_files)} files for {agent_name}")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to load files for {agent_name}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🔄 Reset", key=f"reset_{rag_name}", use_container_width=True):
                    st.session_state.rag_instance.reset_to_default(rag_name)
                    st.success(f"Reset to default KB")
                    st.rerun()

            st.markdown("---")

    st.markdown("---")

    # Basic settings
    st.subheader("Basic Settings")
    col1, col2 = st.columns(2)
    with col1:
        rounds = st.number_input(
            "Rounds",
            min_value=1,
            max_value=10,
            value=st.session_state.config['rounds'],
            help="Number of dialogue rounds"
        )
    with col2:
        rule_mode = st.selectbox(
            "Rule Mode",
            options=['light', 'struct'],
            index=0 if st.session_state.config['rule_mode'] == 'light' else 1,
            help="Light: simple rules, Struct: structured reasoning"
        )

    st.markdown("---")

    # Weight configuration
    st.subheader("Weight Configuration")

    wT = st.slider(
        "Task Weight (wT)",
        min_value=0.5,
        max_value=2.0,
        value=st.session_state.config['wT'],
        step=0.1,
        help="Identity & stance importance"
    )

    wM = st.slider(
        "Memory Weight (wM)",
        min_value=0.5,
        max_value=2.0,
        value=st.session_state.config['wM'],
        step=0.1,
        help="Historical context importance"
    )

    wD = st.slider(
        "Data Weight (wD)",
        min_value=0.5,
        max_value=2.0,
        value=st.session_state.config['wD'],
        step=0.1,
        help="Evidence & retrieval importance"
    )

    st.markdown("---")

    # Advanced settings
    st.subheader("Advanced Settings")

    use_R = st.checkbox(
        "Use Rules (R)",
        value=st.session_state.config['use_R'],
        help="Enable explicit reasoning rules"
    )

    adaptive_weight = st.checkbox(
        "Adaptive Weights",
        value=st.session_state.config['adaptive_weight'],
        help="Dynamically adjust weights based on agent behavior"
    )

    st.markdown("---")

    # Start dialogue button
    if st.button("🚀 Start Dialogue", type="primary", use_container_width=True):
        # Update config
        st.session_state.config = {
            'query': query,
            'rounds': rounds,
            'use_R': use_R,
            'rule_mode': rule_mode,
            'adaptive_weight': adaptive_weight,
            'wT': wT,
            'wM': wM,
            'wD': wD,
            'human_participation': human_participation,
            'human_frequency': human_frequency if human_participation else 'every_round',
            'human_position': human_position if human_participation else 'after_all'
        }
        st.session_state.is_running = True
        st.session_state.dialogue_results = []
        st.session_state.dialogue_history_text = [f"Initial topic: {query}"]
        st.session_state.current_round = 0
        st.session_state.waiting_for_human = False
        st.session_state.interactive_mode_active = human_participation
        st.session_state.round_agents_done = False  # 重置
        st.rerun()

    # Clear button
    if st.button("🗑️ Clear Results", use_container_width=True):
        st.session_state.dialogue_results = []
        st.session_state.current_message_index = 0
        st.session_state.is_running = False
        st.session_state.waiting_for_human = False
        st.session_state.interactive_mode_active = False
        st.session_state.dialogue_history_text = []
        st.session_state.current_round = 0
        st.session_state.round_agents_done = False  # 重置
        st.rerun()

    st.markdown("---")

    # Current config display
    with st.expander("📋 Current Configuration"):
        st.json(st.session_state.config)

# Main content - Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "💬 Dialogue", "🗣️ Interactive", "📈 Metrics", "📚 KB Info", "📥 Export"
])

with tab1:
    st.header("📊 System Overview")

    # Agent descriptions
    col1, col2, col3, col4 = st.columns(4)

    agents_list = list(AGENTS.items())
    cols = [col1, col2, col3, col4]
    agent_classes = ["agent-conservation", "agent-farmer", "agent-community", "agent-human"]

    for idx, (col, (agent_name, agent_config)) in enumerate(zip(cols, agents_list)):
        with col:
            st.markdown(f"""
            <div class="{agent_classes[idx]}">
                <strong>{agent_name}</strong>
                <p>{agent_config['description']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # System information
    st.subheader("🔧 QTMD Framework")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Framework Components:**
        - **Q (Query)**: The debate question/topic
        - **T (Task)**: Agent identity & perspective
        - **M (Memory)**: Historical dialogue context
        - **D (Data)**: Retrieved evidence from RAG
        - **R (Rules)**: Optional reasoning constraints
        """)

    with col2:
        st.markdown("""
        **Key Features:**
        - 🔄 Real-time dialogue generation
        - 👤 **Human participation mode**
        - 📤 Custom knowledge base upload
        - 📊 Multi-dimensional metrics tracking
        - ⚖️ Adaptive weight adjustment
        - 🎯 RAG-enhanced responses
        """)

    st.divider()

    # Metrics explanation
    st.subheader("📏 Evaluation Metrics")

    metrics_col1, metrics_col2 = st.columns(2)

    with metrics_col1:
        st.markdown("""
        **Dialogue Quality Metrics:**
        - **Responsive**: Does the agent respond to previous statements?
        - **Rebuttal**: Does the agent oppose previous arguments?
        - **Non-repetition**: How unique is this response vs. agent's previous response?
        """)

    with metrics_col2:
        st.markdown("""
        **Content Metrics:**
        - **Evidence Usage**: Does the response cite retrieved evidence?
        - **Stance Shift**: How consistent is the response with agent's persona?
        """)

with tab2:
    st.header("💬 Dialogue History (Auto Mode)")

    if st.session_state.config.get('human_participation', False):
        st.info("👉 Human participation is enabled. Go to the **Interactive** tab to participate in the dialogue.")

    # Run dialogue if triggered (non-interactive mode)
    if st.session_state.is_running and not st.session_state.config.get('human_participation', False):
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        messages_container = st.container()

        st.session_state.dialogue_results = []
        st.session_state.current_message_index = 0

        try:
            total_expected = st.session_state.config['rounds'] * 3

            progress_bar = progress_placeholder.progress(0)
            status_text = status_placeholder.empty()

            dialogue_gen = run_dialogue(
                query=st.session_state.config['query'],
                use_R=st.session_state.config['use_R'],
                rule_mode=st.session_state.config['rule_mode'],
                adaptive_weight=st.session_state.config['adaptive_weight'],
                rounds=st.session_state.config['rounds']
            )

            for idx, result in enumerate(dialogue_gen):
                st.session_state.dialogue_results.append(result)
                st.session_state.current_message_index = idx + 1

                progress = (idx + 1) / total_expected
                progress_bar.progress(progress)
                status_text.text(f"Round {result['round'] + 1}, Agent: {result['agent']} ({idx + 1}/{total_expected})")

                with messages_container:
                    agent_class = "agent-conservation" if "Conservation" in result['agent'] else \
                        "agent-farmer" if "Farmer" in result['agent'] else "agent-community"

                    metrics = result['metrics']
                    metrics_str = (
                        f"Resp:{metrics['responsive']} | "
                        f"Reb:{metrics['rebuttal']} | "
                        f"NR:{metrics['non_repetition']:.2f} | "
                        f"Ev:{metrics['evidence_usage']} | "
                        f"Stance:{metrics['stance_shift']:.2f}"
                    )

                    st.markdown(f"""
                    <div class="{agent_class}">
                        <strong>{result['agent']}</strong> 
                        <small>Round {result['round']} | wT:{result['weights']['wT']:.2f} wM:{result['weights']['wM']:.2f} wD:{result['weights']['wD']:.2f}</small>
                        <p>{result['response']}</p>
                        <small style='color: #6b7280;'>{metrics_str}</small>
                    </div>
                    """, unsafe_allow_html=True)

                time.sleep(0.1)

            progress_bar.progress(1.0)
            status_text.text("✅ Dialogue completed!")
            st.session_state.is_running = False

            st.success(
                f"✅ Completed {len(st.session_state.dialogue_results)} messages "
                f"across {st.session_state.config['rounds']} rounds!"
            )

        except Exception as e:
            st.error(f"❌ Error running dialogue: {str(e)}")
            import traceback

            with st.expander("Show error details"):
                st.code(traceback.format_exc())
            st.session_state.is_running = False

    elif st.session_state.dialogue_results and not st.session_state.config.get('human_participation', False):
        st.info(f"📝 Showing {len(st.session_state.dialogue_results)} messages from previous dialogue")

        for r in st.session_state.dialogue_results:
            agent_class = "agent-conservation" if "Conservation" in r['agent'] else \
                "agent-farmer" if "Farmer" in r['agent'] else \
                    "agent-human" if "Human" in r['agent'] else "agent-community"

            if 'metrics' in r and r['metrics']:
                metrics = r['metrics']
                metrics_str = (
                    f"Resp:{metrics['responsive']} | "
                    f"Reb:{metrics['rebuttal']} | "
                    f"NR:{metrics['non_repetition']:.2f} | "
                    f"Ev:{metrics['evidence_usage']} | "
                    f"Stance:{metrics['stance_shift']:.2f}"
                )
            else:
                metrics_str = "Human input"

            st.markdown(f"""
            <div class="{agent_class}">
                <strong>{r['agent']}</strong> 
                <small>Round {r['round']} | wT:{r['weights']['wT']:.2f} wM:{r['weights']['wM']:.2f} wD:{r['weights']['wD']:.2f}</small>
                <p>{r['response']}</p>
                <small style='color: #6b7280;'>{metrics_str}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👈 Click 'Start Dialogue' in the sidebar to begin")

# New Interactive Tab for Human Participation
with tab3:
    st.header("🗣️ Interactive Dialogue")

    if not st.session_state.config.get('human_participation', False):
        st.warning("👤 Human participation is disabled. Enable it in the sidebar to use this feature.")
        st.info("Go to **Configuration** → **Human Participation** → Enable the checkbox")
    else:
        # Display current dialogue state
        st.markdown(f"**Topic:** {st.session_state.config['query']}")
        st.markdown(f"**Current Round:** {st.session_state.current_round + 1} / {st.session_state.config['rounds']}")

        st.divider()

        # Display all messages so far
        messages_container = st.container()
        with messages_container:
            if st.session_state.dialogue_results:
                for r in st.session_state.dialogue_results:
                    agent_class = "agent-conservation" if "Conservation" in r['agent'] else \
                        "agent-farmer" if "Farmer" in r['agent'] else \
                            "agent-human" if "Human" in r['agent'] else "agent-community"

                    if 'metrics' in r and r['metrics']:
                        metrics = r['metrics']
                        metrics_str = (
                            f"Resp:{metrics.get('responsive', 'N/A')} | "
                            f"Reb:{metrics.get('rebuttal', 'N/A')} | "
                            f"NR:{metrics.get('non_repetition', 0):.2f}"
                        )
                    else:
                        metrics_str = "Human input"

                    st.markdown(f"""
                    <div class="{agent_class}">
                        <strong>{r['agent']}</strong> 
                        <small>Round {r['round']}</small>
                        <p>{r['response']}</p>
                        <small style='color: #6b7280;'>{metrics_str}</small>
                    </div>
                    """, unsafe_allow_html=True)

        st.divider()

        # Interactive controls
        if st.session_state.is_running and st.session_state.interactive_mode_active:

            # Check if waiting for human input
            if st.session_state.waiting_for_human:
                st.markdown("""
                <div class="human-input-box">
                    <p class="waiting-indicator">🎤 Your turn to speak!</p>
                    <p>The agents are waiting for your input. Share your thoughts on the topic.</p>
                </div>
                """, unsafe_allow_html=True)

                # Human input form
                with st.form(key="human_input_form"):
                    human_message = st.text_area(
                        "Your message:",
                        placeholder="Enter your thoughts, questions, or arguments...",
                        height=100
                    )

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        submit_button = st.form_submit_button("📤 Send Message", use_container_width=True)
                    with col2:
                        skip_button = st.form_submit_button("⏭️ Skip", use_container_width=True)

                    if submit_button and human_message.strip():
                        # Add human message to dialogue
                        human_result = {
                            "round": st.session_state.current_round,
                            "agent": "Human 👤",
                            "response": human_message.strip(),
                            "metrics": None,
                            "weights": {"wT": 1.0, "wM": 1.0, "wD": 1.0}
                        }
                        st.session_state.dialogue_results.append(human_result)
                        st.session_state.dialogue_history_text.append(f"Human 👤: {human_message.strip()}")

                        # 后台打印人类输入
                        print(f"\n[Human 👤]: {human_message.strip()}")

                        st.session_state.waiting_for_human = False

                        # 人类输入完成后，推进到下一轮
                        st.session_state.current_round += 1
                        st.session_state.round_agents_done = False  # 重置，准备下一轮agent对话

                        # 检查是否对话结束
                        if st.session_state.current_round >= st.session_state.config['rounds']:
                            st.session_state.is_running = False
                            st.session_state.interactive_mode_active = False
                            print(f"\n{'=' * 50}")
                            print("Dialogue completed!")
                            print(f"{'=' * 50}")

                        st.rerun()

                    if skip_button:
                        print(f"\n[Human 👤]: (skipped)")
                        st.session_state.waiting_for_human = False

                        # 跳过后也推进到下一轮
                        st.session_state.current_round += 1
                        st.session_state.round_agents_done = False

                        if st.session_state.current_round >= st.session_state.config['rounds']:
                            st.session_state.is_running = False
                            st.session_state.interactive_mode_active = False

                        st.rerun()

            else:
                # 检查当前轮的agent是否已经完成
                if st.session_state.round_agents_done:
                    # Agent已完成，等待状态变化（不应该到这里，但作为安全检查）
                    st.info("⏳ Waiting for next action...")
                else:
                    # Generate agent responses
                    st.info("🤖 Agents are thinking...")

                try:
                    # Import agent modules
                    import agent_conservation
                    import agent_farmer
                    import agent_community
                    from adaptive_weight import scheduler
                    from rag_llm import rag_search
                    from evaluator import is_responsive, is_rebuttal, non_repetition, evidence_usage, stance_shift

                    agents = [
                        {"name": "Conservation 🌲", "module": agent_conservation, "rag_name": "ConservationAgent"},
                        {"name": "Farmer 🚜", "module": agent_farmer, "rag_name": "FarmerAgent"},
                        {"name": "Community 🏘", "module": agent_community, "rag_name": "CommunityAgent"}
                    ]

                    # Initialize weights if not present
                    if 'w_pool' not in st.session_state:
                        st.session_state.w_pool = {a["name"]: [1.0, 1.0, 1.0] for a in agents}
                    if 'last_self' not in st.session_state:
                        st.session_state.last_self = {a["name"]: "" for a in agents}

                    r = st.session_state.current_round

                    # 后台打印当前轮次
                    print(f"\n{'=' * 50}")
                    print(f"========== Round {r} ==========")
                    print(f"{'=' * 50}")

                    # Generate responses for each agent in this round
                    for agent_info in agents:
                        name = agent_info["name"]
                        module = agent_info["module"]
                        rag_name = agent_info["rag_name"]

                        # Build history text
                        if r == 0:
                            history_text = st.session_state.config['query']
                            use_R_this = False
                            prev_round_text = ""
                        else:
                            history_text = "\n".join(st.session_state.dialogue_history_text[-4:])
                            use_R_this = st.session_state.config['use_R']
                            prev_round_text = st.session_state.dialogue_history_text[
                                -1] if st.session_state.dialogue_history_text else ""

                        wT, wM, wD = st.session_state.w_pool[name]
                        prev_self = st.session_state.last_self[name]

                        # Generate response
                        response = module.invoke(
                            history=history_text,
                            round_num=r,
                            query=st.session_state.config['query'],
                            use_T=True, use_M=True, use_D=True,
                            wT=wT, wM=wM, wD=wD,
                            use_R=use_R_this,
                            rule_mode=st.session_state.config['rule_mode'],
                            max_sentences=3
                        )

                        # 后台打印agent对话
                        print(f"\n[{name}]: {response}")

                        # Calculate metrics
                        retrieved = rag_search(history_text, agent=rag_name)
                        rag_sents = [robj["content"] for robj in retrieved]

                        resp_score = is_responsive(response, prev_round_text)
                        reb_score = is_rebuttal(response, prev_round_text)
                        nr_score = non_repetition(response, prev_self)
                        evi_score = evidence_usage(response, rag_sents)
                        stance_score = stance_shift(response, module.MY_TASK)

                        # 后台打印metrics
                        print(f"  [metrics] responsive={resp_score}, rebuttal={reb_score}, "
                              f"non_rep={nr_score:.2f}, evidence={evi_score}, stance={stance_score:.2f}")

                        # Update weights if adaptive
                        if st.session_state.config['adaptive_weight']:
                            new_wT, new_wM, new_wD = scheduler(
                                round_num=r,
                                last_response=response,
                                responsive=resp_score,
                                rag_sentences=rag_sents,
                                wT=wT, wM=wM, wD=wD,
                                alpha=0.2
                            )
                            st.session_state.w_pool[name] = [new_wT, new_wM, new_wD]

                        st.session_state.last_self[name] = response

                        # Store result
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
                        st.session_state.dialogue_results.append(result)
                        st.session_state.dialogue_history_text.append(f"{name}: {response}")

                    # 标记当前轮的agent已完成
                    st.session_state.round_agents_done = True

                    # Check if human should participate
                    freq = st.session_state.config.get('human_frequency', 'every_round')
                    should_participate = (
                            freq == 'every_round' or
                            (freq == 'every_other_round' and r % 2 == 0) or
                            freq == 'on_demand'
                    )

                    if should_participate and r < st.session_state.config['rounds'] - 1:
                        st.session_state.waiting_for_human = True
                    else:
                        # 不需要人类参与，直接推进到下一轮
                        st.session_state.current_round += 1
                        st.session_state.round_agents_done = False  # 重置，准备下一轮
                        if st.session_state.current_round >= st.session_state.config['rounds']:
                            st.session_state.is_running = False
                            st.session_state.interactive_mode_active = False
                            st.success("✅ Dialogue completed!")

                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback

                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
                    st.session_state.is_running = False

        elif st.session_state.dialogue_results:
            st.success(f"✅ Dialogue completed with {len(st.session_state.dialogue_results)} messages!")

            # Option to continue
            if st.button("🔄 Start New Interactive Dialogue"):
                st.session_state.dialogue_results = []
                st.session_state.dialogue_history_text = [f"Initial topic: {st.session_state.config['query']}"]
                st.session_state.current_round = 0
                st.session_state.is_running = True
                st.session_state.interactive_mode_active = True
                st.session_state.waiting_for_human = False
                st.session_state.round_agents_done = False  # 重置
                if 'w_pool' in st.session_state:
                    del st.session_state.w_pool
                if 'last_self' in st.session_state:
                    del st.session_state.last_self
                st.rerun()
        else:
            st.info("👈 Click 'Start Dialogue' in the sidebar to begin interactive mode")

with tab4:
    st.header("📊 Metrics Analysis")

    if st.session_state.dialogue_results:
        # Filter out human results for metrics (they don't have metrics)
        agent_results = [r for r in st.session_state.dialogue_results if r.get('metrics')]

        if agent_results:
            df = pd.DataFrame([
                {
                    'round': r['round'],
                    'agent': r['agent'],
                    **r['metrics'],
                    'wT': r['weights']['wT'],
                    'wM': r['weights']['wM'],
                    'wD': r['weights']['wD']
                }
                for r in agent_results
            ])

            st.subheader("Overall Performance")
            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric("Avg Responsive", f"{df['responsive'].mean():.3f}")
            col2.metric("Avg Rebuttal", f"{df['rebuttal'].mean():.3f}")
            col3.metric("Avg Non-repetition", f"{df['non_repetition'].mean():.3f}")
            col4.metric("Avg Evidence", f"{df['evidence_usage'].mean():.3f}")
            col5.metric("Avg Stance Shift", f"{df['stance_shift'].mean():.3f}")

            st.divider()

            st.subheader("Per-Agent Metrics")
            agent_metrics = df.groupby('agent')[
                ['responsive', 'rebuttal', 'non_repetition', 'evidence_usage', 'stance_shift']].mean()
            st.dataframe(agent_metrics.style.format("{:.3f}"), use_container_width=True)

            st.divider()
            st.subheader("Metric Trends Over Rounds")

            metric_choice = st.selectbox(
                "Select Metric",
                ['responsive', 'rebuttal', 'non_repetition', 'evidence_usage', 'stance_shift']
            )

            fig = px.line(
                df,
                x='round',
                y=metric_choice,
                color='agent',
                markers=True,
                title=f"{metric_choice.replace('_', ' ').title()} by Round",
                labels={'round': 'Round', metric_choice: metric_choice.replace('_', ' ').title()}
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Weight Evolution")

            weight_cols = st.columns(3)
            for idx, weight in enumerate(['wT', 'wM', 'wD']):
                with weight_cols[idx]:
                    fig = px.line(
                        df,
                        x='round',
                        y=weight,
                        color='agent',
                        markers=True,
                        title=f"{weight} Over Rounds"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Metrics Heatmap")

            heatmap_data = df.groupby(['agent', 'round'])[
                ['responsive', 'rebuttal', 'non_repetition', 'evidence_usage', 'stance_shift']].mean()

            for agent in df['agent'].unique():
                if agent in heatmap_data.index:
                    agent_data = heatmap_data.loc[agent]
                    fig = go.Figure(data=go.Heatmap(
                        z=agent_data.values.T,
                        x=agent_data.index,
                        y=agent_data.columns,
                        colorscale='RdYlGn',
                        text=agent_data.values.T,
                        texttemplate='%{text:.2f}',
                        textfont={"size": 10}
                    ))
                    fig.update_layout(title=f"{agent} - Metrics by Round", xaxis_title="Round", yaxis_title="Metric")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No agent metrics available yet")
    else:
        st.info("👈 Run a dialogue first to see metrics")

with tab5:
    st.header("📚 Knowledge Base Information")

    st.markdown("View the current knowledge base status for each agent.")

    for agent_name, agent_config in list(AGENTS.items())[:3]:  # Only AI agents
        rag_name = agent_config["rag_name"]
        kb_info = st.session_state.rag_instance.get_kb_info(rag_name)

        with st.expander(f"**{agent_name}** - Knowledge Base Details", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Status", "✅ Loaded" if kb_info['loaded'] else "❌ Not Loaded")

            with col2:
                kb_type = "Default" if kb_info.get('using_default', True) else "Custom"
                st.metric("Type", kb_type)

            with col3:
                st.metric("Chunks", kb_info.get('num_chunks', 0))

            # Test search
            st.markdown("**Test Retrieval:**")
            test_query = st.text_input(
                "Enter test query",
                key=f"test_query_{rag_name}",
                placeholder="e.g., forest conservation"
            )

            if test_query:
                results = st.session_state.rag_instance.rag_search(test_query, rag_name, top_k=3)

                if results:
                    st.markdown(f"**Top {len(results)} Results:**")
                    for i, result in enumerate(results, 1):
                        st.markdown(f"**Result {i}** (Score: {result['score']:.4f})")
                        st.text_area(
                            f"Content",
                            value=result['content'],
                            height=100,
                            key=f"result_{rag_name}_{i}",
                            disabled=True
                        )
                else:
                    st.warning("No results found")

with tab6:
    st.header("📥 Export Results")

    if st.session_state.dialogue_results:
        # Include human messages in export
        export_data = []
        for r in st.session_state.dialogue_results:
            row = {
                'round': r['round'],
                'agent': r['agent'],
                'response': r['response'],
                'wT': r['weights']['wT'],
                'wM': r['weights']['wM'],
                'wD': r['weights']['wD'],
            }
            if r.get('metrics'):
                row.update({
                    'responsive': r['metrics']['responsive'],
                    'rebuttal': r['metrics']['rebuttal'],
                    'non_repetition': r['metrics']['non_repetition'],
                    'evidence_usage': r['metrics']['evidence_usage'],
                    'stance_shift': r['metrics']['stance_shift'],
                })
            else:
                row.update({
                    'responsive': None,
                    'rebuttal': None,
                    'non_repetition': None,
                    'evidence_usage': None,
                    'stance_shift': None,
                })
            export_data.append(row)

        df_export = pd.DataFrame(export_data)

        st.subheader("Preview")
        st.dataframe(df_export, use_container_width=True)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Export as CSV")
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name='dialogue_results.csv',
                mime='text/csv',
                use_container_width=True
            )

        with col2:
            st.subheader("Export as JSON")
            import json

            json_str = json.dumps(st.session_state.dialogue_results, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name='dialogue_results.json',
                mime='application/json',
                use_container_width=True
            )

        st.divider()
        st.subheader("Summary Statistics")

        # Only for agent results
        agent_df = df_export[df_export['responsive'].notna()]
        if not agent_df.empty:
            summary_stats = agent_df[
                ['responsive', 'rebuttal', 'non_repetition', 'evidence_usage', 'stance_shift']].describe()
            st.dataframe(summary_stats, use_container_width=True)

    else:
        st.info("👈 Run a dialogue first to export results")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; padding: 0.5rem; font-size: 0.75rem;">
    <p>QTMD Multi-Agent Dialogue System with Human Participation | Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)