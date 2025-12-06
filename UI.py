import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dialogue_runner import run_dialogue
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

# Custom CSS (same as before)
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
    .agent-conservation, .agent-farmer, .agent-community {
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
    .agent-conservation strong, .agent-farmer strong, .agent-community strong {
        font-size: 0.75rem;
    }
    .agent-conservation p, .agent-farmer p, .agent-community p {
        font-size: 0.7rem;
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
        line-height: 1.3;
    }
    .agent-conservation small, .agent-farmer small, .agent-community small {
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
        'wD': 1.5
    }
if 'rag_instance' not in st.session_state:
    st.session_state.rag_instance = get_rag_instance()
if 'kb_info' not in st.session_state:
    st.session_state.kb_info = {}

# Agent
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
    }
}

# Header
st.title("🧠 QTMD Multi-Agent Dialogue System")
st.markdown(
    "<p style='font-size: 0.9rem; color: #6b7280;'><strong>Query-Task-Memory-Data Framework</strong> for Multi-Agent Debate with Custom Knowledge Base</p>",
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

    # 📚 Knowledge Base Upload Section
    st.subheader("📚 Knowledge Base")

    with st.expander("📤 Upload Custom Knowledge Base", expanded=False):
        st.markdown("**Upload files for each agent's knowledge base**")
        st.markdown("Supported formats: `.txt`, `.md`, `.pdf`, `.docx`")

        for agent_name, agent_config in AGENTS.items():
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

            # 文件上传器
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
            'wD': wD
        }
        st.session_state.is_running = True
        st.session_state.dialogue_results = []
        st.rerun()

    # Clear button
    if st.button("🗑️ Clear Results", use_container_width=True):
        st.session_state.dialogue_results = []
        st.session_state.current_message_index = 0
        st.session_state.is_running = False
        st.rerun()

    st.markdown("---")

    # Current config display
    with st.expander("📋 Current Configuration"):
        st.json(st.session_state.config)

# Main content - Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "💬 Dialogue", "📈 Metrics", "📚 KB Info", "📥 Export"])

with tab1:
    st.header("📊 System Overview")

    # Agent descriptions
    col1, col2, col3 = st.columns(3)

    agents_list = list(AGENTS.items())
    for idx, (col, (agent_name, agent_config)) in enumerate(zip([col1, col2, col3], agents_list)):
        with col:
            agent_class = ["agent-conservation", "agent-farmer", "agent-community"][idx]
            st.markdown(f"""
            <div class="{agent_class}">
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
    st.header("💬 Dialogue History")

    # Run dialogue if triggered
    if st.session_state.is_running:
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

    elif st.session_state.dialogue_results:
        st.info(f"📝 Showing {len(st.session_state.dialogue_results)} messages from previous dialogue")

        for r in st.session_state.dialogue_results:
            agent_class = "agent-conservation" if "Conservation" in r['agent'] else \
                "agent-farmer" if "Farmer" in r['agent'] else "agent-community"

            metrics = r['metrics']
            metrics_str = (
                f"Resp:{metrics['responsive']} | "
                f"Reb:{metrics['rebuttal']} | "
                f"NR:{metrics['non_repetition']:.2f} | "
                f"Ev:{metrics['evidence_usage']} | "
                f"Stance:{metrics['stance_shift']:.2f}"
            )

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

with tab3:
    st.header("📊 Metrics Analysis")

    if st.session_state.dialogue_results:
        df = pd.DataFrame([
            {
                'round': r['round'],
                'agent': r['agent'],
                **r['metrics'],
                'wT': r['weights']['wT'],
                'wM': r['weights']['wM'],
                'wD': r['weights']['wD']
            }
            for r in st.session_state.dialogue_results
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
        st.info("👈 Run a dialogue first to see metrics")

with tab4:
    st.header("📚 Knowledge Base Information")

    st.markdown("View the current knowledge base status for each agent.")

    for agent_name, agent_config in AGENTS.items():
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

with tab5:
    st.header("📥 Export Results")

    if st.session_state.dialogue_results:
        df_export = pd.DataFrame([
            {
                'round': r['round'],
                'agent': r['agent'],
                'responsive': r['metrics']['responsive'],
                'rebuttal': r['metrics']['rebuttal'],
                'non_repetition': r['metrics']['non_repetition'],
                'evidence_usage': r['metrics']['evidence_usage'],
                'stance_shift': r['metrics']['stance_shift'],
                'wT': r['weights']['wT'],
                'wM': r['weights']['wM'],
                'wD': r['weights']['wD'],
                'response': r['response']
            }
            for r in st.session_state.dialogue_results
        ])

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

        summary_stats = df_export[
            ['responsive', 'rebuttal', 'non_repetition', 'evidence_usage', 'stance_shift']].describe()
        st.dataframe(summary_stats, use_container_width=True)

    else:
        st.info("👈 Run a dialogue first to export results")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; padding: 0.5rem; font-size: 0.75rem;">
    <p>QTMD Multi-Agent Dialogue System with Custom KB | Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)