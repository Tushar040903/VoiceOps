"""VoiceOps Interactive Web Application.

Demonstrates real-time voice & text agent interaction, identity verification,
knowledge-base RAG exploration, and official benchmark evaluation metrics.

Run:
    streamlit run app.py
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd

from src.data.loader import DataLoader
from src.agent.conversational_agent import ConversationalAgent
from src.agent.tools import execute_lookup_order, execute_create_ticket
from src.rag.retriever import HybridRetriever
from src.voice.gnani_voice import VoiceManager, generate_synthetic_wav

# Page configuration
st.set_page_config(
    page_title="VoiceOps 2026 | ZentroMart AI Support",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
    }
    .chat-bubble-user {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
    .chat-bubble-agent {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
        padding: 0.8rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
    .citation-tag {
        display: inline-block;
        background-color: #EDE9FE;
        color: #6D28D9;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_services():
    loader = DataLoader()
    retriever = HybridRetriever(data_loader=loader)
    agent = ConversationalAgent(data_loader=loader, retriever=retriever)
    voice = VoiceManager()
    return loader, retriever, agent, voice


loader, retriever, agent, voice = get_services()

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/headset.png", width=64)
    st.markdown("## **VoiceOps 2026**")
    st.markdown("*LLM & GenAI Support Agent*")
    st.caption("IIT Guwahati × Gnani.ai Capstone")
    st.divider()

    nav = st.radio(
        "Navigation",
        [
            "🎙️ Live Agent (Voice & Chat)",
            "🔐 Order Verification Gate",
            "📚 Knowledge Base & RAG",
            "📊 Benchmark Metrics",
            "ℹ️ Architecture & About",
        ],
        index=0,
    )

    st.divider()
    st.markdown("### System Status")
    st.write(f"**Reference Date:** 20 Oct 2026")
    st.write(f"**Knowledge Chunks:** {len(retriever.chunks)} sections")
    st.write(f"**Synthetic Orders:** {len(loader.orders):,} rows")
    st.write(f"**Gnani Speech:** {'🟢 Live API' if voice.is_live else '🟡 Simulation Mode'}")


# -------------------------------------------------------------
# TAB 1: Live Agent (Voice & Chat)
# -------------------------------------------------------------
if nav == "🎙️ Live Agent (Voice & Chat)":
    st.markdown('<div class="main-header">🎙️ ZentroMart Voice & Chat Support Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-turn grounded conversational support with caller verification, tool actions, and Gnani TTS voice replies.</div>',
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = "streamlit_web_session"

    col_chat, col_tools = st.columns([3, 2])

    with col_chat:
        st.subheader("Conversation")

        # Display history
        for turn in st.session_state.chat_history:
            if turn["role"] == "user":
                st.markdown(
                    f'<div class="chat-bubble-user"><strong>Customer:</strong> {turn["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                citations_html = "".join([f'<span class="citation-tag">[{c}]</span>' for c in turn.get("evidence", [])])
                st.markdown(
                    f'<div class="chat-bubble-agent"><strong>ZentroMart AI:</strong> {turn["content"]}<br><small style="color:#6B7280;">Decision: {turn.get("decision")} | Evidence: {citations_html}</small></div>',
                    unsafe_allow_html=True,
                )
                if turn.get("audio_file") and Path(turn["audio_file"]).exists():
                    st.audio(turn["audio_file"])

        st.divider()

        # Input options
        input_mode = st.radio("Input Channel:", ["💬 Text Message", "🎙️ Spoken Audio (.wav)"], horizontal=True)

        if input_mode == "💬 Text Message":
            with st.form("chat_form", clear_on_submit=True):
                user_msg = st.text_input("Type your message to ZentroMart support:")
                submitted = st.form_submit_button("Send Message")

                if submitted and user_msg.strip():
                    # Process turn
                    res = agent.respond(
                        user_msg,
                        session_id=st.session_state.session_id,
                        mode="conversational",
                        voice_mode=True,
                    )

                    # Synthesize audio reply
                    audio_out = Path("sample_outputs") / f"reply_{len(st.session_state.chat_history)}.wav"
                    voice.synthesize(res["spoken_reply"], output_file=audio_out)

                    st.session_state.chat_history.append({"role": "user", "content": user_msg})
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": res["reply"],
                        "spoken": res["spoken_reply"],
                        "decision": res["decision"],
                        "evidence": res["evidence_ids"],
                        "tool_calls": res["tool_calls"],
                        "audio_file": str(audio_out),
                    })
                    st.rerun()

        else:
            uploaded_audio = st.file_uploader("Upload customer audio (.wav)", type=["wav", "mp3"])
            if uploaded_audio:
                temp_in = Path("sample_outputs") / "temp_customer_voice.wav"
                temp_in.parent.mkdir(parents=True, exist_ok=True)
                temp_in.write_bytes(uploaded_audio.read())

                if st.button("Transcribe & Process Audio"):
                    audio_out = Path("sample_outputs") / f"reply_{len(st.session_state.chat_history)}.wav"
                    res = voice.voice_turn(
                        audio_path=temp_in,
                        agent=agent,
                        output_wav_path=audio_out,
                        session_id=st.session_state.session_id,
                    )

                    st.session_state.chat_history.append({"role": "user", "content": res["transcript"]})
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": res["reply"],
                        "spoken": res["spoken_reply"],
                        "decision": res["decision"],
                        "evidence": res["evidence_ids"],
                        "tool_calls": res["tool_calls"],
                        "audio_file": str(audio_out),
                    })
                    st.rerun()

        if st.button("Clear Conversation"):
            st.session_state.chat_history = []
            agent.reset_session(st.session_state.session_id)
            st.rerun()

    with col_tools:
        st.subheader("Agent Brain State & Tools")
        sess = agent.get_or_create_session(st.session_state.session_id)

        st.markdown(f"**Session ID:** `{sess.conversation_id}`")
        st.markdown(f"**Turn Count:** `{sess.turn}`")
        st.markdown(f"**Detected Order ID:** `{sess.order_id or 'None'}`")
        st.markdown(f"**Verification Status:** {'✅ Verified' if sess.is_verified else '🔒 Unverified'}")
        st.markdown(f"**Failed Verification Tries:** `{sess.verification_attempts}`")
        st.markdown(f"**Escalation Ticket ID:** `{sess.ticket_id or 'None'}`")

        if sess.order_data:
            st.divider()
            st.markdown("#### Unlocked Order Details")
            st.json({
                "order_id": sess.order_data.get("order_id"),
                "customer": sess.order_data.get("customer_first_name"),
                "is_plus": sess.order_data.get("is_plus"),
                "status": sess.order_data.get("status"),
                "total_inr": sess.order_data.get("total_inr"),
                "payment_method": sess.order_data.get("payment_method"),
                "items_count": len(sess.order_data.get("items", [])),
            })


# -------------------------------------------------------------
# TAB 2: Order Verification Gate
# -------------------------------------------------------------
elif nav == "🔐 Order Verification Gate":
    st.markdown('<div class="main-header">🔐 Identity Verification Gatekeeper</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Simulates ZentroMart identity verification [IDV-1, IDV-2, IDV-3]. Private order details remain locked until phone suffix or delivery pincode matches.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Test Order Lookup")
        test_oid = st.text_input("Enter 8-digit Order ID:", value="89782837")
        test_verif = st.text_input("Verification Value (Last 4 digits of phone OR 6-digit pincode):", value="")

        btn_unverified = st.button("1. Lookup Without Verification (Order ID Only)")
        btn_verified = st.button("2. Lookup With Verification Value")

        if btn_unverified:
            res = execute_lookup_order(test_oid)
            st.warning("Tool Response (Unverified):")
            st.json(res)

        if btn_verified:
            res = execute_lookup_order(test_oid, verification_value=test_verif)
            if res.get("verified"):
                st.success("Verification Succeeded! Full Order Details Disclosed:")
                st.json(res)
            else:
                st.error("Verification Failed! Order details withheld:")
                st.json(res)

    with c2:
        st.subheader("Synthetic Verification Rules")
        st.markdown("""
        - **[IDV-1] Rule:** Never disclose order items, amounts, or address before customer verification.
        - **[IDV-2] Verification Factors:**
          - 4-digit suffix of registered phone number (e.g. `4533`)
          - 6-digit delivery pincode (e.g. `781006`)
        - **[IDV-3] Retry Limit:** Maximum **1 retry** allowed. If the second attempt fails, order details are permanently withheld for that session.
        - **[ESC-1] High-Value Escalation:** Damaged or defective items > ₹10,000 immediately trigger a priority handover ticket.
        """)


# -------------------------------------------------------------
# TAB 3: Knowledge Base & RAG
# -------------------------------------------------------------
elif nav == "📚 Knowledge Base & RAG":
    st.markdown('<div class="main-header">📚 Knowledge Base & Precedence Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Explore all 21 company documents, section IDs, and test hybrid semantic retrieval with document precedence arbitration.</div>',
        unsafe_allow_html=True,
    )

    search_q = st.text_input("Search Policy Knowledge Base:", value="What is the return window for clothing and footwear?")
    k_val = st.slider("Top Chunks (k):", min_value=1, max_value=8, value=4)

    if search_q:
        results = retriever.retrieve(search_q, top_k=k_val)
        st.markdown(f"**Retrieved Evidence String:** `{results['evidence_str']}`")

        for idx, chunk in enumerate(results["chunks"], 1):
            with st.expander(f"#{idx} [{chunk.section_id}] {chunk.section_title} ({chunk.doc_type}, Status: {chunk.status})"):
                st.markdown(f"**Document:** `{chunk.doc_title}` | **Effective From:** `{chunk.effective_from}`")
                st.write(chunk.content)


# -------------------------------------------------------------
# TAB 4: Benchmark Metrics
# -------------------------------------------------------------
elif nav == "📊 Benchmark Metrics":
    st.markdown('<div class="main-header">📊 Official Capstone Benchmark Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Evaluation of Step 1 (Prompting Baseline) vs Step 2 (Grounded RAG) on 52 practice questions, and Step 3 multi-turn conversations.</div>',
        unsafe_allow_html=True,
    )

    eval_json_path = Path("sample_outputs") / "eval_results.json"
    if eval_json_path.exists():
        with open(eval_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        s1 = data["step1_baseline"]
        s2 = data["step2_rag"]
        s3 = data["step3_conversations"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Step 1 Baseline Acc.", f"{s1['accuracy']*100:.1f}%")
        m2.metric("Step 2 Grounded RAG Acc.", f"{s2['accuracy']*100:.1f}%", f"+{(s2['accuracy']-s1['accuracy'])*100:.1f}%")
        m3.metric("RAG Evidence Hit Rate", f"{s2['hit_rate']*100:.1f}%")
        m4.metric("Step 3 Decision Match", f"{s3['decision_accuracy']*100:.1f}%")

        st.divider()
        st.subheader("Step 1 vs Step 2 Grounding Comparison")

        comp_df = pd.DataFrame([
            {"Metric": "Fact Groundedness Accuracy", "Step 1 (Prompt-Only)": f"{s1['accuracy']*100:.1f}%", "Step 2 (Grounded RAG)": f"{s2['accuracy']*100:.1f}%", "Gain": f"+{(s2['accuracy']-s1['accuracy'])*100:.1f}%"},
            {"Metric": "Evidence Retrieval Recall", "Step 1 (Prompt-Only)": "0.0%", "Step 2 (Grounded RAG)": f"{s2['retrieval_recall']*100:.1f}%", "Gain": f"+{s2['retrieval_recall']*100:.1f}%"},
            {"Metric": "Retrieval Precision", "Step 1 (Prompt-Only)": "0.0%", "Step 2 (Grounded RAG)": f"{s2['retrieval_precision']*100:.1f}%", "Gain": f"+{s2['retrieval_precision']*100:.1f}%"},
            {"Metric": "Retrieval Hit Rate", "Step 1 (Prompt-Only)": "0.0%", "Step 2 (Grounded RAG)": f"{s2['hit_rate']*100:.1f}%", "Gain": f"+{s2['hit_rate']*100:.1f}%"},
        ])
        st.table(comp_df)

        st.subheader("Step 3 Multi-Turn Dialogue Benchmark (96 Customer Turns)")
        st.write(f"- **Decision Match Accuracy:** `{s3['decision_accuracy']*100:.1f}%`")
        st.write(f"- **Tool Invocation Accuracy:** `{s3['tool_call_accuracy']*100:.1f}%`")
        st.write(f"- **Fact Groundedness:** `{s3['fact_groundedness']*100:.1f}%`")
        st.write(f"- **Evidence Recall:** `{s3['average_evidence_recall']*100:.1f}%`")

    else:
        st.info("Run `python eval_pipeline.py` to generate the latest benchmark report.")


# -------------------------------------------------------------
# TAB 5: Architecture & About
# -------------------------------------------------------------
elif nav == "ℹ️ Architecture & About":
    st.markdown('<div class="main-header">ℹ️ System Architecture & Specifications</div>', unsafe_allow_html=True)
    st.markdown(
        """
        ### System Architecture
        ```mermaid
        flowchart LR
            CustomerAudio["Customer Audio (.wav)"] --> GnaniSTT["Gnani STT (REST)"]
            GnaniSTT --> ASRNorm["ASR & Query Normalizer"]
            ASRNorm --> ConvAgent["Conversational Agent Engine"]
            
            subgraph AgentBrain ["Agent Brain (Step 2 & 3)"]
                ConvAgent <--> VerifGate["IDV Verification Gate"]
                ConvAgent <--> Tools["Order & Ticket Tools"]
                ConvAgent <--> HybridRAG["Hybrid RAG Retriever"]
                HybridRAG <--> Precedence["Document Precedence Arbiter"]
                HybridRAG <--> KB["Knowledge Base (21 Docs)"]
            end
            
            ConvAgent --> SpokenNorm["Spoken Text Normalizer"]
            SpokenNorm --> GnaniTTS["Gnani TTS (timbre-v2.5)"]
            GnaniTTS --> OutputAudio["Spoken Audio Reply (.wav)"]
        ```

        ### Core Capabilities
        1. **End-to-End Multilingual Speech:** Powered by Gnani.ai STT & TTS with text normalization ('writing for the ear').
        2. **Document Grounded RAG:** Hybrid dense/lexical search over 82 policy sections with strict precedence arbitration.
        3. **Strict Identity Verification Gate:** Discloses order information only upon verified phone suffix or delivery pincode.
        4. **Business Day & Date Rules:** Computes exact refund and delivery SLAs using 20 October 2026 reference date and warehouse holiday calendar.
        5. **Deterministic Human Escalation:** Automated ticket generation for high-value damages (> ₹10,000), repeat contacts (≥ 2), and fraud reports.
        """
    )

