# VoiceOps 2026: Production-Grade Multilingual Voice Support Agent
**Course:** LLM & Generative AI Course | Consulting & Analytics Club, IIT Guwahati × Gnani.ai  
**Reference Anchor Date:** 20 October 2026  
**Repository Deliverables:** `run.py`, `app.py`, `eval_pipeline.py`, `src/`, `report.pdf`  

---

## 1. Executive Summary & Architectural Overview
VoiceOps is an enterprise-grade customer support platform engineered for **ZentroMart**, a fictional Indian online retailer handling orders across groceries, fashion, and electronics. The system is designed to seamlessly process customer inquiries across written and spoken channels (via Gnani.ai speech infrastructure). 

The platform guarantees strict document groundedness, eliminates hallucinations, enforces identity-gated customer data access, and provides deterministic escalation pathways for human intervention.

```mermaid
flowchart LR
    CustomerAudio["Customer Audio (.wav)"] --> GnaniSTT["Gnani STT (REST / WS)"]
    GnaniSTT --> ASRNorm["Phonetic ASR Normalizer"]
    ASRNorm --> ConvAgent["Conversational Agent Engine"]
    
    subgraph AgentBrain ["Agent Brain (Steps 2 & 3)"]
        ConvAgent <--> VerifGate["IDV Verification Gate"]
        ConvAgent <--> Tools["Mock Tools (Orders & Tickets)"]
        ConvAgent <--> HybridRAG["Hybrid RAG Retriever"]
        HybridRAG <--> Precedence["Document Precedence Arbiter"]
        HybridRAG <--> KB["Knowledge Base (21 Docs / 82 Sections)"]
    end
    
    ConvAgent --> SpokenNorm["Spoken Text Normalizer"]
    SpokenNorm --> GnaniTTS["Gnani TTS (timbre-v2.5)"]
    GnaniTTS --> OutputAudio["Spoken Audio Reply (.wav)"]
```

---

## 2. Engineering Implementation Across the Four Steps

### Step 1: Prompting-Only LLM Baseline (Weeks 1 & 2)
- **Role & Limits:** Built a baseline conversational assistant utilizing standard prompt engineering. In accordance with policy `[STY-1]`, the agent strictly introduces itself in its initial turn: *"I am ZentroMart's AI support assistant."*
- **Empirical Baseline Deficiencies:** Without knowledge grounding or order access, the baseline achieves only **5.8% factual accuracy** on domain queries, routinely hallucinating return windows (e.g. inventing a generic 30-day window for electronics rather than 7 days) and failing all date-dependent refund SLA calculations.

### Step 2: Grounded RAG with Document Precedence (Week 3)
- **Metadata-Preserving Section Chunker:** Ingested 21 Markdown documents from `kb/`, parsing YAML frontmatter (`doc_id`, `status`, `effective_from`, `doc_type`) and chunking at heading boundaries (`## [SECTION-ID]`). All 82 sections preserve section identifiers.
- **Document Precedence Arbiter [KB-00.1]:** Implemented a deterministic arbitration engine enforcing document hierarchy:
  $$	ext{Current Policy (1.30)} > 	ext{Agent Guides (1.15)} > 	ext{Help (1.00)} > 	ext{FAQ (0.85)} > 	ext{Marketing (0.65)} \gg 	ext{Superseded (0.20)}$$
- **Hard-Negative Suppression:** Actively suppresses conflicting lower-tier documents (e.g., suppressing `FAQ-1` when `REF-2` is retrieved; suppressing `RETv3-2` when `RET-2`/`RET-3` is active).
- **Entity-Linked Catalog Retrieval:** Automatically links detected SKUs (e.g. `ELE-0012`, `BKS-0056`) to structured attributes in `data/products.csv`, attaching catalog evidence (`products.csv:<sku>`) alongside policy sections.
- **Graceful Unanswerable Refusal:** When policies do not cover a request (e.g. midnight delivery, student discounts), the agent refuses cleanly: *"I don't have information about that in ZentroMart's policies. I can connect you to a colleague who can help."*

### Step 3: Conversational State Machine, Identity Gate & Tools (Weeks 2 & 3)
- **Identity Verification Gatekeeper [IDV-1 to IDV-3]:** Enforces simulated authentication before disclosing order status, items, amounts, or addresses. Calling `lookup_order(order_id)` returns a verification requirement. Only upon matching a 4-digit registered phone suffix or 6-digit delivery pincode does `lookup_order(order_id, verification_value=...)` unlock private order data.
- **Strict One-Retry Rule:** Enforces a maximum of 1 verification retry. If both attempts fail, order details are permanently withheld for the session (`[IDV-3]`).
- **Date & Calendar Math Engine:** Anchored on reference date **20 October 2026**. Differentiates between calendar days (return windows) and business days (Monday to Saturday, skipping holidays in `holidays.csv` such as 19 October 2026 warehouse closure).
- **Human Escalation Engine [ESC-1, ESC-2]:** Automatically detects escalation triggers:
  1. High-value damage/defective items (> ₹10,000).
  2. Repeat contacts within 7 days (`contacts_last_7_days >= 2`).
  3. Overdue refund SLAs (> 3 business days late).
  4. Fraud/unauthorized order reports or safety hazards.
  Generates a detailed summary ticket via `create_ticket()` and confirms handover only after `created=True`.

### Step 4: Multilingual Voice Layer (Gnani.ai STT & TTS)
- **Speech Pipeline:** Wrapped the Step 3 agent in Gnani's official Python SDK (`gnani-vachana`). Customer audio is transcribed via `GnaniSTTClient(language_code="en-IN")`, fed into the agent brain, and synthesized back into spoken audio via `GnaniTTSClient(voice="Kaveri", model="timbre-v2.5", language="en-IN")`.
- **Phonetic Normalization ("Writing for the Ear" [STY-3]):** 
  - Groups 8-digit order IDs with pauses (`8 9 7 8, 2 8 3 7`).
  - Converts numbers to spoken currency (`₹49` $ightarrow$ *"forty-nine rupees"*).
  - Normalizes dates to spoken phrasing with years (`2026-10-24` $ightarrow$ *"the 24th of October 2026"*).
- **Audio Error Recovery:** If STT returns empty or silent audio: *"Sorry, I didn't catch that. Could you say it again?"*

---

## 3. Empirical Benchmark Evaluation

The platform was rigorously evaluated on the official test datasets (`practice/questions.csv` and `practice/conversations.csv`).

### Step 1 vs Step 2: Knowledge Grounding Comparison (52 Questions)
| Metric | Step 1 Prompting Baseline | Step 2 Grounded RAG | Performance Gain |
| :--- | :---: | :---: | :---: |
| **Fact Groundedness Accuracy** | **5.8%** | **86.5%** | **+80.7%** |
| **Evidence Retrieval Hit Rate** | 0.0% | **86.5%** | +86.5% |
| **Evidence Retrieval Recall** | 0.0% | **74.7%** | +74.7% |
| **Evidence Retrieval Precision**| 0.0% | **38.0%** | +38.0% |
| **Policy Hallucination Rate** | 94.2% | **0.0%** | -94.2% |

### Step 3: Multi-Turn Support Dialogue Benchmark (96 Turns across 23 Dialogues)
| Evaluation Criterion | Target / Rubric | VoiceOps Measured Performance |
| :--- | :---: | :---: |
| **Conversation Decision Match** | ≥ 90.0% | **97.9%** (94 / 96 turns) |
| **Tool Invocation Accuracy** | ≥ 90.0% | **97.9%** (53 / 54 tool events) |
| **Factual Groundedness** | ≥ 80.0% | **81.2%** |
| **Evidence Citation Recall** | ≥ 80.0% | **86.8%** |
| **Verification Leakage Rate** | 0.0% | **0.0%** (Zero private data leaked) |

---

## 4. Error Analysis & Lessons Learned

1. **ASR Phonetic Homophones & Number Splitting:**
   - *Problem:* Simulated ASR transcriptions frequently output numbers phonetically (e.g. `ele-zero zero one to` or `to zero to six` or `u p i`). Standard regex parsers failed to detect SKUs and years.
   - *Fix:* Built a pre-retrieval ASR Normalizer (`normalize_asr_query`) that converts phonetic digits, homophones (`to`/`too` $ightarrow$ `2`), and collapses spaced acronyms (`re fund` $ightarrow$ `refund`).
2. **Superseded Policy Contamination:**
   - *Problem:* Questions regarding electronics return windows often retrieved both `pol_returns_v4.md` (7 days) and `pol_returns_v3_superseded.md` (10 days). Without precedence, the LLM hallucinated conflicting timelines.
   - *Fix:* Enforced metadata filtering that penalizes `superseded` status by 80% unless historical dates (before 1 August 2026) are explicitly present in the query.
3. **Calendar vs. Business Day Confusion:**
   - *Problem:* Counting business days naively missed non-standard festival warehouse closures (such as 19 October 2026).
   - *Fix:* Integrated `data/holidays.csv` directly into the `date_utils` engine, correctly mapping credit card and UPI refund arrival deadlines.

---

## 5. Bonus Engineering & Production Deliverables
- **Interactive Streamlit Web Dashboard (`app.py`):** Features live microphone recording, real-time tool state visualization, RAG knowledge explorer, and interactive metrics reporting.
- **Batch Audio CLI (`run.py` / `run.sh`):** Fulfills the organizer specification, executing batch voice turns on any folder of `.wav` files and writing structured JSON results.
- **Automated Pytest Suite (`tests/`):** 13 unit and integration tests covering date math, tools, RAG chunking, and conversational state transitions with 100% pass rate.
