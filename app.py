import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import re

st.set_page_config(
    page_title = "Smart MCQ Solver",
    layout     = "centered"
)

@st.cache_resource
def load_model():
    token     = st.secrets["HF_TOKEN"]
    tokenizer = AutoTokenizer.from_pretrained(
        "AnishSPIT/deberta-mcq-solver",
        token = token
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        "AnishSPIT/deberta-mcq-solver",
        token = token
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
OPTIONS = ["A", "B", "C", "D", "E"]

def clean_prompt(text):
    text = re.sub(r'(?i)pick the best possible answer[:\s]*', '', text)
    text = re.sub(r'(?i)choose the best answer[:\s]*', '', text)
    text = re.sub(r'(?i)which of the following[:\s]*', '', text)
    text = re.sub(r'(?i)determine the correct option[:\s]*', '', text)
    text = re.sub(r'(?i)select the most accurate option[:\s]*', '', text)
    return text.strip()

def predict(prompt, options):
    prompt    = clean_prompt(prompt)
    encodings = []
    for opt in options:
        enc = tokenizer(
            "Question: " + prompt,
            "Answer choice: " + opt,
            max_length     = 384,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt"
        )
        encodings.append({k: v.squeeze(0) for k, v in enc.items()})

    input_ids      = torch.stack([e["input_ids"]      for e in encodings]).unsqueeze(0)
    attention_mask = torch.stack([e["attention_mask"] for e in encodings]).unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask
        )

    probs  = torch.softmax(outputs.logits[0], dim=-1).numpy()
    ranked = sorted(range(5), key=lambda i: probs[i], reverse=True)
    return ranked, probs

# ── PAGE HEADER ───────────────────────────────────────────────
st.title(" Smart MCQ Solver")
st.markdown("""
**Fine-tuned DeBERTa-v3-base for Multiple Choice Question Answering**  
*BSDA2001P — Introduction to DL and GenAI | IITM BS Data Science*
""")
st.markdown(f"**Kaggle Test MAP@3: 0.7596**")
st.divider()

# ── INPUT SECTION ─────────────────────────────────────────────
st.subheader("Enter Question")
prompt = st.text_area(
    "Question Prompt",
    placeholder = "Enter your multiple choice question here...",
    height      = 100
)

st.subheader("Enter Options")
col1, col2 = st.columns(2)

with col1:
    opt_a = st.text_input("Option A", placeholder="Enter option A...")
    opt_b = st.text_input("Option B", placeholder="Enter option B...")
    opt_c = st.text_input("Option C", placeholder="Enter option C...")

with col2:
    opt_d = st.text_input("Option D", placeholder="Enter option D...")
    opt_e = st.text_input("Option E", placeholder="Enter option E...")

st.write("")
predict_btn = st.button(" Predict Top-3 Answers", use_container_width=True, type="primary")

# ── PREDICTION ────────────────────────────────────────────────
if predict_btn:
    options = [opt_a, opt_b, opt_c, opt_d, opt_e]

    if not prompt.strip():
        st.error("Please enter a question prompt!")
    elif not all(o.strip() for o in options):
        st.error("Please fill in all 5 options!")
    else:
        with st.spinner("Model is thinking..."):
            ranked, probs = predict(prompt, options)

        st.divider()
        st.subheader(" Top-3 Predictions")

        medals = ["🥇", "🥈", "🥉"]
        colors = ["#FFD700", "#C0C0C0", "#CD7F32"]

        for rank, idx in enumerate(ranked[:3]):
            letter     = OPTIONS[idx]
            confidence = probs[idx] * 100
            opt_text   = options[idx]

            st.markdown(
                f"{medals[rank]} **Rank {rank+1}: Option {letter}** — "
                f"`{confidence:.1f}%` confidence"
            )
            st.caption(f"*{opt_text}*")
            st.progress(float(probs[idx]))
            st.write("")

        st.divider()

        # full ranking
        full_rank = " > ".join([OPTIONS[i] for i in ranked])
        st.markdown(f"**Full Ranking:** `{full_rank}`")

        # submission format
        top3_str = " ".join([OPTIONS[i] for i in ranked[:3]])
        st.markdown(f"**Kaggle Submission Format:** `{top3_str}`")

# ── EXAMPLE ───────────────────────────────────────────────────
st.divider()
st.subheader(" Try an Example")

if st.button("Load Example Question"):
    st.session_state["example_loaded"] = True

if st.session_state.get("example_loaded"):
    st.info("""
    **Question:** What is the primary function of mitochondria in a cell?

    - **A:** They produce energy in the form of ATP
    - **B:** They control cell division and replication
    - **C:** They synthesize proteins from amino acids
    - **D:** They store and protect genetic information
    - **E:** They regulate the cell membrane permeability
    """)
    st.caption("Copy these into the fields above and click Predict!")

# ── MODEL INFO ────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ Model & Training Details"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Model Architecture**
        - Base: microsoft/deberta-v3-base
        - Task: AutoModelForMultipleChoice
        - Parameters: ~86M
        - Tokenizer: SentencePiece
        """)
    with col2:
        st.markdown("""
        **Training Configuration**
        - Optimizer: AdamW (lr=2e-5)
        - Epochs: 2
        - Batch size: 4 (grad accum)
        - Max length: 384 tokens
        """)

    st.markdown("""
    **Results**

    | Model | Kaggle MAP@3 |
    |---|---|
    | DeBERTa-v3-base (fine-tuned) | **0.7596** |
    | TF-IDF + Logistic Regression | 0.7438 |
    | RAG + Groq LLaMA | 0.7300 |
    | Simple RNN (from scratch) | 0.4244 |
    """)
