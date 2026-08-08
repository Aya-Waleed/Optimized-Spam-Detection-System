"""
Spam Detection - Streamlit Web App

Usage
-----
    streamlit run deploy/app.py

Must be run from the project root (the folder containing `models/`).
"""

import io
from datetime import datetime

import pandas as pd
import streamlit as st

from spam_system import (
    load_pipeline,
    load_explainer,
    classify_one,
    classify_many,
    explain_prediction,
)

st.set_page_config(page_title="Spam Detection System", page_icon="📩", layout="centered")

st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    .stApp {
        background: linear-gradient(180deg, #f4f7fb 0%, #eef2fb 100%);
    }
    html, body, [class*="css"] { font-family: 'Segoe UI', 'Inter', sans-serif; }

    /* Force readable dark text everywhere in the main content area,
       regardless of the user's light/dark system theme. */
    [data-testid="stAppViewContainer"] .main,
    [data-testid="stAppViewContainer"] .main p,
    [data-testid="stAppViewContainer"] .main span,
    [data-testid="stAppViewContainer"] .main label,
    [data-testid="stAppViewContainer"] .main div,
    [data-testid="stMarkdownContainer"] {
        color: #1f2333 !important;
    }

    /* ---------- Header ---------- */
    .app-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
    }
    .app-header h1 {
        color: #ffffff !important;
        margin: 0 0 4px 0;
        font-size: 1.7rem;
    }
    .app-header p {
        color: #e0e7ff !important;
        margin: 0;
        font-size: 0.95rem;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #1e1b4b;
    }
    section[data-testid="stSidebar"] * { color: #e0e7ff !important; }
    section[data-testid="stSidebar"] .stButton>button {
        background: #4f46e5;
        color: #fff !important;
        border: none;
        border-radius: 8px;
    }

    /* ---------- Buttons ---------- */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        border: none;
        color: #1f2333 !important;
        background-color: #ffffff;
        transition: transform 0.05s ease-in-out;
    }
    .stButton>button:hover { transform: translateY(-1px); }
    button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 10px 10px 0 0;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"] p { color: #1f2333 !important; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    }
    .stTabs [aria-selected="true"] p { color: #ffffff !important; }

    /* ---------- Cards / containers ---------- */
    div[data-testid="stTextArea"] textarea {
        border-radius: 10px;
        border: 1.5px solid #d8ddf0;
        background-color: #ffffff !important;
        color: #1f2333 !important;
    }
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #e2e6f5;
        background-color: #ffffff;
    }
    div[data-testid="stExpander"] summary p { color: #1f2333 !important; }

    /* ---------- File uploader / selectbox ---------- */
    div[data-testid="stFileUploader"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 4px;
    }
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #1f2333 !important;
    }

    /* ---------- Word chips ---------- */
    .word-chip {
        display: inline-block;
        padding: 5px 12px;
        margin: 3px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .chip-spam { background-color: #fde2e2; color: #b42318 !important; }
    .chip-ham  { background-color: #dcfce7; color: #15803d !important; }

    /* ---------- History ---------- */
    .history-row {
        padding: 10px 14px;
        margin-bottom: 6px;
        border-radius: 10px;
        background-color: #ffffff;
        border: 1px solid #eef0fa;
        font-size: 0.9rem;
        color: #1f2333 !important;
    }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #eef0fa;
        border-radius: 12px;
        padding: 10px 6px;
        box-shadow: 0 2px 6px rgba(30, 27, 75, 0.05);
    }
    div[data-testid="stMetric"] * { color: #1f2333 !important; }

    /* ---------- Dataframe / table ---------- */
    div[data-testid="stDataFrame"] { background-color: #ffffff; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    return load_pipeline()


@st.cache_resource
def get_explainer():
    return load_explainer()


pipeline, config = get_pipeline()

st.markdown(
    """
    <div class="app-header">
        <h1>📩 Spam Detection System</h1>
        <p>SMS spam classifier — TF-IDF + calibrated Linear SVM</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if pipeline is None:
    st.error(
        "Deployment artifacts not found. Run "
        "`notebooks/Model_Optimization_Deployment.ipynb` first to generate "
        "`models/spam_detection_pipeline.pkl` and `models/deployment_config.json`."
    )
    st.stop()

tfidf, svm_model = get_explainer()

if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar: decision threshold control -----------------------------------
default_threshold = config["threshold"]

st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(default_threshold),
    step=0.01,
    help=(
        "A message is flagged as spam when its spam probability is at or "
        "above this value. Lower = catches more spam but more false "
        "alarms. Higher = fewer false alarms but more spam gets through. "
        f"Recommended default: {default_threshold}"
    ),
)
if st.sidebar.button("Reset to recommended default"):
    threshold = default_threshold

st.sidebar.caption(f"Model: {config['model']}")

if st.session_state.history:
    if st.sidebar.button("🗑️ Clear history"):
        st.session_state.history = []
        st.rerun()

# --- Tabs: single message vs. batch CSV -------------------------------------
tab_single, tab_batch = st.tabs(["✉️ Classify a message", "📄 Batch (CSV) classification"])

with tab_single:
    message = st.text_area(
        "Enter a message to classify",
        placeholder="e.g. Congratulations! You won a free prize! Click here to claim now",
        height=120,
    )

    classify_clicked = st.button("Classify", type="primary")

    if classify_clicked:
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            result = classify_one(pipeline, config, message, threshold=threshold)
            proba = result["spam_probability"]

            if result["label"] == "spam":
                st.error(f"🚫 **Spam** — {proba:.1%} spam probability")
            else:
                st.success(f"✅ **Ham** — {proba:.1%} spam probability")

            st.caption("Spam probability")
            st.progress(min(max(proba, 0.0), 1.0))

            if tfidf is not None and svm_model is not None:
                contributions = explain_prediction(tfidf, svm_model, message)
                if contributions:
                    st.markdown("**Words that influenced this decision:**")
                    chips_html = ""
                    for word, contribution in contributions:
                        css_class = "chip-spam" if contribution > 0 else "chip-ham"
                        arrow = "↑ spam" if contribution > 0 else "↓ ham"
                        chips_html += (
                            f'<span class="word-chip {css_class}">{word} ({arrow})</span>'
                        )
                    st.markdown(chips_html, unsafe_allow_html=True)
                else:
                    st.caption("No individual words in the vocabulary strongly influenced this message.")

            st.session_state.history.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": message.strip(),
                "label": result["label"],
                "spam_probability": proba,
            })
            st.session_state.history = st.session_state.history[:10]

    with st.expander("Try example messages"):
        examples = [
            "Congratulations! You won a free prize! Click here to claim now",
            "Hey, are we still meeting tomorrow?",
            "URGENT! Your account has been suspended, verify now to avoid loss",
            "Can you send me the report before end of day?",
        ]
        for example in examples:
            if st.button(example, key=f"ex_{example}"):
                result = classify_one(pipeline, config, example, threshold=threshold)
                proba = result["spam_probability"]
                label_text = "🚫 Spam" if result["label"] == "spam" else "✅ Ham"
                st.write(f"{label_text} — {proba:.1%} spam probability")

    if st.session_state.history:
        st.divider()
        st.markdown("**Recent classifications**")
        for entry in st.session_state.history:
            tag = "🚫" if entry["label"] == "spam" else "✅"
            short_message = entry["message"] if len(entry["message"]) <= 70 else entry["message"][:70] + "..."
            st.markdown(
                f'<div class="history-row">{tag} <b>{entry["spam_probability"]:.1%}</b> '
                f'&nbsp;·&nbsp; {entry["time"]} &nbsp;·&nbsp; {short_message}</div>',
                unsafe_allow_html=True,
            )

with tab_batch:
    st.markdown(
        "Upload a CSV file containing a column of messages. Every message "
        "is classified using the pipeline and the current threshold."
    )
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read this file as CSV: {exc}")
            batch_df = None

        if batch_df is not None:
            if batch_df.empty:
                st.warning("The uploaded file has no rows.")
            else:
                text_columns = [c for c in batch_df.columns if batch_df[c].dtype == object]
                default_col = text_columns[0] if text_columns else batch_df.columns[0]

                message_column = st.selectbox(
                    "Which column contains the messages?",
                    options=list(batch_df.columns),
                    index=list(batch_df.columns).index(default_col),
                )

                if st.button("Classify all rows", type="primary"):
                    with st.spinner(f"Classifying {len(batch_df)} messages..."):
                        results = classify_many(
                            pipeline, config, batch_df[message_column].tolist(), threshold=threshold
                        )

                    result_df = batch_df.copy()
                    result_df["predicted_label"] = [r["label"] for r in results]
                    result_df["spam_probability"] = [r["spam_probability"] for r in results]

                    spam_count = sum(1 for r in results if r["label"] == "spam")
                    ham_count = sum(1 for r in results if r["label"] == "ham")
                    skipped = sum(1 for r in results if r["label"] is None)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Spam", spam_count)
                    col2.metric("Ham", ham_count)
                    col3.metric("Skipped (empty)", skipped)

                    st.dataframe(result_df, use_container_width=True)

                    csv_buffer = io.StringIO()
                    result_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "⬇️ Download results as CSV",
                        data=csv_buffer.getvalue(),
                        file_name="spam_classification_results.csv",
                        mime="text/csv",
                    )

st.divider()
st.caption(f"Decision threshold in use: {threshold:.4f}  •  Recommended default: {default_threshold}")
