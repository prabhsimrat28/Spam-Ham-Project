import streamlit as st
import requests
import time

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam / Ham Classifier",
    page_icon="📩",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ─────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    /* ── Title ──────────────────────────────── */
    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00c6ff, #0072ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #a0aec0;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* ── Result Cards ───────────────────────── */
    .result-card {
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-top: 1.5rem;
        text-align: center;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .spam-card {
        background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(185,28,28,0.12));
        box-shadow: 0 0 30px rgba(239,68,68,0.15);
    }
    .ham-card {
        background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(22,163,74,0.12));
        box-shadow: 0 0 30px rgba(34,197,94,0.15);
    }
    .result-label {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .spam-label { color: #f87171; }
    .ham-label  { color: #4ade80; }

    .confidence-text {
        font-size: 1.1rem;
        color: #cbd5e1;
    }

    /* ── Probability Bars ───────────────────── */
    .prob-container {
        margin-top: 1.5rem;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
    }
    .prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.6rem;
    }
    .prob-row:last-child { margin-bottom: 0; }
    .prob-label {
        width: 60px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #e2e8f0;
    }
    .prob-bar-bg {
        flex: 1;
        height: 22px;
        background: rgba(255,255,255,0.06);
        border-radius: 11px;
        overflow: hidden;
        margin: 0 12px;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 11px;
        transition: width 0.6s ease;
    }
    .prob-value {
        width: 58px;
        text-align: right;
        font-weight: 700;
        font-size: 0.95rem;
        color: #f1f5f9;
    }

    /* ── Footer ─────────────────────────────── */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.82rem;
        margin-top: 3rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── API Config ───────────────────────────────────────────────────────────
API_URL = "http://51.20.133.25:8000"

# ── Header ───────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📩 Spam / Ham Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Paste any SMS or message below to find out if it\'s spam or legitimate.</p>',
    unsafe_allow_html=True,
)

# ── Input Area ───────────────────────────────────────────────────────────
user_text = st.text_area(
    "Enter your message",
    height=140,
    placeholder="e.g.  Congratulations! You've won a free iPhone. Click here to claim now...",
    max_chars=500,
)

classify_btn = st.button("🔍  Classify", use_container_width=True, type="primary")

# ── Prediction Logic ────────────────────────────────────────────────────
if classify_btn:
    stripped = user_text.strip() if user_text else ""

    if not stripped:
        st.warning("⚠️  Please enter a non-empty message before classifying.")
    else:
        with st.spinner("Analysing message…"):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": stripped},
                    timeout=15,
                )

                if response.status_code == 200:
                    result = response.json()
                    label = result["predicted_result"]       # "Spam" or "Ham"
                    confidence = result["confidence"]
                    probs = result["class_probabilities"]     # {"Ham": …, "Spam": …}

                    # ── Result Card ──
                    is_spam = label == "Spam"
                    card_cls = "spam-card" if is_spam else "ham-card"
                    label_cls = "spam-label" if is_spam else "ham-label"
                    icon = "🚫" if is_spam else "✅"

                    st.markdown(
                        f"""
                        <div class="result-card {card_cls}">
                            <div class="result-label {label_cls}">{icon}  {label}</div>
                            <div class="confidence-text">Confidence: <strong>{confidence * 100:.1f}%</strong></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # ── Probability Bars ──
                    ham_pct = probs.get("Ham", 0) * 100
                    spam_pct = probs.get("Spam", 0) * 100

                    st.markdown(
                        f"""
                        <div class="prob-container">
                            <div class="prob-row">
                                <span class="prob-label">Ham</span>
                                <div class="prob-bar-bg">
                                    <div class="prob-bar-fill" style="width:{ham_pct:.1f}%; background:linear-gradient(90deg,#22c55e,#4ade80);"></div>
                                </div>
                                <span class="prob-value">{ham_pct:.1f}%</span>
                            </div>
                            <div class="prob-row">
                                <span class="prob-label">Spam</span>
                                <div class="prob-bar-bg">
                                    <div class="prob-bar-fill" style="width:{spam_pct:.1f}%; background:linear-gradient(90deg,#ef4444,#f87171);"></div>
                                </div>
                                <span class="prob-value">{spam_pct:.1f}%</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                elif response.status_code == 422:
                    detail = response.json().get("detail", "Validation error")
                    st.error(f"⚠️  Validation Error: {detail}")
                else:
                    st.error(f"❌  Server returned status {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "🔌  Could not connect to the API. "
                    "Make sure the FastAPI server is running:\n\n"
                    "```\nuvicorn app:app --reload\n```"
                )
            except requests.exceptions.Timeout:
                st.error("⏱️  Request timed out. The server might be overloaded.")
            except Exception as e:
                st.error(f"❌  Unexpected error: {e}")

# ── Footer ───────────────────────────────────────────────────────────────
st.markdown(
    '<p class="footer">Built with ❤️ by Prabhsimrat Singh  ·  LSTM + PyTorch + FastAPI + Streamlit</p>',
    unsafe_allow_html=True,
)
