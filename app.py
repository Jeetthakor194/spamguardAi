
import streamlit as st
import pandas as pd
from spam_classifier import train_model, predict_message, clean_text

st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- DESIGN ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #080d18;
    color: #e5eaf5;
}
[data-testid="stSidebar"] {
    background: #0c1424;
    border-right: 1px solid #1e293b;
}
[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
}
.hero {
    background: linear-gradient(120deg,#102442,#111827 65%,#172554);
    border: 1px solid #263c5d;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 24px;
}
.hero h1 {
    font-size: 34px;
    margin: 0;
    color: #f8fafc;
}
.hero p {
    color: #9fb1cc;
    margin-bottom: 0;
}
.metric-card {
    background: #111c2e;
    border: 1px solid #26364e;
    padding: 20px;
    border-radius: 15px;
    min-height: 112px;
}
.metric-label {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}
.metric-value {
    color: #f8fafc;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}
.panel {
    background: #111c2e;
    border: 1px solid #26364e;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 16px;
}
.result-spam {
    background: #351823;
    border: 1px solid #a33c50;
    padding: 18px;
    border-radius: 12px;
}
.result-safe {
    background: #102d29;
    border: 1px solid #23836a;
    padding: 18px;
    border-radius: 12px;
}
.small-muted {
    color: #94a3b8;
    font-size: 13px;
}
div.stButton > button[kind="primary"] {
    background: #2563eb;
    border: 0;
    border-radius: 9px;
}
h1,h2,h3 {
    color: #f1f5f9;
}
</style>
""", unsafe_allow_html=True)


# ---------- MODEL ----------
@st.cache_resource
def load_model():
    return train_model()


def classify(message):
    result = predict_message(model, message)

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            [clean_text(message)]
        )[0]

        classes = list(model.classes_)
        predicted_class = "spam" if result == "Spam" else "ham"

        confidence = float(
            probabilities[classes.index(predicted_class)]
        )

    return result, confidence


try:
    with st.spinner("Loading dataset and training SpamGuard AI..."):
        model, metrics, df = load_model()

except Exception as exc:
    st.error("The model could not be loaded.")
    st.code(str(exc))
    st.info("Check the dataset URL and your deployment logs.")
    st.stop()


# ---------- SESSION HISTORY ----------
if "history" not in st.session_state:
    st.session_state.history = []


def analyze(message, source="SMS"):
    result, confidence = classify(message)

    st.session_state.history.insert(0, {
        "Source": source,
        "Message": message[:75],
        "Prediction": result,
        "Confidence": (
            f"{confidence:.1%}" if confidence is not None else "N/A"
        )
    })

    st.session_state.history = st.session_state.history[:20]

    return result, confidence


# ---------- SIDEBAR ----------
st.sidebar.markdown("# 🛡️ SpamGuard AI")
st.sidebar.caption("INTELLIGENT MESSAGE SECURITY")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "SMS Analyzer",
        "Email Inbox",
        "Activity History",
        "Project Information"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("**MODEL STATUS**")
st.sidebar.success("Model loaded")
st.sidebar.caption("TF-IDF + Multinomial Naive Bayes")
st.sidebar.caption("UCI SMS Spam Collection")


# ---------- HELPERS ----------
def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_prediction(result, confidence):
    if result == "Spam":
        st.markdown(
            """
            <div class="result-spam">
                <h2 style="color:#fb7185;margin:0">🚨 SPAM DETECTED</h2>
                <p style="color:#fecdd3">
                This message resembles patterns associated with spam.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            """
            <div class="result-safe">
                <h2 style="color:#34d399;margin:0">✓ NOT SPAM</h2>
                <p style="color:#a7f3d0">
                The model classified this message as normal.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    if confidence is not None:
        st.write(
            f"Model probability for the predicted class: **{confidence:.1%}**"
        )
        st.progress(confidence)

    st.caption(
        "Predictions are not guarantees. Always check suspicious links "
        "and requests for personal information independently."
    )


# ---------- DASHBOARD ----------
if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>🛡️ SpamGuard AI</h1>
        <p>Intelligent message security powered by machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Model performance")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("TEST ACCURACY", f"{metrics['accuracy']:.2%}")

    with c2:
        metric_card("SPAM PRECISION", f"{metrics['precision']:.2%}")

    with c3:
        metric_card("SPAM RECALL", f"{metrics['recall']:.2%}")

    with c4:
        metric_card("SPAM F1 SCORE", f"{metrics['f1']:.2%}")

    st.caption(
        f"Evaluated on {metrics['test_size']:,} held-out messages "
        f"from a dataset of {metrics['dataset_size']:,} messages."
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Dataset distribution")

        distribution = pd.DataFrame({
            "Category": ["Not Spam", "Spam"],
            "Messages": [
                metrics["ham_count"],
                metrics["spam_count"]
            ]
        })

        st.vega_lite_chart(
            distribution,
            {
                "mark": {"type": "arc", "innerRadius": 65},
                "encoding": {
                    "theta": {
                        "field": "Messages",
                        "type": "quantitative"
                    },
                    "color": {
                        "field": "Category",
                        "type": "nominal",
                        "scale": {
                            "domain": ["Not Spam", "Spam"],
                            "range": ["#38bdf8", "#fb7185"]
                        }
                    },
                    "tooltip": ["Category", "Messages"]
                },
                "view": {"stroke": None}
            },
            use_container_width=True
        )

        st.dataframe(
            distribution,
            hide_index=True,
            use_container_width=True
        )

    with right:
        st.subheader("Spam model evaluation")

        performance = pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1 Score"],
            "Score": [
                metrics["precision"],
                metrics["recall"],
                metrics["f1"]
            ]
        }).set_index("Metric")

        st.bar_chart(
            performance,
            x="Metric",
            y="Score"
        )

    st.divider()

    st.subheader("Confusion matrix")
    st.caption("Rows = actual labels | Columns = predicted labels")

    cm = metrics["confusion_matrix"]

    cm_display = pd.DataFrame(
        cm,
        index=["Actual Not Spam", "Actual Spam"],
        columns=["Predicted Not Spam", "Predicted Spam"]
    )

    def cell(value, maximum):
        intensity = int(25 + 190 * value / max(maximum, 1))

        return (
            f"background-color: rgb(15, {intensity}, "
            f"{min(220, intensity + 35)}); color: white; "
            "font-weight: bold; text-align: center"
        )

    st.dataframe(
        cm_display.style.map(
            lambda value: cell(int(value), int(cm.max()))
        ),
        use_container_width=True
    )

    st.divider()

    st.subheader("Try SpamGuard AI")
    st.write("Test the classifier with a message of your choice.")

    quick_message = st.text_input(
        "Enter a message",
        placeholder="Congratulations! You won a free prize..."
    )

    if st.button("Analyze message", type="primary"):
        if quick_message.strip():
            result, confidence = analyze(quick_message)
            show_prediction(result, confidence)
        else:
            st.warning("Enter a message first.")


# ---------- SMS ANALYZER ----------
elif page == "SMS Analyzer":

    st.markdown("## ✉️ SMS Security Analyzer")
    st.caption(
        "Analyze SMS messages using your trained machine-learning model."
    )

    examples = {
        "Normal message": "Hey, are we meeting after school today?",
        "Spam example": "Congratulations! You have won a free prize. Claim now!",
        "Delivery message": "Your parcel has been shipped and is arriving tomorrow."
    }

    selected = st.selectbox("Try an example", list(examples.keys()))

    message = st.text_area(
        "Message content",
        value=examples[selected],
        height=160
    )

    if st.button("🛡️ Scan message", type="primary"):
        if message.strip():
            result, confidence = analyze(message, "SMS")
            show_prediction(result, confidence)
        else:
            st.warning("Please enter a message.")


# ---------- EMAIL INBOX ----------
elif page == "Email Inbox":

    st.markdown("## 📧 Smart Email Security")
    st.caption(
        "Demonstration inbox — sample emails are analyzed as message text."
    )

    emails = [
        {
            "sender": "Rewards Department",
            "subject": "You have won a free gift!",
            "preview": "Claim your exclusive reward now...",
            "body": "Congratulations! You have won a free prize. Claim now!"
        },
        {
            "sender": "School Group",
            "subject": "Tomorrow's homework",
            "preview": "Please complete the science assignment...",
            "body": "Please complete the science assignment before tomorrow's class."
        },
        {
            "sender": "Unknown Sender",
            "subject": "Urgent account notification",
            "preview": "You have been selected for a special offer...",
            "body": "You have been selected for a special offer. Claim your free reward now."
        },
        {
            "sender": "Friend",
            "subject": "Weekend plans",
            "preview": "Are we meeting this weekend?",
            "body": "Hey, are we meeting this weekend? Let me know."
        }
    ]

    for i, email in enumerate(emails):

        with st.container(border=True):
            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"**{email['sender']}**")
                st.write(email["subject"])
                st.caption(email["preview"])

            with col2:
                if st.button("Analyze", key=f"email_{i}"):
                    result, confidence = analyze(
                        email["body"], "Email demo"
                    )
                    show_prediction(result, confidence)


# ---------- HISTORY ----------
elif page == "Activity History":

    st.markdown("## 📊 Recent Analysis History")
    st.caption("Recent predictions from this active app session.")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info("No messages analyzed in this session yet.")


# ---------- PROJECT INFORMATION ----------
elif page == "Project Information":

    st.markdown("## About SpamGuard AI")

    st.markdown("""
    <div class="panel">
        <h3>Project objective</h3>
        <p>To classify SMS messages as Spam or Not Spam using
        machine-learning techniques.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Technology stack")

    st.write("""
    - Python
    - pandas
    - scikit-learn
    - TF-IDF Vectorization
    - Multinomial Naive Bayes
    """)

    st.subheader("How the AI works")

    steps = [
        ("01", "Dataset", "The model uses labelled SMS messages."),
        ("02", "Text cleaning", "Messages are converted to lowercase and punctuation is removed."),
        ("03", "TF-IDF", "Text is converted into numerical features."),
        ("04", "Training", "Multinomial Naive Bayes learns patterns from the training data."),
        ("05", "Prediction", "The model classifies new messages as Spam or Not Spam.")
    ]

    for number, title, description in steps:
        with st.container(border=True):
            st.markdown(f"**{number} · {title}**")
            st.caption(description)

    st.subheader("Dataset and evaluation")

    st.write(
        f"Dataset size: **{metrics['dataset_size']:,}** messages"
    )

    st.write(
        f"Spam messages: **{metrics['spam_count']:,}**"
    )

    st.write(
        f"Not Spam messages: **{metrics['ham_count']:,}**"
    )

    st.write(
        f"Test accuracy: **{metrics['accuracy']:.2%}**"
    )

    st.subheader("Important limitations")

    st.write("""
    - The model learns from its training dataset and may not recognize every new spam pattern.
    - The email inbox is a demonstration, not a connection to a real email account.
    - A Not Spam prediction does not guarantee that a message is safe.
    - Evaluation scores describe performance on the held-out test set, not every real-world message.
    """)


# ---------- FOOTER ----------
st.divider()

st.markdown(
    "<p style='text-align:center;color:#64748b;font-size:12px'>"
    "SPAMGUARD AI · Class 10 Artificial Intelligence Project<br>"
    "Python · pandas · scikit-learn · TF-IDF · Multinomial Naive Bayes"
    "</p>",
    unsafe_allow_html=True
)
