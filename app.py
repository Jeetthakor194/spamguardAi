import streamlit as st
import pandas as pd

from spam_classifier import train_model, predict_message

st.set_page_config(
    page_title="AI Spam Message Classifier",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ AI Spam Message Classifier")
st.caption("A Class 10 Artificial Intelligence project using Python and Machine Learning.")
st.write(
    "Enter an SMS or short message below. The trained model will predict "
    "whether it looks like **Spam** or **Not Spam**."
)

@st.cache_resource
def load_trained_model():
    return train_model()

try:
    with st.spinner("Loading the dataset and training the model..."):
        model, metrics, df = load_trained_model()
except Exception as exc:
    st.error("The dataset could not be loaded. Check your internet connection and try again.")
    st.caption(f"Technical detail: {exc}")
    st.stop()

with st.expander("📊 Model evaluation"):
    col1, col2 = st.columns(2)
    col1.metric("Test accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Test messages", metrics["test_size"])
    col3, col4 = st.columns(2)
    col3.metric("Spam precision", f"{metrics['precision']:.2%}")
    col4.metric("Spam recall", f"{metrics['recall']:.2%}")
    st.write("**Classification report**")
    st.code(metrics["report"])
    cm = metrics["confusion_matrix"]
    st.write("**Confusion matrix** (actual labels in rows, predictions in columns)")
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual Not Spam", "Actual Spam"],
            columns=["Predicted Not Spam", "Predicted Spam"]
        ),
        use_container_width=True
    )
    st.caption(
        f"Dataset: {metrics['dataset_size']} messages "
        f"({metrics['ham_count']} not spam and {metrics['spam_count']} spam). "
        "The test set is held out from training."
    )

st.divider()
st.subheader("✉️ Check a message")
with st.form("message_form"):
    message = st.text_area(
        "Message text",
        placeholder="Example: Congratulations! You have won a free prize. Claim now!",
        height=130
    )
    submitted = st.form_submit_button("Check message", type="primary")

if submitted:
    if not message.strip():
        st.warning("Please type a message first.")
    else:
        result = predict_message(model, message)
        if result == "Spam":
            st.error("🚨 Prediction: SPAM")
            st.write("The model found patterns associated with spam in its training data.")
        else:
            st.success("✅ Prediction: NOT SPAM")
            st.write("The model classified this message as a normal message.")
        st.caption(
            "This is a machine-learning prediction, not a guarantee. "
            "Do not click links or share personal information just because a message is marked Not Spam."
        )

st.divider()
st.caption("Built with Python • pandas • scikit-learn • TF-IDF • Multinomial Naive Bayes")
