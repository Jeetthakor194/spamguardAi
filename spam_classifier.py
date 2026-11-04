import re
import string
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "SMSSpamCollection"
DATA_URL = DATA_URL = (
    "https://archive.ics.uci.edu/static/public/228/"
    "sms+spam+collection.zip"
)

def clean_text(text):
    """Lowercase text and remove punctuation and extra whitespace."""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_dataset():
    """Load the SMS Spam Collection; download it from UCI if needed."""
    DATA_DIR.mkdir(exist_ok=True)

    if not DATA_FILE.exists():
        zip_path = DATA_DIR / "sms_spam_collection.zip"
        print("Downloading the SMS Spam Collection dataset from UCI...")
        urllib.request.urlretrieve(DATA_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as archive:
            member = next(
                (name for name in archive.namelist()
                 if name.endswith("SMSSpamCollection")),
                None
            )
            if member is None:
                raise FileNotFoundError(
                    "Could not find SMSSpamCollection inside the downloaded ZIP."
                )
            DATA_FILE.write_bytes(archive.read(member))

    df = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].str.lower().str.strip()
    df["message"] = df["message"].astype(str)
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["clean_message"] = df["message"].apply(clean_text)
    return df


def train_model():
    df = get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", MultinomialNB())
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label="spam", zero_division=0),
        "recall": recall_score(y_test, predictions, pos_label="spam", zero_division=0),
        "f1": f1_score(y_test, predictions, pos_label="spam", zero_division=0),
        "report": classification_report(
            y_test, predictions, labels=["ham", "spam"],
            target_names=["Not Spam", "Spam"], zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["ham", "spam"]),
        "test_size": len(y_test),
        "dataset_size": len(df),
        "spam_count": int((df["label"] == "spam").sum()),
        "ham_count": int((df["label"] == "ham").sum())
    }
    return model, metrics, df


def predict_message(model, message):
    cleaned = clean_text(message)
    if not cleaned:
        raise ValueError("Please enter a message first.")
    prediction = model.predict([cleaned])[0]
    return "Spam" if prediction == "spam" else "Not Spam"


def run_console():
    print("\n===================================")
    print("       AI SPAM MESSAGE CHECKER")
    print("===================================")
    model, metrics, _ = train_model()
    print(f"Dataset messages: {metrics['dataset_size']}")
    print(f"Test accuracy: {metrics['accuracy']:.2%}")
    print("\nType a message to classify it. Type 'quit' to exit.\n")

    while True:
        message = input("Enter message: ").strip()
        if message.lower() in {"quit", "exit"}:
            print("Thank you for using the Spam Message Classifier!")
            break
        try:
            result = predict_message(model, message)
            print(f"Prediction: {result}\n")
        except ValueError as error:
            print(f"{error}\n")


if __name__ == "__main__":
    run_console()
