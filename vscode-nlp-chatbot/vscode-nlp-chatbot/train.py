import json
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import random

DATA_PATH = Path(__file__).parent / "intents.json"
MODEL_PATH = Path(__file__).parent / "model.joblib"

def load_data():
    data = json.loads(Path(DATA_PATH).read_text(encoding="utf-8"))
    X, y = [], []
    for intent in data["intents"]:
        for p in intent["patterns"]:
            X.append(p)
            y.append(intent["tag"])
    return X, y, data

def train():
    X, y, data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=None))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds))

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH.resolve()}")

if __name__ == "__main__":
    train()
