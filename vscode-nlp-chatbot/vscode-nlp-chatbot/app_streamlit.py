import json, random
from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.joblib"
INTENTS_PATH = ROOT / "intents.json"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_intents():
    intents = json.loads(INTENTS_PATH.read_text(encoding="utf-8"))
    tag_to_responses = {it["tag"]: it["responses"] for it in intents["intents"]}
    fallback = intents.get("fallback_responses", ["Sorry, I didn't understand that."])
    return tag_to_responses, fallback

def predict_tag(model, text):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        tag = model.classes_[proba.argmax()]
        conf = float(proba.max())
        return tag, conf
    else:
        tag = model.predict([text])[0]
        return tag, 1.0

def generate_reply(model, tag_to_responses, fallback, user_text, threshold=0.45):
    tag, conf = predict_tag(model, user_text)
    if conf < threshold or tag not in tag_to_responses:
        return random.choice(fallback), conf, "fallback"
    return random.choice(tag_to_responses[tag]), conf, tag

st.title("NLP Intent Chatbot")
st.caption("Simple TF-IDF + Logistic Regression bot")

if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, text)

tag_to_responses, fallback = load_intents()
model = load_model()

user_text = st.text_input("Type a message", "")
if st.button("Send") and user_text.strip():
    reply, conf, tag = generate_reply(model, tag_to_responses, fallback, user_text.strip())
    st.session_state.history.append(("user", user_text.strip()))
    st.session_state.history.append(("bot", f"[{tag} | {conf:.2f}] {reply}"))

for role, text in st.session_state.history:
    if role == "user":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")
