import json
import random
import re
import joblib
from pathlib import Path
from datetime import datetime
import pytz

# Paths
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.joblib"
INTENTS_PATH = ROOT / "intents.json"

user_name = None


# Load model and intents
def load_components():
    clf, vectorizer = joblib.load(MODEL_PATH)
    intents = json.loads(INTENTS_PATH.read_text(encoding="utf-8"))
    tag_to_responses = {it["tag"]: it["responses"] for it in intents["intents"]}
    fallback = intents.get("fallback_responses", ["Sorry, I didn't understand that."])
    return clf, vectorizer, tag_to_responses, fallback


# Handle math
def handle_math(user_input):
    try:
        text = user_input.lower()
        text = (
            text.replace("plus", "+")
            .replace("minus", "-")
            .replace("times", "*")
            .replace("into", "*")
            .replace("divide", "/")
        )
        expr = re.findall(r"[\d\+\-\*\/\.\(\)]+", text)
        if expr:
            result = eval("".join(expr), {"__builtins__": {}})
            return f"The result is {result}"
    except Exception:
        return "Sorry, I couldn't solve that math problem."
    return None


# Handle science
def handle_science(user_input):
    text = user_input.lower()

    if "gravity" in text:
        if any(w in text for w in ["more", "detail", "explain"]):
            return (
                "Gravity is a fundamental force that pulls masses together. "
                "Newton described it as a universal force, while Einstein showed it’s the bending of space-time. "
                "It keeps planets in orbit and objects on Earth."
            )
        return "Gravity is the force that attracts two bodies toward each other. On Earth, it keeps us on the ground."

    if "photosynthesis" in text:
        if any(w in text for w in ["more", "detail", "explain"]):
            return (
                "Photosynthesis converts light energy into chemical energy. "
                "Plants use chlorophyll, water, and CO₂ to produce glucose and oxygen. "
                "It is the foundation of life on Earth."
            )
        return "Photosynthesis is the process by which green plants use sunlight to make food from carbon dioxide and water."

    if "atom" in text:
        if any(w in text for w in ["more", "detail", "explain"]):
            return (
                "An atom has a nucleus of protons and neutrons, with electrons orbiting around. "
                "It is the smallest unit that retains the properties of an element."
            )
        return "An atom is the smallest unit of matter, made up of protons, neutrons, and electrons."

    if "relativity" in text:
        if any(w in text for w in ["more", "detail", "explain"]):
            return (
                "Einstein’s relativity has two parts: special relativity (E=mc², time dilation) "
                "and general relativity (gravity is space-time curvature)."
            )
        return "Einstein's theory of relativity explains how space and time are linked for objects moving at constant speeds."

    if "speed of light" in text:
        if any(w in text for w in ["more", "detail", "explain"]):
            return (
                "Light in vacuum travels at 299,792 km/s (186,282 miles/s). "
                "It is the universe’s speed limit and central to relativity."
            )
        return "The speed of light in vacuum is about 299,792 kilometers per second."

    if "newton" in text:
        if any(w in text for w in ["more", "detail", "explain"]):
            return (
                "Sir Isaac Newton (1642–1727) formulated the three laws of motion and law of gravitation, "
                "founded classical mechanics, and contributed to optics and calculus."
            )
        return "Sir Isaac Newton formulated the laws of motion and universal gravitation."

    return None


# Time-based greeting
def handle_time_greeting():
    tz = pytz.timezone("Asia/Kolkata")
    hour = datetime.now(tz).hour
    if 5 <= hour < 12:
        return f"Good morning, {user_name}! ☀️"
    elif 12 <= hour < 18:
        return f"Good afternoon, {user_name}! 🌞"
    elif 18 <= hour < 22:
        return f"Good evening, {user_name}! 🌆"
    else:
        return f"Good night, {user_name}! 🌙"


# Predict intent
def predict_intent(model, vectorizer, user_input):
    X_vec = vectorizer.transform([user_input.lower()])
    proba = model.predict_proba(X_vec)[0]
    tag = model.classes_[proba.argmax()]
    conf = float(proba.max())
    return tag, conf


# Response generator
def chatbot_response(user_input, model, vectorizer, tag_to_responses, fallback, threshold=0.3):
    math_reply = handle_math(user_input)
    if math_reply:
        return math_reply

    science_reply = handle_science(user_input)
    if science_reply:
        return science_reply

    tag, conf = predict_intent(model, vectorizer, user_input)
    if tag == "time_greeting":
        return handle_time_greeting()
    if conf < threshold or tag not in tag_to_responses:
        return random.choice(fallback)
    return random.choice(tag_to_responses[tag])


# Main loop
if __name__ == "__main__":
    model, vectorizer, tag_to_responses, fallback = load_components()
    print("Chatbot is ready! Type 'exit' to quit.\n")

    # Ask name at start
    while not user_name:
        user_input = input("Bot: Hi, can you please give me your name?\nYou: ")
        if user_input.strip():
            user_name = user_input.split()[-1].capitalize()
            print(f"Bot: Nice to meet you, {user_name}!")
            print("Bot: How can I help you today?")

    # Chat loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Bot: Goodbye! Take care.")
            break
        response = chatbot_response(user_input, model, vectorizer, tag_to_responses, fallback)
        print("Bot:", response)
