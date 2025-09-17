# VS Code NLP Chatbot (Python + scikit-learn)

A minimal intent-based chatbot using TF‑IDF + Logistic Regression. Train once, then chat from the terminal or use a simple Streamlit UI.

## 1) Prerequisites
- Install **Python 3.9+**
- Install **VS Code** and the **Python** extension
- (Optional) Install **Git**

## 2) Open in VS Code
- Extract this folder.
- In VS Code: **File → Open Folder…** and select the project directory.

## 3) Create & activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4) Install dependencies
```bash
pip install -r requirements.txt
```

## 5) Train the model
```bash
python train.py
```
This creates `model.joblib` in the project folder.

## 6) Chat in the terminal
```bash
python chat.py
```
Type messages; enter `exit` to quit.

## 7) Optional: Run the Streamlit UI
```bash
streamlit run app_streamlit.py
```
A local web app will open in your browser.

## 8) Modify intents
- Edit `intents.json` to add new intents.
- Each intent has a `tag`, training `patterns`, and possible `responses`.
- After changes, re-run: `python train.py`

## 9) Tips
- Increase `ngram_range` or add more patterns for better accuracy.
- Adjust the confidence `threshold` in `chat.py` if you see too many fallbacks.
- Keep patterns short and natural.

---
Made for learning and quick demos.
