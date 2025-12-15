from flask import Flask, render_template, request
import pandas as pd
from joblib import load
import os
import glob

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../src
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))   # raíz del repo

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")        # ✅ src/templates
MODELS_DIR = os.path.join(ROOT_DIR, "models")              # ✅ models en la raíz

# =========================
# Flask app
# =========================
app = Flask(__name__, template_folder=TEMPLATES_DIR)

# Debug (útil en Render)
print("BASE_DIR:", BASE_DIR)
print("ROOT_DIR:", ROOT_DIR)
print("TEMPLATES_DIR:", TEMPLATES_DIR)
print("Existe index.html?:", os.path.exists(os.path.join(TEMPLATES_DIR, "index.html")))
print("MODELS_DIR:", MODELS_DIR)
print("Archivos en models/:", os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else "NO EXISTE")

# =========================
# Load model
# =========================
MODEL_CANDIDATES = (
    glob.glob(os.path.join(MODELS_DIR, "*.joblib")) +
    glob.glob(os.path.join(MODELS_DIR, "*.pkl")) +
    glob.glob(os.path.join(MODELS_DIR, "*.sav"))
)

if not MODEL_CANDIDATES:
    raise FileNotFoundError(f"No encontré modelos en: {MODELS_DIR}")

# Prioriza el optimizado si existe
opt_models = [p for p in MODEL_CANDIDATES if "opt" in os.path.basename(p).lower()]
MODEL_PATH = opt_models[0] if opt_models else MODEL_CANDIDATES[0]

print("Usando modelo:", MODEL_PATH)
model = load(MODEL_PATH)

# =========================
# Features
# =========================
DEFAULT_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

def get_features(m):
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    if hasattr(m, "named_steps"):
        for step in reversed(list(m.named_steps.values())):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return DEFAULT_FEATURES

FEATURES = get_features(model)

# =========================
# Routes
# =========================
@app.get("/")
def home():
    return render_template("index.html", features=FEATURES, result=None)

@app.post("/predict")
def predict():
    data = {}
    for f in FEATURES:
        val = request.form.get(f, "")
        try:
            val = float(val)
        except:
            pass
        data[f] = val

    X = pd.DataFrame([data], columns=FEATURES)
    pred = model.predict(X)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X).max())

    return render_template(
        "index.html",
        features=FEATURES,
        result={"pred": pred, "proba": proba, "inputs": data}
    )

if __name__ == "__main__":
    app.run(debug=True)
