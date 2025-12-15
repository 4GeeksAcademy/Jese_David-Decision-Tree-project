from flask import Flask, render_template, request
import pandas as pd
from joblib import load
import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # raíz del repo

MODELS_DIR = os.path.join(ROOT_DIR, "models")
TEMPLATES_DIR = os.path.join(ROOT_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# 1) Buscar modelos (rutas absolutas)
MODEL_CANDIDATES = (
    glob.glob(os.path.join(MODELS_DIR, "*.joblib")) +
    glob.glob(os.path.join(MODELS_DIR, "*.pkl")) +
    glob.glob(os.path.join(MODELS_DIR, "*.sav"))
)

if not MODEL_CANDIDATES:
    raise FileNotFoundError(f"No encontré modelos en: {MODELS_DIR}")

# 2) Priorizar el optimizado (si existe)
opt_models = [p for p in MODEL_CANDIDATES if "opt" in os.path.basename(p).lower()]
MODEL_PATH = opt_models[0] if opt_models else MODEL_CANDIDATES[0]

print("ROOT_DIR:", ROOT_DIR)
print("MODELS_DIR:", MODELS_DIR)
print("Usando modelo:", MODEL_PATH)
print("Archivos en models/:", os.listdir(MODELS_DIR))

model = load(MODEL_PATH)

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

    return render_template("index.html", features=FEATURES,
                           result={"pred": pred, "proba": proba, "inputs": data})

if __name__ == "__main__":
    app.run(debug=True)
