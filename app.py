# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

MODEL_PATH = Path("best_diabetes_model - Copy.pkl")

# ---------------------------------------------------------------------------
# Feature metadata (Pima Indians Diabetes style).
# name -> (label, min, max, default, step, help)
# ---------------------------------------------------------------------------
FEATURE_SPECS = {
    "Pregnancies":              ("Pregnancies",                0,    20,   1,    1,    "Number of times pregnant"),
    "Glucose":                  ("Glucose (mg/dL)",            0,   250, 120,    1,    "Plasma glucose concentration (2h OGTT)"),
    "BloodPressure":            ("Blood Pressure (mm Hg)",     0,   150,  70,    1,    "Diastolic blood pressure"),
    "SkinThickness":            ("Skin Thickness (mm)",        0,   100,  20,    1,    "Triceps skin fold thickness"),
    "Insulin":                  ("Insulin (µU/mL)",            0,   900,  79,    1,    "2-Hour serum insulin"),
    "BMI":                      ("BMI (kg/m²)",                0.0,  70.0, 32.0, 0.1,  "Body mass index"),
    "DiabetesPedigreeFunction": ("Diabetes Pedigree Function", 0.0,   3.0,  0.47, 0.01, "Family-history score"),
    "Age":                      ("Age (years)",                21,  100,  33,    1,    "Age in years"),
}

DEFAULT_ORDER = list(FEATURE_SPECS.keys())


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(f"❌ Could not find **{MODEL_PATH.name}** in `{Path.cwd()}`. "
             f"Put the .pkl file next to `app.py`.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error(f"❌ Failed to load the model: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Figure out the expected feature order
# ---------------------------------------------------------------------------
def infer_feature_order(m) -> list[str]:
    """Try to recover the training-time feature order from the estimator."""
    candidates = [m]
    if hasattr(m, "named_steps"):                     # sklearn Pipeline
        candidates += list(m.named_steps.values())
    for obj in candidates:
        names = getattr(obj, "feature_names_in_", None)
        if names is not None:
            return [str(n) for n in names]
        n_feat = getattr(obj, "n_features_in_", None)
        if n_feat is not None and n_feat == len(DEFAULT_ORDER):
            return DEFAULT_ORDER
    return DEFAULT_ORDER


FEATURES = infer_feature_order(model)


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def predict(df: pd.DataFrame):
    """Returns (labels, probabilities_or_None)."""
    try:
        X = df[FEATURES]
    except KeyError as e:
        raise ValueError(f"Missing column(s) for the model: {e}") from e

    preds = model.predict(X)

    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
        except Exception:
            probs = None
    return np.asarray(preds), probs


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🩺 Diabetes Risk Predictor")
st.caption(f"Model: `{MODEL_PATH.name}`  •  Expected inputs: {len(FEATURES)}")

tab_single, tab_batch = st.tabs(["Single patient", "Batch (CSV)"])

# --------------------------- Single prediction -----------------------------
with tab_single:
    with st.form("patient_form"):
        st.subheader("Patient measurements")
        cols = st.columns(3)
        values = {}
        for i, feat in enumerate(FEATURES):
            label, lo, hi, default, step, help_txt = FEATURE_SPECS.get(
                feat, (feat, 0.0, 1000.0, 0.0, 0.1, "")
            )
            with cols[i % 3]:
                if isinstance(step, int) and isinstance(lo, int) and isinstance(hi, int):
                    values[feat] = st.number_input(
                        label, min_value=lo, max_value=hi,
                        value=int(default), step=step, help=help_txt,
                    )
                else:
                    values[feat] = st.number_input(
                        label, min_value=float(lo), max_value=float(hi),
                        value=float(default), step=float(step), help=help_txt,
                    )

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True, type="primary")

    if submitted:
        row = pd.DataFrame([values])[FEATURES]
        try:
            preds, probs = predict(row)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
        else:
            label = preds[0]
            is_pos = str(label).lower() in {"1", "1.0", "true", "yes", "positive", "diabetes"}

            st.divider()
            c1, c2 = st.columns([1, 1])

            with c1:
                if is_pos:
                    st.error(f"### ⚠️ High risk\nPredicted class: **{label}**")
                else:
                    st.success(f"### ✅ Low risk\nPredicted class: **{label}**")

            with c2:
                if probs is not None and probs.shape[1] >= 2:
                    p_pos = float(probs[0][1])
                    st.metric("Probability of diabetes", f"{p_pos:.1%}")
                    st.progress(min(max(p_pos, 0.0), 1.0))
                else:
                    st.info("This model does not expose `predict_proba`.")

            with st.expander("Input summary"):
                st.dataframe(row.T.rename(columns={0: "value"}))

            st.caption("⚠️ Educational/demo tool only — not a medical diagnosis.")

# --------------------------- Batch prediction ------------------------------
with tab_batch:
    st.subheader("Upload a CSV")
    st.write("The file must contain these columns: " + ", ".join(f"`{f}`" for f in FEATURES))

    up = st.file_uploader("CSV file", type=["csv"])

    if up is not None:
        try:
            data = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
        else:
            st.write("Preview:")
            st.dataframe(data.head(20))

            missing = [f for f in FEATURES if f not in data.columns]
            if missing:
                st.error(f"Missing required column(s): {', '.join(missing)}")
            else:
                try:
                    preds, probs = predict(data)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                else:
                    out = data.copy()
                    out["prediction"] = preds
                    if probs is not None and probs.shape[1] >= 2:
                        out["probability"] = probs[:, 1]

                    st.success(f"Scored {len(out)} row(s).")
                    st.dataframe(out.head(50))

                    st.download_button(
                        "⬇️ Download predictions as CSV",
                        data=out.to_csv(index=False).encode("utf-8"),
                        file_name="diabetes_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        This app loads a pre-trained scikit-learn model and scores
        diabetes risk from patient measurements.

        **Deploy:** push `app.py`, `requirements.txt` and
        `best_diabetes_model.pkl` to a repo, then point
        Streamlit Community Cloud at `app.py`.
        """
    )
    if st.button("Clear cache & reload model"):
        st.cache_resource.clear()
        st.rerun()
