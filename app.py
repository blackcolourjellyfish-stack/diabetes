# =============================================================================
# app.py — Streamlit front-end for best_diabetes_model.pkl
# =============================================================================

# -----------------------------------------------------------------------------
# SHIM — must run BEFORE joblib.load() so the pickled pipeline can find the
# custom `train.DiabetesFeatureEngineer` class at unpickle time.
# -----------------------------------------------------------------------------
import sys, types
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_mod = types.ModuleType("train")

class _DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    ZERO_AS_NAN = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    def __init__(self, zero_as_nan=None, add_interactions=True):
        self.zero_as_nan = zero_as_nan if zero_as_nan is not None else self.ZERO_AS_NAN
        self.add_interactions = add_interactions

    def fit(self, X, y=None):
        self.feature_names_in_ = list(pd.DataFrame(X).columns)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        cols = [c for c in self.zero_as_nan if c in X.columns]
        if cols:
            X[cols] = X[cols].replace(0, np.nan)
        if self.add_interactions:
            X['Glucose_BMI']     = X['Glucose'] * X['BMI']
            X['Age_BMI']         = X['Age'] * X['BMI']
            X['Glucose_Age']     = X['Glucose'] * X['Age']
            X['Insulin_Glucose'] = X['Insulin'] / (X['Glucose'] + 1.0)
            X['BMI_Age_ratio']   = X['BMI'] / (X['Age'] + 1.0)
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None \
               else list(self.feature_names_in_)
        if self.add_interactions:
            base += ['Glucose_BMI', 'Age_BMI', 'Glucose_Age',
                     'Insulin_Glucose', 'BMI_Age_ratio']
        return np.asarray(base, dtype=object)

_mod.DiabetesFeatureEngineer = _DiabetesFeatureEngineer
sys.modules["train"] = _mod
# -----------------------------------------------------------------------------
# END SHIM — now safe to import streamlit + joblib and load the model
# -----------------------------------------------------------------------------

import joblib
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

MODEL_PATH = Path("best_diabetes_model.pkl")

# -----------------------------------------------------------------------------
# Feature metadata (raw Pima input columns — the pipeline handles the rest)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(
        f"❌ Could not find **{MODEL_PATH.name}** in `{Path.cwd()}`. "
        f"Put the .pkl file next to `app.py` in the repo root."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error(f"❌ Failed to load the model: {e}")
    st.exception(e)
    st.stop()

st.write("✅ Model loaded successfully")  # remove once confirmed working


# -----------------------------------------------------------------------------
# Figure out the expected feature order from the estimator
# -----------------------------------------------------------------------------
def infer_feature_order(m) -> list[str]:
    """Try to recover the training-time raw feature order from the estimator."""
    candidates = [m]
    if hasattr(m, "named_steps"):                      # sklearn Pipeline
        candidates += list(m.named_steps.values())
    for obj in candidates:
        names = getattr(obj, "feature_names_in_", None)
        if names is not None:
            names = [str(n) for n in names]
            # Keep only the raw columns the user actually supplies
            raw = [n for n in names if n in FEATURE_SPECS]
            if raw:
                return raw
    return DEFAULT_ORDER


FEATURES = infer_feature_order(model)


# -----------------------------------------------------------------------------
# Prediction helper
# -----------------------------------------------------------------------------
def predict(df: pd.DataFrame):
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


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("🩺 Diabetes Risk Predictor")
st.caption(f"Model: `{MODEL_PATH.name}`  •  Expected inputs: {len(FEATURES)}")

tab_single, tab_batch = st.tabs(["Single patient", "Batch (CSV)"])

# ------------------------------ Single patient -------------------------------
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
                if (isinstance(step, int) and isinstance(lo, int)
                        and isinstance(hi, int)):
                    values[feat] = st.number_input(
                        label, min_value=lo, max_value=hi,
                        value=int(default), step=step, help=help_txt,
                    )
                else:
                    values[feat] = st.number_input(
                        label, min_value=float(lo), max_value=float(hi),
                        value=float(default), step=float(step), help=help_txt,
                    )

        submitted = st.form_submit_button(
            "🔍 Predict", use_container_width=True, type="primary"
        )

    if submitted:
        row = pd.DataFrame([values])[FEATURES]
        try:
            preds, probs = predict(row)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
        else:
            label = preds[0]
            is_pos = str(label).lower() in {"1", "1.0", "true", "yes",
                                            "positive", "diabetes"}

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

# ------------------------------ Batch (CSV) ----------------------------------
with tab_batch:
    st.subheader("Upload a CSV")
    st.write("The file must contain these columns: "
             + ", ".join(f"`{f}`" for f in FEATURES))

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

# ------------------------------ Sidebar --------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        This app loads a pre-trained scikit-learn pipeline and scores
        diabetes risk from patient measurements.

        **Pipeline:** zero→NaN → median impute → standard-scale →
        feature engineering → classifier.

        **Disclaimer:** Educational/demo tool only. Not a medical device
        and not a substitute for professional diagnosis.
        """
    )
    st.divider()
    st.caption(f"Features expected by model: {len(FEATURES)}")
    with st.expander("Show feature order"):
        st.write(FEATURES)
    if st.button("🔄 Clear cache & reload model"):
        st.cache_resource.clear()
        st.rerun()
