# --- shim: let joblib unpickle pipelines saved from train.py ---
import sys, types, numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_mod = types.ModuleType("train")
class _DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    ZERO_AS_NAN = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
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
        if cols: X[cols] = X[cols].replace(0, np.nan)
        if self.add_interactions:
            X['Glucose_BMI']     = X['Glucose'] * X['BMI']
            X['Age_BMI']         = X['Age'] * X['BMI']
            X['Glucose_Age']     = X['Glucose'] * X['Age']
            X['Insulin_Glucose'] = X['Insulin'] / (X['Glucose'] + 1.0)
            X['BMI_Age_ratio']   = X['BMI'] / (X['Age'] + 1.0)
        return X
    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else list(self.feature_names_in_)
        if self.add_interactions:
            base += ['Glucose_BMI','Age_BMI','Glucose_Age','Insulin_Glucose','BMI_Age_ratio']
        return np.asarray(base, dtype=object)

_mod.DiabetesFeatureEngineer = _DiabetesFeatureEngineer
sys.modules["train"] = _mod
# --- end shim ---
