"""
Diabetes Risk Assessment — Model Training & Tuning (v2, one-file)

Fixes vs. v1:
  • Zeros in Glucose/BP/SkinThickness/Insulin/BMI → NaN before imputation
  • class_weight='balanced' to fight 65/35 class imbalance
  • Valid (penalty, solver) combinations only — no wasted grid points
  • Feature engineering baked into the saved pipeline
  • Model selection by mean 10-fold CV ROC-AUC (more stable than a
    single 154-row test set)
  • Reports Recall / F1 / confusion matrix — the metrics that matter
    for a screening model

NOTE: The custom transformer is defined in THIS file. Streamlit must
      be able to `import train` before unpickling best_diabetes_model.pkl.
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore', category=UserWarning)
RANDOM_STATE = 42


# ============================================================
# CUSTOM TRANSFORMER
# ============================================================
class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Replace physiologically-impossible zeros with NaN, then add
    interaction features that help on the Pima dataset."""

    ZERO_AS_NAN = ['Glucose', 'BloodPressure', 'SkinThickness',
                   'Insulin', 'BMI']

    def __init__(self, zero_as_nan=None, add_interactions=True):
        self.zero_as_nan = zero_as_nan if zero_as_nan is not None \
                           else self.ZERO_AS_NAN
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


# ============================================================
# 1. DATA LOADING & SPLIT
# ============================================================
df = pd.read_csv('diabetes.csv')

TARGET = 'Outcome'
X = df.drop(TARGET, axis=1)
y = df[TARGET]

print(f"Rows: {len(df)}  |  Positive rate: {y.mean():.3f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ============================================================
# 2. PREPROCESSING + FEATURE-ENGINEERING PIPELINE
#    (baked into the pickle so Streamlit can take raw input)
# ============================================================
preprocessor = Pipeline([
    ('features', DiabetesFeatureEngineer()),
    ('imputer',  SimpleImputer(strategy='median')),
    ('scaler',   StandardScaler()),
])

# ============================================================
# 3. MODELS + VALID HYPERPARAMETER GRIDS
#    Grids use list-of-dicts so only legal (penalty, solver) pairs run.
# ============================================================
C_GRID = [0.01, 0.1, 1, 10, 100]

models = {
    'Logistic Regression': {
        'model': LogisticRegression(
            max_iter=10000, random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        'params': [
            {'classifier__solver': ['liblinear'],
             'classifier__penalty': ['l1', 'l2'],
             'classifier__C': C_GRID},
            {'classifier__solver': ['lbfgs', 'newton-cg'],
             'classifier__penalty': ['l2'],
             'classifier__C': C_GRID},
            {'classifier__solver': ['saga'],
             'classifier__penalty': ['l1', 'l2', 'elasticnet'],
             'classifier__C': C_GRID,
             'classifier__l1_ratio': [0.1, 0.5, 0.9]},
        ],
    },
    'Random Forest': {
        'model': RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
        ),
        'params': {
            'classifier__n_estimators':      [200, 400, 600],
            'classifier__max_depth':         [None, 4, 6, 8, 12],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf':  [1, 2, 4],
            'classifier__max_features':      ['sqrt', 'log2', None],
        },
    },
    'SVM': {
        'model': SVC(
            probability=True, random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        'params': [
            {'classifier__kernel': ['linear'],
             'classifier__C': [0.01, 0.1, 1, 10]},
            {'classifier__kernel': ['rbf', 'sigmoid', 'poly'],
             'classifier__C': [0.01, 0.1, 1, 10],
             'classifier__gamma': ['scale', 'auto', 0.001, 0.01]},
        ],
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'params': {
            'classifier__n_estimators':  [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth':     [2, 3, 4],
            'classifier__subsample':     [0.8, 1.0],
        },
    },
}

# ============================================================
# 4. TRAIN & TUNE
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}

for name, cfg in models.items():
    print(f"\n{'='*66}\nTraining: {name}\n{'='*66}")

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier',   cfg['model']),
    ])

    grid = GridSearchCV(
        pipe, cfg['params'],
        cv=cv, scoring='roc_auc',
        n_jobs=-1, verbose=0, error_score=np.nan,
    )
    grid.fit(X_train, y_train)

    best    = grid.best_estimator_
    y_pred  = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]

    # 10-fold CV on the FULL dataset for a low-variance estimate
    cv_scores = cross_val_score(
        best, X, y,
        cv=StratifiedKFold(10, shuffle=True, random_state=RANDOM_STATE),
        scoring='roc_auc', n_jobs=-1,
    )

    results[name] = {
        'best_estimator': best,
        'best_params':    grid.best_params_,
        'cv_auc_mean':    cv_scores.mean(),
        'cv_auc_std':     cv_scores.std(),
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'roc_auc':   roc_auc_score(y_test, y_proba),
    }

    print(f"Best params     : {grid.best_params_}")
    print(f"10-fold CV AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Test  AUC       : {results[name]['roc_auc']:.4f}")
    print(f"Test  Recall    : {results[name]['recall']:.4f}")
    print(f"Test  F1        : {results[name]['f1']:.4f}")

# ============================================================
# 5. COMPARE & PICK BEST (by mean CV ROC-AUC)
# ============================================================
comparison = pd.DataFrame({
    name: {
        'CV_ROC_AUC_mean': res['cv_auc_mean'],
        'CV_ROC_AUC_std':  res['cv_auc_std'],
        'Test_Accuracy':   res['accuracy'],
        'Test_Precision':  res['precision'],
        'Test_Recall':     res['recall'],
        'Test_F1':         res['f1'],
        'Test_ROC_AUC':    res['roc_auc'],
    }
    for name, res in results.items()
}).T.sort_values('CV_ROC_AUC_mean', ascending=False)

print("\n" + "="*66)
print("MODEL COMPARISON (sorted by mean 10-fold CV ROC-AUC)")
print("="*66)
print(comparison.round(4))

best_name     = comparison.index[0]
best_pipeline = results[best_name]['best_estimator']

print(f"\n✅ Best model: {best_name}")

print("\n" + classification_report(
    y_test, best_pipeline.predict(X_test),
    target_names=['No Diabetes', 'Diabetes'], digits=4,
))
print("Confusion matrix [ [TN FP] [FN TP] ]:")
print(confusion_matrix(y_test, best_pipeline.predict(X_test)))

# ============================================================
# 6. SAVE ARTIFACTS
# ============================================================
joblib.dump(best_pipeline, 'best_diabetes_model.pkl')
comparison.to_csv('model_comparison.csv')

with open('model_metadata.json', 'w') as f:
    json.dump({
        'best_model':    best_name,
        'best_params':   {k: str(v) for k, v in
                          results[best_name]['best_params'].items()},
        'cv_auc_mean':   float(results[best_name]['cv_auc_mean']),
        'cv_auc_std':    float(results[best_name]['cv_auc_std']),
        'test_roc_auc':  float(results[best_name]['roc_auc']),
        'test_recall':   float(results[best_name]['recall']),
        'test_f1':       float(results[best_name]['f1']),
        'feature_order': list(X.columns),
    }, f, indent=2)

print("💾 Saved: best_diabetes_model.pkl")
print("💾 Saved: model_comparison.csv")
print("💾 Saved: model_metadata.json")
