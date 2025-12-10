# train_model.py
# Train a Gradient Boosting model for Fake Account Risk Checker
# Run with: python train_model.py

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------------------------
# Paths (must match app_fake_checker.py)
# -----------------------------------------------------
PROJECT_ROOT = os.getcwd()
DATA_PATH = os.path.join(PROJECT_ROOT, "fake_dataset.xlsx")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.joblib")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Same keywords as UI
ID_KEYWORDS = ("id", "uuid", "user_id", "account_id", "handle")


# -----------------------------------------------------
# Helpers (aligned with UI logic)
# -----------------------------------------------------
def load_base_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find dataset at '{path}'. "
            f"Place 'fake_dataset.xlsx' in the project root."
        )

    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    print(f"[INFO] Loaded dataset: {path} — {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def find_target_column(df: pd.DataFrame) -> str | None:
    """Try to guess the label/target column."""
    candidates = [
        "label", "is_fake", "fake", "target", "isbot", "bot", "class"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for key in candidates:
        if key in lower_map:
            return lower_map[key]

    # fallback: any column with exactly 2 unique values
    for col in df.columns:
        if df[col].dropna().nunique() == 2:
            return col
    return None


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip + replace spaces with underscores."""
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df


def build_feature_frame(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Drop label + obvious ID columns + super high-cardinality text columns."""
    drop_cols = [target_col]
    drop_cols += [
        c for c in df.columns
        if any(k in c.lower() for k in ID_KEYWORDS)
    ]
    X = df.drop(columns=list(set(drop_cols)), errors="ignore")

    # Drop very high-cardinality object columns
    obj_cols = X.select_dtypes(include=["object"]).columns
    to_drop = [c for c in obj_cols if X[c].nunique() > 50]
    if to_drop:
        print(f"[INFO] Dropping high-cardinality text columns: {to_drop}")
        X = X.drop(columns=to_drop, errors="ignore")

    return X


# -----------------------------------------------------
# Training pipeline
# -----------------------------------------------------
def train_model():
    # 1) Load data
    df_raw = load_base_data(DATA_PATH)

    # 2) Normalize columns (consistent naming)
    df = _normalize_cols(df_raw)

    # 3) Detect target/label column
    target_col = find_target_column(df)
    if target_col is None:
        raise ValueError(
            "Could not identify target/label column. "
            "Add something like 'is_fake', 'label', or 'class' to your dataset."
        )

    print(f"[INFO] Detected target column: '{target_col}'")

    # 4) Build feature frame
    X = build_feature_frame(df, target_col)
    y = df[target_col]

    # Drop rows where target is NaN (just in case)
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    print(f"[INFO] Feature matrix shape: {X.shape}")

    # 5) Split numeric vs categorical
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print(f"[INFO] Numeric features: {numeric_features}")
    print(f"[INFO] Categorical features: {categorical_features}")

    # 6) Preprocessor (ColumnTransformer named 'pre' → used by app_fake_checker.py)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 7) Gradient Boosting classifier
    gb_clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    # 8) Full pipeline
    pipeline = Pipeline(
        steps=[
            ("pre", preprocessor),   # IMPORTANT: name 'pre' (UI introspects this)
            ("clf", gb_clf),
        ]
    )

    # 9) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() <= 10 else None,
    )

    print("[INFO] Training Gradient Boosting model...")
    pipeline.fit(X_train, y_train)

    # 10) Evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Test Accuracy: {acc:.4f}\n")
    print("[RESULT] Classification report:")
    print(classification_report(y_test, y_pred))

    # 11) Save model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n[SAVED] Trained model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
