# app_fake_checker.py
# Streamlit UI for checking whether a social media account looks fake or real.
# Run with: streamlit run app_fake_checker.py

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer

# -----------------------------------------------------
# Basic page config
# -----------------------------------------------------
st.set_page_config(
    page_title="Fake Account Risk Checker",
    page_icon="🕵️‍♂️",
    layout="wide",
)

# -----------------------------------------------------
# Paths (keep same so it works with your pipeline)
# -----------------------------------------------------
PROJECT_ROOT = os.getcwd()
DATA_PATH = os.path.join(PROJECT_ROOT, "fake_dataset.xlsx")
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "best_model.joblib")


# -----------------------------------------------------
# Data / model loading helpers
# -----------------------------------------------------
@st.cache_data
def load_base_data(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


@st.cache_resource
def load_pipeline(path: str):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        st.error(f"Could not load model from '{path}': {exc}")
        return None


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


# -----------------------------------------------------
# Feature inference & alignment
# -----------------------------------------------------
ID_KEYWORDS = ("id", "uuid", "user_id", "account_id", "handle")


def build_feature_frame(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Drop label + obvious ID columns + super high-card text."""
    drop_cols = [target_col]
    drop_cols += [
        c for c in df.columns
        if any(k in c.lower() for k in ID_KEYWORDS)
    ]
    X = df.drop(columns=list(set(drop_cols)), errors="ignore")

    # Drop very high-cardinality object columns (keep them only if model explicitly wants them)
    obj_cols = X.select_dtypes(include=["object"]).columns
    to_drop = [c for c in obj_cols if X[c].nunique() > 50]
    if to_drop:
        X = X.drop(columns=to_drop, errors="ignore")
    return X


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for safe matching."""
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df


def _extract_expected_from_ct(pre: ColumnTransformer, df_example: pd.DataFrame) -> list[str]:
    cols_out: list[str] = []
    for name, transformer, cols in pre.transformers_:
        # cols can be list, tuple, array, slice or string
        if isinstance(cols, (list, tuple, np.ndarray)):
            for c in cols:
                if isinstance(c, str):
                    cols_out.append(str(c).strip())
        else:
            try:
                if isinstance(cols, slice):
                    cols_out.extend(list(df_example.columns[cols]))
                elif isinstance(cols, str):
                    cols_out.append(cols)
            except Exception:
                # ignore weird cases
                pass
    # de-duplicate, keep order
    cleaned = [c for c in dict.fromkeys(cols_out) if isinstance(c, str)]
    return cleaned


def get_expected_feature_names(pipeline, df_example: pd.DataFrame) -> list[str]:
    """
    Introspect the pipeline to find which original columns it expects
    (via a ColumnTransformer inside).
    """
    if pipeline is None:
        return []

    pre = None
    if hasattr(pipeline, "named_steps") and "pre" in getattr(pipeline, "named_steps", {}):
        pre = pipeline.named_steps["pre"]
    elif hasattr(pipeline, "steps"):
        for _, step in pipeline.steps:
            if isinstance(step, ColumnTransformer):
                pre = step
                break

    if pre is None:
        return []

    return _extract_expected_from_ct(pre, df_example)


def align_row_for_model(
    input_row: pd.DataFrame,
    full_df: pd.DataFrame,
    target_col: str,
    pipeline,
) -> pd.DataFrame:
    """
    Make sure the single-row input has all columns expected by the model.
    Missing columns are filled with sensible defaults from the dataset.
    Column order is aligned to the expected order.
    """
    base = _normalize_cols(full_df)
    row = _normalize_cols(input_row)

    numeric_cols = base.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = base.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    cat_cols = [c for c in cat_cols if c != target_col]

    expected = get_expected_feature_names(pipeline, base)
    if not expected:
        # fallback: all columns except target
        expected = [c for c in base.columns if c != target_col]

    for col in expected:
        if col not in row.columns:
            if col in base.columns and np.issubdtype(base[col].dtype, np.number):
                default_val = float(base[col].median()) if not base[col].dropna().empty else 0.0
            elif col in base.columns:
                try:
                    default_val = str(base[col].mode().iloc[0])
                except Exception:
                    default_val = ""
            else:
                # generic guesses
                if col.lower() == "platform":
                    default_val = "unknown"
                elif col.lower().endswith("_count"):
                    default_val = 0.0
                else:
                    default_val = ""
            row[col] = default_val

    ordered = [c for c in expected if c in row.columns]
    remaining = [c for c in row.columns if c not in ordered]
    final_row = row[ordered + remaining].copy()
    return final_row


# -----------------------------------------------------
# UI – header & layout
# -----------------------------------------------------
st.markdown(
    """
    # 🕵️‍♂️ Fake Account Risk Checker

    This tool uses your trained machine learning model to estimate whether a social
    media account looks **fake** or **real** based on its attributes.
    """
)

data_col, form_col = st.columns([1.1, 1.4])

# -----------------------------------------------------
# Left column – dataset & model status
# -----------------------------------------------------
with data_col:
    st.subheader("Project Status")

    df = load_base_data(DATA_PATH)
    model = load_pipeline(MODEL_PATH)

    if df is None:
        st.error(f"Could not find dataset at `{DATA_PATH}`. Make sure `fake_dataset.xlsx` is in the project folder.")
        st.stop()

    st.success(f"Dataset loaded: `{os.path.basename(DATA_PATH)}` — {df.shape[0]} rows × {df.shape[1]} columns")

    if model is None:
        st.warning("Model not found. Train your pipeline and save it as `outputs/best_model.joblib`.")
    else:
        st.info("Model loaded successfully from `outputs/best_model.joblib`.")

    target_col = find_target_column(df)
    if target_col is None:
        st.error("Could not identify the target/label column. Add something like `is_fake`, `label`, or `class`.")
        st.stop()

    st.write(f"**Detected target column:** `{target_col}`")

    with st.expander("Preview a sample of the dataset"):
        st.dataframe(df.head().astype(str))


# -----------------------------------------------------
# Build feature space after we know target
# -----------------------------------------------------
X_features = build_feature_frame(df, target_col)

# Guarantee platform exists and is first (nice for UI)
if "platform" not in X_features.columns:
    if "platform" in df.columns:
        try:
            default_platform = str(df["platform"].mode().iloc[0])
        except Exception:
            default_platform = "unknown"
    else:
        default_platform = "unknown"
    X_features.insert(0, "platform", default_platform)

feature_names = X_features.columns.tolist()

# -----------------------------------------------------
# Right column – input + prediction
# -----------------------------------------------------
with form_col:
    tab_predict, tab_debug = st.tabs(["🔮 Single Account Check", "🛠 Debug / Details"])

    with tab_predict:
        st.subheader("Enter Account Details")

        # We group fields into "Basic" + "Advanced" so UI looks different
        basic_fields = []
        advanced_fields = []
        for name in feature_names:
            if name.lower() in ("platform", "followers_count", "following_count", "friends_count", "statuses_count"):
                basic_fields.append(name)
            else:
                advanced_fields.append(name)

        user_values: dict[str, object] = {}

        st.markdown("### Basic attributes")
        for col in basic_fields:
            series = X_features[col].dropna()
            dtype = X_features[col].dtype

            if np.issubdtype(dtype, np.number):
                try:
                    default_val = float(series.median()) if not series.empty else 0.0
                except Exception:
                    default_val = 0.0
                user_values[col] = st.number_input(
                    label=col,
                    value=default_val,
                    step=1.0,
                    format="%.4g",
                )
            else:
                uniques = series.unique().tolist()
                if 1 < len(uniques) <= 25:
                    user_values[col] = st.selectbox(
                        label=col,
                        options=uniques,
                        index=0,
                    )
                else:
                    default_text = str(series.mode().iloc[0]) if not series.empty else ""
                    user_values[col] = st.text_input(
                        label=col,
                        value=default_text,
                    )

        with st.expander("Advanced attributes (optional)"):
            for col in advanced_fields:
                series = X_features[col].dropna()
                dtype = X_features[col].dtype

                if np.issubdtype(dtype, np.number):
                    try:
                        default_val = float(series.median()) if not series.empty else 0.0
                    except Exception:
                        default_val = 0.0
                    user_values[col] = st.number_input(
                        label=col,
                        value=default_val,
                        step=1.0,
                        format="%.4g",
                        key=f"adv_{col}",
                    )
                else:
                    uniques = series.unique().tolist()
                    if 1 < len(uniques) <= 25:
                        user_values[col] = st.selectbox(
                            label=col,
                            options=uniques,
                            index=0,
                            key=f"adv_{col}",
                        )
                    else:
                        default_text = str(series.mode().iloc[0]) if not series.empty else ""
                        user_values[col] = st.text_input(
                            label=col,
                            value=default_text,
                            key=f"adv_{col}",
                        )

        run_btn = st.button("Run Fake Account Check", type="primary")

        if run_btn:
            if model is None:
                st.error("No trained model is available. Train and save `best_model.joblib` first.")
            else:
                # Build single-row DataFrame from user inputs
                user_df = pd.DataFrame([user_values], columns=feature_names)

                # Align with model
                aligned_row = align_row_for_model(
                    input_row=user_df,
                    full_df=df,
                    target_col=target_col,
                    pipeline=model,
                )

                # Predict
                try:
                    pred = model.predict(aligned_row)[0]
                except Exception as exc:
                    st.error(f"Prediction failed. Check that your pipeline was trained on this dataset. Error: {exc}")
                else:
                    prob_fake = None
                    try:
                        proba = model.predict_proba(aligned_row)[0]
                        if len(proba) >= 2:
                            prob_fake = float(proba[1])
                        else:
                            prob_fake = float(proba[0])
                    except Exception:
                        prob_fake = None

                    label_map = {1: "Fake", 0: "Real"}
                    label_text = label_map.get(int(pred), str(pred))

                    st.markdown("---")
                    st.markdown("### Result")

                    if prob_fake is not None:
                        st.metric(
                            label="Predicted label",
                            value=f"{label_text}",
                            delta=f"Fake probability: {prob_fake:.3f}",
                        )

                        st.progress(int(prob_fake * 100))
                    else:
                        st.write(f"**Predicted label:** {label_text} (probability not available)")

                    with st.expander("Show feature vector sent to model"):
                        st.dataframe(aligned_row.T.astype(str))


    with tab_debug:
        st.markdown("### Model / Feature Debug Info")

        st.write("**Feature columns used in this UI:**")
        st.write(feature_names)

        if model is not None:
            st.write("**Columns the pipeline expects (from ColumnTransformer, if present):**")
            expected_cols = get_expected_feature_names(model, df)
            st.write(expected_cols if expected_cols else "Could not introspect expected columns.")

        st.caption(
            "This debug tab is only for development / verification so that you can "
            "confirm the app and the training notebook use the same feature space."
        )
