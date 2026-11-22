"""Streamlit dashboard for explaining breast-cancer model predictions."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Callable

import joblib
import lime.lime_tabular
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = PROJECT_ROOT / "models" / "breast_cancer.joblib"


def ensure_artifact() -> dict:
    """Load the model artifact, training it automatically if it is missing."""
    if not ARTIFACT_PATH.exists():
        from model_trainer import train_and_save

        artifact, _ = train_and_save()
        return artifact
    return joblib.load(ARTIFACT_PATH)


@st.cache_resource
def load_cached_artifact() -> dict:
    """Cache the model artifact for the Streamlit session."""
    return ensure_artifact()


@st.cache_resource
def build_shap_explainer(_model):
    """Build a SHAP TreeExplainer once for the cached model."""
    return shap.TreeExplainer(_model)


@st.cache_resource
def build_lime_explainer(
    X_train_values: np.ndarray,
    feature_names: tuple[str, ...],
    class_names: tuple[str, ...],
):
    """Build a LIME explainer once from training data."""
    return lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_values,
        feature_names=list(feature_names),
        class_names=list(class_names),
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )


@st.cache_data
def compute_shap_values(_model, X_test_values: np.ndarray, feature_names: tuple[str, ...]):
    """Compute SHAP values once for the test set."""
    X_test = pd.DataFrame(X_test_values, columns=list(feature_names))
    explainer = build_shap_explainer(_model)
    return explainer(X_test)


def get_class_explanation(
    explanation,
    class_index: int,
) -> shap.Explanation:
    """Return a two-dimensional, class-specific SHAP Explanation safely."""
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    data = np.asarray(explanation.data)

    if values.ndim == 3:
        class_values = values[:, :, class_index]
    elif isinstance(explanation.values, list):
        class_values = np.asarray(explanation.values[class_index])
    else:
        class_values = values

    if base_values.ndim == 2:
        class_base_values = base_values[:, class_index]
    elif base_values.ndim == 1:
        class_base_values = base_values
    else:
        class_base_values = np.asarray([base_values.item()] * class_values.shape[0])

    return shap.Explanation(
        values=class_values,
        base_values=class_base_values,
        data=data,
        feature_names=list(explanation.feature_names),
    )


def render_plot(plot_fn: Callable[[], None], width: float, height: float) -> None:
    """Draw one plot in an isolated figure and display it in Streamlit."""
    fig = plt.figure(figsize=(width, height))
    plot_fn()
    try:
        fig.tight_layout()
    except (RuntimeError, ValueError):
        plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    plt.close("all")


def make_dataframe(values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Create a DataFrame with names so downstream predict functions have no warnings."""
    return pd.DataFrame(values.reshape(1, -1), columns=feature_names)


def get_current_row(
    artifact: dict,
    input_mode: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build the current sample from test-row selection or sidebar sliders."""
    X_train = artifact["X_train"]
    X_test = artifact["X_test"]
    feature_names = artifact["feature_names"]

    if input_mode == "Pick a test row":
        row_index = int(
            st.sidebar.selectbox(
                "Test-set row",
                options=list(range(len(X_test))),
                index=0,
            )
        )
        row_frame = X_test.iloc[[row_index]].copy()
    else:
        row_frame = X_train.mean().to_frame().T
        row_frame.columns = feature_names
        st.sidebar.markdown("**Top feature sliders**")
        model = artifact["model"]
        importances = model.feature_importances_
        ranking = np.argsort(importances)[::-1][:10]

        def reset_sliders() -> None:
            """Reset all top-feature slider keys to training means."""
            for index in ranking:
                name = feature_names[index]
                st.session_state[f"slider_{index}"] = float(X_train[name].mean())

        st.sidebar.button("🔄 Reset to mean", on_click=reset_sliders, use_container_width=True)
        for index in ranking:
            name = feature_names[index]
            minimum = float(X_train[name].min())
            maximum = float(X_train[name].max())
            mean = float(X_train[name].mean())
            key = f"slider_{index}"
            row_frame.at[0, name] = float(
                st.sidebar.slider(
                    name,
                    min_value=minimum,
                    max_value=maximum,
                    value=mean,
                    key=key,
                )
            )
        row_index = 0

    return row_frame.values[0], row_frame


def sentence_about_top_features(
    row_frame: pd.DataFrame,
    class_explanation: shap.Explanation,
    predicted_index: int,
    predicted_probability: float,
    class_names: list[str],
    training_mean: pd.Series,
) -> str:
    """Write a plain-English summary of the three largest SHAP influences."""
    values = np.asarray(class_explanation.values[predicted_index if class_explanation.values.ndim == 2 else 0])
    if values.ndim != 1:
        values = values.reshape(-1)
    feature_names = class_explanation.feature_names
    row_values = row_frame.iloc[0]
    ranked = np.argsort(np.abs(values))[::-1][:3]

    phrases = []
    for index in ranked:
        name = str(feature_names[index])
        level = (
            "low"
            if float(row_values.iloc[index]) < float(training_mean.iloc[index])
            else "high"
        )
        phrases.append(f"**{name}** is {level}")

    result = ", ".join(phrases)
    return (
        f"Predicted **{str(class_names[predicted_index]).upper()}** "
        f"({predicted_probability:.0%}) mainly because {result}."
    )


def main() -> None:
    """Run the Explainable Machine Learning Streamlit app."""
    st.set_page_config(
        page_title="Explainable Machine Learning",
        page_icon="🔬",
        layout="wide",
    )
    st.title("🔬 Explainable Machine Learning Dashboard")
    st.caption("Breast Cancer Diagnosis Interpretability using SHAP & LIME")

    artifact = load_cached_artifact()
    model = artifact["model"]
    feature_names = list(artifact["feature_names"])
    class_names = list(artifact["class_names"])
    X_train = artifact["X_train"]
    X_test = artifact["X_test"]
    y_test = artifact["y_test"]

    st.sidebar.header("⚙️ Explanation Controls")
    input_mode = st.sidebar.radio(
        "Input mode",
        options=["Pick a test row", "Adjust feature sliders"],
        index=0,
    )
    row_values, row_frame = get_current_row(artifact, input_mode)

    st.sidebar.divider()
    explained_class_name = st.sidebar.selectbox(
        "Explained class",
        options=class_names,
        index=class_names.index("benign") if "benign" in class_names else 0,
    )
    lime_features = st.sidebar.slider(
        "LIME number of features",
        min_value=5,
        max_value=15,
        value=10,
        step=1,
    )

    class_index = class_names.index(explained_class_name)
    raw_explanation = compute_shap_values(
        model,
        X_test.values,
        tuple(feature_names),
    )
    class_explanation = get_class_explanation(raw_explanation, class_index)

    predicted_probabilities = model.predict_proba(row_frame)[0]
    predicted_index = int(np.argmax(predicted_probabilities))
    predicted_probability = float(predicted_probabilities[predicted_index])
    predicted_label = str(class_names[predicted_index]).upper()

    tabs = st.tabs([
        "🎯 Prediction",
        "🌍 SHAP: Global",
        "🔍 SHAP: Local",
        "🍋 LIME",
        "📊 Model Performance",
        "ℹ️ About",
    ])

    with tabs[0]:
        with st.spinner("Calculating prediction and explanation..."):
            try:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Predicted Class",
                        value=predicted_label,
                    )
                with col2:
                    st.metric(
                        label="Prediction Confidence",
                        value=f"{predicted_probability:.2%}",
                    )

                st.divider()
                st.markdown("#### Class Probabilities")
                probability_frame = pd.DataFrame(
                    {
                        "Class": [c.capitalize() for c in class_names],
                        "Probability": predicted_probabilities,
                    }
                )
                st.dataframe(
                    probability_frame,
                    column_config={
                        "Class": st.column_config.TextColumn("Diagnosis Class"),
                        "Probability": st.column_config.ProgressColumn(
                            "Probability Score",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

                summary_text = sentence_about_top_features(
                    row_frame,
                    class_explanation,
                    predicted_index,
