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
    # SHAP's TreeExplainer works directly with the trained tree model.
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
    # SHAP Explanation objects preserve values and feature names in the modern API.
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

    # A 2D Explanation contains one row per test sample and one column per feature.
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

        st.sidebar.button("Reset to mean", on_click=reset_sliders)
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
        phrases.append(f"{name} is {level}")

    result = ", ".join(phrases)
    return (
        f"Predicted {str(class_names[predicted_index]).upper()} "
        f"({predicted_probability:.0%}) mainly because {result}."
    )


def main() -> None:
    """Run the Explainable Machine Learning Streamlit app."""
    st.set_page_config(page_title="Explainable Machine Learning", layout="wide")
    st.title("Explainable Machine Learning Dashboard")

    artifact = load_cached_artifact()
    model = artifact["model"]
    feature_names = list(artifact["feature_names"])
    class_names = list(artifact["class_names"])
    X_train = artifact["X_train"]
    X_test = artifact["X_test"]
    y_test = artifact["y_test"]

    st.sidebar.header("Explanation Controls")
    input_mode = st.sidebar.radio(
        "Input mode",
        options=["Pick a test row", "Adjust feature sliders"],
        index=0,
    )
    row_values, row_frame = get_current_row(artifact, input_mode)
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
    shap_explainer = build_shap_explainer(model)
    raw_explanation = compute_shap_values(
        model,
        X_test.values,
        tuple(feature_names),
    )
    class_explanation = get_class_explanation(raw_explanation, class_index)

    predicted_probabilities = model.predict_proba(row_frame)[0]
    predicted_index = int(np.argmax(predicted_probabilities))
    predicted_probability = float(predicted_probabilities[predicted_index])

    tabs = st.tabs([
        "Prediction",
        "SHAP: Global",
        "SHAP: Local",
        "LIME",
        "Model Performance",
        "About",
    ])

    with tabs[0]:
        with st.spinner("Calculating prediction and explanation..."):
            try:
                st.subheader(f"Predicted class: {class_names[predicted_index].upper()}")
                st.metric(
                    "Prediction confidence",
                    f"{predicted_probability:.2%}",
                    delta=None,
                )
                st.progress(min(max(predicted_probability, 0.0), 1.0))
                st.markdown("**Probability for each class**")
                probability_frame = pd.DataFrame(
                    {
                        "Class": class_names,
                        "Probability": predicted_probabilities,
                    }
                )
                st.dataframe(probability_frame, hide_index=True)

                row_position = int(row_frame.index[0]) if row_frame.index.size else 0
                test_matches = np.all(X_test.values == row_values, axis=1)
                match_indices = np.flatnonzero(test_matches)
                if input_mode == "Pick a test row":
                    sample_index = int(
                        st.session_state.get("Test-set row", match_indices[0] if match_indices.size else 0)
                    )
                    if sample_index >= len(X_test):
                        sample_index = 0
                else:
                    sample_index = int(match_indices[0]) if match_indices.size else 0

                explanation_row = shap.Explanation(
                    values=np.asarray(class_explanation.values[sample_index])
                    if class_explanation.values.shape[0] > sample_index
                    else np.asarray(class_explanation.values).reshape(-1),
                    base_values=float(
                        np.asarray(class_explanation.base_values).reshape(-1)[
                            sample_index
                            if np.asarray(class_explanation.base_values).reshape(-1).size > sample_index
                            else 0
                        ]
                    ),
                    data=row_values,
                    feature_names=feature_names,
                )
                st.info(
                    sentence_about_top_features(
                        row_frame,
                        class_explanation,
                        predicted_index,
                        predicted_probability,
                        class_names,
                        X_train.mean(),
                    )
                )
            except Exception as exc:
                st.error(f"Prediction could not be shown: {exc}")

    with tabs[1]:
        with st.spinner("Building global SHAP plots..."):
            try:
                st.subheader("Which features matter most across the test set?")
                render_plot(
                    lambda: shap.plots.beeswarm(
                        class_explanation,
                        max_display=15,
                        show=False,
                    ),
                    10,
                    7,
                )
                st.subheader("Mean absolute SHAP value")
                render_plot(
                    lambda: shap.plots.bar(
                        class_explanation,
                        max_display=15,
                        show=False,
                    ),
                    10,
                    7,
                )
            except Exception as exc:
                st.error(f"Global SHAP plots could not be shown: {exc}")

    with tabs[2]:
        with st.spinner("Building local SHAP plots..."):
            try:
                test_matches = np.all(X_test.values == row_values, axis=1)
                match_indices = np.flatnonzero(test_matches)
                if input_mode == "Pick a test row":
                    sample_index = int(
                        st.session_state.get("Test-set row", match_indices[0] if match_indices.size else 0)
                    )
                else:
                    sample_index = int(match_indices[0]) if match_indices.size else 0
                if sample_index >= len(X_test):
                    sample_index = 0

                shap_values_row = np.asarray(class_explanation.values[sample_index]).reshape(-1)
                base_values = np.asarray(class_explanation.base_values).reshape(-1)
                base_value = float(
                    base_values[sample_index] if base_values.size > sample_index else base_values[0]
                )
                single_explanation = shap.Explanation(
                    values=shap_values_row,
                    base_values=base_value,
                    data=row_values,
                    feature_names=feature_names,
                )
                render_plot(
                    lambda: shap.plots.waterfall(
                        single_explanation,
                        max_display=12,
                        show=False,
                    ),
                    10,
                    6,
                )
                render_plot(
                    lambda: shap.plots.force(
                        base_value,
                        shap_values_row,
                        row_values,
                        feature_names=feature_names,
                        matplotlib=True,
                        show=False,
                    ),
                    12,
                    3,
                )
            except Exception as exc:
                st.error(f"Local SHAP plots could not be shown: {exc}")

    with tabs[3]:
        with st.spinner("Building the LIME explanation..."):
            try:
                lime_explainer = build_lime_explainer(
                    X_train.values,
                    tuple(feature_names),
                    tuple(class_names),
                )

                def lime_predict(values: np.ndarray) -> np.ndarray:
                    # Rebuild a named DataFrame so the model never emits feature-name warnings.
                    frame = pd.DataFrame(values, columns=feature_names)
                    return model.predict_proba(frame)

                lime_result = lime_explainer.explain_instance(
                    row_values,
                    lime_predict,
                    num_features=lime_features,
                    labels=(class_index,),
                )
                lime_items = lime_result.as_list(label=class_index)
                labels = [str(item[0]) for item in lime_items]
                contributions = [float(item[1]) for item in lime_items]

                def draw_lime() -> None:
                    colors = ["green" if value >= 0 else "red" for value in contributions]
                    plt.barh(range(len(labels)), contributions, color=colors)
                    plt.yticks(range(len(labels)), labels)
                    plt.axvline(0, color="black", linewidth=0.8)
                    plt.gca().invert_yaxis()
                    plt.xlabel("Contribution to explained class")
                    plt.ylabel("Feature condition")

                render_plot(draw_lime, 10, 7)
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Feature condition": labels,
                            "Contribution": contributions,
                        }
                    ),
                    hide_index=True,
                )
            except Exception as exc:
                st.error(f"LIME explanation could not be shown: {exc}")

    with tabs[4]:
        with st.spinner("Calculating model performance..."):
            try:
                metrics = artifact["metrics"]
                st.dataframe(
                    pd.DataFrame(metrics).T[
                        ["accuracy", "precision", "recall", "f1", "roc_auc"]
                    ].round(4),
                    hide_index=False,
                )

                left, right = st.columns(2)
                predictions = model.predict(X_test)
                with left:
                    st.subheader("Confusion matrix")

                    def draw_confusion() -> None:
                        matrix = confusion_matrix(y_test, predictions)
                        ax = plt.gca()
                        image = ax.imshow(matrix, cmap="Blues")
                        ax.set_xticks([0, 1], labels=class_names)
                        ax.set_yticks([0, 1], labels=class_names)
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        for row in range(matrix.shape[0]):
                            for column in range(matrix.shape[1]):
                                ax.text(
                                    column,
                                    row,
                                    str(matrix[row, column]),
                                    ha="center",
                                    va="center",
                                )
                        plt.colorbar(image, ax=ax)

                    render_plot(draw_confusion, 6, 5)

                with right:
                    st.subheader("ROC curve")

                    def draw_roc() -> None:
                        probabilities = model.predict_proba(X_test)[:, 1]
                        false_positive, true_positive, _ = roc_curve(y_test, probabilities)
                        auc_value = roc_auc_score(y_test, probabilities)
                        plt.plot(
                            false_positive,
                            true_positive,
                            label=f"Saved model (AUC = {auc_value:.4f})",
                        )
                        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
                        plt.xlabel("False positive rate")
                        plt.ylabel("True positive rate")
                        plt.legend()

                    render_plot(draw_roc, 6, 5)
            except Exception as exc:
                st.error(f"Performance information could not be shown: {exc}")

    with tabs[5]:
        st.subheader("About SHAP")
        st.write(
            "SHAP explains a prediction by estimating how much each feature pushes "
            "it toward or away from a class. It is useful for seeing both global "
            "feature importance and one person's prediction."
        )
        st.subheader("About LIME")
        st.write(
            "LIME builds a small local approximation around one sample and shows "
            "which feature conditions matter most for that case."
        )
        st.subheader("SHAP vs LIME")
        st.table(
            pd.DataFrame(
                {
                    "SHAP": [
                        "Model-consistent local contributions",
                        "Global beeswarm and bar views",
                        "Can be slower for complex models",
                    ],
                    "LIME": [
                        "Local approximation around one row",
                        "Readable feature-condition rules",
                        "Can change with random sampling",
                    ],
                },
                index=["Strength", "View", "Limitation"],
            )
        )


if __name__ == "__main__":
    main()
