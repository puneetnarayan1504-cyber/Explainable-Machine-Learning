"""Explainable Machine Learning dashboard for breast-cancer model predictions."""

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

# ---------------------------------------------------------------- design tokens
INK = "#16302f"
TEAL = "#0f766e"
TEAL_SOFT = "#e6f2f1"
CORAL = "#c2410c"
SLATE = "#64748b"
PAPER = "#f6f8f8"
POSITIVE = TEAL
NEGATIVE = CORAL

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'IBM Plex Sans', sans-serif; }}
.stApp {{ background: {PAPER}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }}
#MainMenu, footer {{ visibility: hidden; }}

.app-header {{
    background: {INK}; color: #fff; padding: 1.4rem 1.8rem;
    border-radius: 10px; margin-bottom: 1.2rem;
}}
.app-header h1 {{ color: #fff; font-size: 1.6rem; font-weight: 600; margin: 0; padding: 0; }}
.app-header p {{ color: #b9d3d1; margin: .3rem 0 0 0; font-size: .95rem; }}

.kpi {{
    background: #fff; border: 1px solid #dbe4e3; border-left: 4px solid {TEAL};
    border-radius: 8px; padding: .9rem 1.1rem; height: 100%;
}}
.kpi.alert {{ border-left-color: {CORAL}; }}
.kpi .label {{ color: {SLATE}; font-size: .82rem; font-weight: 500; }}
.kpi .value {{ color: {INK}; font-size: 1.6rem; font-weight: 700; line-height: 1.3; }}
.kpi .note {{ color: {SLATE}; font-size: .78rem; }}

.insight {{
    background: {TEAL_SOFT}; border-radius: 8px; padding: .9rem 1.1rem;
    color: {INK}; margin: 1rem 0; border: 1px solid #c5e0de;
}}

h2, h3 {{ color: {INK}; font-weight: 600; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .25rem; border-bottom: 1px solid #dbe4e3; }}
.stTabs [data-baseweb="tab"] {{ padding: .6rem 1rem; font-weight: 500; color: {SLATE}; }}
.stTabs [aria-selected="true"] {{ color: {TEAL}; }}
section[data-testid="stSidebar"] {{ background: #fff; border-right: 1px solid #dbe4e3; }}
.disclaimer {{ color: {SLATE}; font-size: .8rem; margin-top: 2rem; }}
</style>
"""

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#cbd5d4",
        "axes.labelcolor": INK,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


# ------------------------------------------------------------------ model/data
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
    """Compute SHAP values once for the whole test set."""
    X_test = pd.DataFrame(X_test_values, columns=list(feature_names))
    return build_shap_explainer(_model)(X_test)


def get_class_explanation(explanation, class_index: int) -> shap.Explanation:
    """Return a class-specific 2D SHAP Explanation (rows x features)."""
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    data = np.asarray(explanation.data)

    class_values = values[:, :, class_index] if values.ndim == 3 else values
    if base_values.ndim == 2:
        class_base = base_values[:, class_index]
    elif base_values.ndim == 1:
        class_base = base_values
    else:
        class_base = np.full(class_values.shape[0], base_values.item())

    return shap.Explanation(
        values=class_values,
        base_values=class_base,
        data=data,
        feature_names=list(explanation.feature_names),
    )


# --------------------------------------------------------------------- helpers
def render_plot(plot_fn: Callable[[], None], width: float, height: float) -> None:
    """Draw one plot in an isolated figure and display it in Streamlit."""
    fig = plt.figure(figsize=(width, height))
    plot_fn()
    try:
        fig.tight_layout()
    except (RuntimeError, ValueError):
        plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close("all")


def kpi(label: str, value: str, note: str = "", alert: bool = False) -> None:
    """Render one metric card."""
    st.markdown(
        f'<div class="kpi{" alert" if alert else ""}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, caption: str = "") -> None:
    """Render a section heading with an optional one-line explanation."""
    st.subheader(title)
    if caption:
        st.caption(caption)


def get_current_row(artifact: dict, input_mode: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Build the current sample from a test-row selection or sidebar sliders."""
    X_train = artifact["X_train"]
    X_test = artifact["X_test"]
    feature_names = list(artifact["feature_names"])

    if input_mode == "Pick a test row":
        row_index = int(
            st.sidebar.selectbox("Test-set row", options=list(range(len(X_test))), index=0)
        )
        row_frame = X_test.iloc[[row_index]].copy()
        return row_frame.values[0], row_frame

    row_frame = X_train.mean().to_frame().T
    row_frame.columns = feature_names
    st.sidebar.caption("Adjust the ten features the model relies on most.")
    ranking = np.argsort(artifact["model"].feature_importances_)[::-1][:10]

    def reset_sliders() -> None:
        for index in ranking:
            st.session_state[f"slider_{index}"] = float(X_train[feature_names[index]].mean())

    st.sidebar.button("Reset to average", on_click=reset_sliders)
    for index in ranking:
        name = feature_names[index]
        row_frame.at[0, name] = float(
            st.sidebar.slider(
                name,
                min_value=float(X_train[name].min()),
                max_value=float(X_train[name].max()),
                value=float(X_train[name].mean()),
                key=f"slider_{index}",
            )
        )
    return row_frame.values[0], row_frame


def summarise_drivers(
    local: shap.Explanation,
    row_frame: pd.DataFrame,
    training_mean: pd.Series,
    explained_class: str,
) -> str:
    """Write a plain-English summary of the three largest SHAP influences."""
    values = np.asarray(local.values).reshape(-1)
    row = row_frame.iloc[0]
    parts = []
    for index in np.argsort(np.abs(values))[::-1][:3]:
        name = str(local.feature_names[index])
        level = "above" if row.iloc[index] > training_mean.iloc[index] else "below"
        effect = "raises" if values[index] > 0 else "lowers"
        parts.append(f"<b>{name}</b> is {level} average and {effect} the score")
    return f"For the <b>{explained_class}</b> class: " + "; ".join(parts) + "."


# ------------------------------------------------------------------------ main
def main() -> None:
    """Run the Explainable Machine Learning Streamlit app."""
    st.set_page_config(
        page_title="Explainable ML Dashboard",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="app-header"><h1>Explainable Machine Learning Dashboard</h1>'
        "<p>Breast-cancer classification with SHAP and LIME explanations</p></div>",
        unsafe_allow_html=True,
    )

    artifact = load_cached_artifact()
    model = artifact["model"]
    feature_names = list(artifact["feature_names"])
    class_names = list(artifact["class_names"])
    X_train, X_test, y_test = artifact["X_train"], artifact["X_test"], artifact["y_test"]

    # ---- sidebar
    st.sidebar.header("Controls")
    input_mode = st.sidebar.radio("Input mode", ["Pick a test row", "Adjust feature sliders"])
    row_values, row_frame = get_current_row(artifact, input_mode)
    st.sidebar.divider()
    explained_class = st.sidebar.selectbox(
        "Explained class",
        options=class_names,
        index=class_names.index("benign") if "benign" in class_names else 0,
    )
    lime_features = st.sidebar.slider("LIME features to show", 5, 15, 10)
    class_index = class_names.index(explained_class)

    # ---- explanations
    raw_global = compute_shap_values(model, X_test.values, tuple(feature_names))
    global_explanation = get_class_explanation(raw_global, class_index)
    # Local explanation is computed for the exact current row, so it also
    # works for slider-edited samples that are not in the test set.
    local_explanation = get_class_explanation(
        build_shap_explainer(model)(row_frame), class_index
    )[0]

    probabilities = model.predict_proba(row_frame)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    predicted_name = class_names[predicted_index]

    tabs = st.tabs(
        ["Prediction", "Global SHAP", "Local SHAP", "LIME", "Model performance", "About"]
    )

    # ---- prediction
    with tabs[0]:
        try:
            test_probs = model.predict_proba(X_test)[:, 1]
            test_acc = float((model.predict(X_test) == y_test).mean())
            test_auc = float(roc_auc_score(y_test, test_probs))

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                kpi("Predicted class", predicted_name.title(), "For the selected sample",
                    alert=predicted_name.lower() == "malignant")
            with c2:
                kpi("Confidence", f"{confidence:.1%}", "Probability of predicted class")
            with c3:
                kpi("Test accuracy", f"{test_acc:.1%}", f"{len(X_test)} held-out samples")
            with c4:
                kpi("ROC AUC", f"{test_auc:.3f}", "Held-out test set")

            st.markdown(
                f'<div class="insight">{summarise_drivers(local_explanation, row_frame, X_train.mean(), explained_class)}</div>',
                unsafe_allow_html=True,
            )

            left, right = st.columns([1, 1.4])
            with left:
                section("Class probabilities")
                st.dataframe(
                    pd.DataFrame({"Class": class_names, "Probability": probabilities}),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Probability": st.column_config.ProgressColumn(
                            "Probability", min_value=0.0, max_value=1.0, format="%.2f"
                        )
                    },
                )
            with right:
                section("Selected sample", "Feature values being explained.")
                st.dataframe(
                    row_frame.T.rename(columns={row_frame.index[0]: "Value"}),
                    use_container_width=True,
                    height=240,
                )
        except Exception as exc:
            st.error(f"Prediction could not be shown: {exc}")

    # ---- global SHAP
    with tabs[1]:
        try:
            section("Feature impact across the test set",
                    "Each dot is one patient; colour shows the feature value.")
            render_plot(
                lambda: shap.plots.beeswarm(global_explanation, max_display=15, show=False), 10, 7
            )
            section("Average importance", "Mean absolute SHAP value per feature.")
            render_plot(
                lambda: shap.plots.bar(global_explanation, max_display=15, show=False), 10, 6
            )
        except Exception as exc:
            st.error(f"Global SHAP plots could not be shown: {exc}")

    # ---- local SHAP
    with tabs[2]:
        try:
            section("Why this prediction?",
                    "Bars show how each feature moves the score away from the baseline.")
            render_plot(
                lambda: shap.plots.waterfall(local_explanation, max_display=12, show=False), 10, 6
            )
            section("Force plot")
            render_plot(
                lambda: shap.plots.force(
                    float(np.asarray(local_explanation.base_values).reshape(-1)[0]),
                    np.asarray(local_explanation.values).reshape(-1),
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

    # ---- LIME
    with tabs[3]:
        try:
            lime_explainer = build_lime_explainer(
                X_train.values, tuple(feature_names), tuple(class_names)
            )

            def lime_predict(values: np.ndarray) -> np.ndarray:
                return model.predict_proba(pd.DataFrame(values, columns=feature_names))

            with st.spinner("Building the LIME explanation..."):
                result = lime_explainer.explain_instance(
                    row_values, lime_predict, num_features=lime_features, labels=(class_index,)
                )
            items = result.as_list(label=class_index)
            labels = [str(i[0]) for i in items]
            contributions = [float(i[1]) for i in items]

            def draw_lime() -> None:
                colors = [POSITIVE if v >= 0 else NEGATIVE for v in contributions]
                plt.barh(range(len(labels)), contributions, color=colors)
                plt.yticks(range(len(labels)), labels)
                plt.axvline(0, color=INK, linewidth=0.8)
                plt.gca().invert_yaxis()
                plt.xlabel(f"Contribution to “{explained_class}”")

            section("Local approximation", "Teal supports the class, orange pushes against it.")
            render_plot(draw_lime, 10, 6)
            st.dataframe(
                pd.DataFrame({"Feature condition": labels, "Contribution": contributions}),
                hide_index=True,
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"LIME explanation could not be shown: {exc}")

    # ---- performance
    with tabs[4]:
        try:
            section("Metrics")
            st.dataframe(
                pd.DataFrame(artifact["metrics"]).T[
                    ["accuracy", "precision", "recall", "f1", "roc_auc"]
                ].round(4),
                use_container_width=True,
            )
            left, right = st.columns(2)
            predictions = model.predict(X_test)

            with left:
                section("Confusion matrix")

                def draw_confusion() -> None:
                    matrix = confusion_matrix(y_test, predictions)
                    ax = plt.gca()
                    image = ax.imshow(matrix, cmap="Greens")
                    ax.set_xticks([0, 1], labels=class_names)
                    ax.set_yticks([0, 1], labels=class_names)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    for r in range(matrix.shape[0]):
                        for c in range(matrix.shape[1]):
                            ax.text(c, r, str(matrix[r, c]), ha="center", va="center", color=INK)
                    plt.colorbar(image, ax=ax)

                render_plot(draw_confusion, 6, 5)

            with right:
                section("ROC curve")

                def draw_roc() -> None:
                    probs = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, probs)
                    plt.plot(fpr, tpr, color=TEAL, linewidth=2,
                             label=f"Model (AUC = {roc_auc_score(y_test, probs):.4f})")
                    plt.plot([0, 1], [0, 1], linestyle="--", color=SLATE)
                    plt.xlabel("False positive rate")
                    plt.ylabel("True positive rate")
                    plt.legend(frameon=False)

                render_plot(draw_roc, 6, 5)
        except Exception as exc:
            st.error(f"Performance information could not be shown: {exc}")

    # ---- about
    with tabs[5]:
        a, b = st.columns(2)
        with a:
            section("SHAP")
            st.write(
                "SHAP estimates how much each feature pushes a prediction toward or away "
                "from a class. It supports both global importance and single-sample explanations."
            )
        with b:
            section("LIME")
            st.write(
                "LIME fits a small, simple model around one sample and reports which feature "
                "conditions matter most for that case."
            )
        section("SHAP vs LIME")
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

    st.markdown(
        '<div class="disclaimer">Educational project. Not a medical device and not for clinical use.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
