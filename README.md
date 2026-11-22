# Explainable Machine Learning (XAI) Dashboard

An interactive, fully offline dashboard for explaining breast-cancer predictions using SHAP and LIME. The project uses only the built-in scikit-learn Breast Cancer dataset.

## Problem statement

Cancer prediction models can be accurate while remaining difficult to understand. This dashboard compares two tree models and explains both global feature importance and individual predictions in a form that is easy to inspect.

## Features

- Trains Random Forest and Gradient Boosting classifiers
- Displays accuracy, precision, recall, F1, and ROC-AUC
- Explains predictions with SHAP beeswarm, bar, waterfall, and force plots
- Explains individual predictions with LIME
- Supports test-row selection and adjustable top-feature sliders
- Shows confusion matrix and ROC curve

## Tech stack

- Python
- Streamlit
- scikit-learn
- pandas and NumPy
- Matplotlib
- SHAP
- LIME
- joblib

## Project structure

```text
model_trainer.py
app.py
requirements.txt
README.md
models/
  breast_cancer.joblib
```

## Installation

### Windows

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train and run

Train the models first, then start the dashboard:

### Windows

```powershell
python model_trainer.py
streamlit run app.py
```

### macOS and Linux

```bash
python model_trainer.py
streamlit run app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`.

## Using the sidebar

- **Pick a test row:** Choose an unseen test sample and explain its saved prediction.
- **Adjust feature sliders:** Start with training-means; ten most important features can be changed while the remaining features stay at their mean. Use **Reset to mean** to restore them.
- **Explained class:** Choose malignant or benign for class-specific SHAP and LIME explanations.
- **LIME number of features:** Choose how many local features appear in the LIME chart.

## SHAP vs LIME

| Aspect | SHAP | LIME |
|---|---|---|
| Main idea | Adds up each feature's contribution to a prediction | Builds a simple local approximation around one sample |
| Useful for | Global and local explanations | Single-prediction explanations |
| Strength | Consistent game-theoretic contributions | Easy-to-read feature conditions |
| Limitation | Large models can require more computation | Local approximation and random sampling can vary |

SHAP explains how features push a prediction. LIME temporarily tests small changes around a sample and learns which nearby feature conditions help explain that prediction.

## Real model results

Metrics are from the stratified 80/20 test split (`random_state=42`).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9931 |
| Gradient Boosting | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9907 |

## Screenshots

Add screenshots here after running the dashboard:

```text
![Prediction tab](screenshots/prediction.png)
![SHAP global tab](screenshots/shap-global.png)
![LIME tab](screenshots/lime.png)
```

## Limitations

- Explanations describe model behavior, not biological truth.
- Feature importance can reflect correlations in the data.
- LIME is a local approximation and can vary with sampling.
- Slider combinations may not represent real patient observations.
- The built-in dataset is a small, offline demonstration rather than a medical device.

## Future work

- Add cross-validation and confidence intervals
- Add model calibration
- Compare additional interpretable models
- Add patient-level data validation
- Export explanation reports

## References

- Lundberg, S. M., and Lee, S.-I. "A Unified Approach to Interpreting Model Predictions." NeurIPS, 2017.
- Ribeiro, M. T., Singh, S., and Guestrin, C. "Why Should I Trust You?: Explaining the Predictions of Any Classifier." KDD, 2016.
- SHAP: https://shap.readthedocs.io/
- LIME: https://github.com/marcotcr/lime
