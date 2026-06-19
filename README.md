# Tumor Classification

Brain tumor MRI classification project comparing several models — Logistic Regression,
a scratch-built CNN, a modified CNN, and a MobileNetV2 transfer-learning model — on
classifying MRI slices into **glioma**, **meningioma**, **pituitary**, or **no tumor**.

**Authors:** Jeremy Kelly, Marco Basile, Tyler Asmussen, Oden Ineza

## Project Status

All four models are trained and exported to `models/`. Validation results (128x128 input):

| Model                  | Accuracy | Val Loss | Misclassified |
|-------------------------|----------|----------|----------------|
| MobileNetV2 Transfer   | 91.4%    | 0.267    | 96 / 1120      |
| Modified CNN            | 90.7%    | 0.249    | 104 / 1120     |
| Scratch CNN             | 87.4%    | 0.356    | 141 / 1120     |
| Logistic Regression     | 84.6%    | 1.176    | 173 / 1120     |

Full analysis, training curves, confusion matrices, and visualizations are in
`final_nb_deliverable3_ml.ipynb` and `outputs/`.

An interactive Streamlit dashboard (`dashboard/app.py`) lets you upload an MRI slice and
compare all four models' predictions side by side — see below.

## Setup

Requires Python 3.11 or 3.12 (`python3 --version`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Deactivate the environment later with `deactivate`.

## Running the Dashboard

The dashboard is an **educational prototype** — not for diagnostic use.

The trained models are already committed in `models/`, so you can run the dashboard
directly:

```bash
streamlit run dashboard/app.py
```

If you retrain or change the models, regenerate them by running
`final_nb_deliverable3_ml.ipynb` end-to-end (with `RUN_CNN=True`), including
**Section 18 — Export Models for Dashboard**, which writes
`models/logistic_regression.joblib`, `models/scratch_cnn.keras`,
`models/mobilenetv2.keras`, and `models/metadata.json`.

## Contributing

```bash
git pull origin main
git checkout -b feature/your-name-task
# make changes
git add .
git commit -m "Add preprocessing pipeline"
git push origin feature/your-name-task
# open a PR on GitHub
```
