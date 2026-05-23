# Cardia — Heart Attack Risk Predictor

> Habiba Mohamed Hassan · 2210019322 

This is the runnable submission for Cardia. Cardia takes 18 lifestyle and medical-history answers from the CDC's 2022 BRFSS extract and returns a probability that the respondent has ever had a heart attack. The submission ships the cleaned CSV, the executed end-to-end notebook, eleven figures, the trained pipeline plus a tuned threshold, and a Flask + Tailwind web app.

This README walks through what's inside the way I worked on it: the question, then the data, then how it's encoded and modelled, then what came out, then how to run it.

## The question

Heart attacks are a leading cause of death worldwide. Self-reported BRFSS indicators are cheap, abundant, and already collected at scale. Can those alone — no ECG, no troponin, no imaging — flag respondents who deserve a closer look from a clinician? A binary classifier is the right shape, and that is what this submission carries.

The positive class is **5.35 %** of cleaned rows, so accuracy is misleading and the operating point matters more than any single hyper-parameter.

## The data

`data/heart.csv` is the BRFSS 2022 heart-disease extract: 445,132 raw rows, 40 raw columns. After dropping rows with any missing value among the 18 modelled columns, 328,047 complete-case rows remain.

The 18 features:

| Group | Count | Examples |
|---|---:|---|
| Numeric / ordinal | 8 | `AgeCategory`, `GeneralHealth`, `SmokerStatus`, `HadDiabetes`, `PhysicalHealthDays`, `MentalHealthDays`, `SleepHours`, `BMI` |
| Binary (Yes/No) | 10 | `Sex`, `PhysicalActivities`, `AlcoholDrinkers`, `HadAngina`, `HadStroke`, `HadCOPD`, `HadKidneyDisease`, `HadArthritis`, `DifficultyWalking`, `ChestScan` |

Target: `HadHeartAttack`. Categoricals are encoded with their natural ordering; the encoder maps live in `heart_metadata.json` so the notebook and the Flask form can never disagree.

## The pipeline

```
ColumnTransformer
├── numeric/ordinal -> SimpleImputer(median) -> StandardScaler
└── binary          -> SimpleImputer(most_frequent)
```

Wrapped in a single scikit-learn `Pipeline` so the scaler is fit only on the training fold inside `pipe.fit(X_train, y_train)`. Stratified 80/20 train/test split (positive rate preserved). Four candidate models:

| Model | Hyper-parameters | Imbalance handling |
|---|---|---|
| Decision Tree | `max_depth=8` | `class_weight='balanced'` |
| AdaBoost | `n_estimators=100, algorithm='SAMME'` | base learner re-weights |
| Gradient Boosting | `n_estimators=120, max_depth=4, learning_rate=0.08` | left to threshold tuning |
| XGBoost | `n_estimators=200, max_depth=5, learning_rate=0.1, tree_method='hist'` | `scale_pos_weight = neg / pos` |

Selection metric: ROC-AUC. Threshold sweep: 0.10 → 0.90 in 0.01 steps; F1-maximiser kept.

## What came out

Gradient Boosting won the bake-off.

**At the tuned threshold ≈ 0.28:**

| Metric | Score |
|---|---:|
| Accuracy | 0.9434 |
| Precision (heart-attack class) | 0.4727 |
| Recall (heart-attack class) | 0.4987 |
| F1 (heart-attack class) | 0.4854 |
| ROC-AUC | 0.8969 |

**At the default 0.5, all four side by side:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.9501 | 0.5784 | 0.2499 | 0.349 | **0.8969** |
| XGBoost | 0.8278 | 0.2078 | 0.7876 | 0.329 | 0.8944 |
| AdaBoost | 0.9494 | 0.5578 | 0.2693 | 0.363 | 0.8936 |
| Decision Tree | 0.7932 | 0.1809 | 0.8110 | 0.296 | 0.8825 |

**Why move the threshold down to 0.28?** Gradient Boosting at 0.5 picks a small, very precise set of positives (precision 0.58) but misses most true cases (recall 0.25). For a screening tool that's the wrong trade-off. Maximising F1 lifts recall to 0.50 with only a small precision drop, at almost identical overall accuracy.

## The figures

Eleven PNGs in `figures/`, all on the same flare-and-rocket palette so the saved figures and the live UI agree on what teal and rose mean.

| File | What it shows |
|---|---|
| `01_missing_values.png` | Top columns by missing count |
| `02_target_pie.png` | Class imbalance |
| `03_general_health_donut.png` | Self-rated general health distribution |
| `04_bmi_violin.png` | BMI by heart-attack status |
| `05_sleep_box.png` | Sleep hours by heart-attack status |
| `06_age_vs_heart.png` | Heart-attack rate by age band |
| `07_smoking_vs_heart.png` | Heart-attack rate by smoking status |
| `08_model_comparison.png` | Five metrics across the four candidates |
| `09_confusion_matrix.png` | Confusion matrix at the tuned threshold |
| `10_roc_curve.png` | ROC curve, AUC ≈ 0.90 |
| `11_precision_recall.png` | F1 across the threshold sweep |

## The web app

`app.py` loads the four artefacts at import time and exposes a JSON `/predict` endpoint. The front-end is a glassmorphism single-page form built on Tailwind via CDN; it sends form values as JSON, gets back `{ probability, prediction, risk, threshold }`, and animates a risk meter from 0 to the returned percentage. The categorical encoder maps come straight out of `heart_metadata.json`, and the flag-coercion helper accepts `1/0`, `'1'/'0'`, `True/False`, and `'Yes'/'No'`.

## Running it locally

```bash
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000>. On Windows, double-clicking `run.bat` does the install and launch in one step.

To re-execute the notebook end-to-end:
```bash
jupyter notebook notebook.ipynb
```

## Submission layout

```
submission/
├── README.md
├── requirements.txt
├── run.bat
│
├── notebook.ipynb
├── app.py
├── templates/index.html
│
├── data/heart.csv
├── figures/  (11 PNGs)
│
├── heart_model.pkl
├── heart_features.pkl
├── heart_threshold.pkl
└── heart_metadata.json
```

## Reproducibility

A single `RANDOM_STATE = 42` is threaded through the train/test split and every tree-based model. The scaler lives **inside** the saved pipeline, so reload-and-predict needs only `joblib.load` on `heart_model.pkl`. Dependencies are version-ranged in `requirements.txt` so the saved pipeline stays loadable across the scikit-learn 1.3-1.4 family.

## Limitations

- BRFSS is self-reported — under-reporting of diagnoses is real. The target captures *ever-had*, not "currently at risk".
- US-only sample, so cross-country generalisation is unverified.
- No clinical measurements (no ECG, troponin, or imaging). The model is a screening aid only.
- ~5 % positives plus a recall-leaning operating point means a noticeable false-positive rate.

---
