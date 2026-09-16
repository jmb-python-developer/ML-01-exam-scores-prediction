# Exam Score Predictor

Predicts a student's exam score from five everyday habits — study hours, attendance, sleep, mental health rating, and whether they hold a part-time job — using scikit-learn, with a small Streamlit app for trying it out interactively.

This was my first ML project taken end-to-end: EDA, feature engineering, model comparison, and a working app on top of the trained model. It's intentionally scoped small — the goal was to get every step of the pipeline right, not to maximize accuracy.

## Dataset

`data/student_habits_performance.csv` — 1,000 student records, 16 columns (demographics, study/lifestyle habits, and `exam_score` as the target). 91 rows had a missing `parental_education_level` and were dropped during cleaning, leaving 909 rows for modeling.
*(Public dataset of synthetic student habits and academic performance — source link added here if I track it back down.)*

Out of the 16 columns, I constrained the model to 5 features that showed the clearest relationship with `exam_score` during EDA: `study_hours_per_day`, `attendance_percentage`, `mental_health_rating`, `sleep_hours`, and `part_time_job`.

## Approach

1. **EDA** — checked for nulls/duplicates, looked at univariate and bivariate distributions, and used a correlation matrix to narrow down features (`study_hours_per_day` correlates with `exam_score` far more than anything else in the dataset).
2. **Feature encoding** — `part_time_job` (Yes/No) is encoded with `OrdinalEncoder`.
3. **Model comparison** — LinearRegression, DecisionTreeRegressor, and RandomForestRegressor, each tuned with `GridSearchCV` (5-fold CV). A `DummyRegressor` (predicts the mean) is included in the same comparison as a baseline, so the real models' scores mean something concrete rather than being judged only against each other.
4. **Selection** — the winner is picked by cross-validated score alone; the held-out test set is only touched once, afterward, to report a final, unbiased number for that one model.
5. **Export** — the winning model and its `OrdinalEncoder` are bundled into a single `sklearn.pipeline.Pipeline` and serialized with `joblib`, so the app never re-implements the encoding by hand.
6. **App** — a small Streamlit UI loads that pipeline and predicts from five slider/select inputs.

## Results

5-fold cross-validated RMSE, training data only (this is what picked the winner):

| Model | CV RMSE | Best params |
|---|---|---|
| **LinearRegression** | **7.38** | — |
| RandomForest | 7.80 | `max_depth=10, n_estimators=100` |
| DecisionTree | 9.27 | `max_depth=5, min_samples_split=5` |
| DummyRegressor (baseline) | 17.03 | — |

LinearRegression won, then was evaluated once against the untouched test set for a final read:

**Test RMSE: 7.19 · Test R²: 0.81**

So the model's typical prediction is off by about 7 points on a 0–100 exam score, and it explains roughly 81% of the variance in scores — a little over twice as accurate as just guessing the class average every time.

## Project structure

```
exam_scores_prediction/
├── README.md
├── requirements.txt
├── setup.sh
├── data/
│   └── student_habits_performance.csv
├── notebooks/
│   └── notebook.ipynb          # EDA, modeling, export — the whole pipeline
├── model_exports/
│   └── best_model.pkl          # exported Pipeline (encoder + model)
└── app/
    └── app.py                  # Streamlit app
```

## Setup

```bash
git clone <this-repo>
cd exam_scores_prediction
source setup.sh        # creates .venv (if missing), activates it, installs requirements.txt
```

## Usage

**Notebook** — open `notebooks/notebook.ipynb` in Jupyter/VS Code and run top to bottom. Re-running it regenerates `model_exports/best_model.pkl`.

**App**
```bash
cd app
streamlit run app.py
```

## Notes

- Model selection deliberately avoids scoring candidates on the test set — that number is reserved for a single, final report on the already-chosen model. Comparing on the test set and then reporting from the same data would quietly bias the reported score toward whichever model got lucky on that split.
- `RandomForestRegressor`/`DecisionTreeRegressor` are seeded with `random_state=42` — without it, which model "wins" this comparison shifts between runs, since RandomForest's own CV score sits close enough to LinearRegression's (7.38 vs. 7.80) that noise alone can flip the ranking.

## Possible next steps

- Try feature interactions or a couple more of the 16 available columns (e.g. `diet_quality`, `internet_quality`) to see if they move the needle beyond the current 5.
- Compare against a properly scaled LinearRegression / Ridge, since scaling was skipped here in favor of the tree-based models in the comparison.
- Add a couple of unit tests around the export bundle (right shape in, right shape out) if this ever needs to run somewhere other than my machine.

## Stack

Python 3.14 · pandas · scikit-learn · matplotlib/seaborn · Streamlit — see `requirements.txt` for exact versions.
