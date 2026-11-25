# uplift-class-transformation (UP02)

**Question:** Does a single model on the transformed outcome match or beat the two-model T-learner on the same test split?

Under randomization with `p = P(T=1)`, the transformed outcome `Z = Y·T/p − Y·(1−T)/(1−p)` satisfies `E[Z | X=x] = τ(x)`, so a regressor fit on `Z` estimates the uplift directly, with one model instead of two. This repository implements that estimator and compares it with the T-learner from UP01 using the same metric and split.

## Decision supported

Whether the simpler single-model estimator is good enough to rank customers for treatment, or whether the two-arm model is worth its extra complexity.

## Data

- **Dataset:** Criteo uplift benchmark, Kaggle slug [`arashnic/uplift-modeling`](https://www.kaggle.com/datasets/arashnic/uplift-modeling). Real randomized incrementality-test data with 12 anonymized features, a binary `treatment`, and a binary `conversion` outcome.
- **License:** Kaggle metadata says CC0. The upstream Criteo terms say CC BY-NC-SA 4.0. This conflict is unresolved, so no figures are committed. See [data/README.md](data/README.md).
- **Read path:** `/kaggle/input/uplift-modeling/` inside a Kaggle Notebook. Nothing is downloaded or committed.
- **Propensity:** `p` is the treated share in the training split (estimated, about the randomized rate). A known value can be passed to `ClassTransformation(propensity=...)`.
- **Split:** no dates, so a random 70/30 split stratified by `treatment`. Seed 42.

## Method

| Element | Choice |
|---|---|
| Transformed outcome | `Z = Y·T/p − Y·(1−T)/(1−p)` |
| Model | `HistGradientBoostingRegressor(random_state=42)` fit on `Z` |
| Reference | T-learner with `StandardScaler` → `LogisticRegression(random_state=42)` per arm (same as UP01) |

## Metrics

- **Qini coefficient** on the same test split for both models, with random targeting as the baseline.
- The notebook writes `reports/up02_comparison.csv` and a Qini-curve figure.

## Results

**Not yet run on the real dataset.** The notebook has not been executed locally. The comparison table will be filled in after a Kaggle run.

## Tests

The tests use tiny hand-built fixtures. They check the transformed-outcome values, the identity that `mean(Z)` equals the difference in arm means when `p` is the observed treated share, the fit target, and the Qini hand calculation.

## Limitation

The transformed outcome has high variance when `p` is extreme, and the randomized rate in this dataset is about 85% treated, which makes the control-side weight `1/(1−p)` large. This can make the single-model estimate noisier than the T-learner, so the comparison depends on the split size.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Then open `notebooks/02_class_transformation.ipynb` in a Kaggle Notebook with `arashnic/uplift-modeling` attached and choose **Run All**.

## Layout

```text
src/uplift_transform/   transformed outcome, class-transformation model, T-learner reference, Qini metrics
notebooks/              02_class_transformation.ipynb (Run All on Kaggle)
tests/                  transformed-outcome identities, fit target, Qini hand calculation
reports/figures/        exported figures (empty until a licensed run is published)
data/                   git-ignored except data/README.md
```
