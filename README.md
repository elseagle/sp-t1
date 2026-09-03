# Freight Rate Prediction

Predicts the posted rate for truckload freight from lane, equipment, weight, date, and market signals. Trained on ten months of labelled loads and used to forecast the following two months.

## Results

Validation is time-ordered: fit on January to August, score on September to October, a two-month forward window matching the prediction task. MAE is mean absolute error in dollars on the raw rate; MAPE is mean absolute percentage error, the same miss expressed as a percentage of the true rate. `clean` excludes the 1.4% of loads whose recorded rate is corrupted, `all` includes them.

| Model | MAE (clean) | MAPE (clean) | MAE (all) | MAPE (all) | Fold-mean MAPE |
|---|---:|---:|---:|---:|---:|
| Lane-median baseline | 82.4 | 3.48% | 138.0 | 5.98% | 4.46% |
| Ridge, log rate | 50.3 | 2.16% | 106.2 | 4.66% | 2.72% |
| Elastic net, log rate | 54.3 | 2.28% | 110.1 | 4.77% | – |
| Hist. gradient boosting | 53.2 | 2.17% | 109.1 | 4.65% | 2.71% |
| LightGBM | 52.2 | 2.09% | 108.0 | 4.58% | 2.59% |
| **Ridge + LightGBM blend** | **48.4** | **2.01%** | **104.4** | **4.50%** | **2.33%** |

Fold-mean MAPE averages three forward-chaining windows (May-Jun, Jul-Aug, Sep-Oct). The blend is best in every window and is the submitted model.

## Approach

**Data quality.** Four defects were found and handled in `freight_rate.data`:
- Negative weights are sign flips of valid values and are restored with an absolute value.
- Missing weights and market index values are imputed (equipment median, same-day mean) with indicator flags.
- About 1.4% of rates sit at three times or a quarter of their lane median with nothing in between. They are excluded from fitting with a factor-of-two rule against the lane, or distance-band, median rate per mile.
- Recorded distance is floored at 70 miles on short lanes. This is not an error and is left alone; circuity is kept as a feature.

**Features.** Log distance, great-circle distance and circuity, equipment, origin and destination city and coordinates, weight, market index, quote signal, day of week, and a linear time trend. The trend matters: rates rise about 0.5% per month independent of the market index, and without it every model under-predicts the later months.

**Models.** All models predict log rate. Ridge uses a spline in log distance and one-hot cities with unknown categories ignored, so unseen cities fall back to their coordinates. LightGBM captures the remaining non-linear effects. The final model averages the two in log space.

**Validation.** A third of validation lanes and eight cities are absent from training, so the holdout checks unseen-lane error separately; it matches seen-lane error.

Full detail is in the two notebooks: [`01_eda.ipynb`](notebooks/01_eda.ipynb) covers the data and [`02_model_selection.ipynb`](notebooks/02_model_selection.ipynb) covers the comparison, residuals, and what the models learned.

## Layout

```
data/                        source data and prediction templates
notebooks/01_eda.ipynb       exploratory analysis
notebooks/02_model_selection.ipynb
src/freight_rate/
  data.py                    loading, cleaning, outlier detection
  features.py                feature construction
  models.py                  baseline, linear, boosted, and blend models
  evaluation.py              metrics and time-based splits
  train.py                   model comparison and final fit  (freight-train)
  predict.py                 validation and December predictions (freight-predict)
tests/                       unit tests for cleaning, features, and models
reports/                     comparison tables, figures, and the report
score.py                     provided validator and chart generator
validation_predictions.csv   final predictions
```

## Setup

Python 3.12 is pinned in `.python-version`; Poetry manages the environment.

```bash
pyenv install 3.12.4
poetry install
```

Without Poetry, a plain virtual environment works too:

```bash
python -m pip install -r requirements.txt -e .
```

LightGBM needs OpenMP on macOS (`brew install libomp`).

## Usage

Compare candidates on the holdout and save the final model:

```bash
poetry run freight-train --models lane_median ridge elastic_net hist_gb lightgbm blend
poetry run freight-train --models blend --folds
poetry run freight-train --models blend --final-model blend
```

Generate the validation predictions and complete the December inputs, then run the provided scorer:

```bash
poetry run freight-predict
poetry run python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Run the tests and re-execute the notebooks:

```bash
poetry run pytest
poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

## Outputs

- `validation_predictions.csv`: `load_id,predicted_rate` for all 12,000 validation loads.
- `data/december_chart_inputs.csv`: the fixed Lexington to Fort Wayne lane priced for every day of December 2025.
- `scorer_results/candidate_december.png`: the chart produced by `score.py`, also copied to `reports/figures/`.
- `reports/holdout.csv`, `reports/folds.csv`: model comparison tables.

## Notes

- The December rows lack a market index and quote signal. They are filled from the same-day mean across December validation loads and the Dry Van training median respectively; see `predict.build_december_frame`.
- Tree models cannot extrapolate the time trend past the last training day; the blend inherits half of Ridge's linear extrapolation, which is deliberate for a two-month horizon and would need revisiting for longer ones.
- Corrupted rates are presumably present in the validation labels at the same rate as in training. That places a floor of roughly 2.5 points of MAPE that no model can remove.
