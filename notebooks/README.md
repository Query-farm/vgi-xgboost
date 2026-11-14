# XGBoost SQL examples

Three executed stories use public worker release **v0.1.0**:

- [Evaluate a prediction before explaining it](01_evaluate_predictions_before_explaining_them.ipynb): preserve a fixed 192/48 route split, inspect the saved model, compare errors and reconcile SHAP contributions. Change `MAX_DEPTH` from 2 to 6.
- [Feature importance answers more than one question](02_three_questions_about_features.ipynb): compare reported split gain with held-out permutation loss, keep an unused constant control, and inspect partial dependence. Change `SWEEP_FEATURE` from `distance_km` to `stops`.
- [A selection score is not the final evaluation](03_selection_scores_and_new_rows.ipynb): generate a synthetic population once, reconcile out-of-fold predictions with fold scores, select among three depths and evaluate the selected model on reserved rows. Change `CV_FOLDS` from 3 to 6.

All inputs and outcomes are explicitly synthetic. The first two stories use an authored route-duration formula; the third calls the released worker's regression generator with a fixed seed. No private data, account credentials, real delivery records or remote model writes are involved.

## Run from this folder or a single notebook

Install Git and [uv](https://docs.astral.sh/uv/). XGBoost also needs a working OpenMP runtime; on macOS the usual prerequisite is `brew install libomp`.

```bash
uv sync --locked
uv run --locked vgi-notebooks 01_evaluate_predictions_before_explaining_them.ipynb
uv run --locked vgi-notebooks 02_three_questions_about_features.ipynb
uv run --locked vgi-notebooks 03_selection_scores_and_new_rows.ipynb
```

The local lock uses shared **v0.3.38** with `[runner]`. A Colab badge opens one notebook and its setup installs only the shared base package in that kernel. `prepare("xgboost")` retrieves two companions from immutable public commit `1a96d927261600930e8f683152c3c95b7ceb95ab` and verifies both SHA-256 hashes. The helpers do not require a surrounding source checkout or root virtual environment. Restart and Run All after editing parameters. For a local editor:

```bash
uv run --locked python -m ipykernel install --user --name xgboost-examples \
  --display-name "Python (vgi-xgboost notebooks)"
```

Every notebook keeps this literal connection visible:

```sql
INSTALL vgi FROM community;
LOAD vgi;
ATTACH 'xgboost' (TYPE vgi, LOCATION 'uv run --no-project --python 3.13 python xgboost_release.py v0.1.0');
```

The release has no executable assets or console script, and its Python distribution omits the root entry point. The small **worker-specific** launcher fills that packaging gap: it clones release tag v0.1.0, verifies immutable commit `eb4f1991e788801aa11d58e231f019f0cf613f44`, publishes the source cache atomically under a bounded lock, verifies cached worker/lock hashes on reuse, and installs the release's own unchanged `uv.lock` with `uv sync --frozen --no-dev --no-install-project`. It then executes the unchanged `xgboost_worker.py` using that isolated Python 3.13 environment. Explicit Python avoids interpreting the script's looser PEP 723 dependency header. There are no transitive version overrides or compilation steps. Install diagnostics stay on stderr; Arrow protocol uses stdout.

The source/environment cache is under `${XDG_CACHE_HOME:-~/.cache}/vgi-xgboost-releases/<commit>`. GitHub, Python packages and the published VGI extension require internet on first use. A local model registry is created separately for each notebook and deleted at cleanup; existing user models are not opened. Shared Haybarn 2.0.0rc11 setup creates an empty temporary SQL-secret directory and disables persistent secrets before ATTACH. Cleanup checks the secret inventory, closes the session, removes only this run's model directory, and restores its environment settings.

## Separate parameter experiments

```bash
uv run --locked vgi-notebooks 01_evaluate_predictions_before_explaining_them.ipynb \
  --parameters '{"MAX_DEPTH":6}' --output-dir artifacts/depth-six
uv run --locked vgi-notebooks 02_three_questions_about_features.ipynb \
  --parameters '{"SWEEP_FEATURE":"stops"}' --output-dir artifacts/stops
uv run --locked vgi-notebooks 03_selection_scores_and_new_rows.ipynb \
  --parameters '{"CV_FOLDS":6}' --output-dir artifacts/six-folds
```

Overrides require a separate destination, preserving the saved defaults. Experiment copies retain branding and documentation; SVG charts render natively in Jupyter/Colab, while saved defaults also carry matching GitHub image links. The presentation helper only draws returned SQL values and validates parameters. Training, selection, scoring, joining and aggregate arithmetic remain visible in SQL.

## Interpretation and scope

The route target is `12 + 0.8*distance_km + 0.04*distance_km² + 2.5*stops + 8*rush + 5*sin(route_id*1.73)` minutes. Every fifth ID is held out. Increasing depth can improve training error while worsening held-out error. Additive SHAP reconstruction verifies the model's arithmetic, not prediction accuracy or causality.

The feature story retains all four inputs, including a constant that has zero split gain and zero permutation loss. Gain shares normalize reported training split gain; they are not shares of accuracy. The permutation scorer is negative RMSE, so a positive score drop means an RMSE increase in minutes. Error bars describe shuffle variability, not confidence intervals. Partial dependence uses the training-feature background and actual returned replacement grid; constructed feature combinations are not observed interventions.

The selection story reserves 48 rows before any tuning. Its 192 development rows are ordered by ID; the released worker uses integer-fold, unshuffled KFold for regression. Three and six folds both have equal populations, allowing pooled out-of-fold RMSE to reconcile with the root mean square of fold RMSEs. The plain mean fold RMSE is different and is the grid's selection criterion. The grid searches depths 1/3/6, then refits the selected model on all development rows. In this release, registry `train_score` means in-sample R² for direct fit but best CV score for search, so the example uses explicitly named leaderboard scores. Repeatedly comparing experiments after opening the reserved set spends its untouched-test role.

Actual runtime metadata contains **24 named functions**; six also appear as same-name tables and count once. No views or COPY formats exist. The three stories cover **13/24** surfaces, exceeding the required 12. Unused dataset readers, classifier/random-forest variants, randomized search and registry deletion remain in the denominator; descriptive metadata or empty diagnostics are not counted as narrative coverage. Fit/search can write a local registry and `drop_model` can delete there. No authenticated APIs are excluded. `california_housing` may fetch a public dataset when scanned; it was only schema-bound for the inventory.

Release v0.1.0 peels to the same worker revision as current public main at baseline, so released and current snapshots use that identical full SHA and exact runtime evidence. [coverage/](coverage/) holds complete catalogs, full arguments and representative dynamic output schemas; [coverage.json](coverage.json) links interpreted successful SQL to named surfaces. Coverage does not claim every overload, prediction mode, estimator or dataset was exercised.

Validation includes all defaults, three detached experiments, 37 SQL relationship checks plus registry assertions, nine runner guards, one fresh compatible Python3.12 kernel with eight libraries already imported, visual/output review and coverage validation. Hosted Colab itself was not manually operated. Worker source and root locks are unchanged; [green source CI](https://github.com/Query-farm/vgi-xgboost/actions/runs/43621934758) is reused. The separate pre-existing deployment workflow failure is outside this notebook publication. See the [shared standard](STANDARD.md).
