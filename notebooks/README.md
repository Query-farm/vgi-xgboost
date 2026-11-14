# XGBoost SQL notebook

[01 · Evaluate predictions before explaining them](01_evaluate_predictions_before_explaining_them.ipynb)
trains a regressor on authored route data, evaluates a held-out split, and checks
that SHAP contributions reconstruct predictions. It includes visible Haybarn SQL,
two charts and a one-parameter depth experiment. All data are synthetic; no service
account, real deliveries or external dataset download is involved.

## Run the saved example

Install [uv](https://docs.astral.sh/uv/) and Git. On macOS, XGBoost also needs the
OpenMP runtime (`brew install libomp`). From this directory:

```sh
uv sync --locked
uv run --locked python execute.py
```

Only this complete `notebooks/` directory is needed. The runner creates a temporary
kernel specification, uses this directory as its working directory, and atomically
replaces the saved notebook only after all cells succeed. It does not install a
global Jupyter kernel. Generated charts and experiment copies are ignored under
`validation/`. For VS Code, choose `notebooks/.venv/bin/python` as the Python kernel.
The connection and analytical SQL stay visible; support code is folded.

To retain the saved default while testing the proposed experiment:

```sh
uv run --locked python execute.py --parameters '{"MAX_DEPTH":6}' --output-dir validation/depth-6
```

Overrides require a distinct output directory. Unknown parameters and invalid
values fail explicitly. Restart the kernel and run all after editing parameters;
rerunning only a fit cell can collide with SQL tables or retain worker state.
Reset `MAX_DEPTH = 2` to reproduce the default.

## Portable, pinned worker

The literal visible `ATTACH` runs:

```text
uv run --no-project --python 3.13 python worker_launcher.py
```

`worker_launcher.py` fetches immutable public
[worker revision 30962ca341b81489ddc23d31807d87d07e996d3e](https://github.com/Query-farm/vgi-xgboost/tree/30962ca341b81489ddc23d31807d87d07e996d3e),
verifies the revision and Python source hashes, and executes its unchanged
`xgboost_worker.py`. This revision has no native release assets and its wheel omits
the entry point, so an ordinary `uvx --from` entry-point command is not available.
No locally built worker or parent checkout is required. Git/uv downloads need
network access on the first run. The content cache uses `XDG_CACHE_HOME` or
`~/.cache/vgi-xgboost-notebooks/`, a bounded install lock and atomic publication;
concurrent starts have been tested against a cold cache.

The worker runs in a separate uv environment with Python 3.13, VGI Python 0.51.1,
VGI RPC 0.53.1, PyArrow 26.0.1, XGBoost 3.3.0, scikit-learn 1.9.0, NumPy 2.5.0,
pandas 3.0.3 and SciPy 1.18.0. The notebook environment is locked in `uv.lock` and
uses Haybarn 2.0.0rc11. Upgrading requires changing the Git/dependency pins and
rerunning both default and experiment, rather than silently following `main`.

A fresh temporary `XGBOOST_MODELS_DIR` isolates model files. The worker stores a
native XGBoost artifact and metadata there when given `model_name`. Cleanup closes
the connection, deletes that temporary directory and restores the previous process
environment. If interrupted, restart the kernel; orphaned temporary directories
can be removed after their processes exit. Existing model registries are untouched.

## Data and interpretation

The visible SQL defines 240 route IDs and three features: distance in kilometres,
number of stops and a numeric rush flag. The synthetic target in minutes is:

```text
12 + 0.8*distance + 0.04*distance² + 2.5*stops + 8*rush + 5*sin(route_id*1.73)
```

Every fifth ID is held out: 192 training rows and 48 held-out rows. The ID, split
and target are excluded from predictors. No real-world sampling or temporal
forecasting claim is made. The baseline uses the training mean for both splits.
The experiment changes depth 2 to 6 while retaining 80 trees, learning rate 0.1,
seed 0 and the exact input relations. Inspecting the held-out result for tuning
uses up its role as an untouched final evaluation set.

SHAP values are contributions to this fitted squared-error regressor's raw margin,
which equals its prediction in minutes. One repeated base value plus the sum of
three contributions reconstructs each held-out prediction within 0.0001 minute.
This is an arithmetic consistency check, not a causal or accuracy guarantee.
Classification probability explanations require a different interpretation.

The official Query.Farm Strata Sun wordmark and VGI logo are bundled unchanged in
`assets/`; the wordmark links to Query.Farm at both ends of the notebook.
