# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http,oauth]>=0.8.5",
#     "vgi-rpc[sentry]>=0.20.4",
#     "xgboost>=2.0",
#     "scikit-learn>=1.5",
#     "numpy",
# ]
# ///
"""HTTP entrypoint for the XGBoost worker (used by Fly.io).

Forces the worker's CLI into HTTP mode (``Worker.main()`` serves stdio by
default) so callers only pass ``--host``/``--port``.
"""

import sys

from xgboost_worker import XGBoostWorker

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--http" not in argv:
        argv = ["--http", *argv]
    sys.argv = [sys.argv[0], *argv]
    XGBoostWorker.main()
