FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# vgi-python / vgi-rpc are published on PyPI, so install everything directly —
# no vendoring or local wheel building required.
RUN pip install --no-cache-dir \
        "vgi-python[http,oauth]>=0.8.2" \
        "vgi-rpc[sentry]>=0.20.4" \
        "xgboost>=2.0" \
        "scikit-learn>=1.5" \
        numpy \
        pandas \
    && pip uninstall -y pip

COPY vgi_xgboost /app/vgi_xgboost
COPY xgboost_worker.py /app/xgboost_worker.py
COPY serve.py /app/serve.py

ARG GIT_COMMIT=unknown
ENV VGI_XGBOOST_GIT_COMMIT=${GIT_COMMIT}
ENV SENTRY_RELEASE=${GIT_COMMIT}

# Where the local-disk model registry persists (mount a Fly volume here in prod).
ENV XGBOOST_MODELS_DIR=/data/models

EXPOSE 8000
CMD ["sh", "-c", "python /app/serve.py --host 0.0.0.0 --port ${PORT:-8000}"]
