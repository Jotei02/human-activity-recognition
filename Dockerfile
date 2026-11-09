FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Serving code + the deployable model artifacts (ONNX model and fitted scaler)
COPY inference/ inference/
COPY utils/ utils/
COPY model/mlp_model.onnx model/scaler.pkl model/

EXPOSE 8000

# One uvicorn process: the service keeps per-user buffers in process memory, so
# multiple uvicorn workers would split a user's samples across separate buffers.
CMD ["uvicorn", "inference.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
