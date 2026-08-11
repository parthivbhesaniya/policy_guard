FROM python:3.11-slim

# libgl1/libglib2.0-0: runtime libs opencv-python (pulled in by rapidocr-onnxruntime) needs.
# All Python deps ship manylinux wheels, so no compiler toolchain is required.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[ui]"

COPY data ./data

EXPOSE 8000 8501

CMD ["uvicorn", "policyguard.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
