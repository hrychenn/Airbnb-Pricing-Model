# --- Stage 1: build the React frontend ---
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build          # → /frontend/dist

# --- Stage 2: Python runtime that serves the API + built frontend ---
FROM python:3.12-slim AS runtime
WORKDIR /app

# System libs XGBoost needs
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

# App code + the trained artifacts (the two .joblib files)
COPY api/ ./api/
COPY src/ ./src/
COPY models/artifacts/xgb_final.joblib models/artifacts/app_metadata.joblib ./models/artifacts/

# Built frontend from stage 1
COPY --from=frontend /frontend/dist ./frontend/dist

# Bind to $PORT (Render injects it at runtime; 7860 is the local/default fallback)
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
