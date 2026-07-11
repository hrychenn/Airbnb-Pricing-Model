# Deploying the dashboard (free) to Render

The app runs as a **single Docker container**: FastAPI serves both the `/api/*` routes
and the built React app on one port. This has been built and verified locally
(`docker build` → `docker run`), including under a **512 MB memory cap** to match
Render's free tier — it peaks at ~156 MB.

> **Why Render, not HuggingFace Spaces?** HF now gates the **Docker** SDK behind a paid
> plan (only Gradio/Static are free). Render's free tier runs our Docker container as-is,
> needs no credit card, and gives a public URL. Trade-off: a free service sleeps after
> 15 min idle and takes ~50 s to wake on the next click.

**What ships in the image** (`Dockerfile` + `.dockerignore`): `api/`, `src/`, the built
`frontend/`, and the two `.joblib` artifacts. SHAP values come from XGBoost's native
`pred_contribs` (no `shap` lib), so the runtime is slim — no data CSVs, no training deps.

---

## Prerequisites (already done in this repo)
- `Dockerfile`, `.dockerignore`, `requirements-deploy.txt`, `render.yaml` — present.
- `models/artifacts/xgb_final.joblib` + `app_metadata.joblib` — **committed** (they're
  un-ignored in `.gitignore` so Render's build has the model). ~5 MB total.
- Everything pushed to GitHub (`main`).

If you re-train, rebuild the metadata and re-commit the artifacts:
```bash
python -m src.build_app_metadata   # refreshes app_metadata.joblib
git add models/artifacts/*.joblib && git commit -m "Update model artifacts" && git push
```

---

## Deploy (you do this — needs your Render login)

### Option A — Blueprint (uses `render.yaml`, one click)
1. Go to https://dashboard.render.com → **New** → **Blueprint**.
2. Connect your GitHub and pick the **Airbnb-Pricing-Model** repo.
3. Render reads `render.yaml`, shows a `web` service on the **Free** plan → **Apply**.
4. First build takes ~5–10 min (it builds React, then the Python image). When it's
   live, the URL is `https://airbnb-dynamic-pricing.onrender.com` — that's your link.

### Option B — Manual (if you'd rather click through)
1. **New** → **Web Service** → connect the repo.
2. Render detects the `Dockerfile`. Set **Instance Type: Free**. Leave the rest default
   (Render injects `PORT`; the container binds to it automatically).
3. **Create Web Service** → wait for the build → open the URL.

---

## After it's live
- **Put the URL on your resume.** First click after idle spins the service up (~50 s);
  add a note like "may take up to a minute to wake" if you want to set expectations.
- **Auto-deploy is on** (`autoDeploy: true`): every push to `main` rebuilds and redeploys.
- **Health check:** Render pings `/api/health`; the service is marked live once it responds.

## Local sanity check before pushing
```bash
docker build -t airbnb-pricing .
docker run --rm -p 7860:7860 airbnb-pricing        # open http://localhost:7860
# simulate Render's memory limit:
docker run --rm --memory=512m -p 7860:7860 airbnb-pricing
```
