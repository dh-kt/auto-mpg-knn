[![CI](https://github.com/dh-kt/auto-mpg-knn/actions/workflows/ci.yml/badge.svg)](https://github.com/dh-kt/auto-mpg-knn/actions/workflows/ci.yml)

# Auto MPG KNN regression demo
Implemented a reproducible K‑Nearest Neighbors regression pipeline on the ISL Auto dataset: cleaned notebook, KNN tuned via cross‑validation (best_k = 21), model artifacts, and a production-ready FastAPI prediction endpoint packaged with Docker. CI (GitHub Actions) runs tests; model artifacts are published as a GitHub Release for reviewers.
Technical highlights (short bullets you can include in README or profile)
Trained a distance‑weighted KNN (best_k = 21); test RMSE ≈ 4.10, R² ≈ 0.67.
Reproducible pipeline: cleaned Colab notebook, saved artifacts (scaler.joblib, knn_weighted.joblib), and model_data/summary.json.
Production-ready serving: FastAPI endpoint (/predict) validated locally and packaged with Docker.
Continuous integration: GitHub Actions workflow runs pytest with CI-friendly dummy artifacts.
Release assets: model binaries uploaded to GitHub Releases with SHA256 checksums for integrity.

Cleaned reproduction of ISL Chapter 3 using the Auto dataset.
- End-to-end: EDA в†’ preprocessing в†’ KNN CV в†’ distance-weighted KNN в†’ evaluation в†’ API в†’ Docker.
- Use `model_data/` to store trained artifacts (scaler.joblib, knn_weighted.joblib).

Quickstart (local):
1. Build image: `docker build -t auto-mpg-knn:latest .`
2. Run container (mount model_data): `docker run --rm -p 8000:8000 -v "<ABS_PATH>/model_data:/content/model_data" auto-mpg-knn:latest`
3. Test:
   - GET `http://127.0.0.1:8000/`
   - POST `http://127.0.0.1:8000/predict` with JSON body:
     `{"displacement":150,"horsepower":95,"weight":2000,"acceleration":15.5,"cylinders":4}`

Notes:
- Consider adding `model_data/` to `.gitignore` if prefer to not commit model binaries.
- See `notebooks/01_auto_knn_clean.ipynb` for the cleaned analysis.

