# 🛒 Product Return Prediction — Streamlit App

An interactive Streamlit app built from the
[e-commerce-product-return](https://github.com/Shinearushi525/e-commerce-product-return-)
project. It trains a classifier on 15,000 synthetic Indian e-commerce orders and predicts
whether a new order is likely to be **returned**.

## What's included
- `train_model.py` — feature engineering + trains 5 models (Logistic Regression, Decision
  Tree, Random Forest, Gradient Boosting, XGBoost), picks the best by ROC-AUC, and saves
  all artifacts (`model.pkl`, `scaler.pkl`, `encoders.pkl`, `feature_cols.pkl`,
  `results.pkl`, `feature_importance.pkl`).
- `app.py` — the Streamlit app with 4 pages:
  1. **Overview & EDA** — return-rate charts by category, discount, payment method,
     delivery days, a correlation heatmap, and return-reason breakdown.
  2. **Model Performance** — leaderboard of all trained models and feature importances.
  3. **Predict a Single Order** — a form to score one order interactively, with a risk gauge.
  4. **Batch Prediction** — upload a CSV of orders and download the scored results.
- `ecommerce_product_returns.csv` — the dataset.
- `requirements.txt` — pinned dependencies.

> **Note on accuracy:** the original README quoted ~82% accuracy / ~89% ROC-AUC. Retraining
> from the raw CSV in this repo, the real numbers come out lower (roughly 58-64% ROC-AUC
> depending on model) — the dataset is synthetic and only weakly predictive. The app reports
> whatever `train_model.py` actually measures, not the earlier marketing numbers, so what you
> see in the "Model Performance" tab is the honest result.

## Run it locally
```bash
pip install -r requirements.txt
python train_model.py      # generates the .pkl artifacts (run once)
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## Deploy it live (free, ~5 minutes) — Streamlit Community Cloud

1. **Push this folder to a GitHub repo** (a new one, or update your existing
   `e-commerce-product-return-` repo) with all the files above, **including the generated
   `.pkl` files** (or let step 3 generate them — see note below).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**, pick your repo/branch, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` and starts the app —
   you'll get a public URL like `https://your-app-name.streamlit.app`.

**About the `.pkl` files:** Streamlit Cloud's build step won't run `train_model.py` for you.
Either:
- commit the `.pkl` files you already generated locally, **or**
- add one line at the top of `app.py`'s artifact loader to call `train_model.main()`
  automatically the first time no `model.pkl` exists (simplest is to just commit the
  `.pkl` files — they're small, a few MB at most).

## Other deployment options
- **Hugging Face Spaces** (free): create a Space with SDK = Streamlit, push the same files.
- **Render / Railway**: add a `Procfile` with
  `web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
- **Docker**: 
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY . .
  RUN pip install -r requirements.txt
  RUN python train_model.py
  EXPOSE 8501
  CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
  ```

## Tested
This app was run locally end-to-end in this environment: training completed successfully
and the Streamlit server started and responded with HTTP 200 / healthy status with no
runtime errors.
