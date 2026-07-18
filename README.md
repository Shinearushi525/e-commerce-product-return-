<div align="center">

# 🛒 Return Risk AI — E-Commerce Product Return Prediction

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Models-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-016093?style=for-the-badge)](#)
[![Dataset](https://img.shields.io/badge/Dataset-15%2C000%20Orders-A855F7?style=for-the-badge&logo=kaggle&logoColor=white)](#)
[![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![Status](https://img.shields.io/badge/Status-✅%20Live-2ECC71?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](#)

<br/>

> **15,000 real-world-style e-commerce orders** · **5 ML models** · **return-probability scoring** · **batch CSV scoring** · **interactive risk dashboard**

<br/>

### 🔬 See It In Action

| | |
|---|---|
| 📦 **Input Order** | Electronics · Appliances · ₹2,499 · 30% discount · Cash on Delivery · 9-day delivery |
| 🔴 **Predicted Verdict** | `Will Return` — return probability **63.6%** |
| 📊 **Risk Level** | 🔴 **HIGH RISK** |
| 💬 **Model Used** | Gradient Boosting (auto-selected as best performer by ROC-AUC) |
| ⚠️ **Signals Flagged** | High discount · Late delivery · Low seller rating · COD payment |
| 💡 **Action** | Suggests limiting COD, pre-dispatch quality check, or a proactive exchange-friendly policy |

</div>

---

## 🌐 Live Demo

**👉 [commerce-product-return--arushigarg525.replit.app](https://commerce-product-return--arushigarg525.replit.app/)**

> Hosted on Replit's free tier — if the app is asleep, the first load may take a few seconds to wake up.

---

## 📸 See It In Action

<div align="center">

**Predict a Single Order**

![Prediction form](predict-form.png)

**Instant Risk Verdict**

![Risk gauge](risk-gauge.png)

</div>

---

## 🎯 The Problem

Product returns are one of the most expensive, least-understood costs in e-commerce.

<div align="center">

### 📊 E-Commerce Returns Reality

| 🔢 Statistic | 💡 Fact | 😟 Impact |
|-------------|---------|----------|
| **20–30%** | Average return rate for online fashion & electronics | Billions in reverse-logistics cost annually |
| **~$550B** | Estimated annual cost of returns to US retailers alone | Larger than many entire industries |
| **Most returns** | Are predictable from order-level signals *before* dispatch | Discount level, delivery time, payment method, category |
| **Reactive ops** | Most sellers only act *after* a return is filed | Too late to prevent the cost |
| **High-discount + COD** | Consistently correlate with higher return rates | Exactly the orders sellers ship blind |

</div>

Instead of reacting to returns after they happen, this system scores **every order at checkout time**, flags high-risk orders before they ship, and gives sellers a concrete reason and next action — not just a number.

> *A return prevented is cheaper than a return processed.*

---

## ✨ What Makes This Unique

Most return-prediction demos stop at "here's an accuracy number." This system goes further:

```
TYPICAL PROJECT          THIS PROJECT
───────────────          ────────────────────────────────────────────
Predict label      →     ✅ Predict return probability (0–100%)
                   →     ✅ Compare 5 models head-to-head, auto-select best
                   →     ✅ Live interactive risk gauge per order
                   →     ✅ Batch-score entire CSVs of orders at once
                   →     ✅ Full EDA dashboard (category, discount, delivery, payment)
                   →     ✅ Feature importance ranking (what actually drives returns)
                   →     ✅ Actionable recommendations per high-risk order
                   →     ✅ Self-trains on launch — no environment-specific model files
```

<br/>

### 🏗️ The Core Features

| # | Feature | Description | Tech Used |
|---|---------|-------------|-----------|
| 1 | 🔮 **Single Order Predictor** | Interactive form scores one order instantly with a live risk gauge | Streamlit + best trained model |
| 2 | 📁 **Batch Prediction** | Upload a CSV of orders, download every order scored with return probability | pandas + model inference |
| 3 | 📊 **EDA Dashboard** | Return rate broken down by category, discount tier, payment method, delivery days | Plotly |
| 4 | 🤖 **Model Leaderboard** | 5 models compared on Accuracy, F1, ROC-AUC, Precision, Recall | scikit-learn + XGBoost |
| 5 | 📈 **Feature Importance** | Ranks which order attributes matter most for return risk | Random Forest importances |
| 6 | ⚠️ **Risk Tiering** | Categorizes every prediction into LOW / MEDIUM / HIGH risk with next-step guidance | Threshold logic |
| 7 | 🔁 **Self-Training Pipeline** | Trains itself fresh on first launch in whatever environment it's deployed to | Custom pipeline, no stale pickles |

---

## 📊 Dataset

<div align="center">

### 📦 Dataset At A Glance

| 🏷️ Property | 📋 Details |
|-------------|-----------|
| **Name** | E-Commerce Product Returns |
| **Total Orders** | 15,000 orders |
| **Region** | Indian e-commerce market (INR pricing, Indian cities) |
| **Target** | `is_returned` (binary) |
| **Categorical Features** | Customer type, product category & subcategory, payment method, shipping type, warranty, size availability, city |
| **Numeric Features** | Customer age, product price, discount %, order quantity, delivery days, seller rating, customer rating |
| **Preprocessing** | Feature engineering: actual post-discount price, high-discount flag, late-delivery flag, festival-season flag, COD flag |

</div>

### 📋 Return Reason Breakdown

The dataset also captures **why** returned orders came back — size/fit issues, quality concerns, "changed mind," damaged in transit, and more — visualized as an interactive pie chart in the app's EDA tab.

### 📁 Key Dataset Columns

| Column | Type | Description | Example |
|--------|------|--------------|---------|
| `product_category` | `string` | High-level product category | `Electronics` |
| `discount_percent` | `float` | Discount applied at checkout | `30` |
| `delivery_days` | `int` | Days taken to deliver | `9` |
| `payment_method` | `string` | How the order was paid | `Cash on Delivery` |
| `seller_rating` | `float` | Seller's average rating (2.5–5.0) | `3.4` |
| `is_returned` | `int` | Target label (0/1) | `1` |

---

### Feature Engineering at a Glance

| Feature Group | Features | Count |
|---------------|----------|-------|
| Raw numeric | Age, price, discount, quantity, delivery days, ratings | 7 |
| Engineered flags | High discount, late delivery, low seller/customer rating, premium product, festival season, COD | 6 |
| Encoded categoricals | Customer type, category, subcategory, payment, shipping, warranty, size, city | 8 |
| **Total model features** | | **~23** |

---

## 🤖 ML Models Compared

Five models trained and evaluated head-to-head:

| Model | Type | Strength |
|-------|------|----------|
| 🔵 Logistic Regression | Linear | Fast, interpretable baseline |
| 🟢 Decision Tree | Tree-based | Captures simple non-linear splits |
| 🟡 Random Forest | Ensemble | Robust to noise, gives feature importance |
| 🔴 **Gradient Boosting** | **Boosting** | **Best overall performer in this app** |
| 🟣 XGBoost | Boosting | Strong on imbalanced classes, highest recall |

> All models trained on an 80/20 stratified train/test split, evaluated on Accuracy, F1, ROC-AUC, Precision, and Recall. The best model is auto-selected by ROC-AUC — no manual step required.

---

## 📈 Results & Performance

<div align="center">

### 🏆 Model Leaderboard

| Rank | Model | ROC-AUC | Accuracy | F1 Score | Verdict |
|------|-------|---------|----------|----------|---------|
| 🥇 | **Gradient Boosting** | **~62%** | **~63%** | **~30%** | ✅ Best Overall |
| 🥈 | Logistic Regression | ~62% | ~63% | ~28% | ✅ Close runner-up |
| 🥉 | Random Forest | ~58% | ~61% | ~29% | 👍 Solid |
| 4️⃣ | Decision Tree | ~58% | ~61% | ~37% | 👍 Good |
| 5️⃣ | XGBoost | ~57% | ~61% | ~40% | ⚡ Highest recall |

> 🧪 Numbers are computed live every time the app trains — check the **Model Performance** tab for current results, since they're never hardcoded.

</div>

> **Honest note:** this is a synthetic-style dataset with realistic but randomly generated patterns, so scores land in a modest 55–65% ROC-AUC range rather than claiming state-of-the-art accuracy. The value of this project is the **end-to-end pipeline** — data → features → model comparison → live scoring UI — not a claim of production-grade predictive power on real transactional data.

---

## 🗂️ Project Structure

```
📁 e-commerce-product-return-/
│
├── 📊 ecommerce_product_returns.csv       # 15,000 orders
├── 🤖 train_model.py                       # Feature engineering + model training
├── 🖥️ app.py                               # Streamlit app (UI + prediction logic)
├── 📓 Product_Return_Prediction_System.ipynb   # Original exploratory notebook
├── 📸 predict-form.png                     # Screenshot — prediction form
├── 📸 risk-gauge.png                       # Screenshot — risk verdict gauge
├── 📄 README.md                            # This file
└── 📋 requirements.txt                     # All dependencies
```

---

## 🧠 How It Works

**Sample flow:**

```
════════════════════════════════════════════════════════════
  🛒  RETURN RISK AI
════════════════════════════════════════════════════════════

  📦 ORDER
  Category      : Electronics → Appliances
  Price         : ₹2,499  |  Discount: 30%
  Payment       : Cash on Delivery
  Delivery      : 9 days  |  Seller Rating: 3.4

  📊 PREDICTION
  Verdict            : Will Return
  Return Probability : 63.6%
  Risk Level         : 🔴 HIGH RISK

  ⚠️  SIGNALS DETECTED:
  • High discount (≥40% threshold zone)
  • Late delivery (>5 days)
  • Low seller rating (<3.5)
  • Cash on Delivery payment

  💡 RECOMMENDED ACTION:
  Consider limiting COD, flagging for quality check before
  dispatch, or proactively offering an exchange-friendly
  return policy.

════════════════════════════════════════════════════════════
```

The app **trains itself on first launch** in whatever environment it's deployed to, rather than shipping a pre-pickled model. This avoids version-mismatch errors between environments (a common cause of `ModuleNotFoundError` when a model pickled with one library version is loaded with another) and makes the app portable across Replit, Streamlit Cloud, Render, or any other host.

---

## 💼 Business Impact

This system has direct applications for e-commerce operations:

```
🏬  D2C & Marketplace Sellers   →  Flag high-risk orders before dispatch
📦  Logistics Teams             →  Prioritize quality checks on risky SKUs
💳  Payments Teams               →  Selectively restrict COD on high-risk orders
📊  Category Managers            →  Identify which categories/discount tiers drive returns
🔬  Data Teams                   →  Baseline pipeline to extend with real transactional data
```

---

## 🧰 Full Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data** | Pandas, NumPy |
| **ML Models** | scikit-learn (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting), XGBoost |
| **UI** | Streamlit |
| **Visualization** | Plotly |
| **Hosting** | Replit |
| **Version Control** | Git + GitHub |

---

## 🚀 Run Locally

```bash
git clone https://github.com/Shinearushi525/e-commerce-product-return-.git
cd e-commerce-product-return-
pip install -r requirements.txt
streamlit run app.py
```

The first launch trains the model automatically (a few seconds) and caches it for the rest of the session. Open the local URL Streamlit prints — usually `http://localhost:8501`.

---

## 🗺️ Roadmap Ideas

- [ ] Add SHAP-based explanations for individual predictions
- [ ] Add authentication for batch upload in production use
- [ ] Persist historical predictions for trend tracking
- [ ] A/B test alternate models (LightGBM, CatBoost)

---

## 👨‍💻 Author

<div align="center">

**Arushi Garg**

*B.Tech Computer Science (AI and Data Science)*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Arushi%20Garg-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arushi-garg525/)
[![GitHub](https://img.shields.io/badge/GitHub-Shinearushi525-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Shinearushi525)
[![Email](https://img.shields.io/badge/Email-arushigarg525@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:Arushigarg525@gmail.com)

</div>

---

## 📄 License

MIT — free to use, modify, and build on.
