"""
train_model.py
--------------
Trains the product-return prediction model from ecommerce_product_returns.csv
and saves all artifacts needed by the Streamlit app:
    - model.pkl        (best trained classifier)
    - scaler.pkl        (StandardScaler, used only for Logistic Regression)
    - encoders.pkl      (dict of LabelEncoders per categorical column)
    - feature_cols.pkl  (ordered list of feature names the model expects)
    - results.pkl       (leaderboard of all trained models' metrics)
    - feature_importance.pkl (Random Forest importances, for the dashboard)

Run once locally:  python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
)

DATA_PATH = "ecommerce_product_returns.csv"

CAT_COLS = [
    "customer_type", "product_category", "product_subcategory",
    "payment_method", "shipping_type", "warranty_available",
    "product_size_available", "customer_city",
]

FEATURE_COLS = [
    "customer_age", "product_price", "discount_percent", "order_quantity",
    "delivery_days", "seller_rating", "customer_rating",
    "actual_price", "high_discount", "late_delivery",
    "low_seller_rating", "low_customer_rating", "premium_product",
    "festival_season", "is_cod", "order_month",
    "customer_type_enc", "product_category_enc", "product_subcategory_enc",
    "payment_method_enc", "shipping_type_enc", "warranty_available_enc",
    "product_size_available_enc", "customer_city_enc",
]


def engineer_features(df: pd.DataFrame, encoders: dict | None = None):
    """Add engineered features + label-encode categoricals.
    If `encoders` is given, reuse those fitted encoders (inference mode).
    Otherwise fit new ones (training mode) and return them.
    """
    df = df.copy()
    df["actual_price"] = df["product_price"] * (1 - df["discount_percent"] / 100)
    df["high_discount"] = (df["discount_percent"] >= 40).astype(int)
    df["late_delivery"] = (df["delivery_days"] > 5).astype(int)
    df["low_seller_rating"] = (df["seller_rating"] < 3.5).astype(int)
    df["low_customer_rating"] = (df["customer_rating"] <= 2).astype(int)
    df["premium_product"] = (df["product_price"] > 10000).astype(int)
    df["order_month"] = pd.to_datetime(df["order_date"]).dt.month
    df["festival_season"] = df["order_month"].isin([10, 11, 12]).astype(int)
    df["is_cod"] = (df["payment_method"] == "Cash on Delivery").astype(int)

    fit_new = encoders is None
    if fit_new:
        encoders = {}

    for col in CAT_COLS:
        if fit_new:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # handle unseen categories gracefully at inference time
            df[col + "_enc"] = df[col].astype(str).map(
                lambda v: le.transform([v])[0] if v in le.classes_ else -1
            )

    return df, encoders


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {df.shape[0]:,} rows, {df.shape[1]} columns")

    df_model, encoders = engineer_features(df)

    X = df_model[FEATURE_COLS]
    y = df_model["is_returned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0
        )
    except ImportError:
        print("xgboost not installed - skipping XGBoost model")

    results = []
    trained_models = {}

    print("\nTraining models...")
    print(f"{'Model':<22}{'Accuracy':>10}{'F1':>9}{'ROC-AUC':>10}{'Precision':>11}{'Recall':>9}")
    print("-" * 71)

    for name, model in models.items():
        use_scaled = name == "Logistic Regression"
        Xtr = X_train_scaled if use_scaled else X_train
        Xte = X_test_scaled if use_scaled else X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        results.append({
            "Model": name, "Accuracy": acc, "F1 Score": f1,
            "ROC-AUC": auc, "Precision": prec, "Recall": rec,
        })
        trained_models[name] = model

        print(f"{name:<22}{acc*100:>9.2f}%{f1*100:>8.2f}%{auc*100:>9.2f}%"
              f"{prec*100:>10.2f}%{rec*100:>8.2f}%")

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name}  (ROC-AUC = {results_df.iloc[0]['ROC-AUC']*100:.2f}%)")

    # feature importance (Random Forest if available, else best model's if it has one)
    importer = trained_models.get("Random Forest", best_model)
    if hasattr(importer, "feature_importances_"):
        fi = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": importer.feature_importances_,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
    else:
        fi = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": np.nan})

    # save all artifacts
    with open("model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open("feature_cols.pkl", "wb") as f:
        pickle.dump(FEATURE_COLS, f)
    with open("results.pkl", "wb") as f:
        pickle.dump({"results_df": results_df, "best_model_name": best_name}, f)
    with open("feature_importance.pkl", "wb") as f:
        pickle.dump(fi, f)

    print("\nSaved: model.pkl, scaler.pkl, encoders.pkl, feature_cols.pkl, "
          "results.pkl, feature_importance.pkl")


if __name__ == "__main__":
    main()
