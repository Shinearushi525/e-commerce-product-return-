"""
Product Return Prediction — Streamlit App
Author: adapted for deployment from Arushi Garg's e-commerce-product-return project
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import train_model
from train_model import engineer_features, FEATURE_COLS, CAT_COLS

st.set_page_config(
    page_title="Product Return Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACT_FILES = [
    "model.pkl", "scaler.pkl", "encoders.pkl",
    "results.pkl", "feature_importance.pkl",
]


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load pickled model artifacts. If they're missing, or were pickled with
    library versions incompatible with what's installed here (a common issue
    when moving between environments), retrain from scratch in this exact
    environment instead of failing.
    """
    def _load():
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        with open("results.pkl", "rb") as f:
            results = pickle.load(f)
        with open("feature_importance.pkl", "rb") as f:
            fi = pickle.load(f)
        return model, scaler, encoders, results, fi

    missing = [f for f in ARTIFACT_FILES if not os.path.exists(f)]

    if not missing:
        try:
            return _load()
        except Exception as e:
            # Version mismatch (e.g. pickled with a different numpy/sklearn/
            # xgboost) or corrupted file - fall through to retraining.
            st.warning(
                f"Saved model files couldn't be loaded in this environment "
                f"({type(e).__name__}). Retraining from the CSV instead - "
                f"this happens once and takes under a minute."
            )

    with st.spinner("Training model for the first time in this environment..."):
        train_model.main()
    return _load()


@st.cache_data
def load_data():
    return pd.read_csv("ecommerce_product_returns.csv")


try:
    model, scaler, encoders, results_obj, feat_importance = load_artifacts()
    results_df = results_obj["results_df"]
    best_model_name = results_obj["best_model_name"]
    ARTIFACTS_OK = True
except FileNotFoundError:
    ARTIFACTS_OK = False

df = load_data()

# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
st.sidebar.title("🛒 Return Prediction")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview & EDA", "🤖 Model Performance", "🔮 Predict a Single Order", "📁 Batch Prediction"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Predicts whether an e-commerce order is likely to be returned, "
    "based on discount, delivery time, payment method, category, and more."
)

if not ARTIFACTS_OK:
    st.error(
        "Model artifacts not found. Run `python train_model.py` once in this "
        "directory to generate model.pkl, scaler.pkl, encoders.pkl, results.pkl "
        "and feature_importance.pkl, then reload the app."
    )
    st.stop()

# ----------------------------------------------------------------------
# PAGE 1 — Overview & EDA
# ----------------------------------------------------------------------
if page == "📊 Overview & EDA":
    st.title("🛒 E-Commerce Product Return — Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Orders", f"{df.shape[0]:,}")
    c2.metric("Return Rate", f"{df['is_returned'].mean()*100:.1f}%")
    c3.metric("Avg. Order Value", f"₹{df['product_price'].mean():,.0f}")
    c4.metric("Avg. Delivery Days", f"{df['delivery_days'].mean():.1f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Return Rate by Product Category")
        cat_return = (
            df.groupby("product_category")["is_returned"].mean().sort_values(ascending=False) * 100
        )
        fig = px.bar(
            cat_return, x=cat_return.index, y=cat_return.values,
            labels={"x": "Category", "y": "Return Rate (%)"},
            color=cat_return.values, color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Return Rate by Discount Level")
        bins = [0, 10, 20, 30, 40, 50, 100]
        labels = ["0-10%", "11-20%", "21-30%", "31-40%", "41-50%", "50%+"]
        disc_bin = pd.cut(df["discount_percent"], bins=bins, labels=labels)
        disc_return = df.groupby(disc_bin, observed=True)["is_returned"].mean() * 100
        fig = px.bar(
            disc_return, x=disc_return.index.astype(str), y=disc_return.values,
            labels={"x": "Discount Range", "y": "Return Rate (%)"},
            color=disc_return.values, color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Return Rate by Payment Method")
        pay_return = (
            df.groupby("payment_method")["is_returned"].mean().sort_values(ascending=False) * 100
        )
        fig = px.bar(
            pay_return, x=pay_return.values, y=pay_return.index, orientation="h",
            labels={"x": "Return Rate (%)", "y": "Payment Method"},
            color=pay_return.values, color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Return Rate by Delivery Days")
        delivery_return = df.groupby("delivery_days")["is_returned"].mean() * 100
        fig = px.line(
            delivery_return, x=delivery_return.index, y=delivery_return.values,
            labels={"x": "Delivery Days", "y": "Return Rate (%)"}, markers=True,
        )
        fig.add_vline(x=7, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Correlation Heatmap")
    numeric_cols = [
        "customer_age", "product_price", "discount_percent",
        "order_quantity", "delivery_days", "seller_rating",
        "customer_rating", "is_returned",
    ]
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn", zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Return Reasons")
    returned_df = df[df["is_returned"] == 1]
    reason_counts = returned_df["return_reason"].value_counts()
    fig = px.pie(reason_counts, values=reason_counts.values, names=reason_counts.index, hole=0.35)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Preview raw dataset"):
        st.dataframe(df.head(50), use_container_width=True)

# ----------------------------------------------------------------------
# PAGE 2 — Model Performance
# ----------------------------------------------------------------------
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.markdown(
        f"**Best model selected: `{best_model_name}`** (ranked by ROC-AUC on a held-out 20% test split)"
    )

    show_df = results_df.copy()
    for col in ["Accuracy", "F1 Score", "ROC-AUC", "Precision", "Recall"]:
        show_df[col] = (show_df[col] * 100).round(2).astype(str) + "%"
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    st.subheader("Metric Comparison")
    metric = st.selectbox("Choose a metric", ["Accuracy", "F1 Score", "ROC-AUC", "Precision", "Recall"])
    fig = px.bar(
        results_df, x="Model", y=metric, color="Model", text_auto=".2%",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance (Random Forest)")
    if feat_importance["Importance"].notna().any():
        top_fi = feat_importance.head(15).sort_values("Importance")
        fig = px.bar(
            top_fi, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance is unavailable for the selected best model type.")

    st.caption(
        "Note: metrics reflect this synthetic dataset and may differ from any figures "
        "quoted elsewhere — they are computed fresh each time `train_model.py` is run."
    )

# ----------------------------------------------------------------------
# PAGE 3 — Single Prediction
# ----------------------------------------------------------------------
elif page == "🔮 Predict a Single Order":
    st.title("🔮 Predict Return Risk for a Single Order")
    st.write("Fill in the order details below and get an instant return-risk prediction.")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            customer_age = st.slider("Customer Age", 18, 65, 30)
            customer_type = st.selectbox("Customer Type", sorted(df["customer_type"].unique()))
            customer_city = st.selectbox("Customer City", sorted(df["customer_city"].unique()))
            product_category = st.selectbox("Product Category", sorted(df["product_category"].unique()))
            product_subcategory = st.selectbox(
                "Product Subcategory", sorted(df["product_subcategory"].unique())
            )

        with col2:
            product_price = st.number_input("Product Price (₹)", 99, 89999, 2500)
            discount_percent = st.slider("Discount (%)", 0, 70, 20)
            order_quantity = st.number_input("Order Quantity", 1, 10, 1)
            payment_method = st.selectbox("Payment Method", sorted(df["payment_method"].unique()))
            shipping_type = st.selectbox("Shipping Type", sorted(df["shipping_type"].unique()))

        with col3:
            delivery_days = st.slider("Delivery Days", 1, 21, 5)
            seller_rating = st.slider("Seller Rating", 2.5, 5.0, 4.0, step=0.1)
            customer_rating = st.slider("Customer Rating (order review)", 1, 5, 3)
            warranty_available = st.selectbox("Warranty Available", sorted(df["warranty_available"].unique()))
            product_size_available = st.selectbox(
                "Size Available", sorted(df["product_size_available"].unique())
            )
            order_month = st.slider("Order Month", 1, 12, 6)

        submitted = st.form_submit_button("Predict Return Risk", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "order_date": f"2024-{order_month:02d}-15",
            "customer_age": customer_age,
            "customer_type": customer_type,
            "customer_city": customer_city,
            "product_category": product_category,
            "product_subcategory": product_subcategory,
            "product_price": product_price,
            "discount_percent": discount_percent,
            "order_quantity": order_quantity,
            "payment_method": payment_method,
            "shipping_type": shipping_type,
            "delivery_days": delivery_days,
            "seller_rating": seller_rating,
            "customer_rating": customer_rating,
            "warranty_available": warranty_available,
            "product_size_available": product_size_available,
        }])

        row_fe, _ = engineer_features(row, encoders=encoders)
        X_input = row_fe[FEATURE_COLS]

        use_scaled = best_model_name == "Logistic Regression"
        X_final = scaler.transform(X_input) if use_scaled else X_input

        prob = model.predict_proba(X_final)[0][1]
        pred = model.predict(X_final)[0]

        if prob > 0.60:
            risk_label, color = "🔴 HIGH RISK", "red"
        elif prob > 0.35:
            risk_label, color = "🟡 MEDIUM RISK", "orange"
        else:
            risk_label, color = "🟢 LOW RISK", "green"

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Prediction", "Will Return" if pred else "Not Returned")
        r2.metric("Return Probability", f"{prob*100:.1f}%")
        r3.markdown(f"### {risk_label}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Return Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 35], "color": "#d4f7d4"},
                    {"range": [35, 60], "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#f8d7da"},
                ],
            },
        ))
        st.plotly_chart(fig, use_container_width=True)

        if risk_label == "🔴 HIGH RISK":
            st.warning(
                "Consider: limiting COD for this order, flagging for quality check before "
                "dispatch, or offering an exchange-friendly return policy proactively."
            )

# ----------------------------------------------------------------------
# PAGE 4 — Batch Prediction
# ----------------------------------------------------------------------
elif page == "📁 Batch Prediction":
    st.title("📁 Batch Prediction from CSV")
    st.write(
        "Upload a CSV with the same columns as the original dataset "
        "(excluding `order_id`, `is_returned`, `return_reason`) to score many orders at once."
    )

    required_cols = [
        "order_date", "customer_age", "customer_type", "customer_city",
        "product_category", "product_subcategory", "product_price",
        "discount_percent", "order_quantity", "payment_method",
        "shipping_type", "delivery_days", "seller_rating", "customer_rating",
        "warranty_available", "product_size_available",
    ]
    st.caption("Required columns: " + ", ".join(required_cols))

    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        missing = [c for c in required_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            batch_fe, _ = engineer_features(batch_df, encoders=encoders)
            X_batch = batch_fe[FEATURE_COLS]
            use_scaled = best_model_name == "Logistic Regression"
            X_final = scaler.transform(X_batch) if use_scaled else X_batch

            probs = model.predict_proba(X_final)[:, 1]
            preds = model.predict(X_final)

            out = batch_df.copy()
            out["return_probability"] = (probs * 100).round(1)
            out["will_return"] = preds.astype(bool)
            out["risk_level"] = np.where(
                probs > 0.60, "HIGH", np.where(probs > 0.35, "MEDIUM", "LOW")
            )

            st.success(f"Scored {len(out):,} orders.")
            st.dataframe(out, use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download scored CSV", csv_bytes, "scored_orders.csv", "text/csv"
            )

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · scikit-learn · plotly")
