# 🛒 Product Return Prediction System
### Machine Learning · Data Science · E-Commerce Analytics

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Complete-2ECC71?style=for-the-badge"/>
</p>

---

## 📌 Problem Statement

E-commerce companies like Amazon, Flipkart, and Meesho lose **billions of rupees annually** due to product returns. High return rates lead to:

- 📦 Increased logistics & reverse-shipping costs
- 🏭 Inventory management problems
- 💸 Revenue loss and reduced profit margins
- 😟 Poor customer experience

**This project builds a Machine Learning model that predicts whether a product order will be returned — before it is shipped.** By identifying high-risk orders early, businesses can take proactive steps to reduce return rates.

---

## 🎯 Objective

> Predict the probability of a product being returned based on order features such as discount percentage, delivery time, payment method, product category, and customer behavior.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | Custom-generated Indian E-Commerce Dataset |
| **Rows** | 15,000 orders |
| **Columns** | 19 features |
| **Time Period** | January 2022 – December 2024 |
| **Target Variable** | `is_returned` (0 = Not Returned, 1 = Returned) |
| **Return Rate** | ~38.2% |

### 📋 Features Overview

| Feature | Description |
|---------|-------------|
| `order_id` | Unique order identifier |
| `order_date` | Date of purchase |
| `customer_age` | Age of customer (18–65) |
| `customer_type` | New / Returning / Loyal |
| `customer_city` | 18 Indian cities |
| `product_category` | Electronics, Fashion, Home & Kitchen, etc. |
| `product_subcategory` | Mobile Phones, Footwear, Cookware, etc. |
| `product_price` | Price in ₹ (99 – 89,999) |
| `discount_percent` | Discount applied (0% – 70%) |
| `order_quantity` | Number of items ordered |
| `payment_method` | UPI, COD, Credit Card, EMI, etc. |
| `shipping_type` | Standard / Express / Same Day / Free |
| `delivery_days` | Days taken to deliver (1–21) |
| `seller_rating` | Seller rating (2.5 – 5.0) |
| `customer_rating` | Customer-given rating (1–5) |
| `warranty_available` | Yes / No |
| `is_returned` | **Target: 0 or 1** |
| `return_reason` | Why the order was returned |

---

## 🔍 Exploratory Data Analysis (EDA)

Key findings from the analysis:

### 📦 Return Rate by Category
> Fashion leads with **42.4%** return rate, primarily due to wrong size/fit issues.

### 💸 Discount Effect
> Orders with **50%+ discount** have nearly **2x higher return rate** than low-discount orders.
> Heavy discounts encourage impulse buying, which leads to returns.

### 🚚 Delivery Days
> Return rate spikes significantly when delivery exceeds **7 days**.
> On-time delivery is strongly correlated with customer satisfaction.

### 💳 Payment Method
> **Cash on Delivery (COD)** orders show the **highest return rate** — customers feel less committed when paying later.

### 👤 Customer Type
> **New customers** return 6% more than loyal customers.
> Loyal customers show the lowest return rate.

### 🔥 Correlation Highlights
> `discount_percent` and `delivery_days` are the **strongest predictors** of product returns.

---

## ⚙️ Feature Engineering

9 new features were engineered to improve model performance:

| New Feature | Logic |
|-------------|-------|
| `actual_price` | Price after applying discount |
| `high_discount` | 1 if discount ≥ 40% |
| `late_delivery` | 1 if delivery days > 5 |
| `low_seller_rating` | 1 if seller rating < 3.5 |
| `low_customer_rating` | 1 if customer rating ≤ 2 |
| `premium_product` | 1 if price > ₹10,000 |
| `order_month` | Month extracted from order date |
| `festival_season` | 1 if order placed in Oct–Dec |
| `is_cod` | 1 if payment method is Cash on Delivery |

---

## 🤖 Models Trained & Compared

Five classification models were trained and evaluated:

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | ~72% | ~70% | ~78% |
| Decision Tree | ~74% | ~73% | ~74% |
| **Random Forest** | **~82%** | **~81%** | **~89%** |
| Gradient Boosting | ~81% | ~80% | ~88% |
| XGBoost | ~82% | ~81% | ~89% |

> ✅ **Random Forest** and **XGBoost** emerged as the best-performing models.

---

## 📈 Results

```
Best Model    : Random Forest / XGBoost
Accuracy      : ~82%
ROC-AUC Score : ~89%
F1 Score      : ~81%
Features Used : 24 (15 original + 9 engineered)
```

### 🔑 Top Predictive Features
1. `discount_percent` — strongest return signal
2. `delivery_days` — delivery delay drives returns
3. `customer_rating` — dissatisfied customers return more
4. `is_cod` — COD linked to higher return behavior
5. `product_category` — Fashion has highest risk
6. `seller_rating` — poor sellers → more returns
7. `high_discount` — engineered flag for >40% discount
8. `late_delivery` — engineered flag for delivery >5 days

---
## 💡 Business Impact

This system enables e-commerce companies to:

- 🎯 **Flag high-risk orders** before dispatch and offer alternatives
- 💬 **Send targeted messages** to customers likely to return
- 🏷️ **Limit COD** on high-discount + high-risk combinations
- 📦 **Optimize packaging** for categories with high return rates
- 📊 **Monitor seller performance** linked to returns

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core programming language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | ML models & evaluation |
| XGBoost | Gradient boosting classifier |
| Google Colab | Development environment |
| GitHub | Version control & portfolio |

---

## 👨‍💻 Author

**Your Name**
- 🎓 B.Tech Computer Science (AI and Data Science)
- 📧 Arushigarg525@gmail.com
- 🐙 [GitHub](https://github.com/Shinearushi525)

