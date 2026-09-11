# Customer Churn Prediction & Retention Strategy

An end-to-end churn prediction pipeline built on the IBM Telco Customer Churn dataset, combining classical ML, explainability, and business-impact analysis to identify at-risk customers and quantify the revenue opportunity in retaining them.

## Problem Statement

Customer churn is one of the most direct threats to recurring revenue in subscription-based businesses. This project builds a model to predict which customers are likely to churn, explains *why* they churn using SHAP, and translates those predictions into a segmented, actionable retention strategy with dollar-value business impact.

## Dataset

- **Source**: IBM Telco Customer Churn dataset (7,043 customers, 21 features)
- **Target**: `Churn` (Yes/No) — 26.58% base churn rate (imbalanced)
- **Features**: Demographics, account info (tenure, contract type, payment method), and services subscribed (internet, streaming, security add-ons)

## Approach

### 1. Exploratory Data Analysis
Segment-wise churn analysis revealed clear patterns before any modeling:
- Month-to-month contracts churn at **42.7%** vs. **2.9%** for two-year contracts
- Customers in their first year churn at **47.7%** vs. **6.6%** for 5+ year tenure
- Fiber optic customers churn at **41.9%** vs. **19.0%** for DSL

### 2. Modeling
| Model | ROC-AUC | Recall (Churn) | Precision (Churn) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.8350 | 0.80 | 0.49 |
| XGBoost | 0.8376 | 0.79 | 0.49 |
| XGBoost (tuned) | 0.8386 | 0.80 | 0.50 |

- Class imbalance handled via `scale_pos_weight` (XGBoost) and `class_weight='balanced'` (Logistic Regression)
- 5-fold stratified cross-validation: **0.8470 ROC-AUC (± 0.0031)** — confirms model stability across folds
- Hyperparameters tuned via RandomizedSearchCV (20 iterations, 3-fold)
- 95% bootstrap confidence interval for ROC-AUC: **[0.8161, 0.8608]**

**Finding**: XGBoost only marginally outperformed Logistic Regression, and the best-tuned XGBoost used a *shallower* tree (`max_depth=3`) than the default — suggesting the dataset's signal is largely captured by simple, near-linear relationships rather than complex interactions.

### 3. Threshold Optimization
Rather than using the default 0.5 classification threshold, the decision threshold was optimized using an F2 score (recall-weighted), since missing an actual churner (lost customer, lost revenue) is costlier than a false alarm (a retention offer sent to a loyal customer):

| Threshold | Precision | Recall |
|---|---|---|
| Default (0.5) | 0.498 | 0.805 |
| F2-optimized (0.314) | 0.443 | 0.896 |

This trade-off catches ~9% more actual churners at a modest precision cost.

### 4. Explainability (SHAP)
Top churn drivers, consistent with the EDA findings:
1. Two-year contract (reduces churn risk)
2. Tenure (longer tenure reduces risk)
3. One-year contract
4. Fiber optic internet service (increases risk)
5. Monthly charges (higher charges increase risk)

### 5. Business Impact & Retention Strategy
Customers were segmented into three risk tiers based on predicted churn probability:

| Tier | Customers | Churn Rate | Monthly Revenue | Recommended Action |
|---|---|---|---|---|
| High | 494 | 54.9% | $36,830.90 | Personal outreach + custom offer, within 48 hours |
| Medium | 283 | 22.6% | $19,806.30 | Automated discount email + survey, within 2 weeks |
| Low | 630 | 6.2% | $33,402.65 | No intervention — monitor quarterly |

**Total addressable revenue at risk (Medium + High tiers): $679,646.40 annually**

Assuming a conservative 30% retention success rate on the top 20% highest-risk segment, targeted intervention could protect an estimated **$49,971 in annual revenue**.

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, SHAP, Matplotlib/Seaborn

## Key Takeaways
- A well-tuned simple model can match a more complex one — the value here came from feature engineering, threshold selection, and explainability, not model complexity.
- Business-cost-aware threshold selection matters more than optimizing for accuracy alone.
- SHAP explainability directly validated the EDA findings, giving confidence that the model is learning genuine signal rather than noise.

## Future Work
- Deploy as a FastAPI/Streamlit app for interactive predictions
- A/B test the recommended retention actions to validate the 30% success-rate assumption
- Incorporate customer lifetime value (CLV) into the prioritization logic instead of monthly revenue alone
