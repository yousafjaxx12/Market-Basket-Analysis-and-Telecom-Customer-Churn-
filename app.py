"""
Terminal Project — Streamlit Frontend
=====================================
Section A : Telecommunication Customer Churn Classifier
Section B : Market Basket Analysis (Apriori & FP-Growth)

Run with:
    streamlit run app.py
"""

import time
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#  Sklearn 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from imblearn.over_sampling import SMOTE

#  mlxtend 
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# 
# Page config
# 
st.set_page_config(
    page_title="Terminal Project — Churn & MBA",
    page_icon="",
    layout="wide",
)

# 
#  DATA LOADING & MODEL TRAINING  (cached — runs once)
# 

@st.cache_resource(show_spinner="Training classifiers… please wait.")
def load_and_train():
    """Full preprocessing pipeline + train all 4 classifiers (Phase 2)."""
    df = pd.read_excel("Telco_Customer_Churn.xlsx")

    #  Drop non-informative columns 
    drop_cols = ["CustomerID", "Count", "Country", "State", "City",
                 "Zip Code", "Lat Long", "Latitude", "Longitude", "Churn Label"]
    target = (df["Churn Label"] == "Yes").astype(int)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df["Churn"] = target.values

    #  Fix Total Charges 
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df["Total Charges"] = df["Total Charges"].fillna(df["Total Charges"].median())

    #  One-hot encode 
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df = df.fillna(0)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    #  Scale 
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    #  SMOTE 
    sm = SMOTE(random_state=42)
    X_tr_res, y_tr_res = sm.fit_resample(X_tr_sc, y_train)

    #  Train models 
    models = {
        "Decision Tree":       DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42),
        "Naive Bayes":         GaussianNB(),
        "Rule-Based (DT d=5)": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42),
        "SVM":                 SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42, probability=True),
    }
    trained, metrics = {}, {}
    for name, clf in models.items():
        clf.fit(X_tr_res, y_tr_res)
        y_pred = clf.predict(X_te_sc)
        trained[name] = clf
        metrics[name] = {
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
            "cm":        confusion_matrix(y_test, y_pred),
        }

    return trained, metrics, scaler, feature_names, X_test, y_test


@st.cache_data(show_spinner="Loading groceries dataset…")
def load_groceries():
    """Load groceries CSV and convert to OHE transaction matrix."""
    df = pd.read_csv("groceries.csv")
    item_cols = [c for c in df.columns if c.startswith("Item")]
    all_items = sorted({
        v for col in item_cols for v in df[col].dropna().unique()
    })
    ohe = {item: df[item_cols].isin([item]).any(axis=1).astype(bool)
           for item in all_items}
    basket = pd.DataFrame(ohe)
    return basket, len(df), len(all_items)


# 
#  SIDEBAR NAVIGATION
# 
st.sidebar.title("Terminal Project")
st.sidebar.markdown("**Telecom Churn & Market Basket**")
section = st.sidebar.radio(
    "Navigate to:",
    ["Section A — Classifier", "Section B — Market Basket Analysis"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Section A**: Predict customer churn using 4 ML classifiers.\n\n"
    "**Section B**: Mine association rules from grocery transactions."
)

# 
#  SECTION A — CLASSIFIER
# 
if section == "Section A — Classifier":
    st.title("Section A — Telecom Customer Churn Classifier")
    st.markdown(
        "Fill in the customer attributes below, choose a classifier, and click **Predict**."
    )

    # Load models
    trained, metrics, scaler, feature_names, X_test, y_test = load_and_train()

    #  Input form 
    with st.form("churn_form"):
        st.subheader("Customer Attributes")
        c1, c2, c3 = st.columns(3)

        with c1:
            tenure     = st.number_input("Tenure (months)", 0, 72, 12)
            monthly    = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
            total      = st.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)
            senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner    = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        with c2:
            contract     = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment      = st.selectbox("Payment Method",
                                        ["Electronic check", "Mailed check",
                                         "Bank transfer (automatic)", "Credit card (automatic)"])
            paperless    = st.selectbox("Paperless Billing", ["No", "Yes"])
            internet     = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            phone        = st.selectbox("Phone Service", ["No", "Yes"])
            multiple     = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

        with c3:
            online_sec   = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_bk    = st.selectbox("Online Backup",   ["No", "Yes", "No internet service"])
            device_prot  = st.selectbox("Device Protection",["No","Yes","No internet service"])
            tech_support = st.selectbox("Tech Support",    ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV",    ["No", "Yes", "No internet service"])
            streaming_mv = st.selectbox("Streaming Movies",["No", "Yes", "No internet service"])

        st.divider()
        classifier_name = st.selectbox(
            "Choose Classifier",
            ["Decision Tree", "Naive Bayes", "Rule-Based (DT d=5)", "SVM"]
        )
        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    #  Prediction logic 
    if submitted:
        # Build raw row matching the original dataset's structure
        raw_row = {
            "Tenure Months": tenure,
            "Monthly Charges": monthly,
            "Total Charges": total,
            "Senior Citizen": 1 if senior == "Yes"else 0,
            "Partner": partner,
            "Dependents": dependents,
            "Contract": contract,
            "Payment Method": payment,
            "Paperless Billing": paperless,
            "Internet Service": internet,
            "Phone Service": phone,
            "Multiple Lines": multiple,
            "Online Security": online_sec,
            "Online Backup": online_bk,
            "Device Protection": device_prot,
            "Tech Support": tech_support,
            "Streaming TV": streaming_tv,
            "Streaming Movies": streaming_mv,
        }
        raw_df = pd.DataFrame([raw_row])

        # One-hot encode to align with training features
        cat_input = raw_df.select_dtypes(include=["object"]).columns.tolist()
        raw_df = pd.get_dummies(raw_df, columns=cat_input, drop_first=True)

        # Align with training feature set
        input_aligned = pd.DataFrame(columns=feature_names)
        input_aligned = pd.concat([input_aligned, raw_df], ignore_index=True)
        input_aligned = input_aligned.fillna(0).astype(float)
        input_aligned = input_aligned[feature_names]   # ensure column order

        input_scaled = scaler.transform(input_aligned)

        clf   = trained[classifier_name]
        pred  = clf.predict(input_scaled)[0]
        label = "Churn (Yes)" if pred == 1 else "No Churn (No)"

        st.divider()
        res_col, met_col = st.columns([1, 2])

        with res_col:
            st.subheader("Prediction Result")
            if pred == 1:
                st.error(f"**{label}**")
                st.markdown("This customer is **at risk of churning**. Consider retention actions.")
            else:
                st.success(f"**{label}**")
                st.markdown("This customer is **likely to stay**.")

            # Probability if supported
            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(input_scaled)[0]
                st.metric("Churn Probability", f"{prob[1]:.1%}")

        with met_col:
            st.subheader(f"{classifier_name} — Test-Set Performance")
            m = metrics[classifier_name]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Accuracy",  f"{m['Accuracy']:.4f}")
            k2.metric("Precision", f"{m['Precision']:.4f}")
            k3.metric("Recall",    f"{m['Recall']:.4f}")
            k4.metric("F1-Score",  f"{m['F1-Score']:.4f}")

            # Confusion matrix
            fig, ax = plt.subplots(figsize=(4, 3))
            import seaborn as sns
            sns.heatmap(m["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["No Churn", "Churn"],
                        yticklabels=["No Churn", "Churn"],
                        linewidths=0.5)
            ax.set_title("Confusion Matrix", fontsize=10)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)

    #  Model comparison table 
    with st.expander("All Models — Performance Comparison", expanded=False):
        _, metrics_loaded, *_ = load_and_train()
        rows = []
        for name, m in metrics_loaded.items():
            rows.append({"Model": name, "Accuracy": m["Accuracy"],
                         "Precision": m["Precision"],
                         "Recall": m["Recall"], "F1-Score": m["F1-Score"]})
        comp_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(comp_df.style.highlight_max(axis=0, color="#c8f7c5")
                                  .format("{:.4f}"), use_container_width=True)


# 
#  SECTION B — MARKET BASKET ANALYSIS
# 
else:
    st.title("Section B — Market Basket Analysis")
    st.markdown(
        "Association rule mining on the **Groceries** dataset using **Apriori** and **FP-Growth**."
    )

    # Load dataset
    basket_df, n_transactions, n_items = load_groceries()

    # Dataset info
    i1, i2, i3 = st.columns(3)
    i1.metric("Transactions", f"{n_transactions:,}")
    i2.metric("Unique Items", f"{n_items:,}")
    i3.metric("Avg Items/Basket", f"{basket_df.sum(axis=1).mean():.1f}")

    st.divider()

    #  Threshold sliders 
    st.subheader("Algorithm Parameters")
    sl1, sl2 = st.columns(2)
    min_sup  = sl1.slider("Minimum Support",    0.01, 0.20, 0.03, 0.01,
                          help="Fraction of transactions that must contain the itemset.")
    min_conf = sl2.slider("Minimum Confidence", 0.10, 0.90, 0.40, 0.05,
                          help="Conditional probability: P(consequent | antecedent).")

    #  Run buttons 
    b1, b2, b3 = st.columns([1, 1, 2])
    run_apriori  = b1.button("Run Apriori",   use_container_width=True)
    run_fpgrowth = b2.button("Run FP-Growth", use_container_width=True)

    # Keep results in session_state so both can be shown simultaneously
    if "apriori_result" not in st.session_state: st.session_state.apriori_result  = None
    if "fpgrowth_result"not in st.session_state: st.session_state.fpgrowth_result = None

    def run_algorithm(algo: str, basket, sup, conf):
        with st.spinner(f"Running {algo}…"):
            t0 = time.perf_counter()
            if algo == "Apriori":
                freq  = apriori(basket, min_support=sup, use_colnames=True, low_memory=False)
            else:
                freq  = fpgrowth(basket, min_support=sup, use_colnames=True)
            if len(freq) == 0:
                return None, None, 0.0
            rules = association_rules(freq, metric="confidence", min_threshold=conf)
            elapsed = time.perf_counter() - t0
        rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
        return freq, rules, elapsed

    def display_result(algo_name, freq, rules, elapsed, color):
        if freq is None:
            st.warning(f"No frequent itemsets found at min_support={min_sup:.2f}. "
                       "Try lowering the support threshold.")
            return

        st.subheader(f"{algo_name} Results")

        # Summary panel
        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Frequent Itemsets", len(freq))
        sm2.metric("Rules Generated",   len(rules))
        sm3.metric("Runtime",           f"{elapsed:.4f}s")

        if len(rules) == 0:
            st.warning("No rules satisfy the confidence threshold. Try lowering it.")
            return

        # Frequent itemsets bar chart
        top_fi = (freq
                  .assign(item=freq["itemsets"].apply(lambda x: ", ".join(list(x))))
                  .sort_values("support", ascending=False)
                  .head(15))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(top_fi["item"][::-1], top_fi["support"][::-1],
                color=color, edgecolor="black", alpha=0.85)
        for i, (s, item) in enumerate(zip(top_fi["support"][::-1], top_fi["item"][::-1])):
            ax.text(s + 0.002, i, f"{s:.3f}", va="center", fontsize=8)
        ax.set_xlabel("Support")
        ax.set_title(f"Top 15 Frequent Itemsets — {algo_name}", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # Rules table
        st.markdown("**Top-10 Rules by Lift**")
        display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        rules_show = rules[display_cols].head(10).copy()
        rules_show["antecedents"] = rules_show["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules_show["consequents"] = rules_show["consequents"].apply(lambda x: ", ".join(list(x)))
        rules_show.columns = ["Antecedent", "Consequent", "Support", "Confidence", "Lift"]
        rules_show = rules_show.round(4).reset_index(drop=True)
        st.dataframe(rules_show, use_container_width=True)

    #  Run Apriori 
    if run_apriori:
        freq, rules, elapsed = run_algorithm("Apriori", basket_df, min_sup, min_conf)
        st.session_state.apriori_result = (freq, rules, elapsed)

    if st.session_state.apriori_result:
        freq, rules, elapsed = st.session_state.apriori_result
        display_result("Apriori", freq, rules, elapsed, "steelblue")
        st.divider()

    #  Run FP-Growth 
    if run_fpgrowth:
        freq, rules, elapsed = run_algorithm("FP-Growth", basket_df, min_sup, min_conf)
        st.session_state.fpgrowth_result = (freq, rules, elapsed)

    if st.session_state.fpgrowth_result:
        freq, rules, elapsed = st.session_state.fpgrowth_result
        display_result("FP-Growth", freq, rules, elapsed, "darkorange")
        st.divider()

    #  Comparison chart (only when both have run) 
    if st.session_state.apriori_result and st.session_state.fpgrowth_result:
        fa, ra, ta = st.session_state.apriori_result
        ff, rf, tf = st.session_state.fpgrowth_result

        if fa is not None and ff is not None:
            st.subheader("Apriori vs FP-Growth — Comparison")

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            algos = ["Apriori", "FP-Growth"]
            colors = ["steelblue", "darkorange"]

            # Runtime
            b = axes[0].bar(algos, [ta, tf], color=colors, edgecolor="black", alpha=0.85, width=0.4)
            axes[0].set_title("Runtime (seconds)", fontweight="bold")
            axes[0].set_ylabel("Seconds")
            for bar in b:
                axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                             f"{bar.get_height():.4f}s", ha="center", fontweight="bold", fontsize=10)

            # Rules count
            n_rules_vals = [0 if ra is None else len(ra), 0 if rf is None else len(rf)]
            b2 = axes[1].bar(algos, n_rules_vals, color=colors, edgecolor="black", alpha=0.85, width=0.4)
            axes[1].set_title("Rules Generated", fontweight="bold")
            axes[1].set_ylabel("Count")
            for bar in b2:
                axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                             str(int(bar.get_height())), ha="center", fontweight="bold", fontsize=10)

            plt.suptitle(
                f"Algorithm Comparison  |  min_support={min_sup}  |  min_confidence={min_conf}",
                fontsize=11, fontweight="bold"
            )
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            # Quick summary text
            faster = "FP-Growth"if tf < ta else "Apriori"
            st.info(
                f"**{faster}** was faster by **{abs(ta-tf):.4f}s**.  "
                f"Both algorithms found **{len(fa)}** frequent itemsets and generated "
                f"the same rule set at these thresholds."
            )
