# ══════════════════════════════════════════════════════════════════════════════
# CKD-Insight App 
# Explainable Machine Learning for Chronic Kidney Disease Prediction with SHAP Explanations
# BME6938 Medical AI · Project 1 · Group 6
# Author: Joseph Luper Tsenum, Riley Bendure and Gopal Viraj Koundinya Vutukuru 
#
# Usage:
#   Terminal 1:  python backend.py
#   Terminal 2:  streamlit run app.py
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go

# ── Configuration ────────────────────────────────────────────────────────────
BACKEND_URL = "http://127.0.0.1:5000"
CKD_RED = "#C44E52"
NOTCKD_BLUE = "#4C72B0"
RF_GREEN = "#55A868"
XGB_RED = "#C44E52"
MODEL_COLORS = {"Random Forest":"#55A868","XGBoost":"#C44E52",
                "Logistic Regression":"#4C72B0","SVM":"#8172B2","Decision Tree":"#CCB974"}

def call_backend(endpoint, body=None):
    url = f"{BACKEND_URL}{endpoint}"
    try:
        if body is None:
            r = requests.get(url, timeout=10)
        else:
            r = requests.post(url, json=body, timeout=10)
        return r.json()
    except Exception as e:
        st.error(f"⚠ Backend not reachable: {e}")
        return None


# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CKD-Insight App: CKD Prediction & Explainability",
                   page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .pred-ckd { background: linear-gradient(135deg, #C44E52, #e07070); color: white;
                padding: 20px; border-radius: 12px; text-align: center; }
    .pred-notckd { background: linear-gradient(135deg, #4C72B0, #6a9fd8); color: white;
                   padding: 20px; border-radius: 12px; text-align: center; }
    .metric-card { background: #f8f9fa; border-radius: 8px; padding: 15px;
                   border-left: 4px solid #4C72B0; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🩺 CKD-Insight App: CKD Prediction & Explainability")
st.caption("BME6938 Medical AI · Project 1 · Group 6 · Joseph Luper Tsenum, Riley Bendure and Gopal Viraj Koundinya Vutukuru")

tabs = st.tabs(["🔬 Patient Prediction", "📊 Model Performance",
                "🌍 SHAP Global", "🔍 SHAP Local", "📈 SHAP Dependence",
                "🗂 Dataset Explorer", "ℹ About"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: PATIENT PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Patient Prediction")
    st.markdown("Enter patient clinical features below, then click **Predict** to get a CKD risk assessment with SHAP explanation.")

    # Sample patient buttons
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        load_ckd = st.button("📋 Load Sample CKD Patient", use_container_width=True)
    with col_s2:
        load_healthy = st.button("📋 Load Sample Healthy Patient", use_container_width=True)
    with col_s3:
        model_choice = st.selectbox("Model", ["rf", "xgb"],
                                     format_func=lambda x: "Random Forest" if x=="rf" else "XGBoost")

    # Defaults
    defaults = {"age":50,"bp":80,"sg":1.020,"al":0,"su":0,"bgr":120,"bu":40,
                "sc":1.0,"sod":140.0,"pot":4.5,"hemo":13.0,"pcv":40,"wc":8000,"rc":4.5,
                "rbc":"normal","pc":"normal","pcc":"notpresent","ba":"notpresent"}
    if load_ckd:
        defaults = {"age":60,"bp":90,"sg":1.010,"al":3,"su":0,"bgr":200,"bu":100,
                    "sc":3.5,"sod":130.0,"pot":5.0,"hemo":9.0,"pcv":28,"wc":10000,"rc":3.5,
                    "rbc":"abnormal","pc":"abnormal","pcc":"present","ba":"notpresent"}
    elif load_healthy:
        defaults = {"age":35,"bp":70,"sg":1.025,"al":0,"su":0,"bgr":100,"bu":25,
                    "sc":0.8,"sod":142.0,"pot":4.2,"hemo":15.0,"pcv":45,"wc":7500,"rc":5.2,
                    "rbc":"normal","pc":"normal","pcc":"notpresent","ba":"notpresent"}

    # Input form
    col1, col2, col3 = st.columns(3)
    with col1:
        age  = st.number_input("Age (years)", 2, 90, defaults["age"])
        bp   = st.number_input("Blood Pressure (mm Hg)", 50, 180, defaults["bp"])
        sg   = st.number_input("Specific Gravity", 1.005, 1.025, defaults["sg"], step=0.005, format="%.3f")
        al   = st.slider("Albumin (0-5)", 0, 5, defaults["al"])
        su   = st.slider("Sugar (0-5)", 0, 5, defaults["su"])
    with col2:
        bgr  = st.number_input("Blood Glucose (mg/dL)", 22, 490, defaults["bgr"])
        bu   = st.number_input("Blood Urea (mg/dL)", 1, 400, defaults["bu"])
        sc   = st.number_input("Serum Creatinine (mg/dL)", 0.1, 80.0, defaults["sc"], step=0.1)
        sod  = st.number_input("Sodium (mEq/L)", 4.5, 163.0, defaults["sod"])
        pot  = st.number_input("Potassium (mEq/L)", 2.5, 47.0, defaults["pot"], step=0.1)
    with col3:
        hemo = st.number_input("Hemoglobin (g/dL)", 3.1, 17.8, defaults["hemo"], step=0.5)
        pcv  = st.number_input("Packed Cell Volume (%)", 9, 54, defaults["pcv"])
        wc   = st.number_input("White Blood Cells (/cmm)", 2200, 26400, defaults["wc"], step=500)
        rc   = st.number_input("Red Blood Cells (M/cmm)", 2.1, 8.0, defaults["rc"], step=0.1)

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1: rbc = st.selectbox("Red Blood Cells", ["normal","abnormal"], index=["normal","abnormal"].index(defaults["rbc"]))
    with col_c2: pc  = st.selectbox("Pus Cell", ["normal","abnormal"], index=["normal","abnormal"].index(defaults["pc"]))
    with col_c3: pcc = st.selectbox("Pus Cell Clumps", ["notpresent","present"], index=["notpresent","present"].index(defaults["pcc"]))
    with col_c4: ba  = st.selectbox("Bacteria", ["notpresent","present"], index=["notpresent","present"].index(defaults["ba"]))

    if st.button("🔍 Predict CKD Risk", type="primary", use_container_width=True):
        patient = {"age":age,"bp":bp,"sg":sg,"al":al,"su":su,"bgr":bgr,"bu":bu,"sc":sc,
                   "sod":sod,"pot":pot,"hemo":hemo,"pcv":pcv,"wc":wc,"rc":rc,
                   "rbc":rbc,"pc":pc,"pcc":pcc,"ba":ba,"model":model_choice}
        result = call_backend("/predict", patient)
        if result and "error" not in result:
            prob = result["probability"]
            pred = result["prediction"]
            model_name = "Random Forest" if result["model"]=="rf" else "XGBoost"

            # Prediction card
            if pred == 1:
                st.markdown(f'<div class="pred-ckd"><h2>⚠ CKD Detected</h2>'
                            f'<h3>Probability: {prob:.1%}</h3><p>Model: {model_name}</p></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pred-notckd"><h2>✓ No CKD Detected</h2>'
                            f'<h3>Probability: {prob:.1%}</h3><p>Model: {model_name}</p></div>',
                            unsafe_allow_html=True)

            # SHAP waterfall
            st.subheader("SHAP Explanation — Why This Prediction?")
            shap_df = pd.DataFrame({"Feature": result["feature_names"],
                                     "SHAP Value": result["shap_values"],
                                     "Feature Value": result["feature_values"]})
            shap_df["abs_shap"] = shap_df["SHAP Value"].abs()
            shap_df = shap_df.sort_values("abs_shap").tail(15)
            shap_df["Direction"] = shap_df["SHAP Value"].apply(
                lambda v: "→ CKD Risk ↑" if v > 0 else "→ CKD Risk ↓")

            fig, ax = plt.subplots(figsize=(10, 7))
            colors = [CKD_RED if v > 0 else NOTCKD_BLUE for v in shap_df["SHAP Value"]]
            bars = ax.barh(range(len(shap_df)), shap_df["SHAP Value"], color=colors, edgecolor="white", height=0.7)
            ax.set_yticks(range(len(shap_df)))
            ax.set_yticklabels([f"{f}  ({v:.2f})" for f, v in
                                zip(shap_df["Feature"], shap_df["Feature Value"])], fontsize=11)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP Value (impact on CKD prediction)", fontsize=12)
            ax.set_title(f"Feature Contributions — {model_name}\nBase value: {result['base_value']:.3f}",
                         fontweight="bold", fontsize=14)
            red_patch = mpatches.Patch(color=CKD_RED, label="Pushes toward CKD")
            blue_patch = mpatches.Patch(color=NOTCKD_BLUE, label="Pushes toward NotCKD")
            ax.legend(handles=[red_patch, blue_patch], loc="lower right", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)

            # Feature table
            with st.expander("📋 Full Feature Values & SHAP Contributions"):
                display_df = pd.DataFrame({"Feature": result["feature_names"],
                                            "Value": [round(v,3) for v in result["feature_values"]],
                                            "SHAP": [round(v,4) for v in result["shap_values"]]})
                display_df = display_df.sort_values("SHAP", key=abs, ascending=False)
                st.dataframe(display_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Model Performance")

    # Metrics
    metrics = call_backend("/metrics")
    if metrics:
        df_m = pd.DataFrame(metrics["metrics"])

        # Summary cards
        best = df_m.iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Best Model", best["Model"])
        c2.metric("Accuracy", f"{best['Accuracy']:.1%}")
        c3.metric("ROC-AUC", f"{best['ROC_AUC']:.3f}")
        c4.metric("F1-Score", f"{best['F1_Score']:.3f}")

        st.subheader("Test Set Metrics — All 5 Models")
        st.dataframe(df_m.style.format({"Accuracy":"{:.3f}","Precision":"{:.3f}",
                     "Recall":"{:.3f}","F1_Score":"{:.3f}","ROC_AUC":"{:.3f}"}
                     ).background_gradient(cmap="YlGn", subset=["ROC_AUC"]),
                     use_container_width=True, hide_index=True)

        # Comparison bar chart
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.subheader("Performance Comparison")
            fig = go.Figure()
            for metric, color in [("ROC_AUC","#4C72B0"),("F1_Score","#55A868"),("Accuracy","#C44E52")]:
                fig.add_trace(go.Bar(name=metric, x=df_m["Model"], y=df_m[metric],
                                     marker_color=color))
            fig.update_layout(barmode="group", yaxis=dict(range=[0.85,1.02], title="Score"),
                              height=400, legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, use_container_width=True)

        with col_p2:
            st.subheader("ROC Curves")
            roc = call_backend("/roc_data")
            if roc:
                fig = go.Figure()
                for c in roc["curves"]:
                    fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines",
                                             name=f'{c["model"]} (AUC={c["auc"]})',
                                             line=dict(width=2)))
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                         line=dict(dash="dash", color="grey"), name="Chance"))
                fig.update_layout(height=400, xaxis_title="FPR", yaxis_title="TPR",
                                  legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
                st.plotly_chart(fig, use_container_width=True)

        # Confusion matrices
        st.subheader("Confusion Matrices")
        cm_cols = st.columns(5)
        for i, (mk, mname) in enumerate([("rf","Random Forest"),("xgb","XGBoost"),
                                          ("lr","Logistic Reg."),("svm","SVM"),("dt","Decision Tree")]):
            with cm_cols[i]:
                cm_data = call_backend(f"/confusion_matrix?model={mk}")
                if cm_data:
                    cm = np.array(cm_data["matrix"])
                    fig, ax = plt.subplots(figsize=(3,3))
                    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
                    for r in range(2):
                        for c in range(2):
                            ax.text(c, r, str(cm[r][c]), ha="center", va="center",
                                    fontsize=16, fontweight="bold",
                                    color="white" if cm[r][c] > cm.max()/2 else "black")
                    ax.set_xticks([0,1]); ax.set_yticks([0,1])
                    ax.set_xticklabels(["NotCKD","CKD"], fontsize=8)
                    ax.set_yticklabels(["NotCKD","CKD"], fontsize=8)
                    ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("Actual", fontsize=9)
                    ax.set_title(mname, fontweight="bold", fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig)

        # Feature Importance — all 5 models
        st.subheader("Feature Importance — All 5 Models")
        fi_data = call_backend("/feature_importance")
        if fi_data:
            fi_model = st.selectbox("Select model for feature importance:",
                                     list(fi_data["models"].keys()),
                                     format_func=lambda k: fi_data["models"][k]["name"])
            imp = fi_data["models"][fi_model]["importance"]
            fi_df = pd.DataFrame({"Feature": fi_data["features"], "Importance": imp})
            fi_df = fi_df.sort_values("Importance", ascending=True).tail(15)

            fig, ax = plt.subplots(figsize=(10, 6))
            color = list(MODEL_COLORS.values())[list(fi_data["models"].keys()).index(fi_model)]
            ax.barh(fi_df["Feature"], fi_df["Importance"], color=color, edgecolor="white")
            ax.set_xlabel("Normalized Importance", fontsize=12)
            ax.set_title(f"Feature Importance — {fi_data['models'][fi_model]['name']}",
                         fontweight="bold", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)

            # Side-by-side all models
            with st.expander("📊 Compare All Models Side-by-Side"):
                all_imp = []
                for mk, mdata in fi_data["models"].items():
                    for f, v in zip(fi_data["features"], mdata["importance"]):
                        all_imp.append({"Feature":f, "Importance":v, "Model":mdata["name"]})
                all_df = pd.DataFrame(all_imp)
                # Top 10 by average
                avg = all_df.groupby("Feature")["Importance"].mean().nlargest(10).index
                plot_df = all_df[all_df["Feature"].isin(avg)]
                fig = px.bar(plot_df, x="Importance", y="Feature", color="Model",
                             orientation="h", barmode="group", height=500,
                             color_discrete_map=MODEL_COLORS)
                fig.update_layout(yaxis=dict(categoryorder="total ascending"))
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: SHAP GLOBAL
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("SHAP Global Explanations")

    shap_model = st.selectbox("Select Model:", ["rf","xgb"],
                               format_func=lambda x: "Random Forest" if x=="rf" else "XGBoost",
                               key="shap_global_model")

    col_g1, col_g2 = st.columns(2)

    # Bar plot
    with col_g1:
        st.subheader("Mean |SHAP Value| — Feature Importance")
        shap_data = call_backend(f"/shap_global?model={shap_model}")
        if shap_data:
            sdf = pd.DataFrame({"Feature":shap_data["features"],"Importance":shap_data["importance"]})
            sdf = sdf.sort_values("Importance", ascending=True).tail(15)
            bar_color = RF_GREEN if shap_model=="rf" else XGB_RED
            fig, ax = plt.subplots(figsize=(8,6))
            ax.barh(sdf["Feature"], sdf["Importance"], color=bar_color, edgecolor="white")
            ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
            ax.set_title("SHAP Feature Importance", fontweight="bold", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)

    # Beeswarm
    with col_g2:
        st.subheader("SHAP Beeswarm — Feature Impact Distribution")
        bee = call_backend(f"/shap_beeswarm?model={shap_model}")
        if bee:
            # Build DataFrame
            n_samples = len(bee["shap_matrix"][0])
            rows = []
            for i, feat in enumerate(bee["features"]):
                for j in range(n_samples):
                    rows.append({"Feature":feat, "SHAP":bee["shap_matrix"][i][j],
                                 "Value":bee["feature_matrix"][i][j]})
            bdf = pd.DataFrame(rows)
            # Top 15 features by mean |SHAP|
            top15 = bdf.groupby("Feature")["SHAP"].apply(lambda x: np.abs(x).mean()).nlargest(15).index
            bdf = bdf[bdf["Feature"].isin(top15)]
            rank_order = bdf.groupby("Feature")["SHAP"].apply(lambda x: np.abs(x).mean()).sort_values()
            bdf["Feature"] = pd.Categorical(bdf["Feature"], categories=rank_order.index, ordered=True)

            fig, ax = plt.subplots(figsize=(8,6))
            scatter = ax.scatter(bdf["SHAP"], bdf["Feature"], c=bdf["Value"],
                                  cmap="coolwarm", alpha=0.5, s=8, edgecolors="none")
            ax.axvline(0, color="grey", linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label="Feature Value", shrink=0.8)
            ax.set_xlabel("SHAP Value", fontsize=12)
            ax.set_title("SHAP Beeswarm", fontweight="bold", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)

    # RF vs XGB comparison
    st.subheader("Random Forest vs. XGBoost — SHAP Comparison")
    comp = call_backend("/shap_comparison")
    if comp:
        cdf = pd.DataFrame({"Feature":comp["features"],
                             "Random Forest":comp["rf"], "XGBoost":comp["xgb"]})
        cdf["Average"] = (cdf["Random Forest"]+cdf["XGBoost"])/2
        cdf = cdf.sort_values("Average", ascending=True).tail(12)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Random Forest", y=cdf["Feature"], x=cdf["Random Forest"],
                              orientation="h", marker_color=RF_GREEN))
        fig.add_trace(go.Bar(name="XGBoost", y=cdf["Feature"], x=cdf["XGBoost"],
                              orientation="h", marker_color=XGB_RED))
        fig.update_layout(barmode="group", height=450, xaxis_title="Mean |SHAP Value|",
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: SHAP LOCAL — Individual Patient Explanations
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("SHAP Local Explanations — Individual Patients")
    st.markdown("Browse test-set patients and see **why** each prediction was made. "
                "Particularly useful for understanding **misclassified** cases.")

    local_model = st.selectbox("Model:", ["rf","xgb"],
                                format_func=lambda x: "Random Forest" if x=="rf" else "XGBoost",
                                key="local_model")

    # Load patient list
    pts = call_backend(f"/test_patients?model={local_model}")
    if pts:
        pt_list = pts["patients"]

        # Filter options
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            filter_type = st.multiselect("Filter by type:", ["TP","TN","FN","FP"],
                                          default=["TP","TN","FN","FP"])
        with col_f2:
            n_tp = sum(1 for p in pt_list if p["type"]=="TP")
            n_tn = sum(1 for p in pt_list if p["type"]=="TN")
            n_fn = sum(1 for p in pt_list if p["type"]=="FN")
            n_fp = sum(1 for p in pt_list if p["type"]=="FP")
            st.markdown(f"**TP:** {n_tp} · **TN:** {n_tn} · **FN:** {n_fn} · **FP:** {n_fp}")
        with col_f3:
            st.markdown(f"**Model:** {pts['model']} · **Total:** {len(pt_list)}")

        filtered = [p for p in pt_list if p["type"] in filter_type]

        # Patient selector
        pt_options = {f"Patient {p['idx']} — True: {p['true']}, Pred: {p['pred']} ({p['type']}) — P(CKD)={p['prob']:.3f}": p["idx"]
                      for p in filtered}

        if pt_options:
            selected = st.selectbox("Select patient:", list(pt_options.keys()))
            pidx = pt_options[selected]

            # Fetch SHAP for this patient
            local_data = call_backend(f"/shap_local?model={local_model}&patient_idx={pidx}")
            if local_data:
                # Info cards
                ic1, ic2, ic3, ic4 = st.columns(4)
                ic1.metric("True Label", local_data["true_label_str"] if "true_label_str" in local_data
                           else ("CKD" if local_data["true_label"]==1 else "NotCKD"))
                ic2.metric("Predicted", "CKD" if local_data["pred_label"]==1 else "NotCKD")
                ic3.metric("P(CKD)", f"{local_data['probability']:.3f}")
                correct = local_data["true_label"] == local_data["pred_label"]
                ic4.metric("Result", "✅ Correct" if correct else "❌ Misclassified")

                # Waterfall plot
                st.subheader(f"SHAP Waterfall — Patient {pidx}")
                shap_df = pd.DataFrame({
                    "Feature": local_data["feature_names"],
                    "SHAP": local_data["shap_values"],
                    "Value": local_data["feature_values"]
                })
                shap_df = shap_df.reindex(shap_df["SHAP"].abs().sort_values().index).tail(15)

                fig, ax = plt.subplots(figsize=(10, 7))
                colors = [CKD_RED if v > 0 else NOTCKD_BLUE for v in shap_df["SHAP"]]
                ax.barh(range(len(shap_df)), shap_df["SHAP"], color=colors,
                        edgecolor="white", height=0.7)
                ax.set_yticks(range(len(shap_df)))
                ax.set_yticklabels([f"{f}  ({v:.2f})" for f, v in
                                    zip(shap_df["Feature"], shap_df["Value"])], fontsize=11)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP Value", fontsize=12)
                true_str = "CKD" if local_data["true_label"]==1 else "NotCKD"
                pred_str = "CKD" if local_data["pred_label"]==1 else "NotCKD"
                ax.set_title(f"Patient {pidx} — True: {true_str} | Pred: {pred_str} | "
                             f"P(CKD)={local_data['probability']:.3f}\n"
                             f"Base value: {local_data['base_value']:.3f}",
                             fontweight="bold", fontsize=13)
                red_patch = mpatches.Patch(color=CKD_RED, label="→ CKD")
                blue_patch = mpatches.Patch(color=NOTCKD_BLUE, label="→ NotCKD")
                ax.legend(handles=[red_patch, blue_patch], loc="lower right")
                plt.tight_layout()
                st.pyplot(fig)

                # Feature values table
                with st.expander("📋 All Feature Values & SHAP"):
                    full_df = pd.DataFrame({
                        "Feature": local_data["feature_names"],
                        "Value": [round(v,3) for v in local_data["feature_values"]],
                        "SHAP": [round(v,4) for v in local_data["shap_values"]]
                    }).sort_values("SHAP", key=abs, ascending=False)
                    st.dataframe(full_df, use_container_width=True, hide_index=True)
        else:
            st.info("No patients match the selected filter.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: SHAP DEPENDENCE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("SHAP Dependence Plots")
    st.markdown("Explore how individual feature values relate to their SHAP contribution. "
                "Non-linear patterns and thresholds become visible.")

    dep_col1, dep_col2 = st.columns([1,3])
    with dep_col1:
        dep_model = st.selectbox("Model:", ["rf","xgb"],
                                  format_func=lambda x: "Random Forest" if x=="rf" else "XGBoost",
                                  key="dep_model")
        dep_feature = st.selectbox("Feature:", ["sg","hemo","sc","al","pcv","bu","rc",
                                                 "bgr","age","bp","sod","pot","su","wc"],
                                    key="dep_feature")

    with dep_col2:
        dep = call_backend(f"/shap_dependence?model={dep_model}&feature={dep_feature}")
        if dep:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.scatter(dep["feature_values"], dep["shap_values"],
                       alpha=0.5, c=NOTCKD_BLUE, s=20, edgecolors="none")
            # Trend line
            z = np.polyfit(dep["feature_values"], dep["shap_values"], 3)
            xline = np.linspace(min(dep["feature_values"]), max(dep["feature_values"]), 100)
            ax.plot(xline, np.polyval(z, xline), color=CKD_RED, linewidth=2, alpha=0.7)
            ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
            ax.set_xlabel(dep_feature, fontsize=13)
            ax.set_ylabel("SHAP Value", fontsize=13)
            model_name = "Random Forest" if dep_model=="rf" else "XGBoost"
            ax.set_title(f"SHAP Dependence: {dep_feature} — {model_name} (Training Set)",
                         fontweight="bold", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)

    # Multi-feature dependence grid
    st.subheader("Top 6 Features — Dependence Grid")
    shap_data = call_backend(f"/shap_global?model={dep_model}")
    if shap_data:
        imp_df = pd.DataFrame({"f":shap_data["features"],"i":shap_data["importance"]})
        top6 = imp_df.nlargest(6, "i")["f"].tolist()

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for feat, ax in zip(top6, axes.flat):
            dep_d = call_backend(f"/shap_dependence?model={dep_model}&feature={feat}")
            if dep_d:
                ax.scatter(dep_d["feature_values"], dep_d["shap_values"],
                           alpha=0.4, c=NOTCKD_BLUE, s=12, edgecolors="none")
                ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
                ax.set_title(feat, fontweight="bold", fontsize=12)
                ax.set_xlabel(""); ax.set_ylabel("SHAP")
        model_name = "Random Forest" if dep_model=="rf" else "XGBoost"
        plt.suptitle(f"SHAP Dependence — Top 6 Features ({model_name})",
                     fontweight="bold", fontsize=15, y=1.01)
        plt.tight_layout()
        st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("Dataset Explorer")

    data = call_backend("/dataset")
    if data:
        df = pd.DataFrame(data["data"])
        st.dataframe(df, use_container_width=True, height=400)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.subheader("Feature Distribution by Class")
            dist_feat = st.selectbox("Feature:", ["hemo","sg","sc","al","bu","bgr",
                                                   "pcv","rc","age","bp","sod","pot"],
                                      key="dist_feat")
            dist = call_backend(f"/feature_distribution?feature={dist_feat}")
            if dist:
                ddf = pd.DataFrame({"value":dist["values"],"class":dist["classes"]})
                fig = px.histogram(ddf, x="value", color="class", barmode="overlay",
                                    color_discrete_map={"ckd":CKD_RED,"notckd":NOTCKD_BLUE},
                                    opacity=0.7, labels={"value":dist_feat})
                st.plotly_chart(fig, use_container_width=True)

        with col_d2:
            st.subheader("Missing Data Summary")
            miss = call_backend("/missing_data")
            if miss:
                mdf = pd.DataFrame({"Feature":miss["features"],"Pct":miss["missing_pct"]})
                mdf = mdf[mdf["Pct"]>0].sort_values("Pct")
                fig, ax = plt.subplots(figsize=(8,5))
                ax.barh(mdf["Feature"], mdf["Pct"], color=CKD_RED, edgecolor="white")
                ax.set_xlabel("% Missing", fontsize=12)
                ax.set_title("Missing Values by Feature", fontweight="bold", fontsize=14)
                plt.tight_layout()
                st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("About This Application")

    st.markdown("""
### CKD-Insight App: Explainable Machine Learning for CKD Risk Prediction

This CKD-Insight App was developed as part of **Project 1 for BME6938: Medical AI** at the University of Florida.
It demonstrates an end-to-end machine learning pipeline for predicting Chronic Kidney Disease (CKD)
and providing model interpretability using SHAP (SHapley Additive exPlanations).

### Project Pipeline

| Phase | Description |
|-------|-------------|
| **Phase I** | Exploratory Data Analysis — data cleaning, feature inspection, leakage identification |
| **Phase II** | 5 ML models trained with GridSearchCV — LR, DT, RF, SVM, XGBoost |
| **Phase III** | SHAP explainability — global & local interpretability for RF and XGBoost |
| **Phase IV** | This interactive Streamlit application |

### Dataset
UCI Chronic Kidney Disease dataset — **400 patients**, **24 features**, binary classification (CKD vs NotCKD).
7 post-diagnosis leakage features removed; **18 features retained** (14 numeric, 4 categorical).

### Key Results
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Random Forest | 97.5% | 0.999 |
| XGBoost | 97.5% | 0.998 |
| Logistic Regression | 95.0% | 0.994 |
| SVM | 92.5% | 0.995 |
| Decision Tree | 93.8% | 0.981 |

### Authors
**Joseph Luper Tsenum**:
Ph.D. Researcher in Biomedical Engineering (Modeling & Biomedical Data Science Specialization), University of Florida.
Joseph develops Generative AI platforms for designing novel oligonucleotides and applies machine learning methods
to biomedical data analysis and drug discovery.

**Riley Bendure**:
M.S. Researcher in Biomedical Engineering (Brain Signal Processing), University of Florida.
Riley utilizes machine learning methods improving modulation of non-motor symptoms in Parkinson's for adaptive deep brain stimulation with previous experience in Cochlear implant temporal signal processing. His aim is to bridge gaps between patient perception and effective neuromodulation in implantable neurostimulators.

**Gopal Viraj Koundinya Vutukuru**:
M.S. Student in Biomedical Engineering, University of Florida.
Gopal Viraj is a first‑year M.S. student in Biomedical Engineering at the University of Florida with a strong interest in biomaterials and regenerative medicine. He is open to pursuing both industry work and academic research in these areas in the future. His overall aim is to work in the healthcare industry to solve day‑to‑day diagnostic problems by applying the latest technologies in biomedical engineering.

## Individual Contributions 

Each team member contributed to the preparation of the written report and the development of the project deliverables. 

1. **Joseph Luper Tsenum** – Responsible for writing the Abstract and coordinating the project, ensuring that the different components of the analysis were well integrated and consistent across all four phases of the work. 

2. **Riley Bendure** – Responsible for writing the Introduction, providing background on chronic kidney disease (CKD) and motivating the importance of applying machine learning methods for early risk prediction. 

3. **Gopal Viraj Koundinya Vutukuru** – Responsible for writing the Literature Review, summarizing existing research on machine learning approaches for CKD prediction and identifying the motivation for explainable machine learning models. 

## Collaboration 

Throughout the four phases of the project, the team maintained a highly collaborative workflow, meeting regularly to discuss progress, make decisions, and coordinate tasks. As a group, we collectively selected the UCI Chronic Kidney Disease (CKD) dataset and worked together across all stages of the project, including exploratory data analysis (EDA), model development, evaluation, report preparation, and application development. 

Most of the work was conducted during in-person meetings, where team members jointly reviewed analyses, implemented modeling approaches, and refined the outputs for each phase. The final notebooks and project artifacts were compiled collaboratively to ensure consistency and reproducibility across the entire pipeline. 

Joseph was responsible for ensuring that the various components of the project—including data preprocessing, modeling outputs, explainability analyses, and the interactive application remained aligned and coherent across phases. At the same time, the collaborative contributions of the entire team made it possible to efficiently develop the README documentation and the “About” section of the application, as different sections written by team members were integrated into a unified narrative. 

This project reflects the type of collaborative environment commonly encountered in real-world industry and research settings, where interdisciplinary teams contribute complementary expertise. Team members were able to work together productively, resolve challenges constructively, and learn from one another’s technical strengths and soft skills, resulting in a cohesive and well-executed final product. 

## Declaration of AI Use 

We acknowledge the use of Claude Opus 4.6 AI and ChatGPT 5.2 tools to support our understanding and assist with coding, particularly in implementing the project in Python.

## References: 

- E. M. Chouit, M. Rachdi, M. Bellafkih, and B. Raouyane, “Interpretable machine learning for chronic kidney disease prediction: Insights from SHAP and LIME analyses,” PLoS One, vol. 21, no. 2, Art. no. e0343205, Feb. 2026.

- I. Balikci Cicek and Z. Kucukakcali, “Explainable Artificial Intelligence Method SHAP’s Prediction of Risk Factors Associated with Chronic Kidney Disease Combined with Black Box Methods,” J. Comm. Med. and Pub. Health Rep., vol. 4, no. 10, Nov. 2023.

- M. A. Islam, M. Z. H. Majumder, and M. A. Hussein, “Chronic kidney disease prediction based on machine learning algorithms,” J. Pathol. Inform., vol. 14, Art. no. 100189, Jan. 2023.

- S. Sharma et al., “Machine Learning Algorithm for Detecting and Predicting Chronic Kidney Disease,” Biomed. & Pharmacol. J., vol. 18, no. 2, pp. 1230–1245, June 2025.

- E. M. Senan et al., “Diagnosis of Chronic Kidney Disease Using Effective Classification Algorithms and Recursive Feature Elimination Techniques,” J. Healthc. Eng., vol. 2021, Art. no. 1004767, June 2021.

- R. K. Halder et al., “ML-CKDP: Machine learning-based chronic kidney disease prediction with smart web application,” J. Pathol. Inform., vol. 15, Art. no. 100371, Feb. 2024.

- B. Metherall, A. K. Berryman, and G. S. Brennan, “Machine learning for classifying chronic kidney disease and predicting creatinine levels using at-home measurements,” Sci. Rep., vol. 15, Art. no. 4330, Feb. 2025.

- P. B. Mark et al., “Global, regional, and national burden of chronic kidney disease in adults, 1990–2023,” The Lancet, vol. 406, no. 10518, pp. 2461–2482, 2025.

- L. Rubini, P. Soundarapandian, and P. Eswaran, “Chronic Kidney Disease,” UCI Machine Learning Repository, 2015. https://doi.org/10.24432/C5G020

- S. M. Lundberg and S. I. Lee, “A Unified Approach to Interpreting Model Predictions,” in Proc. NeurIPS, 2017.

- S. M. Lundberg et al., “From local explanations to global understanding with explainable AI for trees,” Nature Machine Intelligence, vol. 2, pp. 56–67, 2020.

- F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” J. Machine Learning Res., vol. 12, pp. 2825–2830, 2011.

- T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” in Proc. KDD, 2016.

- L. Breiman, “Random Forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.

- A. S. Levey and J. Coresh, “Chronic kidney disease,” The Lancet, vol. 379, no. 9811, pp. 165–180, 2012.

- L. S. Shapley, “A value for n-person games,” Contributions to the Theory of Games, vol. 2, no. 28, pp. 307–317, 1953.

- O. Troyanskaya et al., “Missing value estimation methods for DNA microarrays,” Bioinformatics, vol. 17, no. 6, pp. 520–525, 2001.

- D. W. Hosmer, S. Lemeshow, and R. X. Sturdivant, Applied Logistic Regression, 3rd ed., Wiley, 2013.

- L. Breiman, J. Friedman, R. Olshen, and C. Stone, Classification and Regression Trees, Wadsworth, 1984.

- C. Cortes and V. Vapnik, “Support-vector networks,” Machine Learning, vol. 20, pp. 273–297, 1995.

- L. Breiman, “Random Forests,” Machine Learning, vol. 45, pp. 5–32, 2001.

- T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016.

- S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” Advances in Neural Information Processing Systems (NeurIPS), 2017.

## License

This project is for educational purposes. The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

⚠ **Disclaimer:** This application is intended for educational and research purposes only
and is not designed for clinical use or medical decision-making.
""")
