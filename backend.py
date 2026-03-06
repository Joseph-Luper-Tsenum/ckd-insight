"""
══════════════════════════════════════════════════════════════════════════════
Phase IV — Python Backend (Flask) for CKD Prediction App
BME6938 Medical AI · Project 1 · Group 6
Author: Joseph Luper Tsenum

Usage:  python backend.py
══════════════════════════════════════════════════════════════════════════════
"""

import os, warnings, numpy as np, pandas as pd, joblib
from pathlib import Path
from flask import Flask, request, jsonify
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import shap

warnings.filterwarnings("ignore")
app = Flask(__name__)

# ── Paths & Load ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(".")
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR  = PROJECT_ROOT / "data"

print("Loading artifacts...")
models = {
    "rf":  joblib.load(MODEL_DIR / "ckd_random_forest_tuned.joblib"),
    "xgb": joblib.load(MODEL_DIR / "ckd_xgboost_tuned.joblib"),
    "lr":  joblib.load(MODEL_DIR / "ckd_logistic_regression_tuned.joblib"),
    "svm": joblib.load(MODEL_DIR / "ckd_svm_tuned.joblib"),
    "dt":  joblib.load(MODEL_DIR / "ckd_decision_tree_tuned.joblib"),
}
model_names = {"rf":"Random Forest","xgb":"XGBoost","lr":"Logistic Regression",
               "svm":"SVM","dt":"Decision Tree"}

preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
X_train = np.load(MODEL_DIR / "X_train_processed.npy")
X_test  = np.load(MODEL_DIR / "X_test_processed.npy")
y_train = np.load(MODEL_DIR / "y_train.npy")
y_test  = np.load(MODEL_DIR / "y_test.npy")
feature_names = pd.read_csv(MODEL_DIR / "feature_names.csv")["feature"].tolist()

# Precomputed SHAP values
shap_values = {}
for m in ["rf","xgb"]:
    for split in ["test","train"]:
        fpath = MODEL_DIR / f"{m}_shap_values_{split}.npy"
        if fpath.exists():
            sv = np.load(fpath)
            if sv.ndim == 3: sv = sv[:,:,1]
            shap_values[f"{m}_{split}"] = sv

rf_explainer  = shap.TreeExplainer(models["rf"])
xgb_explainer = shap.TreeExplainer(models["xgb"])
explainers = {"rf": rf_explainer, "xgb": xgb_explainer}

dataset_path = DATA_DIR / "ckd_cleaned.csv"
dataset_df = pd.read_csv(dataset_path) if dataset_path.exists() else pd.DataFrame()

X_test_df  = pd.DataFrame(X_test, columns=feature_names)
X_train_df = pd.DataFrame(X_train, columns=feature_names)

# Precompute predictions for local explanation tab
y_preds = {k: m.predict(X_test) for k,m in models.items()}
y_probas = {k: m.predict_proba(X_test)[:,1] for k,m in models.items()}

print(f"Models: {list(models.keys())} | Features: {len(feature_names)}")
print(f"SHAP: {list(shap_values.keys())}")
print("Backend ready.\n")

# ── Helper ───────────────────────────────────────────────────────────────────
def safe_shap_row(sv, idx):
    """Extract 1D SHAP row from potentially 3D array."""
    row = sv[idx]
    if row.ndim > 1: row = row[:,1]
    return row.ravel()

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    model_key = data.pop("model", "rf")
    numeric_cols = ["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"]
    cat_cols = ["rbc","pc","pcc","ba"]
    row = {c: float(data.get(c,0)) for c in numeric_cols}
    row.update({c: str(data.get(c,"normal")) for c in cat_cols})
    raw_df = pd.DataFrame([row])
    try:
        X_proc = preprocessor.transform(raw_df)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {e}"}), 400

    model = models.get(model_key, models["rf"])
    pred = int(model.predict(X_proc)[0])
    prob = float(model.predict_proba(X_proc)[0][1])

    explainer = explainers.get(model_key)
    if explainer:
        sv = explainer.shap_values(X_proc)
        if isinstance(sv, list): sv = sv[1]
        sv = np.array(sv)
        if sv.ndim == 3: sv = sv[:,:,1]
        shap_vals = sv[0].ravel().tolist()
        bv = explainer.expected_value
        if isinstance(bv, (list, np.ndarray)): bv = float(bv[1])
        else: bv = float(bv)
    else:
        shap_vals = [0.0]*len(feature_names); bv = 0.5

    return jsonify({"prediction":pred,"probability":prob,"model":model_key,
                    "shap_values":shap_vals,"base_value":bv,
                    "feature_names":feature_names,
                    "feature_values":X_proc[0].tolist()})


@app.route("/metrics", methods=["GET"])
def metrics():
    results = []
    for key, model in models.items():
        yp = y_preds[key]; ypr = y_probas[key]
        results.append({"Model":model_names[key],
            "Accuracy":round(accuracy_score(y_test,yp),3),
            "Precision":round(precision_score(y_test,yp),3),
            "Recall":round(recall_score(y_test,yp),3),
            "F1_Score":round(f1_score(y_test,yp),3),
            "ROC_AUC":round(roc_auc_score(y_test,ypr),3)})
    results.sort(key=lambda x: x["ROC_AUC"], reverse=True)
    return jsonify({"metrics": results})


@app.route("/confusion_matrix", methods=["GET"])
def get_confusion_matrix():
    mk = request.args.get("model","rf")
    cm = confusion_matrix(y_test, y_preds.get(mk, y_preds["rf"])).tolist()
    return jsonify({"matrix":cm, "model_name":model_names.get(mk,mk)})


@app.route("/roc_data", methods=["GET"])
def roc_data():
    curves = []
    for key in models:
        fpr,tpr,_ = roc_curve(y_test, y_probas[key])
        auc = roc_auc_score(y_test, y_probas[key])
        step = max(1, len(fpr)//100)
        curves.append({"model":model_names[key],"fpr":fpr[::step].tolist(),
                       "tpr":tpr[::step].tolist(),"auc":round(auc,3)})
    return jsonify({"curves":curves})


@app.route("/feature_importance", methods=["GET"])
def feature_importance():
    """Return feature importance for all 5 models."""
    result = {}
    for key, model in models.items():
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0])
        else:
            # SVM with RBF — use permutation importance proxy via SHAP if available
            if f"{key}_test" in shap_values:
                imp = np.abs(shap_values[f"{key}_test"]).mean(axis=0).ravel()
            else:
                imp = np.zeros(len(feature_names))
        imp = (imp / imp.sum()).tolist() if imp.sum() > 0 else imp.tolist()
        result[key] = {"name": model_names[key], "importance": imp}
    return jsonify({"features": feature_names, "models": result})


@app.route("/shap_global", methods=["GET"])
def shap_global():
    mk = request.args.get("model","rf")
    key = f"{mk}_test"
    if key not in shap_values:
        return jsonify({"error":f"SHAP not found for {mk}"}),404
    imp = np.abs(shap_values[key]).mean(axis=0).ravel().tolist()
    return jsonify({"features":feature_names,"importance":imp,"model":mk})


@app.route("/shap_beeswarm", methods=["GET"])
def shap_beeswarm():
    mk = request.args.get("model","rf")
    key = f"{mk}_test"
    if key not in shap_values:
        return jsonify({"error":"SHAP not found"}),404
    sv = shap_values[key]
    return jsonify({"features":feature_names,
        "shap_matrix":[sv[:,i].tolist() for i in range(sv.shape[1])],
        "feature_matrix":[X_test[:,i].tolist() for i in range(X_test.shape[1])]})


@app.route("/shap_dependence", methods=["GET"])
def shap_dependence():
    mk = request.args.get("model","rf")
    feat = request.args.get("feature","sg")
    key = f"{mk}_train"
    if key not in shap_values or feat not in feature_names:
        return jsonify({"error":"Not found"}),404
    idx = feature_names.index(feat)
    return jsonify({"feature":feat,
        "feature_values":X_train[:,idx].tolist(),
        "shap_values":shap_values[key][:,idx].tolist()})


@app.route("/shap_local", methods=["GET"])
def shap_local():
    """Return SHAP values for a specific test patient."""
    mk = request.args.get("model","rf")
    pidx = int(request.args.get("patient_idx",0))
    key = f"{mk}_test"
    if key not in shap_values or pidx >= len(y_test):
        return jsonify({"error":"Invalid index or model"}),404

    sv = shap_values[key]
    row = sv[pidx].ravel().tolist()

    bv = explainers[mk].expected_value if mk in explainers else 0.5
    if isinstance(bv, (list,np.ndarray)): bv = float(bv[1])
    else: bv = float(bv)

    return jsonify({
        "patient_idx": pidx,
        "true_label": int(y_test[pidx]),
        "pred_label": int(y_preds.get(mk, y_preds["rf"])[pidx]),
        "probability": float(y_probas.get(mk, y_probas["rf"])[pidx]),
        "shap_values": row,
        "base_value": bv,
        "feature_names": feature_names,
        "feature_values": X_test[pidx].tolist()
    })


@app.route("/shap_comparison", methods=["GET"])
def shap_comparison():
    """Return SHAP importance for RF and XGB side by side."""
    result = {}
    for mk in ["rf","xgb"]:
        key = f"{mk}_test"
        if key in shap_values:
            result[mk] = np.abs(shap_values[key]).mean(axis=0).ravel().tolist()
        else:
            result[mk] = [0]*len(feature_names)
    return jsonify({"features":feature_names, "rf":result["rf"], "xgb":result["xgb"]})


@app.route("/test_patients", methods=["GET"])
def test_patients():
    """Return summary of test patients for local explanation browser."""
    mk = request.args.get("model","rf")
    preds = y_preds.get(mk, y_preds["rf"])
    probs = y_probas.get(mk, y_probas["rf"])
    patients = []
    for i in range(len(y_test)):
        true_l = int(y_test[i])
        pred_l = int(preds[i])
        correct = true_l == pred_l
        patients.append({
            "idx":i, "true":"CKD" if true_l==1 else "NotCKD",
            "pred":"CKD" if pred_l==1 else "NotCKD",
            "prob":round(float(probs[i]),3), "correct":correct,
            "type": ("TP" if true_l==1 and pred_l==1 else
                     "TN" if true_l==0 and pred_l==0 else
                     "FN" if true_l==1 and pred_l==0 else "FP")
        })
    return jsonify({"patients":patients, "model":model_names.get(mk,mk)})


@app.route("/dataset", methods=["GET"])
def dataset():
    if dataset_df.empty: return jsonify({"error":"Dataset not found"}),404
    return jsonify({"data":dataset_df.head(400).to_dict(orient="list")})

@app.route("/feature_distribution", methods=["GET"])
def feature_distribution():
    feat = request.args.get("feature","hemo")
    if dataset_df.empty or feat not in dataset_df.columns:
        return jsonify({"error":"Not available"}),404
    df = dataset_df[[feat,"classification"]].dropna()
    return jsonify({"values":df[feat].tolist(),"classes":df["classification"].tolist()})

@app.route("/missing_data", methods=["GET"])
def missing_data():
    if dataset_df.empty: return jsonify({"error":"Not found"}),404
    pct = (dataset_df.isna().mean()*100).round(1)
    return jsonify({"features":pct.index.tolist(),"missing_pct":pct.values.tolist()})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","models":list(models.keys()),"features":len(feature_names)})

if __name__ == "__main__":
    print("="*60)
    print("CKD Backend — http://127.0.0.1:5000")
    print("="*60)
    app.run(host="127.0.0.1", port=5000, debug=False)
