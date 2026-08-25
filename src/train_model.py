"""
Model Training & Comparison - Bitcoin Price Direction Prediction
===================================================================
ทดลอง 3 โมเดล: Logistic Regression, Random Forest, Gradient Boosting
แล้วเปรียบเทียบผลลัพธ์ด้วย Accuracy / Precision / Recall / F1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Daily_Return", "High_Low_Range", "MA_7", "MA_21", "MA_Ratio",
    "Volatility_7", "RSI_14", "Volume_Change",
]
TARGET = "Target"


def main():
    df = pd.read_csv("data/bitcoin_features.csv", parse_dates=["Date"])

    # แบ่งข้อมูลตามลำดับเวลา (ไม่ shuffle) เพราะเป็น time series
    # 80% แรก (เก่า) = train, 20% หลัง (ใหม่) = test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }

    results = []
    trained_models = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        predictions[name] = pred
        trained_models[name] = model

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1-Score": f1_score(y_test, pred),
        })

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    print("=== ตารางเปรียบเทียบโมเดล ===")
    print(results_df.to_string(index=False))
    results_df.to_csv("data/model_comparison.csv", index=False)

    best_model_name = results_df.iloc[0]["Model"]
    print(f"\nโมเดลที่ดีที่สุด: {best_model_name}")

    # === กราฟที่ 1: เปรียบเทียบ metric ทั้ง 4 ตัว ===
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(results_df))
    width = 0.2
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, results_df[m], width, label=m)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df["Model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("ML Model Comparison - Bitcoin Price Direction Prediction")
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/chart_model_comparison.png", dpi=150)
    plt.close()

    # === กราฟที่ 2: Confusion Matrix ของโมเดลที่ดีที่สุด ===
    best_pred = predictions[best_model_name]
    cm = confusion_matrix(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Down (0)", "Up (1)"],
                yticklabels=["Down (0)", "Up (1)"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig("data/chart_confusion_matrix.png", dpi=150)
    plt.close()

    # === กราฟที่ 3: Feature Importance (Random Forest) ===
    rf_model = trained_models["Random Forest"]
    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color="#2a9d8f")
    ax.set_title("Feature Importance (Random Forest)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("data/chart_feature_importance.png", dpi=150)
    plt.close()

    # บันทึกโมเดล + scaler สำหรับใช้ใน Streamlit
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(trained_models[best_model_name], f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/all_models.pkl", "wb") as f:
        pickle.dump(trained_models, f)

    meta = {
        "features": FEATURES,
        "best_model": best_model_name,
        "test_size": len(test_df),
        "train_size": len(train_df),
    }
    with open("models/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nบันทึกโมเดลและกราฟเรียบร้อยแล้วที่ models/ และ data/")


if __name__ == "__main__":
    main()
