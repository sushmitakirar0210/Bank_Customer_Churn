import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from src.explainability import build_shap_artifacts
from src.preprocessing import build_preprocessor
from xgboost import XGBClassifier


def train_and_evaluate(data_path="data/Churn_Modelling.csv"):
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=neg_pos_ratio,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    results = []
    trained_pipelines = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in models.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", clf),
        ])

        cv_auc = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="roc_auc"
        ).mean()
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "CV_ROC_AUC": round(cv_auc, 4),
            "Test_Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1_Score": round(f1_score(y_test, y_pred), 4),
            "Test_ROC_AUC": round(roc_auc_score(y_test, y_prob), 4),
        })
        trained_pipelines[name] = pipe

    results_df = pd.DataFrame(results).sort_values(
        by="Test_ROC_AUC", ascending=False
    )
    print("\n=== Model Benchmarking Results ===")
    print(results_df.to_string(index=False))

    # Select top tree-based/boosting model so SHAP TreeExplainer works seamlessly
    tree_models = results_df[results_df["Model"] != "Logistic Regression"]
    best_model_name = tree_models.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]
    print(
        f"\nSelected Production Model: {best_model_name} (Test ROC-AUC:"
        f" {tree_models.iloc[0]['Test_ROC_AUC']})"
    )

    explainer, shap_values, feature_names = build_shap_artifacts(
        best_pipeline, X_test
    )

    joblib.dump(best_pipeline, "models/churn_pipeline.pkl")
    joblib.dump(explainer, "models/shap_explainer.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")
    joblib.dump(shap_values, "models/test_shap_values.pkl")
    results_df.to_csv("models/model_benchmark.csv", index=False)

    test_eval_df = X_test.copy()
    test_eval_df["Actual_Exited"] = y_test.values
    test_eval_df["Churn_Probability"] = best_pipeline.predict_proba(X_test)[
        :, 1
    ]
    test_eval_df.to_csv("models/test_predictions.csv", index=False)

    print("All model artifacts saved successfully inside /models.")


if __name__ == "__main__":
    train_and_evaluate()
