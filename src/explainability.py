import pandas as pd
import shap


def get_feature_names(pipeline):
    """Extracts transformed column names from the fitted preprocessor."""
    col_transformer = pipeline.named_steps["preprocessor"].named_steps[
        "column_transformer"
    ]
    num_cols = col_transformer.transformers_[0][2]
    cat_cols = (
        col_transformer.transformers_[1][1]
        .named_steps["encoder"]
        .get_feature_names_out(["Geography", "Gender"])
        .tolist()
    )
    return num_cols + cat_cols


def build_shap_artifacts(pipeline, X_test):
    """Creates a SHAP TreeExplainer and computes test set SHAP values."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_feature_names(pipeline)

    X_test_transformed = preprocessor.transform(X_test)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_test_df)

    return explainer, shap_values, feature_names


def explain_single_customer(pipeline, explainer, feature_names, customer_df):
    """Computes SHAP explanation object for a single customer row."""
    transformed = pipeline.named_steps["preprocessor"].transform(customer_df)
    transformed_df = pd.DataFrame(transformed, columns=feature_names)
    shap_exp = explainer(transformed_df)

    if len(shap_exp.shape) == 3:
        return shap_exp[0, :, 1]
    return shap_exp[0]
