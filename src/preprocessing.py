import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Generates behavioral and financial interaction features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        cols_to_drop = [
            c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns
        ]
        df = df.drop(columns=cols_to_drop)

        df["Balance_to_Salary"] = df["Balance"] / (df["EstimatedSalary"] + 1.0)
        df["Product_Density"] = df["NumOfProducts"] / (df["Tenure"] + 1.0)
        df["Engagement_Product_Interaction"] = (
            df["IsActiveMember"] * df["NumOfProducts"]
        )
        df["Age_Tenure_Ratio"] = df["Tenure"] / (df["Age"] + 1e-5)
        df["Is_Zero_Balance"] = (df["Balance"] == 0).astype(int)

        return df


def build_preprocessor():
    """Creates the complete preprocessing pipeline."""
    num_features = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Balance_to_Salary",
        "Product_Density",
        "Engagement_Product_Interaction",
        "Age_Tenure_Ratio",
        "Is_Zero_Balance",
    ]
    cat_features = ["Geography", "Gender"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                drop="first", handle_unknown="ignore", sparse_output=False
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("feature_engineering", ChurnFeatureEngineer()),
        ("column_transformer", preprocessor),
    ])
