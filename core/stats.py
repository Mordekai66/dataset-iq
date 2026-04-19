import os
import pandas as pd
import json
import numpy as np


def detect_problem_type(target_col):
    if target_col.dtype == "object":
        return "classification"

    unique_values = target_col.nunique()
    total_values = len(target_col)

    if unique_values <= 20 or unique_values / total_values < 0.05:
        return "classification"

    return "regression"

def suggest_preprocessing(df, missing_pct, high_corr_cols):
    suggestions = []

    if missing_pct.mean() > 10:
        suggestions.append("Handle missing values (mean/median/mode or drop columns)")

    if len(high_corr_cols) > 0:
        suggestions.append("Drop highly correlated features")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        suggestions.append("Apply feature scaling (Standardization or Normalization)")

    categorical_cols = df.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        suggestions.append("Encode categorical variables (One-hot or Label encoding)")

    low_variance = [col for col in df.columns if df[col].nunique() <= 1]
    if low_variance:
        suggestions.append("Remove low variance features")

    return suggestions

def recommend_model(problem_type, df, target_data):
    rows = len(df)

    if problem_type == "classification":
        imbalance = target_data.value_counts(normalize=True).max()

        if rows < 1000:
            return "Logistic Regression or Random Forest"
        elif imbalance > 0.8:
            return "XGBoost with class_weight or SMOTE"
        else:
            return "Random Forest or XGBoost"

    else:
        skewness = target_data.skew()

        if rows < 1000:
            return "Linear Regression or Ridge"
        elif abs(skewness) > 1:
            return "XGBoost with log transformation"
        else:
            return "Random Forest Regressor or XGBoost"


def estimate_complexity(df, problem_type):
    rows = len(df)
    cols = len(df.columns)

    if rows < 1000 and cols < 10:
        return "easy"
    elif rows < 10000 and cols < 50:
        return "medium"
    else:
        return "hard"


def generate_stats(df):
    target_column = df.columns[-1]
    target_data = df[target_column]
    problem_type = detect_problem_type(target_data)

    missing_pct = (df.isnull().sum() / len(df)) * 100

    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    high_corr_cols = []

    if len(numeric_df.columns) > 1:
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_cols = [column for column in upper.columns if any(upper[column] > 0.90)]

    base_score = 100
    deductions = (
        (df.isnull().mean().mean() * 50)
        + (df.duplicated().mean() * 20)
        + (len(high_corr_cols) * 5)
    )
    data_quality_score = max(0, min(100, base_score - deductions))

    # Regression analysis
    regression_analysis = {}
    if problem_type == "regression" and pd.api.types.is_numeric_dtype(target_data):
        skewness = float(target_data.skew())
        kurtosis = float(target_data.kurt())

        Q1 = target_data.quantile(0.25)
        Q3 = target_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = target_data[(target_data < lower_bound) | (target_data > upper_bound)]
        outlier_pct = (len(outliers) / len(target_data)) * 100

        regression_analysis = {
            "skewness": round(skewness, 3),
            "kurtosis": round(kurtosis, 3),
            "outlier_percentage": round(outlier_pct, 2),
        }

    # Suggestions + model + complexity
    preprocessing_suggestions = suggest_preprocessing(df, missing_pct, high_corr_cols)
    model_recommendation = recommend_model(problem_type, df, target_data)
    complexity = estimate_complexity(df, problem_type)

    stats = {
        "summary": {
            "rows": len(df),
            "columns": len(df.columns),
            "data_quality_score": round(data_quality_score, 2),
            "problem_type": problem_type,
            "target": target_column,
            "complexity": complexity,
            "model_recommendation": model_recommendation,
        },
        "target_analysis": regression_analysis,
        "preprocessing": preprocessing_suggestions,
        "issues": {
            "missing_values_total": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "highly_correlated_columns": high_corr_cols,
            "columns_with_high_missing": missing_pct[missing_pct > 30].index.tolist(),
        },
        "schema": [],
    }

    for col in df.columns:
        col_info = {
            "name": col,
            "type": str(df[col].dtype),
            "missing_pct": round(missing_pct[col], 2),
            "unique_values": df[col].nunique(),
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["stats"] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
            }

        stats["schema"].append(col_info)

    return stats


def process_file(file_path):
    ext = file_path.split(".")[-1]

    if ext == "csv":
        df = pd.read_csv(file_path)
    elif ext in ["xlsx", "xls"]:
        df = pd.read_excel(file_path)
    else:
        return

    stats = generate_stats(df)

    out_path = file_path.replace(f".{ext}", ".stats.json")

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)


def run_all(data_dir="data/ml"):
    for file in os.listdir(data_dir):
        if file.endswith((".csv", ".xlsx", ".xls")):
            print(f"Processing {file}...")
            process_file(os.path.join(data_dir, file))
run_all()