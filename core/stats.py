import os
import pandas as pd
import json

def detect_problem_type(target_col):
    if target_col.dtype == "object":
        return "classification"
    
    unique_values = target_col.nunique()
    total_values = len(target_col)

    if unique_values <= 20 or unique_values / total_values < 0.05:
        return "classification"

    return "regression"


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
    deductions = (df.isnull().mean().mean() * 50) + (df.duplicated().mean() * 20) + (len(high_corr_cols) * 5)
    data_quality_score = max(0, min(100, base_score - deductions))

    stats = {
        "summary": {
            "rows": len(df),
            "columns": len(df.columns),
            "data_quality_score": round(data_quality_score, 2),
            "problem_type": problem_type,
            "target": target_column
        },
        "issues": {
            "missing_values_total": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "highly_correlated_columns": high_corr_cols,
            "columns_with_high_missing": missing_pct[missing_pct > 30].index.tolist()
        },
        "schema": []
    }

    for col in df.columns:
        col_info = {
            "name": col,
            "type": str(df[col].dtype),
            "missing_pct": round(missing_pct[col], 2),
            "unique_values": df[col].nunique()
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["stats"] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean())
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
            process_file(os.path.join(data_dir, file))
