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

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric_columns": len(df.select_dtypes(include=["number"]).columns),
        "categorical_columns": len(df.select_dtypes(exclude=["number"]).columns),
        "target_column": target_column,
        "problem_type": problem_type
    }


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
