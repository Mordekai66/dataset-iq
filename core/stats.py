import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_CLASSIFICATION_UNIQUE_THRESHOLD = 20
_CLASSIFICATION_RATIO_THRESHOLD = 0.05
_CORRELATION_LEAKAGE_THRESHOLD = 0.95
_CORRELATION_HIGH_THRESHOLD = 0.90
_OUTLIER_IQR_FACTOR = 1.5
_MISSING_HIGH_THRESHOLD = 0.30
_QUALITY_MISSING_WEIGHT = 50
_QUALITY_DUPLICATE_WEIGHT = 20
_QUALITY_LEAKAGE_WEIGHT = 10
_QUALITY_OUTLIER_WEIGHT = 20
_MI_SAMPLE_SIZE = 50_000  # cap rows fed into mutual_info to bound O(n*d) cost


def detect_problem_type(target_col: pd.Series) -> str:
    if target_col.dtype == "object" or pd.api.types.is_bool_dtype(target_col):
        return "classification"
    unique_ratio = target_col.nunique() / max(len(target_col), 1)
    if target_col.nunique() <= _CLASSIFICATION_UNIQUE_THRESHOLD or unique_ratio < _CLASSIFICATION_RATIO_THRESHOLD:
        return "classification"
    return "regression"

def analyze_target(target: pd.Series, problem_type: str) -> dict:
    if problem_type == "classification":
        counts = target.value_counts()
        return {
            "num_classes": int(len(counts)),
            "class_distribution": {str(k): round(float(v / len(target)), 4) for k, v in counts.items()},
            "imbalance_ratio": round(float(counts.max() / counts.min()), 4) if len(counts) > 1 else 1.0,
            "majority_class": str(counts.idxmax()),
            "minority_class": str(counts.idxmin()),
        }
    return {
        "mean": float(target.mean()),
        "std": float(target.std()),
        "min": float(target.min()),
        "max": float(target.max()),
        "median": float(target.median()),
        "skewness": float(target.skew()),
        "kurtosis": float(target.kurt()),
        "iqr": float(target.quantile(0.75) - target.quantile(0.25)),
    }

# Vectorized: compute all quantiles in one pass instead of per-column loop
def detect_outliers(df: pd.DataFrame) -> dict[str, int]:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return {}
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - _OUTLIER_IQR_FACTOR * iqr
    upper = q3 + _OUTLIER_IQR_FACTOR * iqr
    mask = (numeric < lower) | (numeric > upper)
    return {col: int(mask[col].sum()) for col in numeric.columns}


# Compute correlation once; reuse for both leakage and high-correlation report
def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame | None:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None
    return numeric.corr()

def detect_leakage(corr_matrix: pd.DataFrame | None, target_col: str) -> list[str]:
    if corr_matrix is None or target_col not in corr_matrix.columns:
        return []
    series = corr_matrix[target_col].abs()
    return [col for col in series.index if col != target_col and series[col] > _CORRELATION_LEAKAGE_THRESHOLD]

def detect_high_correlation_pairs(corr_matrix: pd.DataFrame | None) -> list[dict]:
    if corr_matrix is None:
        return []
    pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = abs(corr_matrix.iloc[i, j])
            if val > _CORRELATION_HIGH_THRESHOLD:
                pairs.append({"col_a": cols[i], "col_b": cols[j], "correlation": round(float(val), 4)})
    return pairs

# Sample rows to cap MI computation at O(sample * d * log d) instead of O(n * d * log d)
def compute_mutual_info(df: pd.DataFrame, target: pd.Series, problem_type: str) -> dict[str, float]:
    X = df.drop(columns=[target.name])
    X = pd.get_dummies(X, drop_first=True)
    if X.empty:
        return {}

    if len(X) > _MI_SAMPLE_SIZE:
        idx = np.random.default_rng(42).choice(len(X), _MI_SAMPLE_SIZE, replace=False)
        X = X.iloc[idx]
        target = target.iloc[idx]

    try:
        fn = mutual_info_classif if problem_type == "classification" else mutual_info_regression
        mi = fn(X.fillna(0), target, random_state=42)
        total = mi.sum() or 1.0
        return {col: round(float(score / total), 6) for col, score in zip(X.columns, mi)}
    except Exception as exc:
        logger.warning("mutual_info failed: %s", exc)
        return {}

# Compute missing_pct once as a Series; reuse across all columns — O(n*d) single pass
def analyze_features(df: pd.DataFrame) -> list[dict]:
    n = len(df)
    missing_pct = df.isnull().mean() * 100  # single vectorized pass
    unique_counts = df.nunique()            # single vectorized pass

    feature_info = []
    for col in df.columns:
        s = df[col]
        info = {
            "name": col,
            "dtype": str(s.dtype),
            "missing_pct": round(float(missing_pct[col]), 4),
            "unique_count": int(unique_counts[col]),
            "cardinality_ratio": round(float(unique_counts[col] / max(n, 1)), 4),
            "high_missing": bool(missing_pct[col] / 100 > _MISSING_HIGH_THRESHOLD),
        }
        if pd.api.types.is_numeric_dtype(s):
            info.update({
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": round(float(s.mean()), 6),
                "median": float(s.median()),
                "std": round(float(s.std()), 6),
                "skewness": round(float(s.skew()), 4),
                "kurtosis": round(float(s.kurt()), 4),
                "variance": round(float(s.var()), 6),
                "q25": float(s.quantile(0.25)),
                "q75": float(s.quantile(0.75)),
            })
        elif s.dtype == "object":
            top = s.value_counts().head(5)
            info["top_values"] = {str(k): int(v) for k, v in top.items()}

        feature_info.append(info)

    return feature_info

# Outlier penalty uses per-column outlier rate (mean) not raw count sum — scale-invariant
def compute_quality(df: pd.DataFrame,leakage_cols: list[str],outliers: dict[str, int],) -> float:
    n = max(len(df), 1)
    missing_penalty = df.isnull().mean().mean() * _QUALITY_MISSING_WEIGHT
    duplicate_penalty = df.duplicated().mean() * _QUALITY_DUPLICATE_WEIGHT
    leakage_penalty = len(leakage_cols) * _QUALITY_LEAKAGE_WEIGHT
    outlier_rates = [v / n for v in outliers.values()]
    outlier_penalty = (np.mean(outlier_rates) if outlier_rates else 0.0) * _QUALITY_OUTLIER_WEIGHT
    score = 100 - (missing_penalty + duplicate_penalty + leakage_penalty + outlier_penalty)
    return round(float(np.clip(score, 0, 100)), 2)


def generate_recommendations(df: pd.DataFrame, problem_type: str, leakage_cols: list[str]) -> dict:
    rec: dict = {
        "drop_columns": list(leakage_cols),
        "encoding": {},
        "imputation": {},
        "scaling": problem_type == "regression",
        "handle_outliers": False,
        "notes": [],
    }

    for col in df.columns:
        s = df[col]
        missing_ratio = s.isnull().mean()

        if s.nunique() <= 1:
            if col not in rec["drop_columns"]:
                rec["drop_columns"].append(col)
            continue

        if missing_ratio > 0:
            if missing_ratio > _MISSING_HIGH_THRESHOLD:
                rec["imputation"][col] = "consider_dropping" if missing_ratio > 0.6 else "median_or_mode"
            else:
                rec["imputation"][col] = "median" if pd.api.types.is_numeric_dtype(s) else "mode"

        if s.dtype == "object":
            rec["encoding"][col] = "one-hot" if s.nunique() < 15 else "target_encode"

        if pd.api.types.is_numeric_dtype(s) and abs(s.skew()) > 1:
            rec["handle_outliers"] = True

    if any(df[col].isnull().mean() > _MISSING_HIGH_THRESHOLD for col in df.columns):
        rec["notes"].append("Multiple columns exceed 30% missing — evaluate dropping before imputation.")
    if problem_type == "classification":
        target = df.iloc[:, -1]
        if len(target.value_counts()) > 1:
            imbalance = target.value_counts().max() / target.value_counts().min()
            if imbalance > 3:
                rec["notes"].append(f"Class imbalance ratio {imbalance:.1f}x — consider SMOTE or class_weight.")

    return rec

def estimate_suitability(df: pd.DataFrame) -> dict:
    rows, cols = df.shape
    suitable = []
    rationale = []

    if rows < 10_000:
        suitable.append("classical_ml")
        rationale.append("Small dataset — tree ensembles or SVMs preferred over deep learning.")
    if rows >= 10_000:
        suitable.append("gradient_boosting")
    if rows > 100_000:
        suitable.append("deep_learning")
        rationale.append("Large row count supports neural network training.")
    if cols < 50:
        suitable.append("linear_models")
    else:
        suitable.append("tree_models")
        rationale.append("High dimensionality — tree-based models handle feature interaction implicitly.")

    return {"models": suitable, "rationale": rationale}

def estimate_risk(df: pd.DataFrame) -> dict:
    rows, cols = df.shape
    risks = []

    overfitting = "low"
    if cols > rows:
        overfitting = "high"
        risks.append("More features than rows — regularization mandatory.")
    elif cols > rows * 0.5:
        overfitting = "medium"
        risks.append("Feature count is high relative to row count.")

    underfitting = "low"
    if rows < 500:
        underfitting = "high"
        risks.append("Very few rows — model may not generalize.")
    elif rows < 2_000:
        underfitting = "medium"

    return {
        "overfitting_risk": overfitting,
        "underfitting_risk": underfitting,
        "warnings": risks,
    }

def generate_stats(df: pd.DataFrame) -> dict:
    target_col_name = df.columns[-1]
    target_data = df[target_col_name]
    problem_type = detect_problem_type(target_data)

    # Compute correlation matrix once — shared by leakage and pair detection
    corr_matrix = compute_correlation_matrix(df)

    target_analysis = analyze_target(target_data, problem_type)
    outliers = detect_outliers(df)
    leakage_cols = detect_leakage(corr_matrix, target_col_name)
    high_corr_pairs = detect_high_correlation_pairs(corr_matrix)
    mi_scores = compute_mutual_info(df, target_data, problem_type)
    features = analyze_features(df)
    quality = compute_quality(df, leakage_cols, outliers)
    recommendations = generate_recommendations(df, problem_type, leakage_cols)
    suitability = estimate_suitability(df)
    risk = estimate_risk(df)

    # Top 10 MI features sorted descending
    top_features = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "summary": {
            "rows": len(df),
            "columns": len(df.columns),
            "problem_type": problem_type,
            "target": target_col_name,
            "quality_score": quality,
            "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1e6), 4),
        },
        "target_analysis": target_analysis,
        "issues": {
            "missing_total": int(df.isnull().sum().sum()),
            "missing_pct_overall": round(float(df.isnull().mean().mean() * 100), 4),
            "duplicates": int(df.duplicated().sum()),
            "leakage_columns": leakage_cols,
            "high_correlation_pairs": high_corr_pairs,
            "outliers_per_column": outliers,
            "columns_high_missing": [
                col for col in df.columns if df[col].isnull().mean() > _MISSING_HIGH_THRESHOLD
            ],
        },
        "feature_importance": {
            "top_features": [{"name": k, "normalized_mi": v} for k, v in top_features],
            "all_scores": mi_scores,
        },
        "features": features,
        "recommendations": recommendations,
        "suitability": suitability,
        "risk": risk,
    }

def process_file(file_path: str) -> None:
    ext = os.path.splitext(file_path)[1].lstrip(".")
    try:
        if ext == "csv":
            df = pd.read_csv(file_path)
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(file_path)
        else:
            logger.warning("Unsupported file type: %s", file_path)
            return

        if df.empty or df.shape[1] < 2:
            logger.warning("Skipping %s — insufficient columns.", file_path)
            return

        stats = generate_stats(df)
        out_path = os.path.splitext(file_path)[0] + ".stats.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info("Stats written: %s", out_path)

    except Exception as exc:
        logger.error("Failed to process %s: %s", file_path, exc)


def run_all(data_dir: str = r"data/ml") -> None:
    if not os.path.isdir(data_dir):
        logger.error("Directory not found: %s", data_dir)
        return
    for file in os.listdir(data_dir):
        if file.endswith((".csv", ".xlsx", ".xls")):
            print(f"Processing {file}...")
            process_file(os.path.join(data_dir, file))
