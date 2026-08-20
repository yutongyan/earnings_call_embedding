#!/usr/bin/env python3
"""Run the PEAD.txt rolling-window elastic net regression.

Replicates the core methodology of Meursault et al. (2022):
- Rolling 8-quarter training windows
- Elastic net multinomial logistic regression (alpha=0.5)
- SUE.txt = log-odds(H) - log-odds(L)

Usage:
    python3 src/run_pead_regression.py --input data/merged_dataset.parquet --output data/sue_txt_results.parquet
"""

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")


def preprocess_text(text):
    """Lowercase text and replace numbers with #."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\d+\.?\d*", "#", text)
    return text


def build_features(train_texts, test_texts, max_features=1000, ngram_range=(1, 1)):
    """Build log-frequency features from text using training-set vocabulary."""
    vec = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        min_df=5,
    )
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    X_train = np.log1p(X_train.toarray())
    X_test = np.log1p(X_test.toarray())
    return X_train, X_test, vec


def categorize_returns_train(ar_series):
    """Categorize returns into H/F/L based on training set terciles."""
    abs_ar = ar_series.abs()
    cutoff = abs_ar.quantile(1 / 3)
    cats = pd.Series("F", index=ar_series.index)
    cats[ar_series > cutoff] = "H"
    cats[ar_series < -cutoff] = "L"
    return cats, cutoff


def compute_sue_txt(model, X, classes):
    """Compute SUE.txt = log-odds(H) - log-odds(L)."""
    probs = model.predict_proba(X)
    eps = 1e-10
    probs = np.clip(probs, eps, 1 - eps)

    idx_H = list(classes).index("H")
    idx_L = list(classes).index("L")

    log_odds_H = np.log(probs[:, idx_H] / (1 - probs[:, idx_H]))
    log_odds_L = np.log(probs[:, idx_L] / (1 - probs[:, idx_L]))

    return log_odds_H - log_odds_L


def run_rolling_window(df, n_train_quarters=8):
    """Run the full rolling-window estimation."""
    df = df.copy()
    df["call_date"] = pd.to_datetime(df["call_date"])
    df["yq"] = df["call_date"].dt.to_period("Q")

    all_quarters = sorted(df["yq"].unique())
    print(f"Total quarters: {len(all_quarters)}")
    print(f"Range: {all_quarters[0]} to {all_quarters[-1]}")

    test_quarters = [q for q in all_quarters if q >= pd.Period("2010Q1")]
    print(f"Test quarters: {len(test_quarters)}")

    results = []

    for i, test_q in enumerate(test_quarters):
        train_quarters = [q for q in all_quarters if q < test_q][-n_train_quarters:]

        if len(train_quarters) < n_train_quarters:
            print(f"  Skipping {test_q}: only {len(train_quarters)} training quarters")
            continue

        train_mask = df["yq"].isin(train_quarters)
        test_mask = df["yq"] == test_q

        train_data = df[train_mask].copy()
        test_data = df[test_mask].copy()

        if len(test_data) == 0:
            continue

        print(
            f"  [{i+1}/{len(test_quarters)}] {test_q}: "
            f"train={len(train_data)}, test={len(test_data)}",
            flush=True,
        )

        y_train, cutoff = categorize_returns_train(train_data["abnormal_return"])

        train_pres = train_data["presentation_text"].apply(preprocess_text).tolist()
        train_qa = train_data["qa_text"].apply(preprocess_text).tolist()
        test_pres = test_data["presentation_text"].apply(preprocess_text).tolist()
        test_qa = test_data["qa_text"].apply(preprocess_text).tolist()

        feature_configs = [
            (train_pres, test_pres, 1000, (1, 1)),
            (train_pres, test_pres, 1000, (2, 2)),
            (train_qa, test_qa, 1000, (1, 1)),
            (train_qa, test_qa, 1000, (2, 2)),
        ]

        X_train_parts = []
        X_test_parts = []

        for tr_texts, te_texts, max_feat, ngram in feature_configs:
            try:
                Xtr, Xte, _ = build_features(tr_texts, te_texts, max_feat, ngram)
                X_train_parts.append(Xtr)
                X_test_parts.append(Xte)
            except ValueError:
                n_tr = len(tr_texts)
                n_te = len(te_texts)
                X_train_parts.append(np.zeros((n_tr, 1)))
                X_test_parts.append(np.zeros((n_te, 1)))

        X_train = np.hstack(X_train_parts)
        X_test = np.hstack(X_test_parts)

        C_values = np.logspace(-3, 1, 10)

        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            multi_class="multinomial",
            max_iter=3000,
            random_state=42,
            tol=1e-3,
        )

        cv = GridSearchCV(
            model,
            {"C": C_values},
            cv=5,
            scoring="neg_log_loss",
            n_jobs=-1,
            refit=True,
        )

        try:
            cv.fit(X_train, y_train)
            best_model = cv.best_estimator_
            sue_txt = compute_sue_txt(best_model, X_test, best_model.classes_)

            for j, (idx, row) in enumerate(test_data.iterrows()):
                results.append({
                    "event_id": row["event_id"],
                    "permno": row["permno"],
                    "call_date": row["call_date"],
                    "yq": str(test_q),
                    "sue_txt": sue_txt[j],
                    "abnormal_return": row["abnormal_return"],
                    "best_C": cv.best_params_["C"],
                })

            print(
                f"    Best C={cv.best_params_['C']:.4f}, "
                f"SUE.txt mean={np.mean(sue_txt):.3f}, std={np.std(sue_txt):.3f}"
            )

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    results_df = pd.DataFrame(results)
    return results_df


def compute_pead(results_df, merged_df):
    """Compute PEAD.txt quintile spread returns."""
    print("\n=== PEAD.txt Results ===\n")

    results_df = results_df.merge(
        merged_df[["event_id", "car_63d", "car_126d", "car_189d", "car_252d"]],
        on="event_id",
        how="left",
    )

    results_df["yq_period"] = results_df["call_date"].dt.to_period("Q")
    results_df["sue_txt_quintile"] = results_df.groupby("yq_period")["sue_txt"].transform(
        lambda x: pd.qcut(x, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    )

    print("Quintile spread CARs (Top - Bottom):")
    for horizon in ["car_63d", "car_126d", "car_189d", "car_252d"]:
        q5 = results_df[results_df["sue_txt_quintile"] == 5][horizon].mean()
        q1 = results_df[results_df["sue_txt_quintile"] == 1][horizon].mean()
        spread = q5 - q1
        days = horizon.replace("car_", "").replace("d", "")
        print(f"  {days:>3} days: {spread*100:.2f}% (Q5={q5*100:.2f}%, Q1={q1*100:.2f}%)")

    print("\nQuintile mean CARs (63 days):")
    for q in range(1, 6):
        qdata = results_df[results_df["sue_txt_quintile"] == q]
        car = qdata["car_63d"].mean()
        print(f"  Q{q}: {car*100:.2f}% (n={len(qdata):,})")

    print(f"\nSample size: {len(results_df):,}")
    print(f"Quarters: {results_df['yq'].nunique()}")
    print(f"SUE.txt distribution: mean={results_df['sue_txt'].mean():.3f}, "
          f"std={results_df['sue_txt'].std():.3f}")

    return results_df


def run_panel_regression(results_df):
    """Run panel regression of CAR on SUE.txt."""
    print("\n=== Panel Regression: CAR_63d ~ SUE.txt ===\n")

    df = results_df.dropna(subset=["car_63d", "sue_txt"]).copy()

    sue_std = (df["sue_txt"] - df["sue_txt"].mean()) / df["sue_txt"].std()
    car = df["car_63d"]

    X = np.column_stack([np.ones(len(sue_std)), sue_std.values])
    y = car.values

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat
    n, k = X.shape
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (resid @ resid / (n - k))))
    t_stats = beta / se

    print(f"  Intercept: {beta[0]*100:.3f}% (t={t_stats[0]:.2f})")
    print(f"  SUE.txt:   {beta[1]*100:.3f}% (t={t_stats[1]:.2f})")
    print(f"  N = {n:,}")
    print(f"  R² = {1 - np.sum(resid**2) / np.sum((y - y.mean())**2):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/merged_dataset.parquet")
    parser.add_argument("--output", default="data/sue_txt_results.parquet")
    args = parser.parse_args()

    print("Loading merged dataset...", flush=True)
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} observations")

    required = ["event_id", "permno", "call_date", "abnormal_return",
                 "presentation_text", "qa_text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["abnormal_return", "presentation_text"])
    df["qa_text"] = df["qa_text"].fillna("")
    print(f"  After dropna: {len(df):,}")

    results_df = run_rolling_window(df)
    results_df.to_parquet(args.output, index=False)
    print(f"\nSaved SUE.txt results to {args.output}")

    results_df = compute_pead(results_df, df)

    run_panel_regression(results_df)

    summary_path = args.output.replace(".parquet", "_summary.csv")
    summary = results_df.groupby("yq").agg(
        n_obs=("sue_txt", "count"),
        sue_txt_mean=("sue_txt", "mean"),
        sue_txt_std=("sue_txt", "std"),
        car_63d_spread=("car_63d", lambda x: x[results_df.loc[x.index, "sue_txt_quintile"] == 5].mean()
                        - x[results_df.loc[x.index, "sue_txt_quintile"] == 1].mean()
                        if len(x) > 10 else np.nan),
    ).reset_index()
    summary.to_csv(summary_path, index=False)
    print(f"Summary by quarter saved to {summary_path}")


if __name__ == "__main__":
    main()
