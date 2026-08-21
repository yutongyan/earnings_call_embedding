#!/usr/bin/env python3
"""Analyze PEAD results from glmnet SUE.txt output.

Usage:
    python3 -u src/analyze_pead.py --results data/sue_txt_glmnet.csv --metadata data/merged_metadata_v4.parquet
"""

import argparse
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/sue_txt_glmnet.csv")
    parser.add_argument("--metadata", default="data/merged_metadata_v4.parquet")
    args = parser.parse_args()

    results = pd.read_csv(args.results)
    results["call_date"] = pd.to_datetime(results["call_date"])
    print(f"Results: {len(results):,} events", flush=True)

    meta = pd.read_parquet(args.metadata)
    results = results.merge(
        meta[["event_id", "car_32d", "car_63d", "car_126d", "car_189d", "car_252d"]],
        on="event_id", how="left",
    )

    # Training-set quintile cutoffs per quarter
    results["yq_period"] = results["call_date"].dt.to_period("Q")

    # Use training-set cutoffs: for each test quarter, compute cutoffs from
    # the 8 training quarters' SUE.txt values
    all_quarters = sorted(results["yq_period"].unique())
    quintile_assignments = []

    for test_q in all_quarters:
        train_qs = [q for q in all_quarters if q < test_q][-8:]
        train_data = results[results["yq_period"].isin(train_qs)]
        test_data = results[results["yq_period"] == test_q]

        if len(train_data) < 100 or len(test_data) == 0:
            cutoffs = np.percentile(test_data["sue_txt"], [20, 40, 60, 80])
        else:
            cutoffs = np.percentile(train_data["sue_txt"], [20, 40, 60, 80])

        for _, row in test_data.iterrows():
            q_assign = np.searchsorted(cutoffs, row["sue_txt"]) + 1
            quintile_assignments.append({
                "event_id": row["event_id"],
                "sue_txt_quintile": int(q_assign),
            })

    qa_df = pd.DataFrame(quintile_assignments)
    results = results.merge(qa_df, on="event_id", how="left")

    # PEAD results
    print("\n=== PEAD.txt glmnet Quintile Spread (Q5-Q1) ===")
    for h in ["car_63d", "car_126d", "car_189d", "car_252d"]:
        q5 = results[results["sue_txt_quintile"] == 5][h].mean()
        q1 = results[results["sue_txt_quintile"] == 1][h].mean()
        d = h.replace("car_", "").replace("d", "")
        print(f"  {d:>3}d: {(q5 - q1) * 100:.2f}% (Q5={q5 * 100:.2f}%, Q1={q1 * 100:.2f}%)")

    print("\n=== Quintile Mean CARs (63d) ===")
    for q in range(1, 6):
        qd = results[results["sue_txt_quintile"] == q]
        print(f"  Q{q}: {qd['car_63d'].mean() * 100:.2f}% (n={len(qd):,})")

    print("\n=== SUE.txt Distribution ===")
    print(f"  mean: {results['sue_txt'].mean():.4f}")
    print(f"  std:  {results['sue_txt'].std():.4f}")
    print(f"  min:  {results['sue_txt'].min():.4f}")
    print(f"  max:  {results['sue_txt'].max():.4f}")

    print("\n=== Panel Regression: CAR_63d ~ SUE.txt ===")
    df = results.dropna(subset=["car_63d", "sue_txt"]).copy()
    sue_z = (df["sue_txt"] - df["sue_txt"].mean()) / df["sue_txt"].std()
    X = np.column_stack([np.ones(len(sue_z)), sue_z.values])
    y = df["car_63d"].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    n = len(y)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (resid @ resid / (n - 2))))
    print(f"  Intercept: {beta[0] * 100:.3f}% (t={beta[0] / se[0]:.2f})")
    print(f"  SUE.txt:   {beta[1] * 100:.3f}% (t={beta[1] / se[1]:.2f})")
    print(f"  N={n:,}, R2={1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2):.4f}")

    results.to_parquet("data/pead_glmnet_final.parquet", index=False)
    print(f"\nSaved data/pead_glmnet_final.parquet ({len(results):,} rows)")


if __name__ == "__main__":
    main()
