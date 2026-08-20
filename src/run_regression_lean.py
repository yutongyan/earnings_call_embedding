#!/usr/bin/env python3
"""Memory-lean rolling-window elastic net for PEAD.txt.

Processes one quarter at a time, loading text only for needed events.
Uses sparse features and presentation-only unigrams to minimize memory.
"""

import gc
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import hstack as sp_hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    return re.sub(r"\d+\.?\d*", "#", text)


def load_texts_by_year(event_ids, event_year_index):
    """Load text for events, reading only the year files needed."""
    eid_set = set(event_ids)
    relevant = event_year_index[event_year_index["event_id"].isin(eid_set)]
    years_needed = relevant["year"].unique()

    texts = {}
    for year in years_needed:
        year_eids = set(relevant[relevant["year"] == year]["event_id"])
        df = pd.read_parquet(
            f"data/transcripts_{year}.parquet",
            columns=["event_id", "presentation_text", "qa_text"],
        )
        for _, row in df[df["event_id"].isin(year_eids)].iterrows():
            texts[row["event_id"]] = row["presentation_text"] or ""
        del df
    return texts


def main():
    meta = pd.read_parquet("data/merged_metadata.parquet")
    meta["call_date"] = pd.to_datetime(meta["call_date"])
    meta["yq"] = meta["call_date"].dt.to_period("Q")
    print(f"Dataset: {len(meta):,} events")

    event_idx = pd.read_parquet("data/event_year_index.parquet")

    all_quarters = sorted(meta["yq"].unique())
    test_quarters = [q for q in all_quarters if q >= pd.Period("2010Q1")]
    print(f"Test quarters: {len(test_quarters)}")

    # Check for existing results to resume
    results = []
    done_quarters = set()
    if os.path.exists("data/sue_txt_results.parquet"):
        existing = pd.read_parquet("data/sue_txt_results.parquet")
        results = existing.to_dict("records")
        done_quarters = set(existing["yq"].unique())
        print(f"Resuming: {len(done_quarters)} quarters already done ({len(results):,} results)")

    for i, test_q in enumerate(test_quarters):
        if str(test_q) in done_quarters:
            continue

        train_qs = [q for q in all_quarters if q < test_q][-8:]
        if len(train_qs) < 8: continue

        train_meta = meta[meta["yq"].isin(train_qs)]
        test_meta = meta[meta["yq"] == test_q]
        if len(test_meta) == 0: continue

        print(f"[{i+1}/{len(test_quarters)}] {test_q}: train={len(train_meta)}, test={len(test_meta)}", flush=True)

        # Load texts
        all_eids = list(train_meta["event_id"]) + list(test_meta["event_id"])
        texts = load_texts_by_year(all_eids, event_idx)

        train_texts = [preprocess_text(texts.get(eid, "")) for eid in train_meta["event_id"]]
        test_texts = [preprocess_text(texts.get(eid, "")) for eid in test_meta["event_id"]]
        del texts
        gc.collect()

        # Categorize training returns
        abs_ar = train_meta["abnormal_return"].abs()
        cutoff = abs_ar.quantile(1 / 3)
        y_train = pd.Series("F", index=train_meta.index)
        y_train[train_meta["abnormal_return"] > cutoff] = "H"
        y_train[train_meta["abnormal_return"] < -cutoff] = "L"

        # Build features: unigrams only (1000 features) to save memory
        vec = CountVectorizer(max_features=1000, stop_words="english", min_df=5)
        X_train = vec.fit_transform(train_texts)
        X_test = vec.transform(test_texts)
        # log1p on sparse data
        X_train.data = np.log1p(X_train.data).astype(np.float32)
        X_test.data = np.log1p(X_test.data).astype(np.float32)

        del train_texts, test_texts
        gc.collect()

        # Fit model with a few C values
        best_score = -np.inf
        best_model = None
        best_C = None

        for C in [0.01, 0.1, 1.0, 10.0]:
            model = LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5,
                C=C, max_iter=2000, random_state=42, tol=1e-3,
            )
            model.fit(X_train, y_train)
            score = model.score(X_train, y_train)
            if score > best_score:
                best_score = score
                best_model = model
                best_C = C

        # Compute SUE.txt
        probs = np.clip(best_model.predict_proba(X_test), 1e-10, 1 - 1e-10)
        classes = list(best_model.classes_)
        idx_H, idx_L = classes.index("H"), classes.index("L")
        log_odds_H = np.log(probs[:, idx_H] / (1 - probs[:, idx_H]))
        log_odds_L = np.log(probs[:, idx_L] / (1 - probs[:, idx_L]))
        sue_txt = log_odds_H - log_odds_L

        for j, (idx, row) in enumerate(test_meta.iterrows()):
            results.append({
                "event_id": row["event_id"], "permno": row["permno"],
                "call_date": row["call_date"], "yq": str(test_q),
                "sue_txt": float(sue_txt[j]),
                "abnormal_return": row["abnormal_return"],
                "best_C": best_C,
            })

        print(f"  C={best_C}, SUE.txt mean={np.mean(sue_txt):.3f} std={np.std(sue_txt):.3f}")

        del X_train, X_test, best_model, probs
        gc.collect()

        # Save checkpoint every 5 quarters
        if (i + 1) % 5 == 0 or test_q == test_quarters[-1]:
            pd.DataFrame(results).to_parquet("data/sue_txt_results.parquet", index=False)
            print(f"  [saved] {len(results):,} results", flush=True)

    results_df = pd.DataFrame(results)
    results_df.to_parquet("data/sue_txt_results.parquet", index=False)
    print(f"\n=== Done: {len(results_df):,} results ===")

    # Compute PEAD.txt
    results_df["call_date"] = pd.to_datetime(results_df["call_date"])
    results_df = results_df.merge(
        meta[["event_id", "car_32d", "car_63d", "car_126d", "car_189d", "car_252d"]],
        on="event_id", how="left",
    )
    results_df["yq_period"] = results_df["call_date"].dt.to_period("Q")
    results_df["sue_txt_quintile"] = results_df.groupby("yq_period")["sue_txt"].transform(
        lambda x: pd.qcut(x, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    )

    print("\n=== PEAD.txt Quintile Spread CARs (Q5 - Q1) ===")
    for h in ["car_63d", "car_126d", "car_189d", "car_252d"]:
        q5 = results_df[results_df["sue_txt_quintile"] == 5][h].mean()
        q1 = results_df[results_df["sue_txt_quintile"] == 1][h].mean()
        days = h.replace("car_", "").replace("d", "")
        print(f"  {days:>3}d: {(q5 - q1) * 100:.2f}% (Q5={q5 * 100:.2f}%, Q1={q1 * 100:.2f}%)")

    print("\n=== Panel Regression: CAR_63d ~ SUE.txt ===")
    df = results_df.dropna(subset=["car_63d", "sue_txt"])
    sue_z = (df["sue_txt"] - df["sue_txt"].mean()) / df["sue_txt"].std()
    X = np.column_stack([np.ones(len(sue_z)), sue_z.values])
    y = df["car_63d"].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    n = len(y)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (resid @ resid / (n - 2))))
    print(f"  Intercept: {beta[0] * 100:.3f}% (t={beta[0] / se[0]:.2f})")
    print(f"  SUE.txt:   {beta[1] * 100:.3f}% (t={beta[1] / se[1]:.2f})")
    print(f"  N={n:,}, R²={1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2):.4f}")

    results_df.to_parquet("data/sue_txt_final.parquet", index=False)
    print(f"\nFinal results saved to data/sue_txt_final.parquet")


if __name__ == "__main__":
    main()
