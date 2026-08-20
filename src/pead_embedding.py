#!/usr/bin/env python3
"""PEAD regression using DatedGPT embeddings instead of bag-of-words.

Replaces the CountVectorizer features with pre-computed 2048-dim
embeddings from point-in-time DatedGPT models.

Usage:
    python3 -u src/pead_embedding.py
"""

import gc
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV


def load_embeddings(embedding_dir="data/embeddings/"):
    """Load all pre-computed embeddings and build event_id -> embedding mapping."""
    print("Loading embeddings...", flush=True)
    all_eids = []
    all_embs = []
    for f in sorted(os.listdir(embedding_dir)):
        if not f.endswith(".npz"):
            continue
        data = np.load(os.path.join(embedding_dir, f))
        eids = data["event_ids"]
        embs = data["embeddings"]
        all_eids.extend(eids)
        all_embs.append(embs)
        print(f"  {f}: {embs.shape}", flush=True)

    all_embs = np.vstack(all_embs)
    print(f"  Total: {all_embs.shape}")

    eid_to_idx = {eid: i for i, eid in enumerate(all_eids)}
    return all_embs, all_eids, eid_to_idx


def categorize_returns(ar_series):
    abs_ar = ar_series.abs()
    cutoff = abs_ar.quantile(1 / 3)
    cats = pd.Series("F", index=ar_series.index)
    cats[ar_series > cutoff] = "H"
    cats[ar_series < -cutoff] = "L"
    return cats


def compute_sue_txt(model, X):
    probs = np.clip(model.predict_proba(X), 1e-10, 1 - 1e-10)
    classes = list(model.classes_)
    iH, iL = classes.index("H"), classes.index("L")
    return np.log(probs[:, iH]) - np.log(probs[:, iL])


def main():
    print("=== PEAD with DatedGPT Embeddings ===\n", flush=True)

    meta = pd.read_parquet("data/merged_metadata_v3.parquet")
    meta["call_date"] = pd.to_datetime(meta["call_date"])
    meta["yq"] = meta["call_date"].dt.to_period("Q")
    print(f"Metadata: {len(meta):,} events")

    all_embs, all_eids, eid_to_idx = load_embeddings()

    embedded_eids = set(all_eids)
    meta = meta[meta["event_id"].isin(embedded_eids)].copy()
    print(f"Events with embeddings: {len(meta):,}")

    all_quarters = sorted(meta["yq"].unique())
    test_quarters = [q for q in all_quarters if q >= pd.Period("2014Q1")]
    print(f"Test quarters: {len(test_quarters)} (2014Q1-2019Q4)\n")

    results = []
    for i, test_q in enumerate(test_quarters):
        train_qs = [q for q in all_quarters if q < test_q][-8:]
        if len(train_qs) < 8:
            continue

        train_meta = meta[meta["yq"].isin(train_qs)]
        test_meta = meta[meta["yq"] == test_q]
        if len(test_meta) == 0:
            continue

        train_idx = [eid_to_idx[e] for e in train_meta["event_id"] if e in eid_to_idx]
        test_idx = [eid_to_idx[e] for e in test_meta["event_id"] if e in eid_to_idx]
        if len(train_idx) < 100 or len(test_idx) < 10:
            continue

        X_train = all_embs[train_idx]
        X_test = all_embs[test_idx]

        train_eids_matched = [all_eids[j] for j in train_idx]
        test_eids_matched = [all_eids[j] for j in test_idx]

        train_ar = train_meta.set_index("event_id").loc[train_eids_matched, "abnormal_return"]
        y_train = categorize_returns(train_ar)

        base = SGDClassifier(
            loss="log_loss", penalty="elasticnet", l1_ratio=0.5,
            alpha=0.001, max_iter=500, random_state=42, tol=1e-3,
        )
        model = CalibratedClassifierCV(base, cv=3, method="sigmoid", n_jobs=1)
        model.fit(X_train, y_train)

        classes = list(model.classes_)
        if "H" not in classes or "L" not in classes:
            continue

        sue_txt = compute_sue_txt(model, X_test)

        sue_txt_train = compute_sue_txt(model, X_train)
        quintile_cutoffs = np.percentile(sue_txt_train, [20, 40, 60, 80])

        test_meta_matched = test_meta.set_index("event_id").loc[test_eids_matched]
        for j, (eid, row) in enumerate(test_meta_matched.iterrows()):
            q_assign = np.searchsorted(quintile_cutoffs, sue_txt[j]) + 1
            results.append({
                "event_id": eid,
                "permno": row["permno"],
                "call_date": row["call_date"],
                "yq": str(test_q),
                "sue_txt": float(sue_txt[j]),
                "sue_txt_quintile": int(q_assign),
                "abnormal_return": row["abnormal_return"],
            })

        print(f"[{i+1}/{len(test_quarters)}] {test_q}: n={len(test_idx)}, "
              f"mean={np.mean(sue_txt):.3f} std={np.std(sue_txt):.3f}", flush=True)

    results_df = pd.DataFrame(results)
    results_df.to_parquet("data/pead_embedding_results.parquet", index=False)
    print(f"\nSaved {len(results_df):,} results")

    # Merge CARs
    results_df["call_date"] = pd.to_datetime(results_df["call_date"])
    results_df = results_df.merge(
        meta[["event_id", "car_32d", "car_63d", "car_126d", "car_189d", "car_252d"]],
        on="event_id", how="left",
    )

    print("\n=== PEAD Embedding Quintile Spread (Q5 - Q1) ===")
    for h in ["car_63d", "car_126d", "car_189d", "car_252d"]:
        q5 = results_df[results_df["sue_txt_quintile"] == 5][h].mean()
        q1 = results_df[results_df["sue_txt_quintile"] == 1][h].mean()
        d = h.replace("car_", "").replace("d", "")
        print(f"  {d:>3}d: {(q5 - q1) * 100:.2f}% (Q5={q5 * 100:.2f}%, Q1={q1 * 100:.2f}%)")

    print("\n=== Quintile Mean CARs (63d) ===")
    for q in range(1, 6):
        qdata = results_df[results_df["sue_txt_quintile"] == q]
        print(f"  Q{q}: {qdata['car_63d'].mean() * 100:.2f}% (n={len(qdata):,})")

    # Panel regression
    print("\n=== Panel Regression: CAR_63d ~ SUE.emb ===")
    df = results_df.dropna(subset=["car_63d", "sue_txt"]).copy()
    sue_z = (df["sue_txt"] - df["sue_txt"].mean()) / df["sue_txt"].std()
    X = np.column_stack([np.ones(len(sue_z)), sue_z.values])
    y = df["car_63d"].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    n = len(y)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (resid @ resid / (n - 2))))
    print(f"  Intercept: {beta[0] * 100:.3f}% (t={beta[0] / se[0]:.2f})")
    print(f"  SUE.emb:   {beta[1] * 100:.3f}% (t={beta[1] / se[1]:.2f})")
    print(f"  N={n:,}, R²={1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2):.4f}")

    results_df.to_parquet("data/pead_embedding_final.parquet", index=False)
    print(f"\nFinal: data/pead_embedding_final.parquet ({len(results_df):,} rows)")


if __name__ == "__main__":
    main()
