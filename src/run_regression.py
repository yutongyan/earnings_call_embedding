#!/usr/bin/env python3
"""Rolling-window elastic net regression for PEAD.txt replication.

Loads text on-the-fly from per-year transcript files to avoid memory issues.
"""

import glob
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
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"\d+\.?\d*", "#", text)
    return text


def load_text_for_events(event_ids, data_dir="data/"):
    """Load presentation_text and qa_text for given event_ids from per-year files."""
    event_set = set(event_ids)
    results = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "transcripts_20*.parquet"))):
        df = pd.read_parquet(f, columns=["event_id", "presentation_text", "qa_text"])
        matches = df[df["event_id"].isin(event_set)]
        for _, row in matches.iterrows():
            results[row["event_id"]] = {
                "presentation_text": row["presentation_text"] or "",
                "qa_text": row["qa_text"] or "",
            }
        if len(results) >= len(event_set):
            break
    return results


def build_features(train_texts, test_texts, max_features=500, ngram_range=(1, 1)):
    vec = CountVectorizer(max_features=max_features, ngram_range=ngram_range,
                          stop_words="english", min_df=5)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    return X_train, X_test


def compute_sue_txt(model, X):
    probs = np.clip(model.predict_proba(X), 1e-10, 1 - 1e-10)
    classes = list(model.classes_)
    idx_H, idx_L = classes.index("H"), classes.index("L")
    log_odds_H = np.log(probs[:, idx_H] / (1 - probs[:, idx_H]))
    log_odds_L = np.log(probs[:, idx_L] / (1 - probs[:, idx_L]))
    return log_odds_H - log_odds_L


def main():
    meta = pd.read_parquet("data/merged_metadata.parquet")
    meta["call_date"] = pd.to_datetime(meta["call_date"])
    meta["yq"] = meta["call_date"].dt.to_period("Q")
    print(f"Dataset: {len(meta):,} events, {meta['yq'].nunique()} quarters")

    all_quarters = sorted(meta["yq"].unique())
    test_quarters = [q for q in all_quarters if q >= pd.Period("2010Q1")]
    print(f"Test quarters: {len(test_quarters)} (2010Q1 to {test_quarters[-1]})")

    results = []
    for i, test_q in enumerate(test_quarters):
        train_qs = [q for q in all_quarters if q < test_q][-8:]
        if len(train_qs) < 8: continue

        train_mask = meta["yq"].isin(train_qs)
        test_mask = meta["yq"] == test_q
        train_meta = meta[train_mask]
        test_meta = meta[test_mask]
        if len(test_meta) == 0: continue

        print(f"[{i+1}/{len(test_quarters)}] {test_q}: train={len(train_meta)}, test={len(test_meta)}", flush=True)

        all_eids = list(train_meta["event_id"]) + list(test_meta["event_id"])
        texts = load_text_for_events(all_eids)

        train_pres = [preprocess_text(texts.get(eid, {}).get("presentation_text", "")) for eid in train_meta["event_id"]]
        train_qa = [preprocess_text(texts.get(eid, {}).get("qa_text", "")) for eid in train_meta["event_id"]]
        test_pres = [preprocess_text(texts.get(eid, {}).get("presentation_text", "")) for eid in test_meta["event_id"]]
        test_qa = [preprocess_text(texts.get(eid, {}).get("qa_text", "")) for eid in test_meta["event_id"]]

        # Categorize training returns
        abs_ar = train_meta["abnormal_return"].abs()
        cutoff = abs_ar.quantile(1 / 3)
        y_train = pd.Series("F", index=train_meta.index)
        y_train[train_meta["abnormal_return"] > cutoff] = "H"
        y_train[train_meta["abnormal_return"] < -cutoff] = "L"

        # Build 4 feature sets (sparse, 500 each = 2000 total)
        from scipy.sparse import hstack as sp_hstack
        X_train_parts, X_test_parts = [], []
        for tr, te, nf, ng in [
            (train_pres, test_pres, 500, (1,1)),
            (train_pres, test_pres, 500, (2,2)),
            (train_qa, test_qa, 500, (1,1)),
            (train_qa, test_qa, 500, (2,2)),
        ]:
            try:
                Xtr, Xte = build_features(tr, te, nf, ng)
            except ValueError:
                from scipy.sparse import csr_matrix
                Xtr = csr_matrix((len(tr), 1))
                Xte = csr_matrix((len(te), 1))
            X_train_parts.append(Xtr)
            X_test_parts.append(Xte)

        X_train = sp_hstack(X_train_parts, format="csr")
        X_test = sp_hstack(X_test_parts, format="csr")
        # log1p on sparse
        X_train.data = np.log1p(X_train.data)
        X_test.data = np.log1p(X_test.data)

        del texts, train_pres, train_qa, test_pres, test_qa
        del X_train_parts, X_test_parts

        model = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5,
            max_iter=2000, random_state=42, tol=1e-3,
        )
        cv = GridSearchCV(model, {"C": [0.01, 0.1, 1.0, 10.0]},
                          cv=3, scoring="neg_log_loss", n_jobs=1, refit=True)

        try:
            cv.fit(X_train, y_train)
            sue_txt = compute_sue_txt(cv.best_estimator_, X_test)

            for j, (idx, row) in enumerate(test_meta.iterrows()):
                results.append({
                    "event_id": row["event_id"], "permno": row["permno"],
                    "call_date": row["call_date"], "yq": str(test_q),
                    "sue_txt": sue_txt[j], "abnormal_return": row["abnormal_return"],
                    "best_C": cv.best_params_["C"],
                })

            print(f"  C={cv.best_params_['C']:.4f}, SUE.txt mean={np.mean(sue_txt):.3f} std={np.std(sue_txt):.3f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        del X_train, X_test

        # Incremental save every 5 quarters
        if (i + 1) % 5 == 0 or i == len(test_quarters) - 1:
            pd.DataFrame(results).to_parquet("data/sue_txt_results.parquet", index=False)
            print(f"  [checkpoint] {len(results):,} results saved", flush=True)

    results_df = pd.DataFrame(results)
    results_df.to_parquet("data/sue_txt_results.parquet", index=False)
    print(f"\nSaved {len(results_df):,} results to data/sue_txt_results.parquet")

    # Merge CARs and compute PEAD.txt
    results_df = results_df.merge(
        meta[["event_id", "car_32d", "car_63d", "car_126d", "car_189d", "car_252d"]],
        on="event_id", how="left",
    )
    results_df["yq_period"] = results_df["call_date"].dt.to_period("Q")
    results_df["sue_txt_quintile"] = results_df.groupby("yq_period")["sue_txt"].transform(
        lambda x: pd.qcut(x, 5, labels=[1,2,3,4,5], duplicates="drop")
    )

    print("\n=== PEAD.txt Quintile Spread CARs (Q5 - Q1) ===")
    for h in ["car_63d", "car_126d", "car_189d", "car_252d"]:
        q5 = results_df[results_df["sue_txt_quintile"]==5][h].mean()
        q1 = results_df[results_df["sue_txt_quintile"]==1][h].mean()
        days = h.replace("car_","").replace("d","")
        print(f"  {days:>3} days: {(q5-q1)*100:.2f}% (Q5={q5*100:.2f}%, Q1={q1*100:.2f}%)")

    print("\n=== Panel Regression: CAR_63d ~ SUE.txt ===")
    df = results_df.dropna(subset=["car_63d","sue_txt"])
    sue_z = (df["sue_txt"] - df["sue_txt"].mean()) / df["sue_txt"].std()
    X = np.column_stack([np.ones(len(sue_z)), sue_z.values])
    y = df["car_63d"].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta; resid = y - yhat; n = len(y)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (resid @ resid / (n-2))))
    print(f"  Intercept: {beta[0]*100:.3f}% (t={beta[0]/se[0]:.2f})")
    print(f"  SUE.txt:   {beta[1]*100:.3f}% (t={beta[1]/se[1]:.2f})")
    print(f"  N={n:,}, R²={1-np.sum(resid**2)/np.sum((y-y.mean())**2):.4f}")


if __name__ == "__main__":
    main()
