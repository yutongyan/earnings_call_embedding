#!/usr/bin/env python3
"""PEAD.txt regression on full v4 dataset (2001-2024, 211K events).

4 feature blocks, rolling 8-quarter window, SGDClassifier,
training-set quintile cutoffs, FF6 benchmark CARs.
"""

import gc
import os
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from concurrent.futures import ProcessPoolExecutor, as_completed

SNOWBALL_STOPWORDS = [
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","through",
    "during","before","after","above","below","to","from","up","down","in",
    "out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now",
    "d","ll","m","o","re","ve","y","ain",
    "aren","couldn","didn","doesn","hadn","hasn","haven",
    "isn","ma","mightn","mustn","needn","shan","shouldn",
    "wasn","weren","won","wouldn",
]
TOKEN_PATTERN = r"(?u)\b[\w#$]+\b"


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    return re.sub(r"\d+\.?\d*", "#", text)


def load_both_texts(event_ids, event_year_index):
    eid_set = set(event_ids)
    relevant = event_year_index[event_year_index["event_id"].isin(eid_set)]
    pres, qa = {}, {}
    for year in relevant["year"].unique():
        year_eids = set(relevant[relevant["year"] == year]["event_id"])
        df = pd.read_parquet(f"data/transcripts_{year}.parquet",
                             columns=["event_id", "presentation_text", "qa_text"])
        matched = df[df["event_id"].isin(year_eids)].set_index("event_id")
        pres.update(matched["presentation_text"].fillna("").to_dict())
        qa.update(matched["qa_text"].fillna("").to_dict())
        del df, matched
    gc.collect()
    return pres, qa


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


def process_one_quarter(args):
    test_q, train_qs, meta, event_year_index = args

    train_meta = meta[meta["yq"].isin(train_qs)]
    test_meta = meta[meta["yq"] == test_q]
    if len(test_meta) == 0:
        return []

    y_train = categorize_returns(train_meta["abnormal_return"])

    all_eids = list(set(train_meta["event_id"]) | set(test_meta["event_id"]))
    all_pres, all_qa = load_both_texts(all_eids, event_year_index)

    tr_pres = [preprocess_text(all_pres.get(e, "")) for e in train_meta["event_id"]]
    tr_qa = [preprocess_text(all_qa.get(e, "")) for e in train_meta["event_id"]]
    te_pres = [preprocess_text(all_pres.get(e, "")) for e in test_meta["event_id"]]
    te_qa = [preprocess_text(all_qa.get(e, "")) for e in test_meta["event_id"]]
    del all_pres, all_qa
    gc.collect()

    blocks_tr, blocks_te = [], []
    for tr, te, ng in [
        (tr_pres, te_pres, (1, 1)), (tr_pres, te_pres, (2, 2)),
        (tr_qa, te_qa, (1, 1)), (tr_qa, te_qa, (2, 2)),
    ]:
        vec = CountVectorizer(max_features=1000, ngram_range=ng,
                              stop_words=SNOWBALL_STOPWORDS, min_df=5,
                              token_pattern=TOKEN_PATTERN)
        try:
            Xtr = vec.fit_transform(tr)
            Xte = vec.transform(te)
        except ValueError:
            from scipy.sparse import csr_matrix
            Xtr = csr_matrix((len(tr), 1))
            Xte = csr_matrix((len(te), 1))
        blocks_tr.append(Xtr)
        blocks_te.append(Xte)

    X_train = sparse.hstack(blocks_tr, format="csr")
    X_test = sparse.hstack(blocks_te, format="csr")
    X_train.data = np.log1p(X_train.data).astype(np.float32)
    X_test.data = np.log1p(X_test.data).astype(np.float32)
    del blocks_tr, blocks_te, tr_pres, tr_qa, te_pres, te_qa
    gc.collect()

    try:
        base = SGDClassifier(loss="log_loss", penalty="elasticnet", l1_ratio=0.5,
                             alpha=0.001, max_iter=500, random_state=42, tol=1e-3)
        model = CalibratedClassifierCV(base, cv=3, method="sigmoid", n_jobs=1)
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"  {test_q} ERROR: {e}", flush=True)
        return []

    classes = list(model.classes_)
    if "H" not in classes or "L" not in classes:
        return []

    sue_test = compute_sue_txt(model, X_test)
    sue_train = compute_sue_txt(model, X_train)
    quintile_cutoffs = np.percentile(sue_train, [20, 40, 60, 80])

    results = []
    for j, (idx, row) in enumerate(test_meta.iterrows()):
        q_assign = np.searchsorted(quintile_cutoffs, sue_test[j]) + 1
        results.append({
            "event_id": row["event_id"],
            "permno": row["permno"],
            "call_date": row["call_date"],
            "yq": str(test_q),
            "sue_txt": float(sue_test[j]),
            "sue_txt_quintile": int(q_assign),
            "abnormal_return": row["abnormal_return"],
        })

    print(f"  {test_q}: n={len(test_meta)}, mean={np.mean(sue_test):.3f} std={np.std(sue_test):.3f}", flush=True)
    del X_train, X_test, model
    gc.collect()
    return results


def main():
    print("=== PEAD.txt v4 (full dataset) ===", flush=True)

    meta = pd.read_parquet("data/merged_metadata_v4.parquet")
    meta["call_date"] = pd.to_datetime(meta["call_date"])
    meta["yq"] = meta["call_date"].dt.to_period("Q")
    print(f"Events: {len(meta):,}")

    event_idx = pd.read_parquet("data/event_year_index.parquet")

    all_quarters = sorted(meta["yq"].unique())
    test_quarters = [q for q in all_quarters if q >= pd.Period("2003Q1")]
    print(f"Test quarters: {len(test_quarters)} (2003Q1 onwards)")

    todo = []
    for test_q in test_quarters:
        train_qs = [q for q in all_quarters if q < test_q][-8:]
        if len(train_qs) < 8:
            continue
        todo.append((test_q, train_qs, meta, event_idx))

    print(f"Quarters to process: {len(todo)}")

    # Resume from checkpoint
    results = []
    done_quarters = set()
    if os.path.exists("data/pead_v4_checkpoint.parquet"):
        existing = pd.read_parquet("data/pead_v4_checkpoint.parquet")
        results = existing.to_dict("records")
        done_quarters = set(existing["yq"].unique())
        print(f"Resuming: {len(done_quarters)} quarters done ({len(results):,} results)")
        todo = [t for t in todo if str(t[0]) not in done_quarters]
        print(f"Remaining: {len(todo)}")

    batch_size = 8
    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start:batch_start + batch_size]
        print(f"\nBatch {batch_start // batch_size + 1}: {[str(t[0]) for t in batch]}", flush=True)

        with ProcessPoolExecutor(max_workers=len(batch)) as executor:
            futures = {executor.submit(process_one_quarter, args): args[0] for args in batch}
            for future in as_completed(futures):
                results.extend(future.result())

        pd.DataFrame(results).to_parquet("data/pead_v4_checkpoint.parquet", index=False)
        print(f"  [checkpoint] {len(results):,} results", flush=True)

    results_df = pd.DataFrame(results)
    results_df.to_parquet("data/pead_v4_results.parquet", index=False)
    print(f"\nSaved {len(results_df):,} results")

    # PEAD results
    results_df["call_date"] = pd.to_datetime(results_df["call_date"])
    results_df = results_df.merge(
        meta[["event_id", "car_32d", "car_63d", "car_126d", "car_189d", "car_252d"]],
        on="event_id", how="left",
    )

    print("\n=== PEAD.txt v4 Quintile Spread (Q5-Q1) ===")
    for h in ["car_63d", "car_126d", "car_189d", "car_252d"]:
        q5 = results_df[results_df["sue_txt_quintile"] == 5][h].mean()
        q1 = results_df[results_df["sue_txt_quintile"] == 1][h].mean()
        d = h.replace("car_", "").replace("d", "")
        print(f"  {d:>3}d: {(q5 - q1) * 100:.2f}% (Q5={q5 * 100:.2f}%, Q1={q1 * 100:.2f}%)")

    print("\n=== Quintile Mean CARs (63d) ===")
    for q in range(1, 6):
        qd = results_df[results_df["sue_txt_quintile"] == q]
        print(f"  Q{q}: {qd['car_63d'].mean() * 100:.2f}% (n={len(qd):,})")

    print("\n=== Panel Regression: CAR_63d ~ SUE.txt ===")
    df = results_df.dropna(subset=["car_63d", "sue_txt"]).copy()
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

    results_df.to_parquet("data/pead_v4_final.parquet", index=False)
    print(f"\nFinal: data/pead_v4_final.parquet ({len(results_df):,} rows)")


if __name__ == "__main__":
    main()
