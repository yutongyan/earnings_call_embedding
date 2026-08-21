#!/usr/bin/env python3
"""Build features for R glmnet, one rolling window at a time.

For each test quarter, exports:
- data/glmnet_windows/{quarter}/X_train.mtx (Matrix Market sparse)
- data/glmnet_windows/{quarter}/X_test.mtx
- data/glmnet_windows/{quarter}/y_train.csv (H/F/L labels)
- data/glmnet_windows/{quarter}/meta_test.csv (event_id, permno, call_date, AR)
- data/glmnet_windows/{quarter}/feature_names.txt

Usage:
    python3 -u src/build_glmnet_features.py --data_dir data/ --start 2010Q1 --end 2019Q4
"""

import argparse
import gc
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import io as sio
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

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
    """Paper's H/F/L: flat = inner tercile by abs, H = positive rest, L = negative rest."""
    abs_ar = ar_series.abs()
    cutoff = abs_ar.quantile(1 / 3)
    cats = pd.Series("F", index=ar_series.index)
    cats[ar_series > cutoff] = "H"
    cats[ar_series < -cutoff] = "L"
    return cats


def build_one_window(test_q, train_qs, meta, event_year_index, output_dir):
    """Build features for one rolling window and save to disk."""
    out_path = os.path.join(output_dir, str(test_q))
    if os.path.exists(os.path.join(out_path, "X_train.mtx")):
        print(f"  {test_q}: already built, skipping", flush=True)
        return

    train_meta = meta[meta["yq"].isin(train_qs)]
    test_meta = meta[meta["yq"] == test_q]
    if len(test_meta) == 0:
        return

    # Load texts
    all_eids = list(set(train_meta["event_id"]) | set(test_meta["event_id"]))
    all_pres, all_qa = load_both_texts(all_eids, event_year_index)

    tr_pres = [preprocess_text(all_pres.get(e, "")) for e in train_meta["event_id"]]
    tr_qa = [preprocess_text(all_qa.get(e, "")) for e in train_meta["event_id"]]
    te_pres = [preprocess_text(all_pres.get(e, "")) for e in test_meta["event_id"]]
    te_qa = [preprocess_text(all_qa.get(e, "")) for e in test_meta["event_id"]]
    del all_pres, all_qa
    gc.collect()

    # Build 4 feature blocks (1000 each, no min_df)
    blocks_tr, blocks_te, all_names = [], [], []
    for tr, te, ng, label in [
        (tr_pres, te_pres, (1, 1), "pres_uni"),
        (tr_pres, te_pres, (2, 2), "pres_bi"),
        (tr_qa, te_qa, (1, 1), "qa_uni"),
        (tr_qa, te_qa, (2, 2), "qa_bi"),
    ]:
        vec = CountVectorizer(
            max_features=1000, ngram_range=ng,
            stop_words=SNOWBALL_STOPWORDS,
            token_pattern=TOKEN_PATTERN,
        )
        try:
            Xtr = vec.fit_transform(tr)
            Xte = vec.transform(te)
            names = [f"{label}__{n}" for n in vec.get_feature_names_out()]
        except ValueError:
            Xtr = sparse.csr_matrix((len(tr), 0))
            Xte = sparse.csr_matrix((len(te), 0))
            names = []
        blocks_tr.append(Xtr)
        blocks_te.append(Xte)
        all_names.extend(names)

    X_train = sparse.hstack(blocks_tr, format="coo")
    X_test = sparse.hstack(blocks_te, format="coo")

    # Apply log1p to counts
    X_train = X_train.tocsr()
    X_test = X_test.tocsr()
    X_train.data = np.log1p(X_train.data).astype(np.float64)
    X_test.data = np.log1p(X_test.data).astype(np.float64)

    # Categorize returns using training-set cutoffs
    y_train = categorize_returns(train_meta["abnormal_return"])

    # Save
    os.makedirs(out_path, exist_ok=True)
    sio.mmwrite(os.path.join(out_path, "X_train.mtx"), X_train.tocoo())
    sio.mmwrite(os.path.join(out_path, "X_test.mtx"), X_test.tocoo())

    y_train.to_csv(os.path.join(out_path, "y_train.csv"), index=False, header=False)

    test_meta[["event_id", "permno", "call_date", "abnormal_return"]].to_csv(
        os.path.join(out_path, "meta_test.csv"), index=False
    )
    train_meta[["event_id"]].to_csv(
        os.path.join(out_path, "meta_train.csv"), index=False
    )

    with open(os.path.join(out_path, "feature_names.txt"), "w") as f:
        for name in all_names:
            f.write(name + "\n")

    print(f"  {test_q}: train={X_train.shape}, test={X_test.shape}, "
          f"H={sum(y_train=='H')}/F={sum(y_train=='F')}/L={sum(y_train=='L')}", flush=True)

    del X_train, X_test, blocks_tr, blocks_te
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--start", default="2010Q1")
    parser.add_argument("--end", default="2019Q4")
    args = parser.parse_args()

    meta = pd.read_parquet(os.path.join(args.data_dir, "merged_metadata_v4.parquet"))
    meta["call_date"] = pd.to_datetime(meta["call_date"])
    meta["yq"] = meta["call_date"].dt.to_period("Q")
    print(f"Metadata: {len(meta):,} events", flush=True)

    event_idx = pd.read_parquet(os.path.join(args.data_dir, "event_year_index.parquet"))

    all_quarters = sorted(meta["yq"].unique())
    start_q = pd.Period(args.start)
    end_q = pd.Period(args.end)
    test_quarters = [q for q in all_quarters if start_q <= q <= end_q]
    print(f"Test quarters: {len(test_quarters)} ({args.start} to {args.end})")

    output_dir = os.path.join(args.data_dir, "glmnet_windows")
    os.makedirs(output_dir, exist_ok=True)

    for i, test_q in enumerate(test_quarters):
        train_qs = [q for q in all_quarters if q < test_q][-8:]
        if len(train_qs) < 8:
            print(f"  {test_q}: skipping (only {len(train_qs)} training quarters)", flush=True)
            continue
        print(f"[{i+1}/{len(test_quarters)}] Building {test_q}...", flush=True)
        build_one_window(test_q, train_qs, meta, event_idx, output_dir)

    print(f"\nDone. Features saved to {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
