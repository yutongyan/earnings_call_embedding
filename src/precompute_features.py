#!/usr/bin/env python3
"""Pre-compute text features year by year to avoid loading all text at once."""

import gc
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    return re.sub(r"\d+\.?\d*", "#", text)


# Step 1: Build vocabulary from a sample
print("Building vocabulary from 2008-2009 sample...", flush=True)
sample_texts = []
for year in [2008, 2009]:
    df = pd.read_parquet(f"data/transcripts_{year}.parquet", columns=["presentation_text"])
    for t in df["presentation_text"]:
        sample_texts.append(preprocess_text(t))
    del df; gc.collect()

vec = CountVectorizer(max_features=1000, stop_words="english", min_df=5)
vec.fit(sample_texts)
vocab = vec.vocabulary_
del sample_texts; gc.collect()
print(f"  Vocabulary: {len(vocab)} terms")

# Step 2: Process each year and save sparse features
meta = pd.read_parquet("data/merged_metadata.parquet", columns=["event_id"])
meta_eids = set(meta["event_id"])

all_eids = []
all_features = []

for year in range(2008, 2020):
    print(f"Processing {year}...", flush=True)
    df = pd.read_parquet(f"data/transcripts_{year}.parquet", columns=["event_id", "presentation_text"])
    df = df[df["event_id"].isin(meta_eids)]

    texts = [preprocess_text(t) for t in df["presentation_text"]]
    eids = df["event_id"].tolist()

    vec_fixed = CountVectorizer(vocabulary=vocab, stop_words="english")
    X = vec_fixed.transform(texts)
    X.data = np.log1p(X.data).astype(np.float32)

    all_eids.extend(eids)
    all_features.append(X)
    print(f"  {year}: {len(eids):,} events, shape {X.shape}")
    del df, texts; gc.collect()

X_all = sparse.vstack(all_features, format="csr")
print(f"\nTotal: {X_all.shape}")

sparse.save_npz("data/text_features.npz", X_all)
pd.DataFrame({"event_id": all_eids}).to_parquet("data/feature_event_ids.parquet", index=False)
print("Saved data/text_features.npz and data/feature_event_ids.parquet")
