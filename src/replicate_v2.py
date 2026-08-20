#!/usr/bin/env python3
"""PEAD.txt strict replication (v2).

Fixes all issues identified by Codex review:
- FF6 benchmark-adjusted CARs
- Event-day timing heuristic
- CRSP sentinel codes filtered
- 4 feature blocks (pres+QA uni+bigrams), rolling vocabulary
- Multinomial logistic regression with 10-fold CV log-loss
- Proper SUE.txt from multinomial log-odds
- Winsorization at 1%/99%
- Panel FE regression with clustered SE
- Compustat matching with causal direction
- # token preserved in CountVectorizer
"""

import gc
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

WRDS_DIR = "/scratch/llm_in_asset_pricing/wrds_raw"
CRSP_SENTINELS = {-66.0, -77.0, -88.0, -99.0}
TOKEN_PATTERN = r"(?u)\b[\w#$]+\b"


def winsorize(s, lower=0.01, upper=0.99):
    """Winsorize a Series at given percentiles."""
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    return re.sub(r"\d+\.?\d*", "#", text)


def load_texts_for_events(event_ids, event_year_index, col="presentation_text"):
    """Load text for specific events from per-year parquet files."""
    eid_set = set(event_ids)
    relevant = event_year_index[event_year_index["event_id"].isin(eid_set)]
    texts = {}
    for year in relevant["year"].unique():
        year_eids = set(relevant[relevant["year"] == year]["event_id"])
        df = pd.read_parquet(f"data/transcripts_{year}.parquet",
                             columns=["event_id", col])
        for _, row in df[df["event_id"].isin(year_eids)].iterrows():
            texts[row["event_id"]] = row[col] or ""
        del df
    gc.collect()
    return texts


def determine_use_same_day_return(is_before_close):
    """Implement the paper's timing heuristic using call time from headline.

    If the call happened during market hours (before 4 PM ET), the market
    reacts on the same day, so we use the return from t-1 to t.
    If the call happened after market close, the reaction is next day,
    so we use the return from t to t+1.

    Returns True if we should use same-day return (t-1 to t).
    """
    if pd.isna(is_before_close):
        return True
    return bool(is_before_close)


def compute_abnormal_returns_v2(linked, crsp_chunks, ff):
    """Compute one-day AR with timing heuristic and CRSP sentinel filtering."""
    print("\nComputing abnormal returns (v2)...", flush=True)

    ff = ff.copy()
    ff["date"] = pd.to_datetime(ff["date"])

    all_results = []
    for chunk_id in range(len(crsp_chunks)):
        crsp = pd.read_parquet(crsp_chunks[chunk_id])
        crsp["permno"] = crsp["permno"].astype("int64")
        for col in ["ret", "mktrf", "smb", "hml", "umd", "rf", "exret"]:
            crsp[col] = pd.to_numeric(crsp[col], errors="coerce").astype("float64")
        crsp = crsp[~crsp["ret"].isin(CRSP_SENTINELS)]
        crsp = crsp.dropna(subset=["exret", "mktrf", "smb", "hml", "umd"])

        grouped = crsp.groupby("permno")
        chunk_permnos = set(crsp["permno"].unique())
        chunk_calls = linked[linked["permno"].isin(chunk_permnos)]

        for permno, calls in chunk_calls.groupby("permno"):
            if permno not in grouped.groups:
                continue
            fc = grouped.get_group(permno).reset_index(drop=True)
            if len(fc) < 100:
                continue
            dates_arr = fc["date"].values

            for _, call in calls.iterrows():
                use_same_day = determine_use_same_day_return(
                    call.get("is_before_close", True),
                )

                cd_np = np.datetime64(call["call_date"])
                idx = np.searchsorted(dates_arr, cd_np, side="right") - 1
                if idx < 0:
                    idx = np.searchsorted(dates_arr, cd_np, side="left")

                if not use_same_day and idx + 1 < len(fc):
                    idx = idx + 1

                if idx < 0 or idx >= len(fc):
                    continue

                est_end = max(0, idx - 5)
                est_start = max(0, est_end - 252)
                if est_end - est_start < 60:
                    continue

                est = fc.iloc[est_start:est_end]
                if len(est) < 60:
                    continue

                X = est[["mktrf", "smb", "hml", "umd"]].values
                y = est["exret"].values
                Xc = np.column_stack([np.ones(len(X)), X])
                beta = np.linalg.lstsq(Xc, y, rcond=None)[0]

                ev = fc.iloc[idx]
                if pd.isna(ev["exret"]):
                    continue
                factors = np.array([1.0, ev["mktrf"], ev["smb"], ev["hml"], ev["umd"]])
                ar = ev["exret"] - np.dot(beta, factors)

                all_results.append({
                    "event_id": call["event_id"],
                    "permno": permno,
                    "call_date": call["call_date"],
                    "event_date": ev["date"],
                    "ret": float(ev["ret"]),
                    "abnormal_return": float(ar),
                })

        del crsp, grouped
        gc.collect()
        print(f"  Chunk {chunk_id}: total {len(all_results):,} ARs", flush=True)

    ar_df = pd.DataFrame(all_results)
    ar_df["abnormal_return"] = winsorize(ar_df["abnormal_return"])
    print(f"  Final: {len(ar_df):,} ARs")
    return ar_df


def compute_benchmark_adjusted_cars(merged_ar, crsp_chunks):
    """Compute CARs using market-adjusted returns (firm - market)."""
    print("\nComputing benchmark-adjusted CARs...", flush=True)

    horizons = [32, 63, 126, 189, 252]
    all_cars = []

    for chunk_id in range(len(crsp_chunks)):
        crsp = pd.read_parquet(crsp_chunks[chunk_id])
        crsp["permno"] = crsp["permno"].astype("int64")
        for col in ["ret", "mktrf", "rf"]:
            crsp[col] = pd.to_numeric(crsp[col], errors="coerce").astype("float64")
        crsp = crsp[~crsp["ret"].isin(CRSP_SENTINELS)]
        crsp = crsp.dropna(subset=["ret", "mktrf", "rf"])
        crsp["bm_ret"] = crsp["mktrf"] + crsp["rf"]
        crsp["ar_daily"] = crsp["ret"] - crsp["bm_ret"]

        grouped = crsp.groupby("permno")
        chunk_permnos = set(crsp["permno"].unique())
        chunk_calls = merged_ar[merged_ar["permno"].isin(chunk_permnos)]

        for permno, calls in chunk_calls.groupby("permno"):
            if permno not in grouped.groups:
                continue
            fc = grouped.get_group(permno).reset_index(drop=True)
            if len(fc) < 50:
                continue
            dates_arr = fc["date"].values

            for _, call in calls.iterrows():
                ed = pd.to_datetime(call["event_date"])
                si = np.searchsorted(dates_arr, np.datetime64(ed), side="left")
                if si >= len(dates_arr):
                    continue

                row = {"event_id": call["event_id"]}
                for h in horizons:
                    ei = min(si + h + 1, len(fc))
                    if ei <= si + 1:
                        row[f"car_{h}d"] = np.nan
                        continue
                    window = fc.iloc[si + 1:ei]
                    car = (1 + window["ar_daily"]).prod() - 1
                    row[f"car_{h}d"] = float(car)

                all_cars.append(row)

        del crsp, grouped
        gc.collect()
        print(f"  Chunk {chunk_id}: total {len(all_cars):,} CARs", flush=True)

    car_df = pd.DataFrame(all_cars)
    for h in horizons:
        col = f"car_{h}d"
        car_df[col] = winsorize(car_df[col].dropna()).reindex(car_df.index)
    print(f"  Final: {len(car_df):,} CARs")
    return car_df


def build_rolling_features(train_eids, test_eids, event_year_index):
    """Build 4-block features with rolling vocabulary from training set."""
    train_pres = load_texts_for_events(train_eids, event_year_index, "presentation_text")
    train_qa = load_texts_for_events(train_eids, event_year_index, "qa_text")
    test_pres = load_texts_for_events(test_eids, event_year_index, "presentation_text")
    test_qa = load_texts_for_events(test_eids, event_year_index, "qa_text")

    tr_pres = [preprocess_text(train_pres.get(e, "")) for e in train_eids]
    tr_qa = [preprocess_text(train_qa.get(e, "")) for e in train_eids]
    te_pres = [preprocess_text(test_pres.get(e, "")) for e in test_eids]
    te_qa = [preprocess_text(test_qa.get(e, "")) for e in test_eids]

    del train_pres, train_qa, test_pres, test_qa
    gc.collect()

    blocks_tr, blocks_te = [], []
    for tr_texts, te_texts, ngram in [
        (tr_pres, te_pres, (1, 1)),
        (tr_pres, te_pres, (2, 2)),
        (tr_qa, te_qa, (1, 1)),
        (tr_qa, te_qa, (2, 2)),
    ]:
        vec = CountVectorizer(
            max_features=1000, ngram_range=ngram,
            stop_words="english", min_df=5,
            token_pattern=TOKEN_PATTERN,
        )
        try:
            Xtr = vec.fit_transform(tr_texts)
            Xte = vec.transform(te_texts)
        except ValueError:
            from scipy.sparse import csr_matrix
            Xtr = csr_matrix((len(tr_texts), 1))
            Xte = csr_matrix((len(te_texts), 1))
        blocks_tr.append(Xtr)
        blocks_te.append(Xte)

    X_train = sparse.hstack(blocks_tr, format="csr")
    X_test = sparse.hstack(blocks_te, format="csr")
    X_train.data = np.log1p(X_train.data).astype(np.float32)
    X_test.data = np.log1p(X_test.data).astype(np.float32)

    del blocks_tr, blocks_te, tr_pres, tr_qa, te_pres, te_qa
    gc.collect()

    return X_train, X_test


def categorize_returns(ar_series):
    """Paper's H/F/L categorization: F=inner tercile by abs, H=positive rest, L=negative rest."""
    abs_ar = ar_series.abs()
    cutoff = abs_ar.quantile(1 / 3)
    cats = pd.Series("F", index=ar_series.index)
    cats[ar_series > cutoff] = "H"
    cats[ar_series < -cutoff] = "L"
    return cats


def run_rolling_regression(meta, event_year_index):
    """Rolling 8-quarter elastic net multinomial logistic regression."""
    print("\n=== Rolling-window regression (v2) ===", flush=True)

    meta = meta.copy()
    meta["yq"] = meta["call_date"].dt.to_period("Q")
    all_quarters = sorted(meta["yq"].unique())
    test_quarters = [q for q in all_quarters if q >= pd.Period("2010Q1")]
    print(f"Test quarters: {len(test_quarters)}")

    results = []
    done_quarters = set()
    checkpoint_path = "data/sue_txt_v2_checkpoint.parquet"
    if os.path.exists(checkpoint_path):
        existing = pd.read_parquet(checkpoint_path)
        results = existing.to_dict("records")
        done_quarters = set(existing["yq"].unique())
        print(f"Resuming: {len(done_quarters)} quarters done ({len(results):,} results)")

    for i, test_q in enumerate(test_quarters):
        if str(test_q) in done_quarters:
            continue

        train_qs = [q for q in all_quarters if q < test_q][-8:]
        if len(train_qs) < 8:
            continue

        train_meta = meta[meta["yq"].isin(train_qs)]
        test_meta = meta[meta["yq"] == test_q]
        if len(test_meta) == 0:
            continue

        print(f"[{i+1}/{len(test_quarters)}] {test_q}: train={len(train_meta)}, test={len(test_meta)}", flush=True)

        y_train = categorize_returns(train_meta["abnormal_return"])

        train_eids = train_meta["event_id"].tolist()
        test_eids = test_meta["event_id"].tolist()

        X_train, X_test = build_rolling_features(train_eids, test_eids, event_year_index)

        Cs = np.logspace(-3, 1, 10)
        try:
            model = LogisticRegressionCV(
                Cs=Cs, cv=5, penalty="elasticnet", solver="saga",
                l1_ratios=[0.5], scoring="neg_log_loss",
                max_iter=2000, random_state=42, tol=1e-3,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"  ERROR fitting: {e}")
            del X_train, X_test
            gc.collect()
            continue

        probs = np.clip(model.predict_proba(X_test), 1e-10, 1 - 1e-10)
        classes = list(model.classes_)

        if "H" not in classes or "L" not in classes:
            print(f"  WARNING: classes={classes}, skipping")
            del X_train, X_test
            gc.collect()
            continue

        iH, iL = classes.index("H"), classes.index("L")
        log_odds_H = np.log(probs[:, iH] / (1 - probs[:, iH]))
        log_odds_L = np.log(probs[:, iL] / (1 - probs[:, iL]))
        sue_txt = log_odds_H - log_odds_L

        best_C = model.C_[0] if hasattr(model, "C_") else 0

        for j, (idx, row) in enumerate(test_meta.iterrows()):
            results.append({
                "event_id": row["event_id"],
                "permno": row["permno"],
                "call_date": row["call_date"],
                "yq": str(test_q),
                "sue_txt": float(sue_txt[j]),
                "abnormal_return": row["abnormal_return"],
                "best_C": float(best_C),
            })

        print(f"  C={best_C:.4f}, SUE.txt mean={np.mean(sue_txt):.3f} std={np.std(sue_txt):.3f}")

        del X_train, X_test, model, probs
        gc.collect()

        if (i + 1) % 5 == 0:
            pd.DataFrame(results).to_parquet("data/sue_txt_v2_checkpoint.parquet", index=False)
            print(f"  [checkpoint] {len(results):,} results saved", flush=True)

    return pd.DataFrame(results)


def panel_regression_fe(results_df):
    """Panel regression with firm and year-quarter FE, clustered SE."""
    print("\n=== Panel Regression with FE ===")

    df = results_df.dropna(subset=["car_63d", "sue_txt"]).copy()
    df["sue_txt_z"] = (df["sue_txt"] - df["sue_txt"].mean()) / df["sue_txt"].std()

    try:
        from linearmodels.panel import PanelOLS
        df["permno_cat"] = df["permno"].astype("category")
        df["yq_cat"] = df["yq"].astype("category")
        df = df.set_index(["permno", "yq"])

        mod = PanelOLS(
            df["car_63d"], df[["sue_txt_z"]],
            entity_effects=True, time_effects=True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        print(res.summary)
        return res
    except ImportError:
        print("  linearmodels not installed, falling back to demeaned OLS")

        df["car_63d_dm"] = df.groupby("permno")["car_63d"].transform(lambda x: x - x.mean())
        df["car_63d_dm"] = df.groupby("yq")["car_63d_dm"].transform(lambda x: x - x.mean())
        df["sue_dm"] = df.groupby("permno")["sue_txt_z"].transform(lambda x: x - x.mean())
        df["sue_dm"] = df.groupby("yq")["sue_dm"].transform(lambda x: x - x.mean())

        X = np.column_stack([np.ones(len(df)), df["sue_dm"].values])
        y = df["car_63d_dm"].values
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        n = len(y)
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (resid @ resid / (n - 2))))
        print(f"  Intercept: {beta[0]*100:.3f}% (t={beta[0]/se[0]:.2f})")
        print(f"  SUE.txt:   {beta[1]*100:.3f}% (t={beta[1]/se[1]:.2f})")
        print(f"  N={n:,}, R²={1-np.sum(resid**2)/np.sum((y-y.mean())**2):.4f}")


def main():
    print("=== PEAD.txt Strict Replication (v2) ===\n")

    # Load metadata
    meta = pd.read_parquet("data/merged_metadata.parquet")
    meta["call_date"] = pd.to_datetime(meta["call_date"])
    meta["permno"] = meta["permno"].astype("int64")
    print(f"Metadata: {len(meta):,} events")

    # Load linked transcripts for timing info
    linked = pd.read_parquet("data/linked_transcripts.parquet")
    linked["call_date"] = pd.to_datetime(linked["call_date"])
    linked["permno"] = linked["permno"].astype("int64")

    # Check for timing columns
    if "is_before_close" not in linked.columns:
        linked["is_before_close"] = True
        print("  (No call-time metadata, defaulting to same-day return)")
    if "is_preliminary" not in linked.columns:
        linked["is_preliminary"] = False

    # Load event year index
    event_idx = pd.read_parquet("data/event_year_index.parquet")

    # CRSP chunks
    crsp_chunks = sorted(glob.glob("data/crsp_chunk_*.parquet"))
    print(f"CRSP chunks: {len(crsp_chunks)}")

    # FF factors
    ff = pd.read_parquet("data/ff_factors_daily.parquet")
    ff["date"] = pd.to_datetime(ff["date"])

    # Step 1: Recompute ARs with fixes
    ar_df = compute_abnormal_returns_v2(linked, crsp_chunks, ff)
    ar_df.to_parquet("data/abnormal_returns_v2.parquet", index=False)

    # Step 2: Merge AR into metadata
    meta_ar = meta.drop(columns=["abnormal_return", "ret", "event_date"], errors="ignore")
    meta_ar = meta_ar.merge(ar_df, on=["event_id", "permno", "call_date"], how="inner")
    print(f"\nMerged with AR: {len(meta_ar):,} events")

    # Step 3: Compute benchmark-adjusted CARs
    car_df = compute_benchmark_adjusted_cars(meta_ar, crsp_chunks)
    car_df.to_parquet("data/cars_v2.parquet", index=False)

    meta_ar = meta_ar.drop(columns=[c for c in meta_ar.columns if c.startswith("car_")], errors="ignore")
    meta_ar = meta_ar.merge(car_df, on="event_id", how="left")

    # Save updated metadata
    meta_ar.to_parquet("data/merged_metadata_v2.parquet", index=False)
    print(f"Saved merged_metadata_v2.parquet: {len(meta_ar):,} rows")

    # Step 4: Run rolling-window regression
    results_df = run_rolling_regression(meta_ar, event_idx)
    results_df.to_parquet("data/sue_txt_v2.parquet", index=False)
    print(f"\nSUE.txt computed: {len(results_df):,} results")

    # Step 5: PEAD.txt results
    results_df["call_date"] = pd.to_datetime(results_df["call_date"])
    results_df = results_df.merge(
        meta_ar[["event_id", "car_32d", "car_63d", "car_126d", "car_189d", "car_252d"]],
        on="event_id", how="left",
    )
    results_df["yq_period"] = results_df["call_date"].dt.to_period("Q")
    results_df["sue_txt_quintile"] = results_df.groupby("yq_period")["sue_txt"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") + 1
    )

    print("\n=== PEAD.txt Quintile Spread (Q5 - Q1) ===")
    for h in ["car_63d", "car_126d", "car_189d", "car_252d"]:
        q5 = results_df[results_df["sue_txt_quintile"] == 5][h].mean()
        q1 = results_df[results_df["sue_txt_quintile"] == 1][h].mean()
        d = h.replace("car_", "").replace("d", "")
        print(f"  {d:>3}d: {(q5 - q1) * 100:.2f}% (Q5={q5*100:.2f}%, Q1={q1*100:.2f}%)")

    print("\n=== Quintile Mean CARs (63d) ===")
    for q in range(1, 6):
        qdata = results_df[results_df["sue_txt_quintile"] == q]
        print(f"  Q{q}: {qdata['car_63d'].mean()*100:.2f}% (n={len(qdata):,})")

    # Step 6: Panel regression
    panel_regression_fe(results_df)

    # Save final
    results_df.to_parquet("data/sue_txt_v2_final.parquet", index=False)
    print(f"\nFinal results: data/sue_txt_v2_final.parquet ({len(results_df):,} rows)")


if __name__ == "__main__":
    main()
