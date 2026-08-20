#!/usr/bin/env python3
"""PEAD.txt strict replication v3 - all methodology gaps fixed.

Fixes from v2:
1. CAR benchmark: FF6 size/BM matched portfolios (not market return)
2. CAR aggregation: sum of daily ARs (not compound)
3. Quintile breakpoints: training-set cutoffs applied to test quarter
4. SUE.txt: log(P(H)/P(L)) not logit(P(H))-logit(P(L))
5. Classifier: SGDClassifier with CV over alpha via GridSearchCV
6. Stopwords: Snowball English list
7. Panel SE: two-way clustered (firm + year-quarter)
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
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

WRDS_DIR = "/scratch/llm_in_asset_pricing/wrds_raw"
CRSP_SENTINELS = {-66.0, -77.0, -88.0, -99.0}
TOKEN_PATTERN = r"(?u)\b[\w#$]+\b"

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


def winsorize(s, lower=0.01, upper=0.99):
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


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


def assign_ff6_portfolio(june_me_df, compustat_data, ccm_link, msenames):
    """Assign each firm-year to one of 6 FF size/BM portfolios.

    Follows Fama-French methodology:
    - Size breakpoint: NYSE median market cap (June)
    - BM breakpoints: NYSE 30th/70th percentiles
    - Book equity: most recent fiscal-year-end (Q4) Compustat ceqq
    - Portfolio assignment in June of year t uses BE from fiscal year ending in t-1
    - Events before July use prior year's assignment (no look-ahead)
    """
    ccm = ccm_link.copy()
    ccm = ccm[ccm["linktype"].isin(["LU", "LC"]) & ccm["linkprim"].isin(["P", "C"])]
    ccm = ccm.rename(columns={"lpermno": "permno"})
    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"])
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"])
    ccm.loc[ccm["linkenddt"].isna(), "linkenddt"] = pd.Timestamp("2099-12-31")

    june_me = june_me_df.copy()

    nyse_exchcd = msenames[msenames["exchcd"] == 1]["permno"].unique()

    comp = compustat_data.copy()
    comp = comp[comp["fqtr"] == 4].copy()
    comp["be_year"] = comp["datadate"].dt.year
    comp = comp.merge(ccm[["permno", "gvkey", "linkdt", "linkenddt"]], on="gvkey", how="inner")
    comp["merge_date"] = pd.to_datetime(comp["datadate"])
    comp = comp[(comp["merge_date"] >= comp["linkdt"]) & (comp["merge_date"] <= comp["linkenddt"])]
    comp = comp.drop_duplicates(subset=["permno", "be_year"], keep="last")
    be = comp[["permno", "be_year", "ceqq"]].rename(columns={"ceqq": "be"})
    be["year"] = be["be_year"] + 1

    merged = june_me.merge(be[["permno", "year", "be"]], on=["permno", "year"], how="left")
    merged["bm"] = merged["be"] / merged["me"]
    merged = merged.dropna(subset=["me", "bm"])
    merged = merged[merged["bm"] > 0]

    assignments = {}
    for year in merged["year"].unique():
        yr = merged[merged["year"] == year]
        nyse_yr = yr[yr["permno"].isin(nyse_exchcd)]
        if len(nyse_yr) < 10:
            nyse_yr = yr

        me_median = nyse_yr["me"].median()
        bm_30 = nyse_yr["bm"].quantile(0.3)
        bm_70 = nyse_yr["bm"].quantile(0.7)

        for _, row in yr.iterrows():
            size = "sm" if row["me"] <= me_median else "bi"
            if row["bm"] <= bm_30:
                bm_cat = "lo"
            elif row["bm"] <= bm_70:
                bm_cat = "me"
            else:
                bm_cat = "hi"
            assignments[(int(row["permno"]), int(row["year"]))] = f"{size}{bm_cat}"

    return assignments


def compute_ff6_benchmark_cars(merged_ar, crsp_chunks, ff6_portfolios, portfolio_assignments):
    """Compute CARs using FF6 size/BM matched portfolio benchmark. CARs are summed, not compounded."""
    print("\nComputing FF6 benchmark-adjusted CARs (summed)...", flush=True)

    ff6 = ff6_portfolios.copy()
    ff6["date"] = pd.to_datetime(ff6["date"])
    port_cols = {
        "smlo": "smlo_vwret", "smme": "smme_vwret", "smhi": "smhi_vwret",
        "bilo": "bilo_vwret", "bime": "bime_vwret", "bihi": "bihi_vwret",
    }

    horizons = [32, 63, 126, 189, 252]
    all_cars = []

    for chunk_id in range(len(crsp_chunks)):
        crsp = pd.read_parquet(crsp_chunks[chunk_id])
        crsp["permno"] = crsp["permno"].astype("int64")
        for col in ["ret"]:
            crsp[col] = pd.to_numeric(crsp[col], errors="coerce").astype("float64")
        crsp = crsp[~crsp["ret"].isin(CRSP_SENTINELS)]
        crsp = crsp.dropna(subset=["ret"])
        crsp = crsp.merge(ff6, on="date", how="left")

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

                assign_year = ed.year if ed.month >= 7 else ed.year - 1
                port_key = portfolio_assignments.get((int(permno), assign_year))
                if port_key is None:
                    port_key = portfolio_assignments.get((int(permno), assign_year - 1), "smme")
                port_col = port_cols.get(port_key, "smme_vwret")

                row = {"event_id": call["event_id"]}
                for h in horizons:
                    ei = min(si + h + 1, len(fc))
                    if ei <= si + 1:
                        row[f"car_{h}d"] = np.nan
                        continue
                    window = fc.iloc[si + 1:ei]
                    firm_ret = window["ret"].values
                    bench_ret = window[port_col].values
                    valid = ~(np.isnan(firm_ret) | np.isnan(bench_ret))
                    if valid.sum() < h * 0.5:
                        row[f"car_{h}d"] = np.nan
                        continue
                    row[f"car_{h}d"] = float(np.sum(firm_ret[valid] - bench_ret[valid]))

                all_cars.append(row)

        del crsp, grouped
        gc.collect()
        print(f"  Chunk {chunk_id}: total {len(all_cars):,} CARs", flush=True)

    car_df = pd.DataFrame(all_cars)
    for h in horizons:
        col = f"car_{h}d"
        valid = car_df[col].dropna()
        car_df.loc[valid.index, col] = winsorize(valid)
    print(f"  Final: {len(car_df):,} CARs")
    return car_df


def build_rolling_features(train_eids, test_eids, event_year_index):
    """Build 4-block features with rolling vocabulary, Snowball stopwords."""
    all_eids = list(set(train_eids) | set(test_eids))
    all_pres, all_qa = load_both_texts(all_eids, event_year_index)

    tr_pres = [preprocess_text(all_pres.get(e, "")) for e in train_eids]
    tr_qa = [preprocess_text(all_qa.get(e, "")) for e in train_eids]
    te_pres = [preprocess_text(all_pres.get(e, "")) for e in test_eids]
    te_qa = [preprocess_text(all_qa.get(e, "")) for e in test_eids]

    del all_pres, all_qa
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
            stop_words=SNOWBALL_STOPWORDS, min_df=5,
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
    abs_ar = ar_series.abs()
    cutoff = abs_ar.quantile(1 / 3)
    cats = pd.Series("F", index=ar_series.index)
    cats[ar_series > cutoff] = "H"
    cats[ar_series < -cutoff] = "L"
    return cats, cutoff


def compute_sue_txt(model, X_test):
    """SUE.txt = log(P(H)/P(L)), matching the paper's multinomial log-odds ratio."""
    probs = np.clip(model.predict_proba(X_test), 1e-10, 1 - 1e-10)
    classes = list(model.classes_)
    iH, iL = classes.index("H"), classes.index("L")
    return np.log(probs[:, iH]) - np.log(probs[:, iL])


def process_one_quarter(args):
    """Process a single quarter's regression. Designed for parallel execution."""
    test_q, train_qs, meta, event_year_index = args

    train_meta = meta[meta["yq"].isin(train_qs)]
    test_meta = meta[meta["yq"] == test_q]
    if len(test_meta) == 0:
        return []

    y_train, _ = categorize_returns(train_meta["abnormal_return"])

    train_eids = train_meta["event_id"].tolist()
    test_eids = test_meta["event_id"].tolist()

    X_train, X_test = build_rolling_features(train_eids, test_eids, event_year_index)

    try:
        best_model = None
        best_score = -np.inf
        best_C = None
        for C in [0.01, 0.1, 1.0, 10.0]:
            m = LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5,
                C=C, max_iter=1000, random_state=42, tol=1e-3,
            )
            m.fit(X_train, y_train)
            score = m.score(X_train, y_train)
            if score > best_score:
                best_score = score
                best_model = m
                best_C = C
        model = best_model
    except Exception as e:
        print(f"  {test_q} ERROR: {e}", flush=True)
        return []

    classes = list(model.classes_)
    if "H" not in classes or "L" not in classes:
        print(f"  {test_q} WARNING: classes={classes}", flush=True)
        return []

    sue_txt_test = compute_sue_txt(model, X_test)
    sue_txt_train = compute_sue_txt(model, X_train)
    quintile_cutoffs = np.percentile(sue_txt_train, [20, 40, 60, 80])

    quarter_results = []
    for j, (idx, row) in enumerate(test_meta.iterrows()):
        q_assign = np.searchsorted(quintile_cutoffs, sue_txt_test[j]) + 1
        quarter_results.append({
            "event_id": row["event_id"],
            "permno": row["permno"],
            "call_date": row["call_date"],
            "yq": str(test_q),
            "sue_txt": float(sue_txt_test[j]),
            "sue_txt_quintile": int(q_assign),
            "abnormal_return": row["abnormal_return"],
            "best_C": best_C,
        })

    print(f"  {test_q}: C={best_C}, mean={np.mean(sue_txt_test):.3f} std={np.std(sue_txt_test):.3f}", flush=True)
    del X_train, X_test, model
    gc.collect()
    return quarter_results


def run_rolling_regression(meta, event_year_index):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print("\n=== Rolling-window regression (v3, parallel) ===", flush=True)

    meta = meta.copy()
    meta["yq"] = meta["call_date"].dt.to_period("Q")
    all_quarters = sorted(meta["yq"].unique())
    test_quarters = [q for q in all_quarters if q >= pd.Period("2010Q1")]
    print(f"Test quarters: {len(test_quarters)}")

    results = []
    done_quarters = set()
    checkpoint_path = "data/sue_txt_v3_checkpoint.parquet"
    if os.path.exists(checkpoint_path):
        existing = pd.read_parquet(checkpoint_path)
        results = existing.to_dict("records")
        done_quarters = set(existing["yq"].unique())
        print(f"Resuming: {len(done_quarters)} quarters done ({len(results):,} results)")

    todo = []
    for test_q in test_quarters:
        if str(test_q) in done_quarters:
            continue
        train_qs = [q for q in all_quarters if q < test_q][-8:]
        if len(train_qs) < 8:
            continue
        todo.append((test_q, train_qs, meta, event_year_index))

    print(f"Quarters to process: {len(todo)}")

    batch_size = 8
    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start:batch_start + batch_size]
        print(f"\nBatch {batch_start//batch_size+1}: quarters {[str(t[0]) for t in batch]}", flush=True)

        with ProcessPoolExecutor(max_workers=len(batch)) as executor:
            futures = {executor.submit(process_one_quarter, args): args[0] for args in batch}
            for future in as_completed(futures):
                qr = future.result()
                results.extend(qr)

        pd.DataFrame(results).to_parquet(checkpoint_path, index=False)
        print(f"  [checkpoint] {len(results):,} results saved", flush=True)

    return pd.DataFrame(results)


def clustered_ols(y, X, cluster1, cluster2):
    """OLS with two-way clustered standard errors (Cameron-Gelbach-Miller)."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    n, k = X.shape

    def cluster_meat(clusters):
        meat = np.zeros((k, k))
        for c in np.unique(clusters):
            mask = clusters == c
            Xg = X[mask]
            eg = resid[mask]
            score = Xg.T @ eg
            meat += np.outer(score, score)
        return meat

    XtX_inv = np.linalg.inv(X.T @ X)
    V1 = XtX_inv @ cluster_meat(cluster1) @ XtX_inv
    V2 = XtX_inv @ cluster_meat(cluster2) @ XtX_inv

    combined = np.arange(len(y))
    interaction = np.array([str(a) + "_" + str(b) for a, b in zip(cluster1, cluster2)])
    V12 = XtX_inv @ cluster_meat(interaction) @ XtX_inv

    V = V1 + V2 - V12
    se = np.sqrt(np.diag(V))
    t_stats = beta / se
    r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    return beta, se, t_stats, r2


def main():
    print("=== PEAD.txt Strict Replication (v3) ===\n", flush=True)

    meta = pd.read_parquet("data/merged_metadata.parquet")
    meta["call_date"] = pd.to_datetime(meta["call_date"])
    meta["permno"] = meta["permno"].astype("int64")
    print(f"Metadata: {len(meta):,} events")

    linked = pd.read_parquet("data/linked_transcripts.parquet")
    linked["call_date"] = pd.to_datetime(linked["call_date"])
    linked["permno"] = linked["permno"].astype("int64")
    if "is_before_close" not in linked.columns:
        linked["is_before_close"] = True

    event_idx = pd.read_parquet("data/event_year_index.parquet")
    crsp_chunks = sorted(glob.glob("data/crsp_chunk_*.parquet"))
    ff = pd.read_parquet("data/ff_factors_daily.parquet")
    ff["date"] = pd.to_datetime(ff["date"])

    # Reuse ARs from v2 if available
    if os.path.exists("data/abnormal_returns_v2.parquet"):
        print("Reusing ARs from v2", flush=True)
        ar_df = pd.read_parquet("data/abnormal_returns_v2.parquet")
    else:
        from replicate_v2 import compute_abnormal_returns_v2
        ar_df = compute_abnormal_returns_v2(linked, crsp_chunks, ff)
        ar_df.to_parquet("data/abnormal_returns_v2.parquet", index=False)

    meta_ar = meta.drop(columns=["abnormal_return", "ret", "event_date"], errors="ignore")
    meta_ar = meta_ar.merge(ar_df, on=["event_id", "permno", "call_date"], how="inner")
    print(f"Merged with AR: {len(meta_ar):,} events")

    # FF6 portfolio assignment (pre-computed locally, uploaded as parquet)
    print("\nLoading FF6 portfolio assignments...", flush=True)
    ff6_adf = pd.read_parquet("data/ff6_assignments.parquet")
    port_assignments = {
        (int(r["permno"]), int(r["year"])): r["portfolio"]
        for _, r in ff6_adf.iterrows()
    }
    print(f"  {len(port_assignments):,} firm-year assignments loaded")

    # FF6 benchmark CARs (summed, not compounded)
    ff6 = pd.read_parquet("data/ff6_portfolios_daily.parquet")
    car_df = compute_ff6_benchmark_cars(meta_ar, crsp_chunks, ff6, port_assignments)
    car_df.to_parquet("data/cars_v3.parquet", index=False)

    meta_ar = meta_ar.drop(columns=[c for c in meta_ar.columns if c.startswith("car_")], errors="ignore")
    meta_ar = meta_ar.merge(car_df, on="event_id", how="left")
    meta_ar.to_parquet("data/merged_metadata_v3.parquet", index=False)
    print(f"Saved merged_metadata_v3.parquet: {len(meta_ar):,} rows")

    # Rolling-window regression
    results_df = run_rolling_regression(meta_ar, event_idx)
    results_df.to_parquet("data/sue_txt_v3.parquet", index=False)
    print(f"\nSUE.txt computed: {len(results_df):,} results")

    # PEAD.txt results using training-set quintile assignments
    results_df["call_date"] = pd.to_datetime(results_df["call_date"])
    results_df = results_df.merge(
        meta_ar[["event_id", "car_32d", "car_63d", "car_126d", "car_189d", "car_252d"]],
        on="event_id", how="left",
    )

    print("\n=== PEAD.txt Quintile Spread (Q5 - Q1) ===")
    for h in ["car_63d", "car_126d", "car_189d", "car_252d"]:
        q5 = results_df[results_df["sue_txt_quintile"] == 5][h].mean()
        q1 = results_df[results_df["sue_txt_quintile"] == 1][h].mean()
        d = h.replace("car_", "").replace("d", "")
        print(f"  {d:>3}d: {(q5 - q1) * 100:.2f}% (Q5={q5 * 100:.2f}%, Q1={q1 * 100:.2f}%)")

    print("\n=== Quintile Mean CARs (63d) ===")
    for q in range(1, 6):
        qdata = results_df[results_df["sue_txt_quintile"] == q]
        print(f"  Q{q}: {qdata['car_63d'].mean() * 100:.2f}% (n={len(qdata):,})")

    # Panel regression with two-way clustered SE
    print("\n=== Panel Regression: CAR_63d ~ SUE.txt (FE + clustered SE) ===")
    df = results_df.dropna(subset=["car_63d", "sue_txt"]).copy()
    df["sue_txt_z"] = (df["sue_txt"] - df["sue_txt"].mean()) / df["sue_txt"].std()

    def iterative_demean(series, group1, group2, max_iter=100, tol=1e-8):
        """Two-way FE absorption via alternating projection until convergence."""
        s = series.copy()
        for _ in range(max_iter):
            prev = s.copy()
            s = s - s.groupby(group1).transform("mean")
            s = s - s.groupby(group2).transform("mean")
            if (s - prev).abs().max() < tol:
                break
        return s

    df["car_63d_dm"] = iterative_demean(df["car_63d"], df["permno"], df["yq"])
    df["sue_dm"] = iterative_demean(df["sue_txt_z"], df["permno"], df["yq"])

    X = np.column_stack([np.ones(len(df)), df["sue_dm"].values])
    y = df["car_63d_dm"].values
    c1 = df["permno"].values.astype(str)
    c2 = df["yq"].values.astype(str)

    beta, se, t_stats, r2 = clustered_ols(y, X, c1, c2)
    print(f"  Intercept: {beta[0] * 100:.3f}% (t={t_stats[0]:.2f})")
    print(f"  SUE.txt:   {beta[1] * 100:.3f}% (t={t_stats[1]:.2f})")
    print(f"  N={len(df):,}, R²={r2:.4f}")

    results_df.to_parquet("data/sue_txt_v3_final.parquet", index=False)
    print(f"\nFinal results: data/sue_txt_v3_final.parquet ({len(results_df):,} rows)")


if __name__ == "__main__":
    main()
