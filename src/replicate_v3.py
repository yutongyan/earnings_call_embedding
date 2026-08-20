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
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV

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


def assign_ff6_portfolio(permno_data, compustat_data, ccm_link):
    """Assign each firm-year to one of 6 FF size/BM portfolios."""
    ccm = ccm_link.copy()
    ccm = ccm[ccm["linktype"].isin(["LU", "LC"]) & ccm["linkprim"].isin(["P", "C"])]
    ccm = ccm.rename(columns={"lpermno": "permno"})

    june_me = permno_data[permno_data["month"] == 6][["permno", "year", "me"]].copy()

    comp = compustat_data.copy()
    comp["year"] = comp["datadate"].dt.year
    comp = comp.merge(ccm[["permno", "gvkey"]], on="gvkey", how="inner")
    be = comp.groupby(["permno", "year"]).last()[["ceqq"]].reset_index()
    be = be.rename(columns={"ceqq": "be"})
    be["year"] = be["year"] + 1

    merged = june_me.merge(be, on=["permno", "year"], how="left")
    merged["bm"] = merged["be"] / merged["me"]
    merged = merged.dropna(subset=["me", "bm"])
    merged = merged[merged["bm"] > 0]

    assignments = {}
    for year in merged["year"].unique():
        yr = merged[merged["year"] == year]
        me_median = yr["me"].median()
        bm_30 = yr["bm"].quantile(0.3)
        bm_70 = yr["bm"].quantile(0.7)

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

                call_year = ed.year
                port_key = portfolio_assignments.get((int(permno), call_year))
                if port_key is None:
                    port_key = portfolio_assignments.get((int(permno), call_year - 1), "smme")
                port_col = port_cols.get(port_key, "smme_vwret")

                row = {"event_id": call["event_id"]}
                for h in horizons:
                    ei = min(si + h + 1, len(fc))
                    if ei <= si + 1:
                        row[f"car_{h}d"] = np.nan
                        continue
                    window = fc.iloc[si + 1:ei]
                    daily_ar = window["ret"].values - window[port_col].values
                    row[f"car_{h}d"] = float(np.nansum(daily_ar))

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


def run_rolling_regression(meta, event_year_index):
    print("\n=== Rolling-window regression (v3) ===", flush=True)

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

    prev_quintile_cutoffs = None

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

        y_train, _ = categorize_returns(train_meta["abnormal_return"])

        train_eids = train_meta["event_id"].tolist()
        test_eids = test_meta["event_id"].tolist()

        X_train, X_test = build_rolling_features(train_eids, test_eids, event_year_index)

        try:
            base_model = SGDClassifier(
                loss="log_loss", penalty="elasticnet", l1_ratio=0.5,
                max_iter=500, random_state=42, tol=1e-3,
            )
            param_grid = {"alpha": [0.0001, 0.001, 0.01, 0.1]}
            cv = GridSearchCV(base_model, param_grid, cv=3,
                              scoring="neg_log_loss", n_jobs=-1, refit=False)
            cv.fit(X_train, y_train)
            best_alpha = cv.best_params_["alpha"]

            best_base = SGDClassifier(
                loss="log_loss", penalty="elasticnet", l1_ratio=0.5,
                alpha=best_alpha, max_iter=500, random_state=42, tol=1e-3,
            )
            model = CalibratedClassifierCV(best_base, cv=3, method="sigmoid", n_jobs=-1)
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"  ERROR fitting: {e}", flush=True)
            del X_train, X_test
            gc.collect()
            continue

        classes = list(model.classes_)
        if "H" not in classes or "L" not in classes:
            print(f"  WARNING: classes={classes}, skipping", flush=True)
            del X_train, X_test
            gc.collect()
            continue

        sue_txt_test = compute_sue_txt(model, X_test)

        sue_txt_train = compute_sue_txt(model, X_train)
        quintile_cutoffs = np.percentile(sue_txt_train, [20, 40, 60, 80])
        prev_quintile_cutoffs = quintile_cutoffs

        for j, (idx, row) in enumerate(test_meta.iterrows()):
            q_assign = np.searchsorted(quintile_cutoffs, sue_txt_test[j]) + 1
            results.append({
                "event_id": row["event_id"],
                "permno": row["permno"],
                "call_date": row["call_date"],
                "yq": str(test_q),
                "sue_txt": float(sue_txt_test[j]),
                "sue_txt_quintile": int(q_assign),
                "abnormal_return": row["abnormal_return"],
                "best_alpha": best_alpha,
            })

        print(f"  alpha={best_alpha}, SUE.txt mean={np.mean(sue_txt_test):.3f} std={np.std(sue_txt_test):.3f}", flush=True)

        del X_train, X_test, model
        gc.collect()

        if (i + 1) % 5 == 0 or test_q == test_quarters[-1]:
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
    interaction = cluster1.astype(str) + "_" + cluster2.astype(str)
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

    # FF6 portfolio assignment
    print("\nAssigning firms to FF6 portfolios...", flush=True)
    crsp_me = pd.read_parquet(
        f"{WRDS_DIR}/crsp_dsf.parquet",
        columns=["permno", "date", "prc", "shrout"],
    )
    crsp_me["date"] = pd.to_datetime(crsp_me["date"])
    crsp_me = crsp_me[(crsp_me["date"] >= "2007-01-01") & (crsp_me["date"] <= "2020-12-31")]
    crsp_me["prc"] = pd.to_numeric(crsp_me["prc"], errors="coerce").abs()
    crsp_me["shrout"] = pd.to_numeric(crsp_me["shrout"], errors="coerce")
    crsp_me["me"] = crsp_me["prc"] * crsp_me["shrout"] / 1000
    crsp_me["month"] = crsp_me["date"].dt.month
    crsp_me["year"] = crsp_me["date"].dt.year
    june_me = crsp_me.groupby(["permno", "year", "month"]).last().reset_index()
    june_me = june_me[june_me["month"] == 6][["permno", "year", "me"]].dropna()
    del crsp_me
    gc.collect()

    compustat = pd.read_parquet(
        f"{WRDS_DIR}/comp_fundq.parquet",
        columns=["gvkey", "datadate", "ceqq", "indfmt", "datafmt", "popsrc", "consol"],
    )
    compustat = compustat[
        (compustat["indfmt"] == "INDL") & (compustat["datafmt"] == "STD")
        & (compustat["popsrc"] == "D") & (compustat["consol"] == "C")
    ]
    compustat["datadate"] = pd.to_datetime(compustat["datadate"])
    ccm = pd.read_parquet(f"{WRDS_DIR}/crsp_ccmxpf_linktable.parquet")

    port_assignments = assign_ff6_portfolio(june_me, compustat, ccm)
    print(f"  Assigned {len(port_assignments):,} firm-years to portfolios")
    del june_me, compustat, ccm
    gc.collect()

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

    df["car_63d_dm"] = df.groupby("permno")["car_63d"].transform(lambda x: x - x.mean())
    df["car_63d_dm"] = df.groupby("yq")["car_63d_dm"].transform(lambda x: x - x.mean())
    df["sue_dm"] = df.groupby("permno")["sue_txt_z"].transform(lambda x: x - x.mean())
    df["sue_dm"] = df.groupby("yq")["sue_dm"].transform(lambda x: x - x.mean())

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
