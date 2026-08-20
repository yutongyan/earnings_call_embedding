#!/usr/bin/env python3
"""Merge transcripts with CRSP/Compustat/IBES and compute abnormal returns.

Uses existing WRDS data from /scratch/llm_in_asset_pricing/wrds_raw/.

Usage:
    python3 src/build_dataset.py --output data/merged_dataset.parquet
"""

import argparse
import os

import numpy as np
import pandas as pd

WRDS_DIR = "/scratch/llm_in_asset_pricing/wrds_raw"


def load_data(transcript_path):
    """Load transcripts and WRDS data."""
    print("Loading data files...", flush=True)

    if os.path.isdir(transcript_path) or "*" in transcript_path:
        import glob
        files = sorted(glob.glob(os.path.join(transcript_path, "transcripts_20*.parquet")))
        if not files:
            files = sorted(glob.glob(transcript_path))
        dfs = []
        for f in files:
            df = pd.read_parquet(f, columns=[
                "event_id", "call_date", "ticker", "company_name", "headline",
                "presentation_text", "qa_text", "n_words_pres", "n_words_qa",
            ])
            dfs.append(df)
            print(f"  {os.path.basename(f)}: {len(df):,} rows")
        transcripts = pd.concat(dfs, ignore_index=True)
    else:
        transcripts = pd.read_parquet(transcript_path)
    print(f"  transcripts total: {len(transcripts):,} rows")

    msenames = pd.read_parquet(f"{WRDS_DIR}/crsp_msenames.parquet")
    print(f"  crsp_msenames: {len(msenames):,} rows")

    ccm = pd.read_parquet(f"{WRDS_DIR}/crsp_ccmxpf_linktable.parquet")
    print(f"  ccm_link: {len(ccm):,} rows")

    compustat = pd.read_parquet(
        f"{WRDS_DIR}/comp_fundq.parquet",
        columns=["gvkey", "datadate", "fyearq", "fqtr", "rdq",
                  "epspxq", "epsfxq", "ibq", "saleq", "atq", "ceqq",
                  "cshoq", "prccq", "mkvaltq", "indfmt", "datafmt",
                  "popsrc", "consol"],
    )
    compustat = compustat[
        (compustat["indfmt"] == "INDL")
        & (compustat["datafmt"] == "STD")
        & (compustat["popsrc"] == "D")
        & (compustat["consol"] == "C")
    ].drop(columns=["indfmt", "datafmt", "popsrc", "consol"])
    print(f"  compustat_quarterly: {len(compustat):,} rows")

    ff = pd.read_parquet(f"{WRDS_DIR}/ff_factors_daily.parquet")
    print(f"  ff_factors: {len(ff):,} rows")

    print("  crsp_daily: deferred (will load after linking)")

    return transcripts, msenames, ccm, compustat, ff


def link_transcripts_to_crsp(transcripts, msenames):
    """Link transcript tickers to CRSP PERMNOs."""
    print("\nLinking transcripts to CRSP...", flush=True)

    def clean_ticker(t):
        if not isinstance(t, str) or not t:
            return ""
        t = t.split("^")[0]
        if "." in t:
            parts = t.split(".")
            us_ex = {"OQ", "N", "A", "PK", "O", "K", "CB"}
            if parts[-1] in us_ex:
                return ".".join(parts[:-1]).upper()
            return t.upper()
        return t.upper()

    transcripts = transcripts.copy()
    transcripts["ticker_clean"] = transcripts["ticker"].apply(clean_ticker)
    transcripts["call_date"] = pd.to_datetime(transcripts["call_date"])
    transcripts = transcripts.dropna(subset=["call_date"])

    msenames = msenames.copy()
    msenames["ticker"] = msenames["ticker"].str.upper().str.strip()
    msenames["namedt"] = pd.to_datetime(msenames["namedt"])
    msenames["nameendt"] = pd.to_datetime(msenames["nameendt"])

    us_stocks = msenames[msenames["shrcd"].isin([10, 11]) & msenames["exchcd"].isin([1, 2, 3])]
    ticker_permno = us_stocks[["permno", "ticker", "namedt", "nameendt"]].drop_duplicates()

    merged = transcripts.merge(
        ticker_permno,
        left_on="ticker_clean",
        right_on="ticker",
        how="inner",
        suffixes=("", "_crsp"),
    )
    merged = merged[
        (merged["call_date"] >= merged["namedt"])
        & (merged["call_date"] <= merged["nameendt"])
    ]
    merged = merged.drop_duplicates(subset=["event_id"], keep="first")
    merged = merged.drop(columns=["ticker_crsp", "namedt", "nameendt"], errors="ignore")

    print(f"  Linked: {len(merged):,} / {len(transcripts):,} transcripts")
    return merged


def compute_abnormal_returns(merged, crsp_daily, ff_factors):
    """Compute one-day abnormal returns using FF3+momentum."""
    print("\nComputing abnormal returns...", flush=True)

    crsp = crsp_daily.copy()
    crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
    crsp = crsp.dropna(subset=["ret"])

    ff = ff_factors.copy()
    ff["date"] = pd.to_datetime(ff["date"])

    crsp = crsp.merge(ff, on="date", how="inner")
    crsp["exret"] = crsp["ret"] - crsp["rf"]

    crsp = crsp.sort_values(["permno", "date"]).reset_index(drop=True)

    merged_dates = merged[["event_id", "permno", "call_date"]].copy()
    merged_dates["call_date"] = pd.to_datetime(merged_dates["call_date"])

    results = []
    grouped = crsp.groupby("permno")
    permnos = merged_dates["permno"].unique()

    for i, permno in enumerate(permnos):
        if i % 1000 == 0:
            print(f"  AR: permno {i}/{len(permnos)}...", flush=True)

        if permno not in grouped.groups:
            continue

        firm_crsp = grouped.get_group(permno).reset_index(drop=True)
        if len(firm_crsp) < 100:
            continue

        firm_calls = merged_dates[merged_dates["permno"] == permno]
        dates_arr = firm_crsp["date"].values

        for _, call in firm_calls.iterrows():
            cd = call["call_date"]
            cd_np = np.datetime64(cd)

            idx = np.searchsorted(dates_arr, cd_np, side="right") - 1
            if idx < 0:
                idx = np.searchsorted(dates_arr, cd_np, side="left")
            if idx < 0 or idx >= len(firm_crsp):
                continue

            est_end = max(0, idx - 5)
            est_start = max(0, est_end - 252)
            if est_end - est_start < 60:
                continue

            est = firm_crsp.iloc[est_start:est_end].dropna(
                subset=["exret", "mktrf", "smb", "hml", "umd"]
            )
            if len(est) < 60:
                continue

            try:
                X = est[["mktrf", "smb", "hml", "umd"]].values
                y = est["exret"].values
                X_c = np.column_stack([np.ones(len(X)), X])
                beta = np.linalg.lstsq(X_c, y, rcond=None)[0]

                ev = firm_crsp.iloc[idx]
                ev_x = np.array([1.0, ev["mktrf"], ev["smb"], ev["hml"], ev["umd"]])
                ar = ev["exret"] - np.dot(beta, ev_x)

                results.append({
                    "event_id": call["event_id"],
                    "permno": permno,
                    "call_date": call["call_date"],
                    "event_date": ev["date"],
                    "ret": ev["ret"],
                    "abnormal_return": float(ar),
                })
            except Exception:
                continue

    ar_df = pd.DataFrame(results)
    print(f"  Computed AR for {len(ar_df):,} events")
    return ar_df


def compute_long_run_cars(merged_with_ar, crsp_daily):
    """Compute buy-and-hold abnormal returns at multiple horizons."""
    print("\nComputing long-run CARs...", flush=True)

    crsp = crsp_daily[["permno", "date", "ret"]].copy()
    crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
    crsp = crsp.dropna(subset=["ret"])
    crsp = crsp.sort_values(["permno", "date"]).reset_index(drop=True)

    horizons = [32, 63, 126, 189, 252]
    grouped = crsp.groupby("permno")
    permnos = merged_with_ar["permno"].unique()

    car_results = []

    for i, permno in enumerate(permnos):
        if i % 1000 == 0:
            print(f"  CAR: permno {i}/{len(permnos)}...", flush=True)

        if permno not in grouped.groups:
            continue

        firm_crsp = grouped.get_group(permno).reset_index(drop=True)
        if len(firm_crsp) < 50:
            continue

        dates_arr = firm_crsp["date"].values
        firm_calls = merged_with_ar[merged_with_ar["permno"] == permno]

        for _, call in firm_calls.iterrows():
            ed = pd.to_datetime(call["event_date"])
            start_idx = np.searchsorted(dates_arr, np.datetime64(ed), side="left")
            if start_idx >= len(dates_arr):
                continue

            row = {"event_id": call["event_id"]}
            for h in horizons:
                end_idx = min(start_idx + h + 1, len(firm_crsp))
                if end_idx <= start_idx + 1:
                    row[f"car_{h}d"] = np.nan
                    continue
                window = firm_crsp.iloc[start_idx + 1:end_idx]
                bhar = (1 + window["ret"]).prod() - 1
                row[f"car_{h}d"] = float(bhar)

            car_results.append(row)

    car_df = pd.DataFrame(car_results)
    print(f"  Computed CARs for {len(car_df):,} events")
    return car_df


def link_to_compustat(merged, ccm, compustat):
    """Link to Compustat for SUE and fundamentals."""
    print("\nLinking to Compustat...", flush=True)

    ccm = ccm.copy()
    ccm = ccm[ccm["linktype"].isin(["LU", "LC"]) & ccm["linkprim"].isin(["P", "C"])]
    ccm = ccm.rename(columns={"lpermno": "permno"})
    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"])
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"])
    ccm.loc[ccm["linkenddt"].isna(), "linkenddt"] = pd.Timestamp("2099-12-31")

    merged = merged.merge(
        ccm[["permno", "gvkey", "linkdt", "linkenddt"]],
        on="permno",
        how="left",
    )
    merged = merged[
        merged["gvkey"].notna()
        & (merged["call_date"] >= merged["linkdt"])
        & (merged["call_date"] <= merged["linkenddt"])
    ]
    merged = merged.drop_duplicates(subset=["event_id"], keep="first")
    merged = merged.drop(columns=["linkdt", "linkenddt"])

    comp = compustat.copy()
    comp["rdq"] = pd.to_datetime(comp["rdq"])
    comp = comp.sort_values(["gvkey", "rdq"]).drop_duplicates(
        subset=["gvkey", "fyearq", "fqtr"], keep="last"
    )
    comp["epspxq_lag4"] = comp.groupby("gvkey")["epspxq"].shift(4)
    comp["sue1"] = comp["epspxq"] - comp["epspxq_lag4"]

    merged = merged.merge(comp, on="gvkey", how="left", suffixes=("", "_comp"))

    if "rdq" in merged.columns:
        merged["date_diff"] = (merged["call_date"] - merged["rdq"]).dt.days.abs()
        merged = merged.sort_values("date_diff")
        merged = merged.drop_duplicates(subset=["event_id"], keep="first")
        merged = merged.drop(columns=["date_diff"])

    print(f"  After Compustat merge: {len(merged):,} rows")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", default="data/transcripts_parsed.parquet")
    parser.add_argument("--output", default="data/merged_dataset.parquet")
    args = parser.parse_args()

    transcripts, msenames, ccm, compustat, ff = load_data(args.transcripts)

    merged = link_transcripts_to_crsp(transcripts, msenames)
    del transcripts

    needed_permnos = set(merged["permno"].unique())
    print(f"\nLoading CRSP daily for {len(needed_permnos):,} permnos...", flush=True)
    crsp_daily = pd.read_parquet(
        f"{WRDS_DIR}/crsp_dsf.parquet",
        columns=["permno", "date", "ret"],
        filters=[("permno", "in", list(needed_permnos))],
    )
    crsp_daily["date"] = pd.to_datetime(crsp_daily["date"])
    crsp_daily = crsp_daily[
        (crsp_daily["date"] >= "2007-01-01")
        & (crsp_daily["date"] <= "2020-12-31")
    ]
    print(f"  crsp_daily: {len(crsp_daily):,} rows")

    ar_df = compute_abnormal_returns(merged, crsp_daily, ff)

    merged_with_ar = merged.merge(
        ar_df, on=["event_id", "permno", "call_date"], how="inner"
    )

    abs_ar = merged_with_ar["abnormal_return"].abs()
    cutoff = abs_ar.quantile(1 / 3)
    merged_with_ar["return_category"] = "F"
    merged_with_ar.loc[merged_with_ar["abnormal_return"] > cutoff, "return_category"] = "H"
    merged_with_ar.loc[merged_with_ar["abnormal_return"] < -cutoff, "return_category"] = "L"

    car_df = compute_long_run_cars(merged_with_ar, crsp_daily)
    merged_with_ar = merged_with_ar.merge(car_df, on="event_id", how="left")

    merged_with_ar = link_to_compustat(merged_with_ar, ccm, compustat)

    merged_with_ar["year_quarter"] = (
        merged_with_ar["call_date"].dt.year.astype(str)
        + "Q"
        + merged_with_ar["call_date"].dt.quarter.astype(str)
    )

    keep_cols = [
        "event_id", "permno", "gvkey", "call_date", "event_date",
        "ticker_clean", "company_name", "headline",
        "presentation_text", "qa_text", "n_words_pres", "n_words_qa",
        "ret", "abnormal_return", "return_category",
        "car_32d", "car_63d", "car_126d", "car_189d", "car_252d",
        "sue1", "epspxq", "saleq", "atq", "mkvaltq",
        "year_quarter",
    ]
    keep_cols = [c for c in keep_cols if c in merged_with_ar.columns]
    merged_with_ar = merged_with_ar[keep_cols]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    merged_with_ar.to_parquet(args.output, index=False)

    print(f"\n=== Final Dataset ===")
    print(f"Rows: {len(merged_with_ar):,}")
    print(f"Unique firms: {merged_with_ar['permno'].nunique():,}")
    print(f"Date range: {merged_with_ar['call_date'].min()} to {merged_with_ar['call_date'].max()}")
    cats = merged_with_ar["return_category"].value_counts()
    print(f"Return categories: {cats.to_dict()}")
    print(f"Median pres words: {merged_with_ar['n_words_pres'].median():.0f}")
    print(f"Median Q&A words: {merged_with_ar['n_words_qa'].median():.0f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
