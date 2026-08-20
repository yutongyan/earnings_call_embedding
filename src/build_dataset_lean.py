#!/usr/bin/env python3
"""Build merged dataset in a memory-efficient way.

Step 1: Load transcript metadata (no text), link to CRSP, compute ARs
Step 2: Save metadata + ARs
Step 3: Regression script joins text back from per-year files on demand

Usage:
    python3 src/build_dataset_lean.py --output data/merged_metadata.parquet
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

WRDS_DIR = "/scratch/llm_in_asset_pricing/wrds_raw"


def load_transcript_metadata(data_dir):
    """Load only metadata columns from per-year transcript parquets."""
    print("Loading transcript metadata...", flush=True)
    files = sorted(glob.glob(os.path.join(data_dir, "transcripts_20*.parquet")))
    dfs = []
    for f in files:
        df = pd.read_parquet(f, columns=[
            "event_id", "call_date", "ticker", "company_name",
            "n_words_pres", "n_words_qa",
        ])
        dfs.append(df)
        print(f"  {os.path.basename(f)}: {len(df):,}", flush=True)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined):,} transcripts")
    return combined


def link_to_crsp(transcripts, msenames):
    """Link transcript tickers to CRSP PERMNOs."""
    print("\nLinking to CRSP...", flush=True)

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

    ms = msenames.copy()
    ms["ticker"] = ms["ticker"].str.upper().str.strip()
    ms["namedt"] = pd.to_datetime(ms["namedt"])
    ms["nameendt"] = pd.to_datetime(ms["nameendt"])

    us = ms[ms["shrcd"].isin([10, 11]) & ms["exchcd"].isin([1, 2, 3])]
    tp = us[["permno", "ticker", "namedt", "nameendt"]].drop_duplicates()

    merged = transcripts.merge(tp, left_on="ticker_clean", right_on="ticker",
                                how="inner", suffixes=("", "_crsp"))
    merged = merged[
        (merged["call_date"] >= merged["namedt"])
        & (merged["call_date"] <= merged["nameendt"])
    ]
    merged = merged.drop_duplicates(subset=["event_id"], keep="first")
    merged = merged.drop(columns=["ticker_crsp", "namedt", "nameendt"], errors="ignore")

    print(f"  Linked: {len(merged):,} / {len(transcripts):,}")
    return merged


def compute_abnormal_returns(merged, ff):
    """Compute one-day AR using FF3+momentum, loading CRSP per-permno."""
    print("\nComputing abnormal returns...", flush=True)

    permnos = sorted(merged["permno"].unique())
    print(f"  {len(permnos):,} unique permnos to process")

    ff = ff.copy()
    ff["date"] = pd.to_datetime(ff["date"])

    chunk_size = 500
    all_results = []

    for chunk_start in range(0, len(permnos), chunk_size):
        chunk_permnos = permnos[chunk_start:chunk_start + chunk_size]
        print(f"  Chunk {chunk_start//chunk_size + 1}: permnos {chunk_start}-{chunk_start+len(chunk_permnos)-1}...", flush=True)

        crsp_chunk = pd.read_parquet(
            f"{WRDS_DIR}/crsp_dsf.parquet",
            columns=["permno", "date", "ret"],
            filters=[("permno", "in", chunk_permnos)],
        )
        crsp_chunk["date"] = pd.to_datetime(crsp_chunk["date"])
        crsp_chunk = crsp_chunk[
            (crsp_chunk["date"] >= "2007-01-01")
            & (crsp_chunk["date"] <= "2020-12-31")
        ]
        crsp_chunk["ret"] = pd.to_numeric(crsp_chunk["ret"], errors="coerce")
        crsp_chunk = crsp_chunk.dropna(subset=["ret"])
        crsp_chunk = crsp_chunk.merge(ff, on="date", how="inner")
        crsp_chunk["exret"] = crsp_chunk["ret"] - crsp_chunk["rf"]
        crsp_chunk = crsp_chunk.sort_values(["permno", "date"]).reset_index(drop=True)

        grouped = crsp_chunk.groupby("permno")

        for permno in chunk_permnos:
            if permno not in grouped.groups:
                continue
            firm_crsp = grouped.get_group(permno).reset_index(drop=True)
            if len(firm_crsp) < 100:
                continue

            firm_calls = merged[merged["permno"] == permno]
            dates_arr = firm_crsp["date"].values

            for _, call in firm_calls.iterrows():
                cd_np = np.datetime64(call["call_date"])
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
                    Xc = np.column_stack([np.ones(len(X)), X])
                    beta = np.linalg.lstsq(Xc, y, rcond=None)[0]

                    ev = firm_crsp.iloc[idx]
                    ar = ev["exret"] - np.dot(beta, [1, ev["mktrf"], ev["smb"], ev["hml"], ev["umd"]])

                    all_results.append({
                        "event_id": call["event_id"],
                        "permno": permno,
                        "call_date": call["call_date"],
                        "event_date": ev["date"],
                        "ret": float(ev["ret"]),
                        "abnormal_return": float(ar),
                    })
                except Exception:
                    continue

        del crsp_chunk

    ar_df = pd.DataFrame(all_results)
    print(f"  Computed AR for {len(ar_df):,} events")
    return ar_df


def compute_long_run_cars(merged_with_ar):
    """Compute CARs at multiple horizons, loading CRSP in chunks."""
    print("\nComputing long-run CARs...", flush=True)

    permnos = sorted(merged_with_ar["permno"].unique())
    horizons = [32, 63, 126, 189, 252]
    chunk_size = 500
    all_cars = []

    for chunk_start in range(0, len(permnos), chunk_size):
        chunk_permnos = permnos[chunk_start:chunk_start + chunk_size]
        print(f"  CAR chunk {chunk_start//chunk_size + 1}...", flush=True)

        crsp_chunk = pd.read_parquet(
            f"{WRDS_DIR}/crsp_dsf.parquet",
            columns=["permno", "date", "ret"],
            filters=[("permno", "in", chunk_permnos)],
        )
        crsp_chunk["date"] = pd.to_datetime(crsp_chunk["date"])
        crsp_chunk = crsp_chunk[
            (crsp_chunk["date"] >= "2007-01-01")
            & (crsp_chunk["date"] <= "2021-06-30")
        ]
        crsp_chunk["ret"] = pd.to_numeric(crsp_chunk["ret"], errors="coerce")
        crsp_chunk = crsp_chunk.dropna(subset=["ret"])
        crsp_chunk = crsp_chunk.sort_values(["permno", "date"]).reset_index(drop=True)

        grouped = crsp_chunk.groupby("permno")

        for permno in chunk_permnos:
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
                    row[f"car_{h}d"] = float((1 + window["ret"]).prod() - 1)

                all_cars.append(row)

        del crsp_chunk

    car_df = pd.DataFrame(all_cars)
    print(f"  Computed CARs for {len(car_df):,} events")
    return car_df


def link_to_compustat(merged, ccm, compustat):
    """Link to Compustat for SUE."""
    print("\nLinking to Compustat...", flush=True)

    ccm = ccm.copy()
    ccm = ccm[ccm["linktype"].isin(["LU", "LC"]) & ccm["linkprim"].isin(["P", "C"])]
    ccm = ccm.rename(columns={"lpermno": "permno"})
    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"])
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"])
    ccm.loc[ccm["linkenddt"].isna(), "linkenddt"] = pd.Timestamp("2099-12-31")

    merged = merged.merge(
        ccm[["permno", "gvkey", "linkdt", "linkenddt"]],
        on="permno", how="left",
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

    merged = merged.merge(
        comp[["gvkey", "rdq", "fyearq", "fqtr", "epspxq", "sue1", "saleq", "atq", "mkvaltq"]],
        on="gvkey", how="left", suffixes=("", "_comp"),
    )
    if "rdq" in merged.columns:
        merged["date_diff"] = (merged["call_date"] - merged["rdq"]).dt.days.abs()
        merged = merged.sort_values("date_diff")
        merged = merged.drop_duplicates(subset=["event_id"], keep="first")
        merged = merged.drop(columns=["date_diff"])

    print(f"  After Compustat: {len(merged):,} rows")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output", default="data/merged_metadata.parquet")
    args = parser.parse_args()

    transcripts = load_transcript_metadata(args.data_dir)

    msenames = pd.read_parquet(f"{WRDS_DIR}/crsp_msenames.parquet")
    merged = link_to_crsp(transcripts, msenames)
    del transcripts, msenames

    ff = pd.read_parquet(f"{WRDS_DIR}/ff_factors_daily.parquet")
    ar_df = compute_abnormal_returns(merged, ff)

    merged_with_ar = merged.merge(
        ar_df, on=["event_id", "permno", "call_date"], how="inner"
    )
    del merged, ar_df

    abs_ar = merged_with_ar["abnormal_return"].abs()
    cutoff = abs_ar.quantile(1 / 3)
    merged_with_ar["return_category"] = "F"
    merged_with_ar.loc[merged_with_ar["abnormal_return"] > cutoff, "return_category"] = "H"
    merged_with_ar.loc[merged_with_ar["abnormal_return"] < -cutoff, "return_category"] = "L"

    car_df = compute_long_run_cars(merged_with_ar)
    merged_with_ar = merged_with_ar.merge(car_df, on="event_id", how="left")
    del car_df

    ccm = pd.read_parquet(f"{WRDS_DIR}/crsp_ccmxpf_linktable.parquet")
    compustat = pd.read_parquet(
        f"{WRDS_DIR}/comp_fundq.parquet",
        columns=["gvkey", "datadate", "fyearq", "fqtr", "rdq",
                  "epspxq", "saleq", "atq", "mkvaltq",
                  "indfmt", "datafmt", "popsrc", "consol"],
    )
    compustat = compustat[
        (compustat["indfmt"] == "INDL") & (compustat["datafmt"] == "STD")
        & (compustat["popsrc"] == "D") & (compustat["consol"] == "C")
    ].drop(columns=["indfmt", "datafmt", "popsrc", "consol"])

    merged_with_ar = link_to_compustat(merged_with_ar, ccm, compustat)
    del ccm, compustat

    merged_with_ar["year_quarter"] = (
        merged_with_ar["call_date"].dt.year.astype(str)
        + "Q" + merged_with_ar["call_date"].dt.quarter.astype(str)
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    merged_with_ar.to_parquet(args.output, index=False)

    print(f"\n=== Final Metadata Dataset ===")
    print(f"Rows: {len(merged_with_ar):,}")
    print(f"Unique firms: {merged_with_ar['permno'].nunique():,}")
    print(f"Date range: {merged_with_ar['call_date'].min()} to {merged_with_ar['call_date'].max()}")
    cats = merged_with_ar["return_category"].value_counts()
    print(f"Return categories: {cats.to_dict()}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
