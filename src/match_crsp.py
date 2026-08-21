#!/usr/bin/env python3
"""Stage 2: Match earnings calls to CRSP/Compustat and compute returns.

Requires Stage 1 (process_earnings.py) to have been run first.
Requires WRDS access via ~/.pgpass.

Steps:
1. Link to CRSP: CUSIP-first, then ticker fallback
2. Compute abnormal returns with timing heuristic (2 PM ET cutoff)
3. Compute FF6 benchmark-adjusted CARs (summed)
4. Winsorize continuous variables at 1%/99%
5. Save final merged dataset

Usage:
    echo 'yutongyancuhk' | python3 -u src/match_crsp.py --data_dir data/
"""

import argparse
import gc
import glob
import os

import numpy as np
import pandas as pd
import wrds

CRSP_SENTINELS = {-66.0, -77.0, -88.0, -99.0}


def get_wrds_connection(wrds_user):
    """Connect to WRDS, reading password from ~/.pgpass if needed."""
    pgpass = os.path.expanduser("~/.pgpass")
    password = ""
    if os.path.exists(pgpass):
        with open(pgpass) as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) >= 5 and "wrds" in parts[0]:
                    password = ":".join(parts[4:])
                    break
    conn = wrds.Connection.__new__(wrds.Connection)
    conn._hostname = "wrds-pgdata.wharton.upenn.edu"
    conn._port = 9737
    conn._dbname = "wrds"
    conn._username = wrds_user
    conn._password = password
    conn._verbose = False
    conn._connect_args = {}
    conn.engine = None
    conn._Connection__make_sa_engine_conn(raise_err=True)
    return conn


def winsorize(s, lower=0.01, upper=0.99):
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


# ============================================================
# Step 1: Link to CRSP
# ============================================================

def link_to_crsp(data_dir, wrds_user):
    """Link transcripts to CRSP via CUSIP then ticker."""
    print("=== Step 1: Linking to CRSP ===", flush=True)

    dfs = []
    for f in sorted(glob.glob(os.path.join(data_dir, "transcripts_20*.parquet"))):
        df = pd.read_parquet(f, columns=[
            "event_id", "company_id", "call_date", "start_date_str", "last_update",
            "ticker", "cusip", "isin", "company_name",
            "n_words_pres", "n_words_qa", "is_before_close", "story_version",
        ])
        dfs.append(df)
    transcripts = pd.concat(dfs, ignore_index=True)
    transcripts["call_date"] = pd.to_datetime(transcripts["call_date"])
    transcripts = transcripts.dropna(subset=["call_date"])
    print(f"  Transcripts: {len(transcripts):,}", flush=True)

    db = get_wrds_connection(wrds_user)
    msenames = db.raw_sql("""
        SELECT permno, namedt, nameendt, ncusip, ticker, shrcd, exchcd
        FROM crsp.msenames
        WHERE shrcd IN (10, 11) AND exchcd IN (1, 2, 3)
    """)
    db.close()

    msenames["namedt"] = pd.to_datetime(msenames["namedt"])
    msenames["nameendt"] = pd.to_datetime(msenames["nameendt"])
    msenames["ticker"] = msenames["ticker"].str.upper().str.strip()
    msenames["ncusip"] = msenames["ncusip"].str.strip()

    # CUSIP match
    transcripts["cusip8"] = transcripts["cusip"].str[:8].str.strip()
    cusip_link = msenames[["permno", "ncusip", "namedt", "nameendt"]].drop_duplicates()

    matched_cusip = transcripts[transcripts["cusip8"] != ""].merge(
        cusip_link, left_on="cusip8", right_on="ncusip", how="inner", suffixes=("", "_crsp")
    )
    matched_cusip = matched_cusip[
        (matched_cusip["call_date"] >= matched_cusip["namedt"])
        & (matched_cusip["call_date"] <= matched_cusip["nameendt"])
    ]
    matched_cusip = matched_cusip.drop_duplicates(subset=["event_id"], keep="first")
    cusip_eids = set(matched_cusip["event_id"])
    print(f"  CUSIP matched: {len(matched_cusip):,}", flush=True)

    # Ticker fallback
    def clean_ticker(t):
        if not isinstance(t, str) or not t:
            return ""
        t = t.split("^")[0]
        if "." in t:
            parts = t.split(".")
            if parts[-1] in {"OQ", "N", "A", "PK", "O", "K", "CB"}:
                return ".".join(parts[:-1]).upper()
            return t.upper()
        return t.upper()

    unmatched = transcripts[~transcripts["event_id"].isin(cusip_eids)].copy()
    unmatched["ticker_clean"] = unmatched["ticker"].apply(clean_ticker)
    ticker_link = msenames[["permno", "ticker", "namedt", "nameendt"]].drop_duplicates()

    matched_ticker = unmatched[unmatched["ticker_clean"] != ""].merge(
        ticker_link, left_on="ticker_clean", right_on="ticker", how="inner", suffixes=("", "_crsp")
    )
    matched_ticker = matched_ticker[
        (matched_ticker["call_date"] >= matched_ticker["namedt"])
        & (matched_ticker["call_date"] <= matched_ticker["nameendt"])
    ]
    matched_ticker = matched_ticker.drop_duplicates(subset=["event_id"], keep="first")
    print(f"  Ticker matched: {len(matched_ticker):,}", flush=True)

    # Combine
    keep_cols = [
        "event_id", "company_id", "call_date", "start_date_str", "last_update",
        "ticker", "cusip", "company_name", "permno",
        "n_words_pres", "n_words_qa", "is_before_close", "story_version",
    ]
    for c in keep_cols:
        if c not in matched_cusip.columns:
            matched_cusip[c] = ""
        if c not in matched_ticker.columns:
            matched_ticker[c] = ""

    linked = pd.concat([matched_cusip[keep_cols], matched_ticker[keep_cols]], ignore_index=True)
    linked["permno"] = linked["permno"].astype("int64")
    linked = linked.drop_duplicates(subset=["event_id"], keep="first")

    # Drop outlier-length calls
    total_words = linked["n_words_pres"] + linked["n_words_qa"]
    cutoff = total_words.quantile(0.995)
    before = len(linked)
    linked = linked[total_words <= cutoff]
    print(f"  Total linked: {len(linked):,} (dropped {before - len(linked)} outliers)", flush=True)

    linked.to_parquet(os.path.join(data_dir, "linked_transcripts_v4.parquet"), index=False)
    return linked


# ============================================================
# Step 2: Compute abnormal returns
# ============================================================

def compute_abnormal_returns(linked, data_dir, wrds_user):
    """Compute one-day AR using FF3+momentum with timing heuristic."""
    print("\n=== Step 2: Computing abnormal returns ===", flush=True)

    db = get_wrds_connection(wrds_user)

    ff = db.raw_sql("""
        SELECT date, mktrf, smb, hml, rf, umd
        FROM ff.factors_daily
        WHERE date BETWEEN '2000-01-01' AND '2026-12-31'
    """)
    ff["date"] = pd.to_datetime(ff["date"])
    for col in ["mktrf", "smb", "hml", "umd", "rf"]:
        ff[col] = pd.to_numeric(ff[col], errors="coerce").astype("float64")

    permnos = linked["permno"].unique().tolist()
    print(f"  {len(permnos):,} unique permnos", flush=True)

    all_results = []
    batch_size = 500
    for batch_start in range(0, len(permnos), batch_size):
        batch = permnos[batch_start:batch_start + batch_size]
        plist = ",".join(str(int(p)) for p in batch)
        crsp = db.raw_sql(f"""
            SELECT permno, date, ret
            FROM crsp.dsf
            WHERE permno IN ({plist})
              AND date BETWEEN '2000-01-01' AND '2026-12-31'
        """)
        crsp["date"] = pd.to_datetime(crsp["date"])
        crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
        crsp = crsp[~crsp["ret"].isin(CRSP_SENTINELS)]
        crsp = crsp.dropna(subset=["ret"])
        crsp = crsp.merge(ff, on="date", how="inner")
        crsp["exret"] = crsp["ret"] - crsp["rf"]
        crsp = crsp.sort_values(["permno", "date"]).reset_index(drop=True)

        grouped = crsp.groupby("permno")
        batch_calls = linked[linked["permno"].isin(set(batch))]

        for permno, calls in batch_calls.groupby("permno"):
            if permno not in grouped.groups:
                continue
            fc = grouped.get_group(permno).reset_index(drop=True)
            if len(fc) < 100:
                continue
            dates_arr = fc["date"].values

            for _, call in calls.iterrows():
                cd_np = np.datetime64(call["call_date"])
                idx = np.searchsorted(dates_arr, cd_np, side="right") - 1
                if idx < 0:
                    idx = np.searchsorted(dates_arr, cd_np, side="left")

                use_same_day = call.get("is_before_close", True)
                if use_same_day is None:
                    use_same_day = True
                if not use_same_day and idx + 1 < len(fc):
                    idx = idx + 1

                if idx < 0 or idx >= len(fc):
                    continue

                est_end = max(0, idx - 5)
                est_start = max(0, est_end - 252)
                if est_end - est_start < 60:
                    continue
                est = fc.iloc[est_start:est_end].dropna(
                    subset=["exret", "mktrf", "smb", "hml", "umd"]
                )
                if len(est) < 60:
                    continue

                X = est[["mktrf", "smb", "hml", "umd"]].values.astype(np.float64)
                y = est["exret"].values.astype(np.float64)
                beta = np.linalg.lstsq(
                    np.column_stack([np.ones(len(X)), X]), y, rcond=None
                )[0]
                ev = fc.iloc[idx]
                if pd.isna(ev["exret"]):
                    continue
                ar = float(ev["exret"]) - float(
                    np.dot(beta, [1, ev["mktrf"], ev["smb"], ev["hml"], ev["umd"]])
                )
                all_results.append({
                    "event_id": call["event_id"],
                    "permno": int(permno),
                    "call_date": call["call_date"],
                    "event_date": ev["date"],
                    "ret": float(ev["ret"]),
                    "abnormal_return": ar,
                })

        del crsp
        gc.collect()
        print(f"  Batch {batch_start // batch_size + 1}/{(len(permnos) - 1) // batch_size + 1}: "
              f"{len(all_results):,} ARs", flush=True)

    db.close()

    ar_df = pd.DataFrame(all_results)
    ar_df["abnormal_return"] = winsorize(ar_df["abnormal_return"])
    ar_df.to_parquet(os.path.join(data_dir, "abnormal_returns_v4.parquet"), index=False)
    print(f"  Total: {len(ar_df):,} ARs", flush=True)
    return ar_df


# ============================================================
# Step 3: Compute FF6 CARs
# ============================================================

def compute_cars(merged_ar, data_dir, wrds_user):
    """Compute CARs using FF6 size/BM portfolios, summed."""
    print("\n=== Step 3: Computing FF6 CARs ===", flush=True)

    ff6_path = os.path.join(data_dir, "ff6_portfolios_daily.parquet")
    if not os.path.exists(ff6_path):
        print("  Downloading FF6 portfolio returns...", flush=True)
        db = get_wrds_connection(wrds_user)
        ff6 = db.raw_sql("""
            SELECT date, smlo_vwret, smme_vwret, smhi_vwret,
                   bilo_vwret, bime_vwret, bihi_vwret
            FROM ff.portfolios_d
            WHERE date BETWEEN '2000-01-01' AND '2026-12-31'
        """)
        db.close()
        ff6.to_parquet(ff6_path, index=False)
    else:
        ff6 = pd.read_parquet(ff6_path)

    ff6["date"] = pd.to_datetime(ff6["date"])
    port_cols = {
        "smlo": "smlo_vwret", "smme": "smme_vwret", "smhi": "smhi_vwret",
        "bilo": "bilo_vwret", "bime": "bime_vwret", "bihi": "bihi_vwret",
    }

    assign_path = os.path.join(data_dir, "ff6_assignments.parquet")
    if os.path.exists(assign_path):
        assignments = pd.read_parquet(assign_path)
        port_map = {(int(r["permno"]), int(r["year"])): r["portfolio"]
                    for _, r in assignments.iterrows()}
    else:
        print("  WARNING: ff6_assignments.parquet not found, using smme default", flush=True)
        port_map = {}

    db = get_wrds_connection(wrds_user)
    horizons = [32, 63, 126, 189, 252]
    all_cars = []

    permnos = merged_ar["permno"].unique().tolist()
    batch_size = 500
    for batch_start in range(0, len(permnos), batch_size):
        batch = permnos[batch_start:batch_start + batch_size]
        plist = ",".join(str(int(p)) for p in batch)
        crsp = db.raw_sql(f"""
            SELECT permno, date, ret
            FROM crsp.dsf
            WHERE permno IN ({plist})
              AND date BETWEEN '2000-01-01' AND '2027-06-30'
        """)
        crsp["date"] = pd.to_datetime(crsp["date"])
        crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
        crsp = crsp[~crsp["ret"].isin(CRSP_SENTINELS)]
        crsp = crsp.dropna(subset=["ret"])
        crsp = crsp.merge(ff6, on="date", how="left")
        crsp = crsp.sort_values(["permno", "date"]).reset_index(drop=True)

        grouped = crsp.groupby("permno")
        batch_calls = merged_ar[merged_ar["permno"].isin(set(batch))]

        for permno, calls in batch_calls.groupby("permno"):
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
                port_key = port_map.get((int(permno), assign_year))
                if port_key is None:
                    port_key = port_map.get((int(permno), assign_year - 1), "smme")
                port_col = port_cols.get(port_key, "smme_vwret")

                row = {"event_id": call["event_id"]}
                for h in horizons:
                    ei = min(si + h + 1, len(fc))
                    if ei <= si + 1:
                        row[f"car_{h}d"] = np.nan
                        continue
                    window = fc.iloc[si + 1:ei]
                    firm_ret = window["ret"].values.astype(np.float64)
                    bench_ret = pd.to_numeric(window[port_col], errors="coerce").values.astype(np.float64)
                    valid = ~(np.isnan(firm_ret) | np.isnan(bench_ret))
                    if valid.sum() < h * 0.5:
                        row[f"car_{h}d"] = np.nan
                        continue
                    row[f"car_{h}d"] = float(np.sum(firm_ret[valid] - bench_ret[valid]))
                all_cars.append(row)

        del crsp
        gc.collect()
        print(f"  CAR batch {batch_start // batch_size + 1}/{(len(permnos) - 1) // batch_size + 1}: "
              f"{len(all_cars):,}", flush=True)

    db.close()

    car_df = pd.DataFrame(all_cars)
    for h in horizons:
        col = f"car_{h}d"
        valid = car_df[col].dropna()
        car_df.loc[valid.index, col] = winsorize(valid)

    car_df.to_parquet(os.path.join(data_dir, "cars_v4.parquet"), index=False)
    print(f"  Total: {len(car_df):,} CARs", flush=True)
    return car_df


# ============================================================
# Step 4: Assemble final dataset
# ============================================================

def assemble(linked, ar_df, car_df, data_dir):
    print("\n=== Step 4: Assembling final dataset ===", flush=True)

    meta = linked.merge(ar_df, on=["event_id", "permno", "call_date"], how="inner")

    abs_ar = meta["abnormal_return"].abs()
    cutoff = abs_ar.quantile(1 / 3)
    meta["return_category"] = "F"
    meta.loc[meta["abnormal_return"] > cutoff, "return_category"] = "H"
    meta.loc[meta["abnormal_return"] < -cutoff, "return_category"] = "L"

    meta = meta.merge(car_df, on="event_id", how="left")
    meta["year_quarter"] = (
        meta["call_date"].dt.year.astype(str) + "Q" + meta["call_date"].dt.quarter.astype(str)
    )

    meta.to_parquet(os.path.join(data_dir, "merged_metadata_v4.parquet"), index=False)

    print(f"  Final: {len(meta):,} events", flush=True)
    print(f"  Unique firms: {meta['permno'].nunique():,}", flush=True)
    print(f"  Date range: {meta['call_date'].min()} to {meta['call_date'].max()}", flush=True)
    print(f"  Categories: {meta['return_category'].value_counts().to_dict()}", flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--wrds_user", default="yutongyancuhk")
    args = parser.parse_args()

    linked = link_to_crsp(args.data_dir, args.wrds_user)
    ar_df = compute_abnormal_returns(linked, args.data_dir, args.wrds_user)

    merged_ar = linked.merge(ar_df[["event_id", "event_date"]], on="event_id", how="inner")
    car_df = compute_cars(merged_ar, args.data_dir, args.wrds_user)

    assemble(linked, ar_df, car_df, args.data_dir)

    print("\n=== Stage 2 complete ===", flush=True)


if __name__ == "__main__":
    main()
