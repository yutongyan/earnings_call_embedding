#!/usr/bin/env python3
"""Process raw StreetEvents transcripts following docs/data_processing.md.

Steps:
1. Unzip year-level archives
2. Parse XML: eventTypeId=1, startDate (GMT->NYC), company_id, CUSIP
3. Dedup by (company_id, start_date): earliest lastUpdate
4. Link to CRSP: CUSIP-first, then ticker fallback
5. Drop outlier-length calls (>99.5th percentile)
6. Compute abnormal returns with timing heuristic (2 PM ET cutoff)
7. Compute FF6 benchmark-adjusted CARs (summed)
8. Winsorize continuous variables at 1%/99%

Usage:
    python3 -u src/process_data.py --raw_dir ~/large/streetevent_transcripts/streetevent_transcripts_zip
"""

import argparse
import gc
import glob
import os
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import wrds

TZ_UTC = ZoneInfo("UTC")
TZ_NYC = ZoneInfo("America/New_York")
CRSP_SENTINELS = {-66.0, -77.0, -88.0, -99.0}

# ============================================================
# Step 1: Unzip year-level archives
# ============================================================

def unzip_years(raw_dir, output_dir):
    """Unzip each year's zip file into output_dir/YYYY/."""
    print("=== Step 1: Unzipping year archives ===", flush=True)
    os.makedirs(output_dir, exist_ok=True)
    zips = sorted(glob.glob(os.path.join(raw_dir, "*.zip")))
    for zf in zips:
        year = os.path.basename(zf).replace(".zip", "")
        year_dir = os.path.join(output_dir, year)
        if os.path.isdir(year_dir) and len(os.listdir(year_dir)) > 100:
            print(f"  {year}: already unzipped ({len(os.listdir(year_dir))} files)", flush=True)
            continue
        os.makedirs(year_dir, exist_ok=True)
        print(f"  Unzipping {year}...", flush=True)
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(year_dir)
        print(f"  {year}: {len(os.listdir(year_dir))} files", flush=True)


# ============================================================
# Step 2: Parse XML transcripts
# ============================================================

def parse_single_xml(filepath):
    """Parse one StreetEvents XML file."""
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(filepath)
        root = tree.getroot()

        if root.get("eventTypeId", "") != "1":
            return None

        story = root.find("EventStory")
        if story is None:
            return None

        body_el = story.find("Body")
        if body_el is None:
            return None
        body = " ".join(body_el.itertext())
        if not body.strip():
            return None

        event_id = root.get("Id", "")
        story_version = story.get("version", "")
        last_update = root.get("lastUpdate", "")

        # startDate -> call_date (GMT to NYC)
        sd_el = root.find("startDate")
        sd_str = sd_el.text.strip() if sd_el is not None and sd_el.text else ""

        call_date = None
        call_hour_et = None
        if sd_str:
            try:
                clean = re.sub(r"\s*(GMT|EST|EDT|ET)\s*$", "", sd_str).strip()
                dt = pd.to_datetime(clean, format="mixed")
                if "EST" in sd_str:
                    dt_nyc = dt.replace(tzinfo=timezone(timedelta(hours=-5)))
                elif "EDT" in sd_str:
                    dt_nyc = dt.replace(tzinfo=timezone(timedelta(hours=-4)))
                else:
                    dt_nyc = dt.replace(tzinfo=TZ_UTC).astimezone(TZ_NYC)
                call_date = pd.Timestamp(dt_nyc.replace(tzinfo=None)).normalize()
                call_hour_et = dt_nyc.hour
            except Exception:
                pass

        # Headline fallback for date
        headline_el = story.find("Headline")
        headline = headline_el.text.strip() if headline_el is not None and headline_el.text else ""

        if call_date is None:
            dm = re.search(r"(\d{1,2}-\w{3}-\d{2,4})\s+\d{1,2}:\d{2}", headline)
            if dm:
                try:
                    call_date = pd.to_datetime(dm.group(1))
                except Exception:
                    pass

        # Company identifiers
        company_id = _xml_text(root, "companyId")
        company_name = _xml_text(root, "companyName")
        ticker = _xml_text(root, "companyTicker")
        cusip = _xml_text(root, "CUSIP")
        isin = _xml_text(root, "ISIN")

        # Section split
        sections = _split_sections(body)
        if not sections["presentation"] and not sections["qa"]:
            return None

        is_before_close = None if call_hour_et is None else (call_hour_et < 14)

        return {
            "event_id": event_id,
            "company_id": company_id,
            "call_date": call_date,
            "start_date_str": sd_str,
            "last_update": last_update,
            "ticker": ticker,
            "cusip": cusip,
            "isin": isin,
            "company_name": company_name,
            "presentation_text": sections["presentation"],
            "qa_text": sections["qa"],
            "n_words_pres": len(sections["presentation"].split()),
            "n_words_qa": len(sections["qa"].split()),
            "is_before_close": is_before_close,
            "story_version": story_version,
        }
    except Exception:
        return None


def _xml_text(root, tag):
    el = root.find(tag)
    return el.text.strip() if el is not None and el.text else ""


def _split_sections(body):
    pres_pat = re.compile(r"={5,}\s*\n\s*Presentation\s*\n\s*-{5,}", re.IGNORECASE)
    qa_pat = re.compile(r"={5,}\s*\n\s*Questions\s+and\s+Answers\s*\n\s*-{5,}", re.IGNORECASE)
    pm = pres_pat.search(body)
    qm = qa_pat.search(body)
    pres, qa = "", ""
    if pm and qm:
        pres = body[pm.end():qm.start()]
        qa = body[qm.end():]
    elif pm:
        pres = body[pm.end():]
    elif qm:
        pres = body[:qm.start()]
        qa = body[qm.end():]
    else:
        pres = body
    pres = _clean_text(pres)
    qa = _clean_text(qa)
    return {"presentation": pres, "qa": qa}


def _clean_text(text):
    text = re.sub(r"-{5,}", "", text)
    text = re.sub(r"={5,}", "", text)
    text = re.sub(r"\[(\d+)\]", "", text)
    lines = [l.strip() for l in text.split("\n") if l.strip() and not re.match(r"^(Operator|OPERATOR)\s*$", l.strip())]
    return " ".join(lines)


def parse_year(xml_dir, year):
    """Parse all XML files for one year, dedup by (company_id, start_date_str)."""
    year_dir = os.path.join(xml_dir, str(year))
    if not os.path.isdir(year_dir):
        return []
    files = [os.path.join(year_dir, f) for f in os.listdir(year_dir) if f.endswith("_T.xml")]
    results = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for r in ex.map(parse_single_xml, files):
            if r is not None:
                results.append(r)

    # Dedup: (company_id, start_date_str), keep earliest lastUpdate
    grouped = {}
    for r in results:
        if not r["event_id"] or not r["company_id"]:
            continue
        key = (r["company_id"], r["start_date_str"])
        if key not in grouped:
            grouped[key] = r
        else:
            existing = grouped[key]
            r_lu = r["last_update"] or ""
            e_lu = existing["last_update"] or ""
            if r_lu and e_lu and r_lu < e_lu:
                grouped[key] = r
            elif r_lu == e_lu and r["event_id"] < existing["event_id"]:
                grouped[key] = r
            elif not e_lu and r_lu:
                grouped[key] = r
    return list(grouped.values())


def parse_all_years(xml_dir, data_dir, start_year=2008, end_year=2019):
    """Parse all years, save per-year parquets."""
    print("\n=== Step 2: Parsing transcripts ===", flush=True)
    for year in range(start_year, end_year + 1):
        out_path = os.path.join(data_dir, f"transcripts_{year}.parquet")
        if os.path.exists(out_path):
            n = len(pd.read_parquet(out_path, columns=["event_id"]))
            print(f"  {year}: already parsed ({n:,} calls)", flush=True)
            continue
        results = parse_year(xml_dir, year)
        pd.DataFrame(results).to_parquet(out_path, index=False)
        print(f"  {year}: {len(results):,} earnings calls", flush=True)


# ============================================================
# Step 3: Link to CRSP (CUSIP-first, then ticker)
# ============================================================

def link_to_crsp(data_dir, wrds_user):
    """Link transcripts to CRSP via CUSIP then ticker."""
    print("\n=== Step 3: Linking to CRSP ===", flush=True)

    # Load transcript metadata
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
    print(f"  Transcripts: {len(transcripts):,}")

    db = wrds.Connection(wrds_username=wrds_user)

    msenames = db.raw_sql("""
        SELECT permno, namedt, nameendt, ncusip, ticker, shrcd, exchcd
        FROM crsp.msenames
        WHERE shrcd IN (10, 11) AND exchcd IN (1, 2, 3)
    """)
    msenames["namedt"] = pd.to_datetime(msenames["namedt"])
    msenames["nameendt"] = pd.to_datetime(msenames["nameendt"])
    msenames["ticker"] = msenames["ticker"].str.upper().str.strip()
    msenames["ncusip"] = msenames["ncusip"].str.strip()
    db.close()

    # Step 3a: CUSIP match
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
    print(f"  CUSIP matched: {len(matched_cusip):,}")

    # Step 3b: Ticker fallback for unmatched
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
    print(f"  Ticker matched: {len(matched_ticker):,}")

    # Combine
    keep_cols = [
        "event_id", "company_id", "call_date", "start_date_str", "last_update",
        "ticker", "cusip", "company_name", "permno",
        "n_words_pres", "n_words_qa", "is_before_close", "story_version",
    ]
    matched_cusip["ticker_clean"] = matched_cusip["ticker"]
    for c in keep_cols:
        if c not in matched_cusip.columns:
            matched_cusip[c] = ""
        if c not in matched_ticker.columns:
            matched_ticker[c] = ""

    linked = pd.concat([matched_cusip[keep_cols], matched_ticker[keep_cols]], ignore_index=True)
    linked["permno"] = linked["permno"].astype("int64")
    linked = linked.drop_duplicates(subset=["event_id"], keep="first")
    print(f"  Total linked: {len(linked):,}")

    # Drop outlier-length calls (>99.5th percentile)
    total_words = linked["n_words_pres"] + linked["n_words_qa"]
    cutoff = total_words.quantile(0.995)
    before = len(linked)
    linked = linked[total_words <= cutoff]
    print(f"  After length filter (>{cutoff:.0f} words): {len(linked):,} (dropped {before - len(linked)})")

    linked.to_parquet(os.path.join(data_dir, "linked_transcripts_v4.parquet"), index=False)
    print(f"  Saved linked_transcripts_v4.parquet")
    return linked


# ============================================================
# Step 4: Compute abnormal returns with timing
# ============================================================

def compute_abnormal_returns(linked, data_dir, wrds_user):
    """Compute one-day AR using FF3+momentum with timing heuristic."""
    print("\n=== Step 4: Computing abnormal returns ===", flush=True)

    db = wrds.Connection(wrds_username=wrds_user)

    permnos = linked["permno"].unique().tolist()
    print(f"  {len(permnos):,} unique permnos")

    ff = db.raw_sql("""
        SELECT date, mktrf, smb, hml, rf, umd
        FROM ff.factors_daily
        WHERE date BETWEEN '2007-01-01' AND '2020-12-31'
    """)
    ff["date"] = pd.to_datetime(ff["date"])

    all_results = []
    batch_size = 500
    for batch_start in range(0, len(permnos), batch_size):
        batch = permnos[batch_start:batch_start + batch_size]
        plist = ",".join(str(int(p)) for p in batch)
        crsp = db.raw_sql(f"""
            SELECT permno, date, ret
            FROM crsp.dsf
            WHERE permno IN ({plist})
              AND date BETWEEN '2007-01-01' AND '2020-12-31'
        """)
        crsp["date"] = pd.to_datetime(crsp["date"])
        crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
        crsp = crsp[~crsp["ret"].isin(CRSP_SENTINELS)]
        crsp = crsp.dropna(subset=["ret"])
        for col in ["mktrf", "smb", "hml", "umd", "rf"]:
            ff[col] = pd.to_numeric(ff[col], errors="coerce").astype("float64")
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

                # Timing: if call after 2 PM ET, use next day's return
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
                est = fc.iloc[est_start:est_end].dropna(subset=["exret", "mktrf", "smb", "hml", "umd"])
                if len(est) < 60:
                    continue

                X = est[["mktrf", "smb", "hml", "umd"]].values.astype(np.float64)
                y = est["exret"].values.astype(np.float64)
                beta = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]
                ev = fc.iloc[idx]
                if pd.isna(ev["exret"]):
                    continue
                ar = float(ev["exret"]) - float(np.dot(beta, [1, ev["mktrf"], ev["smb"], ev["hml"], ev["umd"]]))

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
        print(f"  Batch {batch_start // batch_size + 1}: {len(all_results):,} ARs", flush=True)

    db.close()

    ar_df = pd.DataFrame(all_results)
    ar_df["abnormal_return"] = _winsorize(ar_df["abnormal_return"])
    ar_df.to_parquet(os.path.join(data_dir, "abnormal_returns_v4.parquet"), index=False)
    print(f"  AR computed: {len(ar_df):,}")
    return ar_df


# ============================================================
# Step 5: Compute FF6 benchmark-adjusted CARs (summed)
# ============================================================

def compute_cars(merged_ar, data_dir, wrds_user):
    """Compute CARs using FF6 size/BM portfolios, summed not compounded."""
    print("\n=== Step 5: Computing FF6 CARs ===", flush=True)

    ff6 = pd.read_parquet(os.path.join(data_dir, "ff6_portfolios_daily.parquet"))
    ff6["date"] = pd.to_datetime(ff6["date"])
    port_cols = {
        "smlo": "smlo_vwret", "smme": "smme_vwret", "smhi": "smhi_vwret",
        "bilo": "bilo_vwret", "bime": "bime_vwret", "bihi": "bihi_vwret",
    }

    assignments = pd.read_parquet(os.path.join(data_dir, "ff6_assignments.parquet"))
    port_map = {(int(r["permno"]), int(r["year"])): r["portfolio"] for _, r in assignments.iterrows()}

    db = wrds.Connection(wrds_username=wrds_user)
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
              AND date BETWEEN '2007-01-01' AND '2021-06-30'
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
        print(f"  CAR batch {batch_start // batch_size + 1}: {len(all_cars):,}", flush=True)

    db.close()

    car_df = pd.DataFrame(all_cars)
    for h in horizons:
        col = f"car_{h}d"
        valid = car_df[col].dropna()
        car_df.loc[valid.index, col] = _winsorize(valid)

    car_df.to_parquet(os.path.join(data_dir, "cars_v4.parquet"), index=False)
    print(f"  CARs computed: {len(car_df):,}")
    return car_df


def _winsorize(s, lower=0.01, upper=0.99):
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


# ============================================================
# Step 6: Assemble final dataset
# ============================================================

def assemble_final(linked, ar_df, car_df, data_dir):
    """Merge everything into the final dataset."""
    print("\n=== Step 6: Assembling final dataset ===", flush=True)

    meta = linked.merge(ar_df, on=["event_id", "permno", "call_date"], how="inner")

    abs_ar = meta["abnormal_return"].abs()
    cutoff = abs_ar.quantile(1 / 3)
    meta["return_category"] = "F"
    meta.loc[meta["abnormal_return"] > cutoff, "return_category"] = "H"
    meta.loc[meta["abnormal_return"] < -cutoff, "return_category"] = "L"

    meta = meta.merge(car_df, on="event_id", how="left")
    meta["year_quarter"] = meta["call_date"].dt.year.astype(str) + "Q" + meta["call_date"].dt.quarter.astype(str)

    meta.to_parquet(os.path.join(data_dir, "merged_metadata_v4.parquet"), index=False)

    print(f"  Final: {len(meta):,} events")
    print(f"  Unique firms: {meta['permno'].nunique():,}")
    print(f"  Date range: {meta['call_date'].min()} to {meta['call_date'].max()}")
    print(f"  Categories: {meta['return_category'].value_counts().to_dict()}")
    print(f"  Saved merged_metadata_v4.parquet")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=os.path.expanduser("~/large/streetevent_transcripts/streetevent_transcripts_zip"))
    parser.add_argument("--xml_dir", default=os.path.expanduser("~/large/streetevent_transcripts/xml"))
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--wrds_user", default="yutongyancuhk")
    parser.add_argument("--start_year", type=int, default=2008)
    parser.add_argument("--end_year", type=int, default=2019)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    unzip_years(args.raw_dir, args.xml_dir)
    parse_all_years(args.xml_dir, args.data_dir, args.start_year, args.end_year)
    linked = link_to_crsp(args.data_dir, args.wrds_user)
    ar_df = compute_abnormal_returns(linked, args.data_dir, args.wrds_user)
    car_df = compute_cars(
        linked.merge(ar_df[["event_id", "event_date"]], on="event_id", how="inner"),
        args.data_dir, args.wrds_user,
    )
    assemble_final(linked, ar_df, car_df, args.data_dir)

    print("\n=== Processing complete ===", flush=True)


if __name__ == "__main__":
    main()
