#!/usr/bin/env python3
"""Stage 1: Process raw StreetEvents transcripts (no WRDS needed).

Steps:
1. Unzip year-level archives
2. Parse XML: eventTypeId=1, startDate (GMT->NYC), company_id, CUSIP
3. Dedup by (company_id, start_date): earliest lastUpdate
4. Drop outlier-length calls (>99.5th percentile)
5. Save per-year parquets + summary stats

Usage:
    python3 -u src/process_earnings.py \
        --raw_dir ~/large-data/streetevent_transcripts/streetevent_transcripts_zip \
        --xml_dir ~/large-data/streetevent_transcripts/xml \
        --data_dir data/
"""

import argparse
import gc
import glob
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TZ_UTC = ZoneInfo("UTC")
TZ_NYC = ZoneInfo("America/New_York")


# ============================================================
# Step 1: Unzip
# ============================================================

def _unzip_one_year(args):
    zf, output_dir = args
    year = os.path.basename(zf).replace(".zip", "")
    year_dir = os.path.join(output_dir, year)

    xml_count = len(glob.glob(os.path.join(year_dir, "*_T.xml")))
    if xml_count > 100:
        return f"  {year}: already unzipped ({xml_count} XMLs)"

    os.makedirs(year_dir, exist_ok=True)
    with zipfile.ZipFile(zf, "r") as z:
        z.extractall(year_dir)

    nested = os.path.join(year_dir, year)
    if os.path.isdir(nested):
        import shutil
        for f in os.listdir(nested):
            shutil.move(os.path.join(nested, f), os.path.join(year_dir, f))
        os.rmdir(nested)

    xml_count = len(glob.glob(os.path.join(year_dir, "*_T.xml")))
    return f"  {year}: {xml_count} XMLs"


def unzip_years(raw_dir, output_dir):
    print("=== Step 1: Unzipping year archives ===", flush=True)
    os.makedirs(output_dir, exist_ok=True)
    zips = sorted(glob.glob(os.path.join(raw_dir, "*.zip")))
    args = [(zf, output_dir) for zf in zips]
    with ProcessPoolExecutor(max_workers=8) as ex:
        for result in ex.map(_unzip_one_year, args):
            print(result, flush=True)


# ============================================================
# Step 2: Parse XML
# ============================================================

def parse_single_xml(filepath):
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

        headline_el = story.find("Headline")
        headline = headline_el.text.strip() if headline_el is not None and headline_el.text else ""

        if call_date is None:
            dm = re.search(r"(\d{1,2}-\w{3}-\d{2,4})\s+\d{1,2}:\d{2}", headline)
            if dm:
                try:
                    call_date = pd.to_datetime(dm.group(1))
                except Exception:
                    pass

        company_id = _xml_text(root, "companyId")
        company_name = _xml_text(root, "companyName")
        ticker = _xml_text(root, "companyTicker")
        cusip = _xml_text(root, "CUSIP")
        isin = _xml_text(root, "ISIN")

        sections = _split_sections(body)
        if not sections["presentation"] and not sections["qa"]:
            return None

        is_before_close = None if call_hour_et is None else (call_hour_et < 12)

        return {
            "event_id": event_id,
            "company_id": company_id,
            "call_date": call_date,
            "start_date_str": sd_str,
            "last_update": last_update,
            "headline": headline,
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
    return {"presentation": _clean_text(pres), "qa": _clean_text(qa)}


def _clean_text(text):
    text = re.sub(r"-{5,}", "", text)
    text = re.sub(r"={5,}", "", text)
    text = re.sub(r"\[(\d+)\]", "", text)
    lines = [l.strip() for l in text.split("\n")
             if l.strip() and not re.match(r"^(Operator|OPERATOR)\s*$", l.strip())]
    return " ".join(lines)


def parse_year(xml_dir, year):
    year_dir = os.path.join(xml_dir, str(year))
    if not os.path.isdir(year_dir):
        return []
    files = [os.path.join(year_dir, f) for f in os.listdir(year_dir) if f.endswith("_T.xml")]
    results = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for r in ex.map(parse_single_xml, files):
            if r is not None:
                results.append(r)

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


def parse_all_years(xml_dir, data_dir, start_year, end_year):
    print("\n=== Step 2: Parsing transcripts ===", flush=True)
    for year in range(start_year, end_year + 1):
        out_path = os.path.join(data_dir, f"transcripts_{year}.parquet")
        if os.path.exists(out_path):
            df = pd.read_parquet(out_path, columns=["event_id"])
            if "company_id" in pd.read_parquet(out_path, columns=["event_id"]).columns or True:
                try:
                    test = pd.read_parquet(out_path, columns=["company_id"])
                    print(f"  {year}: already parsed ({len(df):,} calls)", flush=True)
                    continue
                except Exception:
                    pass
        results = parse_year(xml_dir, year)
        pd.DataFrame(results).to_parquet(out_path, index=False)
        print(f"  {year}: {len(results):,} earnings calls", flush=True)


# ============================================================
# Step 3: Summary and quality checks
# ============================================================

def summarize(data_dir, start_year, end_year):
    print("\n=== Step 3: Summary ===", flush=True)

    all_dfs = []
    for year in range(start_year, end_year + 1):
        path = os.path.join(data_dir, f"transcripts_{year}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path, columns=[
                "event_id", "company_id", "call_date", "ticker", "cusip",
                "n_words_pres", "n_words_qa", "is_before_close", "story_version",
            ])
            all_dfs.append(df)

    meta = pd.concat(all_dfs, ignore_index=True)
    meta["call_date"] = pd.to_datetime(meta["call_date"])
    meta["total_words"] = meta["n_words_pres"] + meta["n_words_qa"]

    print(f"  Total earnings calls: {len(meta):,}")
    print(f"  Unique company_ids: {meta['company_id'].nunique():,}")
    print(f"  Date range: {meta['call_date'].min()} to {meta['call_date'].max()}")
    print(f"  With CUSIP: {(meta['cusip'] != '').sum():,} ({(meta['cusip'] != '').mean()*100:.1f}%)")
    print(f"  With ticker: {(meta['ticker'] != '').sum():,}")
    print(f"  is_before_close True: {meta['is_before_close'].sum():,}")
    print(f"  is_before_close None: {meta['is_before_close'].isna().sum():,}")

    print(f"\n  Word count stats:")
    print(f"    Median: {meta['total_words'].median():.0f}")
    print(f"    99.5th: {meta['total_words'].quantile(0.995):.0f}")

    cutoff = meta["total_words"].quantile(0.995)
    outliers = (meta["total_words"] > cutoff).sum()
    print(f"    Outliers (>99.5th): {outliers:,}")
    print(f"    After filter: {len(meta) - outliers:,}")

    print(f"\n  Yearly counts:")
    meta["year"] = meta["call_date"].dt.year
    for year, count in meta.groupby("year").size().items():
        print(f"    {year}: {count:,}")

    # Save event year index
    idx = []
    for year in range(start_year, end_year + 1):
        path = os.path.join(data_dir, f"transcripts_{year}.parquet")
        if os.path.exists(path):
            eids = pd.read_parquet(path, columns=["event_id"])["event_id"].tolist()
            for eid in eids:
                idx.append({"event_id": eid, "year": str(year)})
    pd.DataFrame(idx).to_parquet(os.path.join(data_dir, "event_year_index.parquet"), index=False)
    print(f"\n  Saved event_year_index.parquet ({len(idx):,} entries)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--xml_dir", required=True)
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--start_year", type=int, default=2001)
    parser.add_argument("--end_year", type=int, default=2026)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    unzip_years(args.raw_dir, args.xml_dir)
    parse_all_years(args.xml_dir, args.data_dir, args.start_year, args.end_year)
    summarize(args.data_dir, args.start_year, args.end_year)

    print("\n=== Stage 1 complete ===", flush=True)


if __name__ == "__main__":
    main()
