#!/usr/bin/env python3
"""Parse StreetEvents XML transcripts into a structured DataFrame.

Extracts presentation text, Q&A text, company metadata, and call dates
from the XML files in the homo_silicus_ceo archive.

Usage:
    python3 src/parse_transcripts.py \
        --input /scratch/homo_silicus_ceo/data/streetevent_transcripts/archive \
        --output data/transcripts_parsed.parquet \
        --start_year 2008 --end_year 2019
"""

import argparse
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

TZ_UTC = ZoneInfo("UTC")
TZ_NYC = ZoneInfo("America/New_York")


def parse_single_xml(filepath):
    """Parse one StreetEvents XML file, return dict or None."""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        event = root
        event_id = event.get("Id", "")
        event_type_id = event.get("eventTypeId", "")

        if event_type_id != "1":
            return None

        story = event.find("EventStory")
        if story is None:
            return None

        headline = ""
        headline_el = story.find("Headline")
        if headline_el is not None and headline_el.text:
            headline = headline_el.text.strip()

        body_el = story.find("Body")
        if body_el is None:
            return None
        body = " ".join(body_el.itertext())
        if not body.strip():
            return None

        story_version = story.get("version", "")

        start_date_el = event.find("startDate")
        start_date_str = start_date_el.text.strip() if start_date_el is not None and start_date_el.text else ""

        call_date = None
        call_hour_et = None
        if start_date_str:
            try:
                clean_str = re.sub(r"\s*(GMT|EST|EDT|ET)\s*$", "", start_date_str).strip()
                call_dt = pd.to_datetime(clean_str, format="mixed")

                if "EST" in start_date_str:
                    from datetime import timezone, timedelta
                    call_dt_nyc = call_dt.replace(tzinfo=timezone(timedelta(hours=-5)))
                elif "EDT" in start_date_str:
                    from datetime import timezone, timedelta
                    call_dt_nyc = call_dt.replace(tzinfo=timezone(timedelta(hours=-4)))
                else:
                    call_dt_utc = call_dt.replace(tzinfo=TZ_UTC)
                    call_dt_nyc = call_dt_utc.astimezone(TZ_NYC)

                call_date = call_dt_nyc.replace(tzinfo=None)
                call_date = pd.Timestamp(call_date).normalize()
                call_hour_et = call_dt_nyc.hour
            except Exception:
                pass

        if call_date is None:
            date_match = re.search(
                r"(\d{1,2}-\w{3}-\d{2,4})\s+\d{1,2}:\d{2}", headline
            )
            if date_match:
                try:
                    call_date = pd.to_datetime(date_match.group(1))
                except Exception:
                    pass

        if call_date is None:
            date_match2 = re.search(r"(\w+ \d{1,2},?\s+\d{4})", headline)
            if date_match2:
                try:
                    call_date = pd.to_datetime(date_match2.group(1))
                except Exception:
                    pass

        company_el = event.find("companyName")
        company_name = company_el.text.strip() if company_el is not None and company_el.text else ""

        ticker_el = event.find("companyTicker")
        ticker = ticker_el.text.strip() if ticker_el is not None and ticker_el.text else ""

        if not ticker:
            ticker_match = re.search(r"of\s+(\S+)\s+earnings", headline, re.IGNORECASE)
            ticker = ticker_match.group(1) if ticker_match else ""

        if not company_name:
            company_match = re.search(r"Transcript of\s+(.+?)\s+earnings", headline, re.IGNORECASE)
            company_name = company_match.group(1) if company_match else ""

        sections = split_presentation_qa(body)

        if not sections["presentation"] and not sections["qa"]:
            return None

        is_before_close = None if call_hour_et is None else (call_hour_et < 16)
        is_preliminary = "preliminary" in headline.lower() or story_version != "Final"

        return {
            "event_id": event_id,
            "call_date": call_date,
            "start_date_str": start_date_str,
            "headline": headline,
            "ticker": ticker,
            "company_name": company_name,
            "presentation_text": sections["presentation"],
            "qa_text": sections["qa"],
            "n_words_pres": len(sections["presentation"].split()),
            "n_words_qa": len(sections["qa"].split()),
            "is_preliminary": is_preliminary,
            "is_before_close": is_before_close,
            "story_version": story_version,
            "filepath": filepath,
        }
    except Exception:
        return None


def split_presentation_qa(body):
    """Split transcript body into presentation and Q&A sections."""
    pres_pattern = re.compile(
        r"={5,}\s*\n\s*Presentation\s*\n\s*-{5,}",
        re.IGNORECASE,
    )
    qa_pattern = re.compile(
        r"={5,}\s*\n\s*Questions\s+and\s+Answers\s*\n\s*-{5,}",
        re.IGNORECASE,
    )

    pres_match = pres_pattern.search(body)
    qa_match = qa_pattern.search(body)

    presentation = ""
    qa = ""

    if pres_match and qa_match:
        pres_start = pres_match.end()
        qa_start = qa_match.end()
        presentation = body[pres_start:qa_match.start()]
        qa = body[qa_start:]
    elif pres_match and not qa_match:
        presentation = body[pres_match.end():]
    elif qa_match and not pres_match:
        presentation = body[:qa_match.start()]
        qa = body[qa_match.end():]
    else:
        presentation = body

    presentation = clean_section_text(presentation)
    qa = clean_section_text(qa)

    return {"presentation": presentation, "qa": qa}


def clean_section_text(text):
    """Remove speaker headers and formatting markers from section text."""
    text = re.sub(r"-{5,}", "", text)
    text = re.sub(r"={5,}", "", text)
    text = re.sub(r"\[(\d+)\]", "", text)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(Operator|OPERATOR)\s*$", line):
            continue
        cleaned.append(line)
    return " ".join(cleaned)


def parse_year(archive_dir, year):
    """Parse all XML files for a given year."""
    year_dir = os.path.join(archive_dir, str(year))
    if not os.path.isdir(year_dir):
        return []

    files = [
        os.path.join(year_dir, f)
        for f in os.listdir(year_dir)
        if f.endswith("_T.xml")
    ]

    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(parse_single_xml, f): f for f in files}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    seen = {}
    for r in results:
        eid = r["event_id"]
        if not eid:
            continue
        if eid not in seen:
            seen[eid] = r
        else:
            existing = seen[eid]
            r_prelim = r["story_version"] != "Final"
            e_prelim = existing["story_version"] != "Final"
            if r_prelim and not e_prelim:
                seen[eid] = r
            elif r_prelim == e_prelim and r["n_words_pres"] > existing["n_words_pres"]:
                seen[eid] = r

    return list(seen.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/scratch/homo_silicus_ceo/data/streetevent_transcripts/archive",
    )
    parser.add_argument("--output", default="data/transcripts_parsed.parquet")
    parser.add_argument("--start_year", type=int, default=2008)
    parser.add_argument("--end_year", type=int, default=2019)
    args = parser.parse_args()

    all_results = []
    for year in range(args.start_year, args.end_year + 1):
        print(f"Parsing {year}...", flush=True)
        year_results = parse_year(args.input, year)
        print(f"  {year}: {len(year_results)} earnings calls", flush=True)
        all_results.extend(year_results)

    df = pd.DataFrame(all_results)
    print(f"\nTotal parsed: {len(df)} earnings calls")
    print(f"Date range: {df['call_date'].min()} to {df['call_date'].max()}")
    print(f"Median presentation words: {df['n_words_pres'].median():.0f}")
    print(f"Median Q&A words: {df['n_words_qa'].median():.0f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
