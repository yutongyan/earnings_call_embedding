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

import pandas as pd


def parse_single_xml(filepath):
    """Parse one StreetEvents XML file, return dict or None."""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        event = root
        event_id = event.get("Id", "")
        event_type = event.get("eventTypeName", "")

        if "earning" not in event_type.lower():
            return None

        story = event.find("EventStory")
        if story is None:
            return None

        headline = ""
        headline_el = story.find("Headline")
        if headline_el is not None and headline_el.text:
            headline = headline_el.text.strip()

        body_el = story.find("Body")
        if body_el is None or not body_el.text:
            return None
        body = body_el.text

        last_update = event.get("lastUpdate", "")
        story_date_str = story.get("expirationDate", "")

        call_date = None
        date_match = re.search(
            r"(\d{1,2}-\w{3}-\d{2,4})\s+\d{1,2}:\d{2}", headline
        )
        if date_match:
            try:
                call_date = pd.to_datetime(date_match.group(1))
            except Exception:
                pass

        if call_date is None:
            date_match2 = re.search(
                r"(\w+ \d{1,2},?\s+\d{4})", headline
            )
            if date_match2:
                try:
                    call_date = pd.to_datetime(date_match2.group(1))
                except Exception:
                    pass

        if call_date is None and last_update:
            try:
                clean = re.sub(
                    r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s*",
                    "", last_update
                )
                call_date = pd.to_datetime(clean, format="mixed")
            except Exception:
                pass

        ticker_match = re.search(r"of\s+(\S+)\s+earnings", headline, re.IGNORECASE)
        ticker = ticker_match.group(1) if ticker_match else ""

        company_match = re.search(
            r"Transcript of\s+(.+?)\s+earnings", headline, re.IGNORECASE
        )
        company_name = company_match.group(1) if company_match else ""

        sections = split_presentation_qa(body)

        if not sections["presentation"] and not sections["qa"]:
            return None

        return {
            "event_id": event_id,
            "call_date": call_date,
            "headline": headline,
            "ticker": ticker,
            "company_name": company_name,
            "presentation_text": sections["presentation"],
            "qa_text": sections["qa"],
            "n_words_pres": len(sections["presentation"].split()),
            "n_words_qa": len(sections["qa"].split()),
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

    return results


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
