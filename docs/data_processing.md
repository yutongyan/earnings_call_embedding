# Earnings Call Transcript Data Processing Guide

This document describes how to properly process StreetEvents earnings call transcripts for academic research, based on lessons learned during the PEAD.txt replication.

## 1. Data Source

StreetEvents transcripts from Refinitiv, accessed via SFTP or Dropbox archive. Each transcript is an XML file named `{event_id}_T.xml`. The raw archive is distributed as year-level zip files (2001.zip through 2026.zip), which are unzipped into `xml/YYYY/` directories for processing.

**WRDS access**: Database queries use credentials stored in `~/.pgpass` (PostgreSQL password file). Once configured, connections are automatic with no Duo push required. Initial setup: copy `.pgpass` to the compute server and `chmod 600`.

## 2. Key XML Fields

```xml
<Event Id="16837828" eventTypeId="1" eventTypeName="Earning Conference Call/Presentation">
  <EventStory version="Final">
    <Headline>...</Headline>
    <Body><![CDATA[...]]></Body>
  </EventStory>
  <eventTitle>Q3 2026 Rogers Sugar Inc Earnings Call</eventTitle>
  <companyName>Rogers Sugar Inc</companyName>
  <companyTicker>RSI.TO</companyTicker>
  <companyId>99000</companyId>
  <startDate>6-Aug-26 12:00pm GMT</startDate>
  <CUSIP>77519R102</CUSIP>
  <ISIN>CA77519R1029</ISIN>
</Event>
```

## 3. Filtering to Earnings Calls

Filter on `eventTypeId="1"` (or `eventTypeName="Earning Conference Call/Presentation"`). This excludes:

| eventTypeId | eventTypeName | Action |
|-------------|--------------|--------|
| 1 | Earning Conference Call/Presentation | **Keep** |
| 5 | Corporate Conference Call/Presentation | Exclude |
| 7 | Conference Presentation | Exclude |
| 8 | Other Corporate Conference Event | Exclude |
| 11 | Shareholder Meeting | Exclude |
| 24 | Syndicated Roadshow | Exclude |
| 30 | Guidance Conference Call/Presentation | Exclude |
| 31 | Analyst Meeting | Exclude |
| 33 | Sales Conference Call/Presentation | Exclude |
| 34 | M&A Conference Call/Presentation | Exclude |

## 4. Call Date and Timing

Use the `<startDate>` element (not `lastUpdate`, not the headline date). Format: `DD-Mon-YY H:MMam/pm GMT`.

**GMT to NYC conversion** (for determining market hours):
- Convert GMT to America/New_York, accounting for DST (UTC-5 in winter, UTC-4 in summer)
- If `startDate` says EST or EDT, use fixed offsets (EST = UTC-5, EDT = UTC-4) to respect the explicit label
- The NYC-local date becomes the `call_date` (important for calls after midnight GMT that are still same day in NYC)

**Market hours determination:**

Earnings calls typically last ~50 minutes (median), with 99th percentile at ~92 minutes. We assume a maximum call duration of 2 hours. The event-day return should capture the market's reaction to the call content.

- `is_before_close`: whether the call **ends** before market close (4 PM ET)
- Since we only have `<startDate>` (no end time), we estimate: `is_before_close = call_hour_et < 14` (call starts before 2 PM ET, so it ends before 4 PM ET with the 2-hour buffer)
- Calls starting between 2-4 PM ET are ambiguous (may span close). Conservatively treat as after-close since the Q&A portion, which contains the most market-moving content, occurs in the second half of the call.
- If call time is unknown, set `is_before_close = None` (not False)

This determines the event-day return window:
- Before close (starts before 2 PM ET): use return from t-1 to t (market reacts same day)
- After close (starts 2 PM ET or later): use return from t to t+1 (market reacts next day)

Note: the original paper (Meursault et al. 2022) uses the Capital IQ transcript **upload timestamp** with a 3 PM ET cutoff. We use `<startDate>` (actual call time) from StreetEvents with a 2 PM ET cutoff to account for call duration. `lastUpdate` in StreetEvents is unreliable (median lag of 2 days from call date, reflecting file edits, not first upload).

## 5. Text Extraction

**Body text:** Use `" ".join(body_el.itertext())` to handle any nested XML elements. Join with space to prevent word concatenation.

**Section splitting:** The transcript body has standard markers:
```
================================================================================
Presentation
--------------------------------------------------------------------------------
[presentation text]
================================================================================
Questions and Answers
--------------------------------------------------------------------------------
[Q&A text]
```

Split on these markers using regex. If markers are missing, treat entire body as presentation.

**Text cleaning:**
- Remove separator lines (`-----`, `=====`)
- Remove speaker turn markers (`[1]`, `[2]`, etc.)
- Remove "Operator" standalone lines
- Join remaining lines with spaces

## 6. Company Identification and CRSP Linking

Extract identifiers from XML elements (not headline parsing):
- `<CUSIP>` (8-digit CUSIP, e.g., `77519R102`)
- `<ISIN>` (12-character, e.g., `CA77519R1029`; extract 8-digit CUSIP as chars 3-10 for US/CA ISINs)
- `<companyTicker>` (e.g., `RSI.TO`)
- `<companyName>`
- `<companyId>` (StreetEvents internal ID)

**Linking to CRSP (two-step, CUSIP first):**

1. **CUSIP match (primary):** Match `<CUSIP>` (first 8 digits) to CRSP `msenames.ncusip`. This is the most reliable link since CUSIPs are security-level identifiers that persist through ticker changes. Filter to `shrcd IN (10,11)` and `exchcd IN (1,2,3)` for US common stocks. Validate that `call_date` falls within `namedt` to `nameendt`.

2. **Ticker match (fallback):** For events without a CUSIP match, clean the `<companyTicker>` by stripping the exchange suffix (`.OQ`, `.N`, `.A`, `.O`, `.K`, `.PK`, `.CB` are US exchanges) and match to CRSP `msenames.ticker`. Same date range validation.

**US exchange suffixes** (Refinitiv convention):
- `.OQ` = NASDAQ, `.N` = NYSE, `.A` = NYSE AMEX
- `.O` = NASDAQ, `.K` = NYSE Arca, `.PK` = OTC
- Non-US: `.TO` (Toronto), `.L` (London), `.T` (Tokyo), etc.

## 7. Handling Duplicates

### 7.1 Multiple versions per event_id

The archive typically stores one file per event_id (the latest version). When multiple versions exist:
- **Preliminary:** uploaded shortly after the call (hours), raw transcription
- **Final:** edited version uploaded days later, corrected

For point-in-time research, prefer the **preliminary (first-uploaded)** version. In the archive, almost all files are Final since the archive stores the latest version. This distinction matters more for real-time/streaming data.

### 7.2 Multiple event_ids per (company_id, start_date)

Found 1,348 cases across all firms, 227 among CRSP-linked US firms. Categories:

1. **Split filings** (most common): The call's presentation and Q&A were filed as separate events. Same `companyId`, same `startDate`, both `eventTypeId=1`. Titles differ: "Earnings Call" vs "Earnings Call - Q&A Session" or "Pre-recorded Management Remarks."

2. **Merged entity duplicates**: Two subsidiaries/merged companies (e.g., SkyWest/ExpressJet, Knight-Swift/Knight Transportation) sharing the same CRSP permno each filed their own call on the same date.

3. **Q4 vs Full Year duplicate**: Same call titled "Q4 2016" and "Full Year 2016."

4. **Exact duplicate filings**: Same content filed under two event_ids.

**Deduplication strategy:** Group by `(company_id, start_date)`. Within each group:
1. Keep the record with the **earliest `lastUpdate`** (first uploaded version)
2. Tie-break by **smallest `event_id`**

This preserves point-in-time consistency: the first-uploaded transcript is what market participants had access to earliest. The `lastUpdate` field is unreliable for absolute timing (median 2-day lag from call date), but its relative ordering within the same call reliably identifies the first-uploaded version. At 0.2% of the US sample, choice of dedup strategy has negligible impact on results.

### 7.3 Multiple calls per (permno, quarter)

Some firms have multiple earnings calls in the same quarter (e.g., preliminary results followed by full results). These are genuinely separate events and should be kept as separate observations.

## 8. Linking to Financial Data

### CRSP (returns)
- Filter: `shrcd IN (10,11)`, `exchcd IN (1,2,3)`
- Filter out CRSP sentinel return codes: -66, -77, -88, -99
- Match by permno + date

### Compustat (fundamentals)
- Filter: `indfmt='INDL'`, `datafmt='STD'`, `popsrc='D'`, `consol='C'`
- Link via CCM (`crsp.ccmxpf_lnkhist`), respecting link date validity
- Match each call to the Compustat quarterly record whose `rdq` (report date) is closest, with priority to `rdq <= call_date`

### IBES (analyst forecasts)
- Link via CRSP-IBES identifier crosswalk
- Use consensus forecast closest to but before the earnings announcement

## 9. Call Duration and Outlier Filtering

The XML only contains `<startDate>` (no end time). We approximate call duration from transcript word count, assuming a speaking pace of ~150 words per minute (standard for conversational English in professional settings). This estimate includes pauses, speaker transitions, and Q&A wait times, which are not captured in the transcript text. The actual wall-clock duration may be 10-20% longer than the word-count estimate.

```
estimated_duration_minutes = total_words / 150
estimated_end_time = startDate + estimated_duration_minutes
```

| Percentile | Words | Est. Duration |
|-----------|-------|---------------|
| 25th | 5,529 | ~37 min |
| 50th (median) | 7,471 | ~50 min |
| 75th | 9,314 | ~62 min |
| 99th | 13,800 | ~92 min |
| 99.5th | 14,776 | ~99 min |

Typical earnings calls last 45-90 minutes. We assume a maximum duration of **2 hours** and drop transcripts above the 99.5th percentile (>14,776 words), which removes 550 outliers that are likely data errors, combined multi-session events, or unusually long special calls.

For the event-day return heuristic, the `<startDate>` hour is a sufficient proxy: a call starting before 4 PM ET will end before market close even at the 99th percentile duration.

## 10. Sample Construction Summary

Starting from raw XML archive:

| Step | Records | Notes |
|------|---------|-------|
| All XML files (2001-2026) | ~700K+ | All event types, all years |
| Filter to eventTypeId=1 | TBD | Earnings calls only |
| Link to CRSP (US common stocks) | TBD | CUSIP-first, then ticker fallback |
| Dedup by (company_id, start_date) | TBD | Earliest lastUpdate, then smallest event_id |
| Drop outlier-length calls (>99.5th pctile) | TBD | Max ~2 hours / ~15K words |
| Merge with abnormal returns | TBD | FF3+momentum with timing heuristic |
| Compute FF6 CARs | TBD | Summed, not compounded |
| Winsorize at 1%/99% | TBD | All continuous variables |

The full archive covers 2001-2026. For the PEAD.txt replication, we use 2008-2019 (8-quarter training window starting 2008, test from 2010). For the embedding pipeline (DatedGPT), we use 2014-2019 (earliest model is 2013). The full archive is processed and stored for future extensions.

## 11. Winsorization

Winsorize all continuous variables at the 1st and 99th percentiles, following standard practice in the PEAD literature.
