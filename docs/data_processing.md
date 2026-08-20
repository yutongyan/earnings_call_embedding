# Earnings Call Transcript Data Processing Guide

This document describes how to properly process StreetEvents earnings call transcripts for academic research, based on lessons learned during the PEAD.txt replication.

## 1. Data Source

StreetEvents transcripts from Refinitiv, accessed via SFTP. Each transcript is an XML file named `{event_id}_T.xml`, organized by year in the archive.

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
- `is_before_close = call_hour_et < 16` (before 4 PM ET)
- If call time is unknown, set `is_before_close = None` (not False)
- This determines the event-day return window:
  - Before close: use return from t-1 to t
  - After close: use return from t to t+1

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

## 6. Ticker and Company Identification

Prefer XML elements over headline parsing:
1. `<companyTicker>` for ticker (e.g., `RSI.TO`)
2. `<companyName>` for company name
3. Fall back to headline regex only if XML elements are missing

**US exchange suffixes** (Refinitiv convention):
- `.OQ` = NASDAQ, `.N` = NYSE, `.A` = NYSE AMEX
- `.O` = NASDAQ, `.K` = NYSE Arca, `.PK` = OTC
- Non-US: `.TO` (Toronto), `.L` (London), `.T` (Tokyo), etc.

**Linking to CRSP:** Match cleaned tickers to CRSP `msenames` by ticker, filtered to `shrcd IN (10,11)` and `exchcd IN (1,2,3)` for US common stocks. Validate that the call date falls within the CRSP name date range (`namedt` to `nameendt`).

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

**Recommended handling:** For split filings, ideally concatenate the texts. For all other cases, keep the longest transcript per (permno, call_date), or the earliest event_id for point-in-time consistency. At 0.2% of the US sample, this has negligible impact on results.

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

## 9. Sample Construction Summary

Starting from raw XML archive:

| Step | Records | Notes |
|------|---------|-------|
| All XML files (2008-2019) | ~300K+ | All event types |
| Filter to eventTypeId=1 | 224,476 | Earnings calls only |
| Link to CRSP (US common stocks) | 110,986 | Ticker matching + date validation |
| Dedup by (permno, call_date) | 110,759 | Keep earliest event_id |
| Merge with abnormal returns | 109,925 | Require valid AR computation |
| Test sample (2010Q1-2019Q4) | ~92,000 | After 8-quarter training window |

## 10. Winsorization

Winsorize all continuous variables at the 1st and 99th percentiles, following standard practice in the PEAD literature.
