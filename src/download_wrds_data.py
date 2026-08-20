#!/usr/bin/env python3
"""Download CRSP, Compustat, IBES data from WRDS for PEAD.txt replication.

Usage:
    python3 src/download_wrds_data.py --output data/
"""

import argparse
import os

import pandas as pd
import wrds


def download_crsp_daily(db, output_dir):
    """Download CRSP daily stock file (2007-2020 for estimation windows)."""
    print("Downloading CRSP daily returns...", flush=True)
    df = db.raw_sql("""
        SELECT a.permno, a.date, a.ret, a.prc, a.shrout, a.vol,
               b.ticker, b.ncusip, b.comnam, b.shrcd, b.exchcd
        FROM crsp.dsf a
        LEFT JOIN crsp.msenames b
            ON a.permno = b.permno
            AND a.date >= b.namedt
            AND a.date <= b.nameendt
        WHERE a.date BETWEEN '2007-01-01' AND '2020-12-31'
          AND b.shrcd IN (10, 11)
          AND b.exchcd IN (1, 2, 3)
    """)
    path = os.path.join(output_dir, "crsp_daily.parquet")
    df.to_parquet(path, index=False)
    print(f"  CRSP daily: {len(df):,} rows -> {path}")
    return df


def download_crsp_msenames(db, output_dir):
    """Download CRSP name history for identifier linking."""
    print("Downloading CRSP msenames...", flush=True)
    df = db.raw_sql("""
        SELECT permno, namedt, nameendt, ncusip, ticker, comnam,
               shrcd, exchcd, siccd
        FROM crsp.msenames
        WHERE shrcd IN (10, 11)
    """)
    path = os.path.join(output_dir, "crsp_msenames.parquet")
    df.to_parquet(path, index=False)
    print(f"  CRSP msenames: {len(df):,} rows -> {path}")
    return df


def download_compustat_quarterly(db, output_dir):
    """Download Compustat quarterly for earnings and fundamentals."""
    print("Downloading Compustat quarterly...", flush=True)
    df = db.raw_sql("""
        SELECT gvkey, datadate, fyearq, fqtr, rdq,
               epspxq, epsfxq, ibq, saleq, atq, ceqq,
               cshoq, prccq, mkvaltq
        FROM comp.fundq
        WHERE datadate BETWEEN '2007-01-01' AND '2020-12-31'
          AND indfmt = 'INDL'
          AND datafmt = 'STD'
          AND popsrc = 'D'
          AND consol = 'C'
    """)
    path = os.path.join(output_dir, "compustat_quarterly.parquet")
    df.to_parquet(path, index=False)
    print(f"  Compustat quarterly: {len(df):,} rows -> {path}")
    return df


def download_ccm_link(db, output_dir):
    """Download Compustat-CRSP linking table."""
    print("Downloading CCM link table...", flush=True)
    df = db.raw_sql("""
        SELECT gvkey, lpermno AS permno, linkdt, linkenddt,
               linktype, linkprim
        FROM crsp.ccmxpf_linkhist
        WHERE linktype IN ('LU', 'LC')
          AND linkprim IN ('P', 'C')
    """)
    path = os.path.join(output_dir, "ccm_link.parquet")
    df.to_parquet(path, index=False)
    print(f"  CCM link: {len(df):,} rows -> {path}")
    return df


def download_ibes(db, output_dir):
    """Download IBES summary statistics for analyst forecasts."""
    print("Downloading IBES summary...", flush=True)
    df = db.raw_sql("""
        SELECT ticker, cusip, statpers, fpedats, fpi,
               meanest, medest, actual, stdev, numest
        FROM ibes.statsumu_epsus
        WHERE fpi = '6'
          AND statpers BETWEEN '2007-01-01' AND '2020-12-31'
          AND measure = 'EPS'
    """)
    path = os.path.join(output_dir, "ibes_summary.parquet")
    df.to_parquet(path, index=False)
    print(f"  IBES summary: {len(df):,} rows -> {path}")
    return df


def download_ibes_id(db, output_dir):
    """Download IBES identifier mapping for CRSP linking."""
    print("Downloading IBES ID table...", flush=True)
    df = db.raw_sql("""
        SELECT ticker, cusip, oftic, cname, sdates
        FROM ibes.id
    """)
    path = os.path.join(output_dir, "ibes_id.parquet")
    df.to_parquet(path, index=False)
    print(f"  IBES ID: {len(df):,} rows -> {path}")
    return df


def download_ff_factors(db, output_dir):
    """Download Fama-French factors from WRDS."""
    print("Downloading FF factors...", flush=True)
    df = db.raw_sql("""
        SELECT date, mktrf, smb, hml, rf, umd
        FROM ff.factors_daily
        WHERE date BETWEEN '2007-01-01' AND '2020-12-31'
    """)
    path = os.path.join(output_dir, "ff_factors_daily.parquet")
    df.to_parquet(path, index=False)
    print(f"  FF factors daily: {len(df):,} rows -> {path}")

    print("Downloading FF 5 factors...", flush=True)
    df5 = db.raw_sql("""
        SELECT date, mktrf, smb, hml, rmw, cma, rf
        FROM ff.fivefactors_daily
        WHERE date BETWEEN '2007-01-01' AND '2020-12-31'
    """)
    path5 = os.path.join(output_dir, "ff5_factors_daily.parquet")
    df5.to_parquet(path5, index=False)
    print(f"  FF5 factors daily: {len(df5):,} rows -> {path5}")

    return df, df5


def download_ff_portfolios(db, output_dir):
    """Download FF size/BM portfolio returns for benchmark CARs."""
    print("Downloading FF 6 size/BM portfolio returns...", flush=True)
    df = db.raw_sql("""
        SELECT date, smalllobm, me1bm2, smallhibm,
               biglobm, me2bm2, bighibm
        FROM ff.portfolios6_daily
        WHERE date BETWEEN '2007-01-01' AND '2020-12-31'
    """)
    path = os.path.join(output_dir, "ff_portfolios6_daily.parquet")
    df.to_parquet(path, index=False)
    print(f"  FF 6 portfolios: {len(df):,} rows -> {path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    db = wrds.Connection(wrds_username="yutongyancuhk")
    print("Connected to WRDS\n")

    try:
        download_crsp_daily(db, args.output)
        download_crsp_msenames(db, args.output)
        download_compustat_quarterly(db, args.output)
        download_ccm_link(db, args.output)
        download_ibes(db, args.output)
        download_ibes_id(db, args.output)
        download_ff_factors(db, args.output)
        download_ff_portfolios(db, args.output)
    finally:
        db.close()

    print("\nAll WRDS downloads complete.")


if __name__ == "__main__":
    main()
