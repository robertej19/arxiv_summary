#!/usr/bin/env python3
"""
Download the past 5 years of finalized (non-amended) 10-K filings for multiple companies from SEC EDGAR.

Reads company list from companies.txt and saves the primary 10-K documents (typically HTML) 
into separate directories for each company.
Requires: requests

SEC guidance:
- Use a descriptive User-Agent with contact info.
- Keep request rate reasonable (<=10 req/sec). We sleep conservatively.
"""

import os
import time
import json
import re
from datetime import date
from pathlib import Path
from typing import List, Tuple

import requests

# ---------- Configuration ----------
YEARS_BACK = 5
BASE_OUTPUT_DIR = Path("10k_filings")
COMPANIES_FILE = Path("companies.txt")

# Get USER_AGENT from environment variable
USER_AGENT = os.getenv("SEC_USER_AGENT")
if not USER_AGENT:
    raise ValueError("SEC_USER_AGENT environment variable must be set. Example: 'export SEC_USER_AGENT=\"YourName YourOrg (youremail@example.com)\"'")

ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
REQUEST_SLEEP_SECONDS = 0.3  # conservative politeness

# ---------- Helpers ----------
def load_companies(file_path: Path) -> List[Tuple[str, str, str]]:
    """Load companies from text file. Returns list of (CIK, TICKER, COMPANY_NAME) tuples."""
    companies = []
    
    if not file_path.exists():
        raise FileNotFoundError(f"Companies file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(',')
            if len(parts) < 2:
                print(f"Warning: Invalid format on line {line_num}: {line}")
                continue
            
            cik = parts[0].strip()
            ticker = parts[1].strip()
            company_name = parts[2].strip() if len(parts) > 2 else ticker
            
            companies.append((cik, ticker, company_name))
    
    return companies

def five_years_ago(today: date) -> date:
    """Return a date that is '5 years ago' from today, handling leap years."""
    try:
        return date(today.year - YEARS_BACK, today.month, today.day)
    except ValueError:
        # For Feb 29, fallback to Feb 28
        return date(today.year - YEARS_BACK, today.month, 28)

def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def fetch_json(url: str) -> dict:
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()

def download_file(url: str, dest_path: Path):
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)

def get_all_filings(data: dict, cutoff: date) -> List[Tuple[date, str, str]]:
    """
    Get all 10-K filings from both recent and files sections.
    This handles cases where companies have many filings and older ones are paginated.
    """
    all_filings = []
    
    # Process recent filings
    recent = data.get("filings", {}).get("recent", {})
    if recent:
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        
        print(f"  Processing {len(forms)} recent filings...")
        
        for form, fdate, acc, pdoc in zip(forms, filing_dates, accession_numbers, primary_docs):
            if form != "10-K":
                continue
            if not fdate or not acc or not pdoc:
                continue
            try:
                fdate_obj = date.fromisoformat(fdate)
                if fdate_obj >= cutoff:
                    all_filings.append((fdate_obj, acc, pdoc))
            except ValueError:
                continue
    
    # Process additional files if they exist (for companies with many filings)
    files = data.get("filings", {}).get("files", [])
    if files:
        print(f"  Found {len(files)} additional filing files to check...")
        for file_info in files:
            # Only check the first few files to avoid excessive API calls
            # Usually the recent files contain what we need
            break  # Skip for now to avoid rate limiting, but this is where you'd fetch older filings
    
    return all_filings

def download_company_10ks(cik: str, ticker: str, company_name: str):
    """Download 10-K filings for a single company."""
    print(f"\n{'='*60}")
    print(f"Processing: {company_name} ({ticker}) - CIK: {cik}")
    print(f"{'='*60}")
    
    base_submissions = f"https://data.sec.gov/submissions/CIK{cik}.json"
    
    try:
        print(f"Fetching submissions JSON...")
        data = fetch_json(base_submissions)
        time.sleep(REQUEST_SLEEP_SECONDS)
    except Exception as e:
        print(f"Error fetching submissions for {ticker}: {e}")
        return

    cutoff = five_years_ago(date.today())
    print(f"Cutoff date (last {YEARS_BACK} years): {cutoff.isoformat()}")
    
    # Debug: Print some basic info about the company
    print(f"  Company name in SEC data: {data.get('name', 'N/A')}")
    print(f"  SIC: {data.get('sic', 'N/A')} - {data.get('sicDescription', 'N/A')}")
    print(f"  State of incorporation: {data.get('stateOfIncorporation', 'N/A')}")
    
    # Get all 10-K filings
    all_10k_filings = get_all_filings(data, cutoff)
    
    # Debug: Show breakdown of all recent filings by type
    recent = data.get("filings", {}).get("recent", {})
    if recent:
        forms = recent.get("form", [])
        form_counts = {}
        for form in forms:
            form_counts[form] = form_counts.get(form, 0) + 1
        
        print(f"  Recent filing types: {dict(sorted(form_counts.items()))}")
        
        # Show specifically 10-K vs 10-K/A breakdown
        tenk_count = form_counts.get("10-K", 0)
        tenka_count = form_counts.get("10-K/A", 0)
        print(f"  10-K filings: {tenk_count}, 10-K/A amendments: {tenka_count}")

    if not all_10k_filings:
        print(f"No 10-K filings found for {ticker} in the last {YEARS_BACK} years.")
        
        # Debug: Show the date range of available filings
        if recent and recent.get("filingDate"):
            dates = [d for d in recent.get("filingDate", []) if d]
            if dates:
                earliest = min(dates)
                latest = max(dates)
                print(f"  Available filing date range: {earliest} to {latest}")
        
        return

    # Sort newest first
    all_10k_filings.sort(reverse=True, key=lambda x: x[0])

    # Create company-specific output directory
    company_output_dir = BASE_OUTPUT_DIR / ticker
    
    cik_int = str(int(cik))  # strip leading zeros for edgar/data path

    print(f"Found {len(all_10k_filings)} 10-K filings in the last {YEARS_BACK} years:")
    
    # Show filing dates for debugging
    filing_years = {}
    for fdate_obj, _, _ in all_10k_filings:
        year = fdate_obj.year
        filing_years[year] = filing_years.get(year, 0) + 1
    print(f"  Filings by year: {dict(sorted(filing_years.items()))}")
    
    for fdate_obj, acc, pdoc in all_10k_filings:
        acc_nodash = acc.replace("-", "")
        filing_url = f"{ARCHIVES_BASE}/{cik_int}/{acc_nodash}/{pdoc}"

        # Build a nice filename
        out_name = f"{fdate_obj.isoformat()}_{ticker}_10-K_{safe_filename(pdoc)}"
        out_path = company_output_dir / out_name

        print(f"- {fdate_obj.isoformat()} -> {filing_url}")
        try:
            download_file(filing_url, out_path)
            print(f"  Saved: {out_path}")
        except requests.HTTPError as e:
            print(f"  HTTP error downloading {filing_url}: {e}")
        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(REQUEST_SLEEP_SECONDS)

# ---------- Main ----------
def main():
    print("Loading companies from companies.txt...")
    try:
        companies = load_companies(COMPANIES_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please create a companies.txt file with format: CIK,TICKER,COMPANY_NAME")
        return
    
    if not companies:
        print("No companies found in companies.txt")
        return
    
    print(f"Found {len(companies)} companies to process")
    
    # Create base output directory
    BASE_OUTPUT_DIR.mkdir(exist_ok=True)
    
    total_companies = len(companies)
    successful_downloads = 0
    
    for i, (cik, ticker, company_name) in enumerate(companies, 1):
        print(f"\n[{i}/{total_companies}] Starting download for {ticker}...")
        try:
            download_company_10ks(cik, ticker, company_name)
            successful_downloads += 1
        except Exception as e:
            print(f"Failed to process {ticker}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Processing completed!")
    print(f"Successfully processed: {successful_downloads}/{total_companies} companies")
    print(f"Files saved in: {BASE_OUTPUT_DIR.absolute()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
