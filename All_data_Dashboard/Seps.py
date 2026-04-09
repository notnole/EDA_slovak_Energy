#!/usr/bin/env python3
"""
DAMAS Data Scraper - Simple single-file scraper for any DAMAS/DAE endpoint.

Usage:
    1. Configure ENDPOINTS below with your VIEW_CODEs
    2. Run: python damas_scraper.py

To add a new endpoint:
    Add an entry to ENDPOINTS dict with:
    - "view_code": The DAMAS view path (e.g., "\\SS_PUB\\PUB\\DATAFLOW\\REAL_SYSTEM_DATA_GUI")
    - "days_back": How many days of history to fetch (default: 1)

Examples of view codes (SEPS Slovakia):
    - Real-time system state:     \\SS_PUB\\PUB\\DATAFLOW\\SYSTEM_STATE_MAP_VIEW
    - 15-min historical data:     \\SS_PUB\\PUB\\DATAFLOW\\REAL_SYSTEM_DATA_GUI
    - Generation by fuel type:    \\SS_PUB\\PUB\\PUB_GENERATION_ES\\AGGREGATED_GENERATION_PER_TYPE_GUI
    - Cross-border flows:         \\SS_PUB\\PUB\\DATAFLOW\\CROSSBOARD_FLOW_OVERVIEW_GUI
    - Load forecasts:             \\SS_PUB\\PUB\\PUB_LOAD_ES\\PREDICTED_LOAD_GUI
"""

import json
import time
import subprocess
import platform
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =============================================================================
# CONFIGURATION - Edit these for your DAMAS instance
# =============================================================================

# Base URL of the DAMAS platform
BASE_URL = "https://dae.sepsas.sk/SK_PROD"

# Login page (for Playwright session establishment)
LOGIN_URL = f"{BASE_URL}/DAEF-GUI/SUC/001System/UCLogin/LoginTest.aspx"

# API endpoint for data fetching
API_URL = f"{BASE_URL}/DAEF-GUI/DAE_MVC/DfeDamas/TsPivotTableComponent/LoadData"

# Cookie cache file (will be created automatically)
COOKIE_FILE = Path(__file__).parent / ".damas_cookies.json"

# =============================================================================
# ENDPOINTS - Add your VIEW_CODEs here
# =============================================================================

ENDPOINTS = {
    "system_realtime": {
        "view_code": "\\SS_PUB\\PUB\\DATAFLOW\\SYSTEM_STATE_MAP_VIEW",
        "days_back": 1,  # Real-time view ignores date range
        "description": "Real-time system state (3-min resolution, no history)"
    },
    "system_15min": {
        "view_code": "\\SS_PUB\\PUB\\DATAFLOW\\REAL_SYSTEM_DATA_GUI",
        "days_back": 7,
        "description": "System data at 15-min resolution (load, production, balance)"
    },
    "generation_by_type": {
        "view_code": "\\SS_PUB\\PUB\\PUB_GENERATION_ES\\AGGREGATED_GENERATION_PER_TYPE_GUI",
        "days_back": 7,
        "description": "Generation by fuel type (hourly, 13 types)"
    },
    "cross_border_flows": {
        "view_code": "\\SS_PUB\\PUB\\DATAFLOW\\CROSSBOARD_FLOW_OVERVIEW_GUI",
        "days_back": 7,
        "description": "Cross-border power flows (15-min resolution)"
    },
}

# =============================================================================
# SCRAPER IMPLEMENTATION
# =============================================================================


def establish_session(headless: bool = True, timeout_ms: int = 30000) -> list | None:
    """Use Playwright to login and get session cookies."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: Playwright not installed. Run:")
        print("  pip install playwright")
        print("  playwright install chromium")
        return None

    print("Establishing browser session...")
    cookies = None

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = browser.new_context(locale="sk-SK", viewport={"width": 1280, "height": 800})
        page = context.new_page()

        try:
            page.goto(LOGIN_URL, wait_until="load", timeout=timeout_ms)

            # Click public access button (adjust selector for your DAMAS instance)
            page.wait_for_selector("a[href*='lbtPublicAccess']", state="attached", timeout=timeout_ms)
            page.evaluate("document.querySelector('a[href*=\"lbtPublicAccess\"]').click()")
            page.wait_for_url("**/default.aspx*", timeout=timeout_ms)

            # Navigate iframe to initialize MVC session
            page.wait_for_selector("iframe[name='frameContent']", timeout=10000)
            page.evaluate("""
                () => {
                    const frame = document.querySelector('iframe[name="frameContent"]');
                    if (frame) {
                        frame.src = '/SK_PROD/DAEF-GUI/DAE_MVC/UC/TS_VIEW_SCREEN?VIEW_CODE=\\\\SS_PUB\\\\PUB\\\\DATAFLOW\\\\SYSTEM_STATE_MAP_VIEW';
                    }
                }
            """)
            page.wait_for_timeout(3000)

            cookies = context.cookies()
            print(f"Session established, got {len(cookies)} cookies")

        except Exception as e:
            print(f"Session error: {e}")
        finally:
            context.close()
            browser.close()

    # Kill orphan chromium processes
    _kill_orphan_chromium()

    return cookies


def _kill_orphan_chromium():
    """Clean up any orphan chromium processes."""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq chromium.exe", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5
            )
            if "chromium.exe" in result.stdout:
                subprocess.run(["taskkill", "/F", "/IM", "chromium.exe"], capture_output=True, timeout=5)
        else:
            result = subprocess.run(
                ["pgrep", "-f", "chromium.*--headless"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                for pid in result.stdout.strip().split("\n"):
                    subprocess.run(["kill", "-9", pid.strip()], capture_output=True, timeout=5)
    except Exception:
        pass


def save_cookies(cookies: list) -> None:
    """Save cookies to disk."""
    COOKIE_FILE.write_text(json.dumps(cookies, ensure_ascii=False), encoding="utf-8")


def load_cookies() -> list | None:
    """Load cached cookies from disk."""
    if COOKIE_FILE.exists():
        try:
            return json.loads(COOKIE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return None


def build_session(cookies: list) -> requests.Session:
    """Build a requests session with cookies."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    for c in cookies:
        session.cookies.set(c["name"], c["value"], domain=c["domain"], path=c["path"])
    return session


def fetch_data(
    session: requests.Session,
    view_code: str,
    days_back: int = 1
) -> dict | None:
    """Fetch data from the DAMAS API."""

    # Get XSRF token from cookies
    xsrf = ""
    for c in session.cookies:
        if "RequestVerification" in c.name:
            xsrf = c.value
            break

    now = datetime.now(timezone.utc)
    interval_till = now.replace(hour=23, minute=0, second=0, microsecond=0)
    interval_from = interval_till - timedelta(days=days_back)

    # URL-encode the view code for QUERY parameter
    view_code_encoded = quote(view_code, safe="")

    headers = {
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest",
        "X-XSRF-TOKEN": xsrf,
        "DFE-Screen-Id": "TS_VIEW_SCREEN",
        "DFE-Component-Id": "PIVOT_TABLE",
        "DFE-Timezone-Id": "CET",
        "Origin": BASE_URL.rsplit("/", 1)[0],
        "Referer": f"{BASE_URL}/DAEF-GUI/DAE_MVC/UC/TS_VIEW_SCREEN",
    }

    payload = {
        "parameters": [
            {"code": "VIEW_CODE", "value": view_code},
            {"code": "TIME_FILTER_UNIT", "value": "DAY"},
            {"code": "INTERVAL_TILL", "value": interval_till.strftime("%Y-%m-%dT%H:%M:%S.000Z")},
            {"code": "INTERVAL_FROM", "value": interval_from.strftime("%Y-%m-%dT%H:%M:%S.000Z")},
            {"code": "USER_COND_FORMULA", "value": "1"},
            {"code": "SKIP_VIEW_CONTEXT", "value": True},
            {"code": "CHART_TYPE", "value": None},
            {"code": "VIEW_TYPE", "value": "TimeseriesView"},
            {"code": "QUERY", "value": f"#?VIEW_CODE={view_code_encoded}"},
        ]
    }

    try:
        start = time.monotonic()
        r = session.post(API_URL, json=payload, headers=headers, verify=False, timeout=30)
        duration = (time.monotonic() - start) * 1000

        if r.status_code == 200:
            data = r.json()
            print(f"  API call: {duration:.0f}ms")
            return data
        else:
            print(f"  API error: HTTP {r.status_code}")
            return None

    except Exception as e:
        print(f"  API error: {e}")
        return None


def parse_response(data: dict) -> tuple[list[str], list[list]]:
    """Parse API response into columns and rows."""
    columns = []
    rows = []

    try:
        grid = data.get("gridConfig", {})
        sheets = grid.get("sheets", [])

        if not sheets:
            return columns, rows

        sheet = sheets[0]

        # Extract column headers
        col_model = sheet.get("colModel", [])
        for col in col_model:
            cell_data = col.get("cellData", {})
            title = cell_data.get("v", col.get("title", ""))
            columns.append(title)

        # Extract data rows
        data_rows = sheet.get("dataModel", {}).get("data", [])
        for row in data_rows:
            parsed_row = []
            for cell in row:
                if isinstance(cell, dict):
                    parsed_row.append(cell.get("v", ""))
                else:
                    parsed_row.append(cell)
            rows.append(parsed_row)

    except Exception as e:
        print(f"Parse error: {e}")

    return columns, rows


def get_session() -> requests.Session | None:
    """Get a valid session, establishing one if needed."""
    # Try cached cookies first
    cookies = load_cookies()
    if cookies:
        session = build_session(cookies)
        # Quick test call
        test_data = fetch_data(session, "\\SS_PUB\\PUB\\DATAFLOW\\SYSTEM_STATE_MAP_VIEW", 1)
        if test_data and not test_data.get("gridConfig", {}).get("noDataFound", True):
            return session
        print("Cached cookies expired, re-establishing session...")

    # Establish new session
    cookies = establish_session()
    if not cookies:
        return None

    save_cookies(cookies)
    return build_session(cookies)


def scrape_endpoint(name: str, config: dict) -> dict:
    """Scrape a single endpoint and return results."""
    print(f"\n{'='*60}")
    print(f"Scraping: {name}")
    print(f"  {config.get('description', '')}")
    print(f"  View: {config['view_code']}")
    print(f"  Days: {config.get('days_back', 1)}")

    session = get_session()
    if not session:
        return {"error": "Failed to establish session"}

    data = fetch_data(
        session,
        config["view_code"],
        config.get("days_back", 1)
    )

    if not data:
        return {"error": "API call failed"}

    columns, rows = parse_response(data)

    print(f"  Columns: {len(columns)}")
    print(f"  Rows: {len(rows)}")

    if columns:
        print(f"  Column names: {columns[:5]}{'...' if len(columns) > 5 else ''}")

    if rows:
        print(f"  Sample row: {rows[0][:5]}{'...' if len(rows[0]) > 5 else ''}")

    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "raw_response": data
    }


def scrape_all() -> dict:
    """Scrape all configured endpoints."""
    results = {}

    for name, config in ENDPOINTS.items():
        results[name] = scrape_endpoint(name, config)

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DAMAS Data Scraper")
    parser.add_argument("--endpoint", "-e", help="Scrape specific endpoint (default: all)")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--list", "-l", action="store_true", help="List available endpoints")
    args = parser.parse_args()

    if args.list:
        print("Available endpoints:")
        for name, config in ENDPOINTS.items():
            print(f"  {name}: {config.get('description', '')}")
        exit(0)

    print("DAMAS Data Scraper")
    print(f"Base URL: {BASE_URL}")

    if args.endpoint:
        if args.endpoint not in ENDPOINTS:
            print(f"Unknown endpoint: {args.endpoint}")
            print(f"Available: {list(ENDPOINTS.keys())}")
            exit(1)
        results = {args.endpoint: scrape_endpoint(args.endpoint, ENDPOINTS[args.endpoint])}
    else:
        results = scrape_all()

    if args.output:
        # Save results (without raw_response to keep file size reasonable)
        output_data = {}
        for name, data in results.items():
            output_data[name] = {
                "columns": data.get("columns", []),
                "rows": data.get("rows", []),
                "row_count": data.get("row_count", 0),
                "error": data.get("error")
            }

        Path(args.output).write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"\nResults saved to: {args.output}")

    print("\nDone!")
