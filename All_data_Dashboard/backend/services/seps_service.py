
import sys
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path to import Seps.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import Seps
except ImportError:
    print("WARNING: Could not import Seps.py. Make sure it exists in the project root.")
    Seps = None

# Cache for SEPS data (simple in-memory cache)
# Key: series_id, Value: (timestamp, data)
_cache = {}

executor = ThreadPoolExecutor(max_workers=3)

async def fetch_seps_data(series_id: str):
    if not Seps:
        return {"error": "Seps module not loaded"}

    # Map series_id to Seps endpoint keys if needed, or use direct keys
    # Seps.ENDPOINTS keys: system_realtime, system_15min, generation_by_type, cross_border_flows
    
    if series_id not in Seps.ENDPOINTS:
        return {"error": f"Unknown series ID: {series_id}"}

    # Run blocking scraper in a thread
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, Seps.scrape_endpoint, series_id, Seps.ENDPOINTS[series_id])
    
    return result
