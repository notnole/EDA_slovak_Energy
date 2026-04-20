# vectord

Thin HTTP client for the Vectord time-series service (SCADA/EDA data).

## Setup

Vectord is not publicly exposed. Open an SSH tunnel first:

```bash
ssh -L8080:10.100.0.70:8080 noel@greenbat1.vps.wbsprt.com
```

Leave that running, then hit `http://localhost:8080` from Python.

## Usage

```python
from vectord import VectordClient

client = VectordClient()  # defaults to localhost:8080 + prod key

# Read as raw list of {time, value} dicts
points = client.read("F.B.Odchylka", "2026-04-20T00:00:00Z", "2026-04-20T12:00:00Z")

# Read as a pandas DataFrame (UTC DatetimeIndex, column 'value')
df = client.read_df("F.B.Odchylka", "2026-04-20T00:00:00Z", "2026-04-20T12:00:00Z")

# Write a single value (server 'now')
client.write_now("F.B.Odchylka", 123.45)

# Write a batch
client.write("F.B.Odchylka", [
    {"time": "2026-04-20T10:00:00Z", "value": 123.45},
])

# Write a pandas Series
client.write("F.B.Odchylka", series)
```

## Config

Override via env vars or constructor args:

- `VECTORD_URL` (default `http://localhost:8080`)
- `VECTORD_API_KEY` (default: prod key baked in)

## Local mock

For testing without the tunnel:

```bash
cd testing/mock_vectord && uvicorn server:app --port 8080
```

Then use `VectordClient(api_key="test-api-key-local")`.

## Notes

- Vector names are **case-sensitive**.
- Timestamps are ISO8601 UTC.
- Only non-null points are returned.
- `time` is optional on write — omit to tag with server time.
