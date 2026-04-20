"""Vectord HTTP client.

Vectord runs on the server, not exposed publicly. Tunnel over SSH first:

    ssh -L8080:10.100.0.70:8080 noel@greenbat1.vps.wbsprt.com

Then the client hits http://localhost:8080. Vector names are case-sensitive.
Timestamps are ISO8601 UTC.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import requests


DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_API_KEY = "ooloogiobub6zaesebahkad1ohYah9su"


class VectordClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = (base_url or os.getenv("VECTORD_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("VECTORD_API_KEY") or DEFAULT_API_KEY
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"X-API-Key": self.api_key})

    def read(
        self,
        vector: str,
        start: str | datetime,
        end: str | datetime,
    ) -> list[dict]:
        """Read raw points for a vector between start and end (UTC)."""
        s = _iso(start)
        e = _iso(end)
        url = f"{self.base_url}/read/{vector}/{s}/{e}"
        r = self._session.get(url, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        if isinstance(body, dict) and "data" in body:
            return body["data"] or []
        return body

    def read_df(
        self,
        vector: str,
        start: str | datetime,
        end: str | datetime,
    ) -> pd.DataFrame:
        """Read a vector as a DataFrame indexed by UTC timestamp."""
        points = self.read(vector, start, end)
        if not points:
            return pd.DataFrame(columns=["value"], index=pd.DatetimeIndex([], tz="UTC"))
        df = pd.DataFrame(points)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.set_index("time").sort_index()

    def write(
        self,
        vector: str,
        points: Iterable[dict] | pd.Series | pd.DataFrame,
    ) -> dict:
        """Write points to a vector. Accepts dicts with time/value, or a
        pandas Series/DataFrame with a DatetimeIndex."""
        payload = _to_payload(points)
        url = f"{self.base_url}/write/{vector}"
        r = self._session.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def write_now(self, vector: str, value: float) -> dict:
        """Write a single value tagged with server 'now'."""
        return self.write(vector, [{"value": float(value)}])


def _iso(ts: str | datetime) -> str:
    if isinstance(ts, str):
        return ts
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_payload(points) -> list[dict]:
    if isinstance(points, pd.Series):
        return [
            {"time": _iso(idx.to_pydatetime()), "value": float(val)}
            for idx, val in points.items()
        ]
    if isinstance(points, pd.DataFrame):
        if "value" not in points.columns:
            raise ValueError("DataFrame must have a 'value' column")
        return [
            {"time": _iso(idx.to_pydatetime()), "value": float(row["value"])}
            for idx, row in points.iterrows()
        ]
    return [dict(p) for p in points]
