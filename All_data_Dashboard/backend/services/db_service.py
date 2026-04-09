
import asyncpg
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# DB Configuration
DB_HOST = "127.0.0.1"
DB_PORT = 5433
DB_USER = "beam"
DB_NAME = "beam-solar"
DB_PASSWORD = os.environ.get("DB_PASSWORD", "s%Upt%H%5vpD2gW@9r&S") 

async def get_db_connection():
    try:
        return await asyncpg.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            host=DB_HOST,
            port=DB_PORT
        )
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

async def fetch_db_series(schema: str, table: str, from_date: datetime, to_date: datetime) -> List[Dict[str, Any]]:
    """
    Generic fetch for time-series data from the DB.
    Handles different table structures based on schema.
    """
    conn = await get_db_connection()
    if not conn:
        return {"error": "Database unreachable"}

    try:
        query = ""
        params = []
        
        # Schema: data2 (Simple time, value structure)
        if schema == "data2":
            query = f"""
                SELECT time, value 
                FROM "{schema}"."{table}"
                WHERE time >= $1 AND time <= $2
                ORDER BY time ASC
            """
            params = [from_date, to_date]
            
        # Schema: data (Complex structure, varies by table)
        elif schema == "data":
            if table == "generation": 
                # Example for generation table
                query = f"""
                    SELECT date + hour * INTERVAL '1 hour' as time, real_value as value
                    FROM "{schema}"."{table}"
                    WHERE date >= $1 AND date <= $2
                    ORDER BY date, hour ASC
                """
                # Note: This is a simplification, date handling might need adjustment based on actual column types
                params = [from_date.date(), to_date.date()]
            elif table == "load":
                 query = f"""
                    SELECT date + hour * INTERVAL '1 hour' as time, real_value as value
                    FROM "{schema}"."{table}"
                    WHERE date >= $1 AND date <= $2
                    ORDER BY date, hour ASC
                """
                 params = [from_date.date(), to_date.date()]

        if not query:
            return {"error": f"Unsupported table configuration: {schema}.{table}"}

        rows = await conn.fetch(query, *params)
        
        # Transform to list of dicts for JSON response
        result = []
        for row in rows:
            record = dict(row)
            # Ensure datetime objects are ISO formatted strings
            if 'time' in record and hasattr(record['time'], 'isoformat'):
                record['time'] = record['time'].isoformat()
            if 'date' in record and hasattr(record['date'], 'isoformat'):
                record['date'] = record['date'].isoformat()
            result.append(record)
            
        return result

    except Exception as e:
        print(f"Query error: {e}")
        return {"error": str(e)}
    finally:
        await conn.close()

async def get_available_db_series():
    # Hardcoded list of useful tables based on documentation
    return [
        {"id": "data2.REAL_SYSTEM_LOAD", "name": "Real System Load (Live)", "schema": "data2", "table": "REAL_SYSTEM_LOAD"},
        {"id": "data2.REAL_SYSTEM_PRODUCTION", "name": "Real System Production (Live)", "schema": "data2", "table": "REAL_SYSTEM_PRODUCTION"},
        {"id": "data.generation", "name": "Generation (Hourly)", "schema": "data", "table": "generation"},
        {"id": "data.load", "name": "Load (Hourly)", "schema": "data", "table": "load"},
    ]
