
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
from datetime import datetime, timedelta
from services.seps_service import fetch_seps_data
from services.db_service import fetch_db_series, get_available_db_series

router = APIRouter()

class DataRequest(BaseModel):
    source: str  # 'seps', 'db', 'entsoe'
    series_id: str
    from_date: Optional[str] = None
    to_date: Optional[str] = None

@router.get("/metadata/sources")
async def get_sources():
    db_series = await get_available_db_series()
    
    # Return available sources and their capabilities
    return {
        "sources": [
            {"id": "seps", "name": "SEPS/Damus", "type": "scraper"},
            {"id": "db", "name": "Private Database", "type": "database", "series": db_series},
            {"id": "entsoe", "name": "Entsoe API", "type": "api"},
        ]
    }

@router.post("/data/series")
async def get_series_data(request: DataRequest):
    if request.source == "seps":
        try:
            data = await fetch_seps_data(request.series_id)
            return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    elif request.source == "db":
        # Parse dates or default to last 24h
        to_date = datetime.now()
        from_date = to_date - timedelta(days=1)
        
        if request.to_date:
            try:
                to_date = datetime.fromisoformat(request.to_date.replace("Z", "+00:00"))
            except ValueError:
                pass
        if request.from_date:
            try:
                from_date = datetime.fromisoformat(request.from_date.replace("Z", "+00:00"))
            except ValueError:
                pass

        # Split series_id into schema and table
        parts = request.series_id.split(".")
        if len(parts) < 2:
             raise HTTPException(status_code=400, detail="Invalid series_id for DB source. Format: schema.table")
        
        schema = parts[0]
        table = ".".join(parts[1:]) 
        
        return await fetch_db_series(schema, table, from_date, to_date)
            
    else:
        raise HTTPException(status_code=400, detail="Unknown source")
