"""Route for dataset upload and schema validation."""

import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.config import RAW_DATA_DIR, MAX_UPLOAD_BYTES, require_api_key, rate_limit_upload
from app.services.preprocessing import validate_csv
from app.db.database import record_dataset
from app.schemas import DatasetUploadResponse

router = APIRouter(prefix="/api", tags=["Data Upload"], dependencies=[Depends(require_api_key), Depends(rate_limit_upload)])


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload CSV dataset, validate required smart grid features, store metadata, and return preview."""
    original_name = file.filename or ""
    if not original_name.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # Save to raw directory
    safe_name = f"{uuid.uuid4().hex}.csv"
    destination = RAW_DATA_DIR / safe_name
    try:
        with destination.open("wb") as buffer:
            total_bytes = 0
            while chunk := file.file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Uploaded file exceeds the maximum allowed size.")
                buffer.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except Exception as e:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    # Validate file
    is_valid, msg, details = validate_csv(destination)
    if not is_valid:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail={"message": msg, "details": details})

    # Record in database
    record_dataset(
        filename=original_name,
        filepath=str(destination),
        row_count=details["row_count"],
        column_count=details["column_count"],
        columns=details["columns"],
    )

    return DatasetUploadResponse(
        status="success",
        message="CSV successfully validated and registered.",
        filename=original_name,
        filepath=str(destination),
        row_count=details["row_count"],
        column_count=details["column_count"],
        columns=details["columns"],
        missing_values=details["missing_values"],
        preview=details["preview"],
    )
