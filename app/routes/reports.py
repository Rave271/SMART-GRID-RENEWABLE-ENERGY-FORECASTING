"""Route for exporting forecast reports as CSV or PDF."""

from datetime import datetime
from fastapi import APIRouter, Query, HTTPException, Depends
from fastapi.responses import FileResponse

from app.config import REPORTS_DIR, require_api_key, rate_limit_report
from app.routes.predict import get_predictions
from app.services.report_generator import export_forecast_csv, export_forecast_pdf

router = APIRouter(prefix="/api", tags=["Reports Export"], dependencies=[Depends(require_api_key), Depends(rate_limit_report)])


@router.get("/report")
async def export_report(
    format: str = Query("pdf", description="Export format: 'csv' or 'pdf'"),
    horizon: str = Query("24h", description="Horizon: '24h' or '7d'"),
    model_type: str = Query("xgboost", description="Model architecture: 'xgboost' or 'mlp'"),
):
    """Export prediction forecast report as CSV or executive PDF with embedded visualizations."""
    fmt = format.lower().strip()
    if fmt not in ("csv", "pdf"):
        raise HTTPException(status_code=400, detail="Invalid format. Must be 'csv' or 'pdf'.")

    # Generate predictions
    pred_res = await get_predictions(horizon=horizon, target="all", model_type=model_type)
    predictions = pred_res.predictions
    summary = pred_res.summary

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    if fmt == "csv":
        csv_path = REPORTS_DIR / f"smart_grid_forecast_{horizon}_{timestamp_str}.csv"
        export_forecast_csv(predictions, csv_path)
        return FileResponse(
            path=str(csv_path),
            filename=f"smart_grid_forecast_{horizon}.csv",
            media_type="text/csv",
        )
    else:
        pdf_path = REPORTS_DIR / f"smart_grid_report_{horizon}_{timestamp_str}.pdf"
        export_forecast_pdf(
            forecast_data=predictions,
            summary_stats=summary,
            model_type=model_type,
            output_path=pdf_path,
        )
        return FileResponse(
            path=str(pdf_path),
            filename=f"smart_grid_forecast_report_{horizon}.pdf",
            media_type="application/pdf",
        )
