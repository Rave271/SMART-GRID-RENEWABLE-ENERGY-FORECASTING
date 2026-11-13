"""Server-side report generator for CSV and PDF reports with Matplotlib visualizations."""

import io
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from app.config import REPORTS_DIR


def generate_forecast_chart(forecast_data: List[Dict[str, Any]], output_path: Path):
    """Render high-resolution Matplotlib chart for the report."""
    timestamps = [d["timestamp"].split(" ")[-1][:5] if " " in d["timestamp"] else d["timestamp"] for d in forecast_data]
    # If horizon is large (e.g. 7d = 168h), sample every 4 or 6 hours for chart readability
    if len(forecast_data) > 36:
        step = len(forecast_data) // 24
        subset = forecast_data[::step]
        timestamps = [d["timestamp"][5:16] for d in subset]
    else:
        subset = forecast_data

    solar = [float(d.get("predicted_solar", 0.0)) for d in subset]
    wind = [float(d.get("predicted_wind", 0.0)) for d in subset]
    gen = [s + w for s, w in zip(solar, wind)]
    demand = [float(d.get("predicted_demand", 0.0)) for d in subset]
    balance = [g - dem for g, dem in zip(gen, demand)]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # Top plot: Generation vs Demand
    x_indices = np.arange(len(timestamps))
    ax1.plot(x_indices, demand, label="Demand (MW)", color="#e63946", linewidth=2.5, marker="o", markersize=4)
    ax1.plot(x_indices, gen, label="Total Renewable Generation (MW)", color="#2a9d8f", linewidth=2.5, marker="s", markersize=4)
    ax1.fill_between(x_indices, 0, solar, label="Solar Portion (MW)", color="#f4a261", alpha=0.35)
    ax1.fill_between(x_indices, solar, gen, label="Wind Portion (MW)", color="#457b9d", alpha=0.35)

    ax1.set_ylabel("Power (MW)", fontsize=11, fontweight="bold")
    ax1.set_title("Smart Grid Power Forecast: Renewable Generation vs. Demand", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="upper right", frameon=True, facecolor="#ffffff", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Bottom plot: Surplus / Deficit balance
    colors_bar = ["#2a9d8f" if b >= 0 else "#e63946" for b in balance]
    ax2.bar(x_indices, balance, color=colors_bar, width=0.6, alpha=0.85, label="Net Grid Balance (MW)")
    ax2.axhline(0, color="#1d3557", linewidth=1.2, linestyle="-")
    ax2.set_ylabel("Balance (MW)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Forecast Timeline", fontsize=11, fontweight="bold")
    ax2.set_xticks(x_indices[:: max(1, len(x_indices) // 10)])
    ax2.set_xticklabels([timestamps[i] for i in range(0, len(timestamps), max(1, len(x_indices) // 10))], rotation=30, ha="right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_forecast_csv(forecast_data: List[Dict[str, Any]], output_path: Path) -> Path:
    """Export predictions to structured CSV file."""
    df = pd.DataFrame(forecast_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def export_forecast_pdf(
    forecast_data: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
    model_type: str,
    output_path: Path,
) -> Path:
    """Generate executive PDF report containing KPI cards, Matplotlib charts, and hourly forecast tables."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate chart image
    chart_image_path = output_path.parent / f"chart_{output_path.stem}.png"
    generate_forecast_chart(forecast_data, chart_image_path)

    # 2. Document Setup
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1b4332"),
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#555555"),
        spaceAfter=14,
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2d6a4f"),
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=8,
    )

    elements = []

    # Title & Subtitle
    elements.append(Paragraph("Smart Grid Renewable Energy Forecasting Report", title_style))
    elements.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Model: {model_type.upper()} | Forecast Horizon: {len(forecast_data)} Hours",
            subtitle_style,
        )
    )
    elements.append(Spacer(1, 10))

    # KPI Summary Table
    elements.append(Paragraph("1. Executive Summary & Grid Balance Metrics", section_style))
    kpi_data = [
        ["Total Renewable Generation", f"{summary_stats.get('total_renewable_mwh', 0):,.1f} MWh", "Peak Clean Surplus", f"{summary_stats.get('max_surplus_mw', 0):,.1f} MW"],
        ["Total Electricity Demand", f"{summary_stats.get('total_demand_mwh', 0):,.1f} MWh", "Peak Deficit Risk", f"{summary_stats.get('max_deficit_mw', 0):,.1f} MW"],
        ["Avg Renewable Penetration", f"{summary_stats.get('avg_penetration_pct', 0):.1f}%", "Peak Shortfall Window", f"{summary_stats.get('peak_deficit_time', 'N/A')}"],
    ]
    t_kpi = Table(kpi_data, colWidths=[2.2 * inch, 1.4 * inch, 2.0 * inch, 1.6 * inch])
    t_kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#212529")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
    ]))
    elements.append(t_kpi)
    elements.append(Spacer(1, 14))

    # Chart Section
    elements.append(Paragraph("2. Generation vs Demand Timeline Forecast", section_style))
    elements.append(RLImage(str(chart_image_path), width=7.2 * inch, height=4.1 * inch))
    elements.append(Spacer(1, 14))

    # Detailed Forecast Table
    elements.append(Paragraph("3. Hourly Dispatch & Forecast Schedule (Preview)", section_style))
    table_headers = ["Timestamp", "Solar (MW)", "Wind (MW)", "Total Gen", "Demand", "Net Balance"]
    table_rows = [table_headers]

    for item in forecast_data[:18]:  # Include first 18 intervals
        solar = float(item.get("predicted_solar", 0.0))
        wind = float(item.get("predicted_wind", 0.0))
        gen = solar + wind
        dem = float(item.get("predicted_demand", 0.0))
        bal = gen - dem
        table_rows.append([
            item.get("timestamp", "")[-8:] if " " in item.get("timestamp", "") else item.get("timestamp", ""),
            f"{solar:.1f}",
            f"{wind:.1f}",
            f"{gen:.1f}",
            f"{dem:.1f}",
            f"{'+' if bal >= 0 else ''}{bal:.1f}",
        ])

    t_schedule = Table(table_rows, colWidths=[1.8 * inch, 1.0 * inch, 1.0 * inch, 1.1 * inch, 1.1 * inch, 1.2 * inch])
    t_schedule.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d6a4f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8f9fa")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_schedule)

    # Build document
    doc.build(elements)

    # Clean up temporary chart file
    if chart_image_path.exists():
        try:
            chart_image_path.unlink()
        except Exception:
            pass

    return output_path
