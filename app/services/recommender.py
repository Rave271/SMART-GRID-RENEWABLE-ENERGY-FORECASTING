"""Rule-based smart grid energy utilization and battery dispatch recommender."""

from typing import List, Dict, Any
import numpy as np


def generate_recommendations(
    forecast_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze generation vs demand forecast and generate operational grid dispatch recommendations."""
    if not forecast_data:
        return {
            "summary": {},
            "recommendations": [],
        }

    recommendations = []
    total_renewable_mwh = 0.0
    total_demand_mwh = 0.0
    penetration_rates = []

    max_surplus = 0.0
    max_deficit = 0.0
    peak_surplus_time = None
    peak_deficit_time = None

    for i, item in enumerate(forecast_data):
        timestamp = item["timestamp"]
        solar = float(item.get("predicted_solar", 0.0))
        wind = float(item.get("predicted_wind", 0.0))
        generation = solar + wind
        demand = float(item.get("predicted_demand", 1.0))

        net_balance = generation - demand  # positive = surplus, negative = deficit
        penetration = (generation / demand * 100.0) if demand > 0 else 0.0

        total_renewable_mwh += generation
        total_demand_mwh += demand
        penetration_rates.append(penetration)

        # Track extremes
        if net_balance > max_surplus:
            max_surplus = net_balance
            peak_surplus_time = timestamp

        if net_balance < -max_deficit:
            max_deficit = abs(net_balance)
            peak_deficit_time = timestamp

        # Rule Evaluations
        # 1. Critical Deficit during peak hours
        if net_balance < -80.0:
            recommendations.append({
                "id": len(recommendations) + 1,
                "hour_offset": i,
                "timestamp": timestamp,
                "category": "deficit",
                "severity": "danger",
                "title": f"Critical Energy Deficit ({abs(net_balance):.1f} MW shortfall)",
                "description": f"At {timestamp}, renewable output ({generation:.1f} MW) falls severely below demand ({demand:.1f} MW). Renewable penetration is only {penetration:.1f}%.",
                "action": "Dispatch fast-ramping peaker generation and discharge BESS reserves. Issue automated demand-response signals to shed non-essential commercial loads.",
            })
        # 2. Moderate Deficit
        elif net_balance < -25.0:
            recommendations.append({
                "id": len(recommendations) + 1,
                "hour_offset": i,
                "timestamp": timestamp,
                "category": "deficit",
                "severity": "warning",
                "title": f"Generation Deficit ({abs(net_balance):.1f} MW)",
                "description": f"Demand exceeds clean generation by {abs(net_balance):.1f} MW at {timestamp}.",
                "action": "Schedule spinning reserves or initiate scheduled battery discharge to maintain grid frequency.",
            })
        # 3. High Clean Surplus
        elif net_balance > 120.0:
            recommendations.append({
                "id": len(recommendations) + 1,
                "hour_offset": i,
                "timestamp": timestamp,
                "category": "curtailment",
                "severity": "warning",
                "title": f"Curtailment Alert (+{net_balance:.1f} MW excess)",
                "description": f"Excess energy threatens transmission line thermal limits and local sub-station capacity.",
                "action": "Signal dynamic negative wholesale pricing or curtail variable wind blade pitch / solar inverter output.",
            })
        elif net_balance > 50.0:
            recommendations.append({
                "id": len(recommendations) + 1,
                "hour_offset": i,
                "timestamp": timestamp,
                "category": "surplus",
                "severity": "success",
                "title": f"Clean Energy Surplus (+{net_balance:.1f} MW)",
                "description": f"Green generation exceeds demand at {timestamp}. Renewable penetration reaches {penetration:.1f}%.",
                "action": "Direct surplus clean energy to recharge BESS battery banks, pump storage facilities, and stimulate flexible EV charging loads.",
            })

    # If no major events triggered, add balanced operational note
    if not recommendations:
        recommendations.append({
            "id": 1,
            "hour_offset": 0,
            "timestamp": forecast_data[0]["timestamp"],
            "category": "normal",
            "severity": "info",
            "title": "Optimal Grid Balance",
            "description": "Renewable energy production closely matches projected grid demand across the forecast window.",
            "action": "Maintain normal baseline operations and preserve standard spinning reserve margins.",
        })

    avg_penetration = float(np.mean(penetration_rates)) if penetration_rates else 0.0

    return {
        "horizon": f"{len(forecast_data)}h",
        "peak_surplus_time": peak_surplus_time,
        "peak_deficit_time": peak_deficit_time,
        "max_surplus_mw": round(max_surplus, 1),
        "max_deficit_mw": round(max_deficit, 1),
        "total_renewable_energy_mwh": round(total_renewable_mwh, 1),
        "total_demand_mwh": round(total_demand_mwh, 1),
        "avg_renewable_penetration_pct": round(avg_penetration, 1),
        "recommendations": recommendations[:12],  # Return the top key operational alerts
    }
