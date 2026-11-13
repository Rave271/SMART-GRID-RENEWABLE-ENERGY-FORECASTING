"""Unit tests for smart grid rule-based recommendation engine."""

from app.services.recommender import generate_recommendations


def test_generate_recommendations_surplus_and_deficit():
    """Test recommendation classification on high surplus and deficit intervals."""
    forecast_data = [
        # Surplus interval: generation (180 MW) > demand (100 MW)
        {
            "timestamp": "2024-06-01 13:00:00",
            "predicted_solar": 100.0,
            "predicted_wind": 80.0,
            "predicted_demand": 100.0,
        },
        # Deficit interval: generation (10 MW) < demand (300 MW)
        {
            "timestamp": "2024-06-01 20:00:00",
            "predicted_solar": 0.0,
            "predicted_wind": 10.0,
            "predicted_demand": 300.0,
        },
    ]

    res = generate_recommendations(forecast_data)
    assert res["horizon"] == "2h"
    assert res["max_surplus_mw"] == 80.0
    assert res["max_deficit_mw"] == 290.0
    assert len(res["recommendations"]) >= 2

    # Check alert types
    categories = [r["category"] for r in res["recommendations"]]
    assert "surplus" in categories
    assert "deficit" in categories
