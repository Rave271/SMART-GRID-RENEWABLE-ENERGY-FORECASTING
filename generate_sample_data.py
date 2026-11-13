"""
Smart Grid Renewable Energy Forecasting - Realistic Sample Data Generator
Generates at least 2 years of hourly smart grid data with:
- Solar irradiance and solar power generation (daylight/seasonal patterns)
- Wind speed and wind power generation (aerodynamic turbine power curve)
- Ambient weather: temperature, humidity, wind speed
- Grid electricity demand with diurnal peaks, weekend drops, and HVAC temperature response.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_smart_grid_data(
    start_date: str = "2024-01-01 00:00:00",
    end_date: str = "2025-12-31 23:00:00",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic synthetic 2-year hourly smart grid dataset."""
    np.random.seed(random_seed)

    # 1. Hourly Timestamp Range
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")
    n_hours = len(timestamps)

    if hasattr(timestamps, "day_of_year"):
        day_of_year = timestamps.day_of_year.values
    else:
        day_of_year = timestamps.dayofyear.values

    hour_of_day = timestamps.hour.values

    if hasattr(timestamps, "day_of_week"):
        day_of_week = timestamps.day_of_week.values
    else:
        day_of_week = timestamps.dayofweek.values
    is_weekend = (day_of_week >= 5).astype(int)

    # 2. Ambient Temperature (Seasonal + Diurnal + Weather shock)
    # Seasonal cycle: lowest in Jan (day ~15), highest in July (day ~195)
    seasonal_temp = 18 + 12 * np.sin(2 * np.pi * (day_of_year - 105) / 365.25)
    # Diurnal cycle: min at 05:00, peak at 15:00
    diurnal_temp = 6 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
    # Multi-day weather waves (autoregressive noise)
    noise_temp = np.zeros(n_hours)
    for i in range(1, n_hours):
        noise_temp[i] = 0.96 * noise_temp[i - 1] + np.random.normal(0, 0.6)

    temperature = seasonal_temp + diurnal_temp + noise_temp
    temperature = np.clip(temperature, -5.0, 42.0).round(2)

    # 3. Relative Humidity (%)
    # Inversely correlated with temperature + autoregressive cloud/rain events
    seasonal_humidity = 65 - 15 * np.sin(
        2 * np.pi * (day_of_year - 105) / 365.25
    )
    diurnal_humidity = -15 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
    noise_humidity = np.zeros(n_hours)
    for i in range(1, n_hours):
        noise_humidity[i] = 0.94 * noise_humidity[i - 1] + np.random.normal(
            0, 2.0
        )

    humidity = seasonal_humidity + diurnal_humidity + noise_humidity
    humidity = np.clip(humidity, 15.0, 98.0).round(1)

    # 4. Wind Speed (m/s)
    # Weibull-like distribution with seasonal breeze and short-term turbulence
    seasonal_wind = 6.0 + 1.8 * np.cos(2 * np.pi * day_of_year / 365.25)
    diurnal_wind = 1.0 * np.sin(2 * np.pi * (hour_of_day - 14) / 24)
    ar_wind = np.zeros(n_hours)
    for i in range(1, n_hours):
        ar_wind[i] = 0.90 * ar_wind[i - 1] + np.random.normal(0, 1.1)

    wind_speed = np.abs(seasonal_wind + diurnal_wind + ar_wind)
    wind_speed = np.clip(wind_speed, 0.2, 24.5).round(2)

    # 5. Solar Irradiance (W/m^2)
    # Zero at night. Peaked at noon. Attenuated by season (higher in summer) and humidity (cloud proxy)
    solar_irradiance = np.zeros(n_hours)
    # Daylight hours approx 06:00 to 19:00 with seasonal adjustment
    day_length_factor = 1.0 + 0.25 * np.sin(
        2 * np.pi * (day_of_year - 80) / 365.25
    )

    for i in range(n_hours):
        h = hour_of_day[i]
        # Daylight roughly 6am to 7pm
        if 6 <= h <= 19:
            # Normalized solar zenith angle proxy
            zenith_sin = np.sin(np.pi * (h - 6) / 13)
            if zenith_sin > 0:
                max_clear_sky = (
                    1050 * zenith_sin * (0.85 + 0.15 * day_length_factor[i])
                )
                # Cloud attenuation correlated with high humidity
                hum_diff = max(0.0, float(humidity[i]) - 30.0) / 70.0
                cloud_attenuation = 1.0 - 0.55 * (
                    hum_diff ** 1.5
                ) + np.random.normal(0, 0.05)
                cloud_attenuation = np.clip(cloud_attenuation, 0.08, 1.0)
                irr = max_clear_sky * cloud_attenuation
                solar_irradiance[i] = max(0.0, irr)

    solar_irradiance = solar_irradiance.round(2)

    # 6. Solar Power Generation (MW) - 100 MW Nominal Plant Capacity
    # Power is proportional to irradiance, with mild thermal derating at high panel temp
    nominal_solar_mw = 100.0
    # Efficiency derating: -0.4% per degree above 25°C
    temp_efficiency = 1.0 - np.maximum(0.0, (temperature - 25.0) * 0.004)
    solar_power = (
        (solar_irradiance / 1000.0)
        * nominal_solar_mw
        * temp_efficiency
        * (1.0 + np.random.normal(0, 0.02, n_hours))
    )
    solar_power = np.clip(solar_power, 0.0, nominal_solar_mw).round(2)

    # 7. Wind Power Generation (MW) - 150 MW Rated Wind Farm
    # Aerodynamic Wind Power Curve:
    # Cut-in: 3.0 m/s
    # Rated speed: 12.0 m/s (reaches 150 MW)
    # Cut-out speed: 22.0 m/s (safety shutdown to 0 MW)
    rated_wind_mw = 150.0
    wind_power = np.zeros(n_hours)
    for i in range(n_hours):
        v = wind_speed[i]
        if v < 3.0:
            p = 0.0
        elif v <= 12.0:
            # Cubic power curve between cut-in and rated
            p = rated_wind_mw * (((v - 3.0) / (12.0 - 3.0)) ** 3)
        elif v <= 22.0:
            # Constant rated power output
            p = rated_wind_mw
        else:
            # High-wind storm shutdown
            p = 0.0
        # Turbine wake loss & mechanical turbulence noise
        p = p * (1.0 + np.random.normal(0, 0.03))
        wind_power[i] = max(0.0, min(rated_wind_mw, p))

    wind_power = wind_power.round(2)

    # 8. Electricity Demand (MW)
    # Baseline grid demand: ~350 MW
    # Diurnal peaks: Morning surge (07:00-11:00) and evening cooking/lighting surge (18:00-22:00)
    # Weekend reduction: Industrial activity drops by 18-25%
    # Temperature effect: HVAC load increases when temp < 15°C (heating) or > 24°C (cooling)
    base_demand = 350.0

    # Hourly pattern: double hump
    hourly_demand_mult = 1.0 + 0.22 * np.exp(
        -0.5 * ((hour_of_day - 9.5) / 2.0) ** 2
    ) + 0.35 * np.exp(-0.5 * ((hour_of_day - 20.0) / 2.5) ** 2)
    # Valley late at night (01:00 - 05:00)
    hourly_demand_mult -= 0.15 * np.exp(-0.5 * ((hour_of_day - 3.5) / 2.0) ** 2)

    # Weekend factor
    weekend_factor = 1.0 - 0.20 * is_weekend

    # Seasonal & weather HVAC load (U-shaped temperature response)
    hvac_cooling = np.maximum(0.0, (temperature - 24.0) * 8.5)
    hvac_heating = np.maximum(0.0, (15.0 - temperature) * 6.0)

    # Long-term multi-day autocorrelation noise
    noise_demand = np.zeros(n_hours)
    for i in range(1, n_hours):
        noise_demand[i] = 0.92 * noise_demand[i - 1] + np.random.normal(
            0, 7.0
        )

    electricity_demand = (
        (base_demand * hourly_demand_mult * weekend_factor)
        + hvac_cooling
        + hvac_heating
        + noise_demand
    )
    electricity_demand = np.clip(electricity_demand, 180.0, 750.0).round(2)

    # Assemble complete DataFrame
    df = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "solar_irradiance": solar_irradiance,
            "solar_power_mw": solar_power,
            "wind_power_mw": wind_power,
            "electricity_demand_mw": electricity_demand,
        }
    )

    return df


def main():
    print("Generating 2-year realistic Smart Grid synthetic dataset...")
    df = generate_smart_grid_data()
    print(f"Generated {len(df):,} hourly records from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}.")

    # Target directories
    sample_dir = Path("data/sample")
    raw_dir = Path("data/raw")
    sample_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    sample_path = sample_dir / "smart_grid_sample.csv"
    raw_path = raw_dir / "smart_grid_sample.csv"

    df.to_csv(sample_path, index=False)
    df.to_csv(raw_path, index=False)

    print(f"Dataset successfully saved to:\n  - {sample_path}\n  - {raw_path}")
    print("\nSummary Statistics:")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
