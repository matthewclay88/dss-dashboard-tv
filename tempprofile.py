"""
Mount Mansfield Observed Slope Profile - Proof of Concept
----------------------------------------------------------

Pulls the latest temperature and wind observations from api.weather.gov,
estimates pressure at each station elevation, and plots a MetPy Skew-T
with temperature observations and wind barbs.

V1 intentionally does NOT include:
- Dewpoint
- Wet bulb
- LCL
- Precipitation type
- Freezing-level calculations
- Inversion diagnostics

Requirements:
    pip install requests numpy matplotlib metpy
"""

import numpy as np
import requests
import matplotlib.pyplot as plt

import metpy.calc as mpcalc
from metpy.units import units
from metpy.plots import SkewT


# ---------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------

STATIONS = {
    "KBTV": 330,
    "D0383": 781,
    "E6664": 958,
    "A3150": 1293,
    "UVM05": 1309,
    "UVM06": 2877,
    # MMNV1 will be added separately if api.weather.gov doesn't provide it
}

# NWS asks API users to identify their application.
HEADERS = {
    "User-Agent": "MountMansfieldPseudoSounding/1.0",
    "Accept": "application/geo+json",
}

API_BASE = "https://api.weather.gov"


# ---------------------------------------------------------------------
# 2. HELPER: READ NWS QUANTITATIVE VALUE
# ---------------------------------------------------------------------

def qv_value(properties, name):
    """
    Return the raw numeric value from an api.weather.gov QuantitativeValue.

    Example:
        properties["temperature"] =
        {
            "unitCode": "wmoUnit:degC",
            "value": 20.0,
            ...
        }
    """

    item = properties.get(name)

    if not item:
        return None

    return item.get("value")


# ---------------------------------------------------------------------
# 3. FETCH LATEST OBSERVATION
# ---------------------------------------------------------------------

def fetch_station(stid):
    """
    Pull latest observation for one station from api.weather.gov.
    """

    url = f"{API_BASE}/stations/{stid}/observations/latest"

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()
        p = data.get("properties", {})

        return {
            "stid": stid,

            # api.weather.gov observation temperature is normally degC
            "temperature_C": qv_value(p, "temperature"),

            # Wind direction in degrees
            "wind_direction_deg": qv_value(p, "windDirection"),

            # Wind speed normally returned in km/h
            "wind_speed_kmh": qv_value(p, "windSpeed"),

            # Pressure fields, if available
            "barometric_pressure_Pa": qv_value(
                p, "barometricPressure"
            ),

            "sea_level_pressure_Pa": qv_value(
                p, "seaLevelPressure"
            ),

            "timestamp": p.get("timestamp"),
        }

    except requests.RequestException as exc:

        print(f"ERROR retrieving {stid}: {exc}")
        return None


# ---------------------------------------------------------------------
# 4. FETCH ALL STATIONS
# ---------------------------------------------------------------------

def fetch_all():

    observations = {}

    print("\nFetching latest observations...\n")

    for stid in STATIONS:

        obs = fetch_station(stid)

        if obs is None:
            continue

        observations[stid] = obs

        print(
            f"{stid:6s}  "
            f"T={str(obs['temperature_C']):>7s} C   "
            f"Wind={str(obs['wind_direction_deg']):>6s}/"
            f"{str(obs['wind_speed_kmh']):>6s} km/h   "
            f"{obs['timestamp']}"
        )

    return observations


# ---------------------------------------------------------------------
# 5. BUILD TEMPERATURE PROFILE
# ---------------------------------------------------------------------

def build_profile(observations):

    profile = []

    for stid, elevation_ft in STATIONS.items():

        obs = observations.get(stid)

        if obs is None:
            print(f"Skipping {stid}: no observation")
            continue

        temperature = obs.get("temperature_C")

        if temperature is None:
            print(f"Skipping {stid}: no temperature")
            continue

        profile.append({
            "stid": stid,
            "elevation_ft": elevation_ft,
            "temperature_C": temperature,
            "wind_direction_deg": obs.get("wind_direction_deg"),
            "wind_speed_kmh": obs.get("wind_speed_kmh"),
            "barometric_pressure_Pa":
                obs.get("barometric_pressure_Pa"),
            "timestamp": obs.get("timestamp"),
        })

    profile.sort(key=lambda x: x["elevation_ft"])

    if not profile:
        raise RuntimeError("No valid station observations available.")

    return profile


# ---------------------------------------------------------------------
# 6. ESTIMATE PRESSURE
# ---------------------------------------------------------------------

def calculate_pressures(profile):
    """
    Estimate pressure at each station.

    Preferred anchor:
        observed KBTV barometric pressure

    Fallback:
        standard pressure at KBTV elevation

    Pressure above KBTV is calculated layer-by-layer with the
    hypsometric equation using mean observed layer temperature.
    """

    Rd = 287.05       # J kg-1 K-1
    g = 9.80665       # m s-2

    # --------------------------------------------------------------
    # Find KBTV
    # --------------------------------------------------------------

    kbtv = next(
        (x for x in profile if x["stid"] == "KBTV"),
        None
    )

    if kbtv is None:
        raise RuntimeError(
            "KBTV is required as the pressure anchor."
        )

    # --------------------------------------------------------------
    # Determine KBTV station pressure
    # --------------------------------------------------------------

    if kbtv["barometric_pressure_Pa"] is not None:

        p0_hpa = (
            kbtv["barometric_pressure_Pa"] / 100.0
        )

        print(
            f"\nUsing observed KBTV station pressure: "
            f"{p0_hpa:.1f} hPa"
        )

    else:

        # Proof-of-concept fallback.
        #
        # Standard atmosphere pressure at KBTV elevation.
        # This keeps the plot working even if the API observation
        # does not contain station pressure.

        z_m = kbtv["elevation_ft"] * 0.3048

        p0_hpa = (
            1013.25 *
            (1 - 2.25577e-5 * z_m) ** 5.25588
        )

        print(
            "\nWARNING: KBTV station pressure unavailable."
        )

        print(
            f"Using standard-atmosphere pressure: "
            f"{p0_hpa:.1f} hPa"
        )

    # --------------------------------------------------------------
    # Integrate upward
    # --------------------------------------------------------------

    profile[0]["pressure_hPa"] = p0_hpa

    for i in range(1, len(profile)):

        lower = profile[i - 1]
        upper = profile[i]

        z1 = lower["elevation_ft"] * 0.3048
        z2 = upper["elevation_ft"] * 0.3048

        dz = z2 - z1

        T1_K = lower["temperature_C"] + 273.15
        T2_K = upper["temperature_C"] + 273.15

        Tmean = (T1_K + T2_K) / 2.0

        p1 = lower["pressure_hPa"]

        p2 = p1 * np.exp(
            -(g * dz) / (Rd * Tmean)
        )

        upper["pressure_hPa"] = p2

    return profile


# ---------------------------------------------------------------------
# 7. PREPARE METPY ARRAYS
# ---------------------------------------------------------------------

def make_metpy_arrays(profile):

    pressure = np.array([
        x["pressure_hPa"]
        for x in profile
    ]) * units.hPa

    temperature = np.array([
        x["temperature_C"]
        for x in profile
    ]) * units.degC

    # --------------------------------------------------------------
    # Winds
    # --------------------------------------------------------------

    wind_pressure = []
    wind_speed = []
    wind_direction = []

    for x in profile:

        speed = x["wind_speed_kmh"]
        direction = x["wind_direction_deg"]

        if speed is None or direction is None:
            continue

        wind_pressure.append(
            x["pressure_hPa"]
        )

        wind_speed.append(
            speed
        )

        wind_direction.append(
            direction
        )

    if wind_speed:

        wspd = (
            np.array(wind_speed) *
            units("km/hour")
        ).to("knots")

        wdir = (
            np.array(wind_direction) *
            units.degree
        )

        u, v = mpcalc.wind_components(
            wspd,
            wdir
        )

        wind_pressure = (
            np.array(wind_pressure) *
            units.hPa
        )

    else:

        wind_pressure = None
        u = None
        v = None

    return (
        pressure,
        temperature,
        wind_pressure,
        u,
        v,
    )


# ---------------------------------------------------------------------
# 8. PRINT QC TABLE
# ---------------------------------------------------------------------

def print_profile(profile):

    print("\n")
    print("=" * 82)

    print(
        f"{'STATION':<8}"
        f"{'ELEV(ft)':>10}"
        f"{'PRESSURE':>12}"
        f"{'TEMP(C)':>10}"
        f"{'WIND':>14}"
        f"   TIME"
    )

    print("=" * 82)

    for x in profile:

        speed = x["wind_speed_kmh"]
        direction = x["wind_direction_deg"]

        if speed is not None and direction is not None:

            speed_kt = (
                speed *
                units("km/hour")
            ).to("knots").m

            wind_text = (
                f"{direction:03.0f}/"
                f"{speed_kt:.0f}kt"
            )

        else:

            wind_text = "MISSING"

        print(
            f"{x['stid']:<8}"
            f"{x['elevation_ft']:>10.0f}"
            f"{x['pressure_hPa']:>10.1f} mb"
            f"{x['temperature_C']:>10.1f}"
            f"{wind_text:>14}"
            f"   {x['timestamp']}"
        )

    print("=" * 82)


# ---------------------------------------------------------------------
# 9. PLOT SKEW-T
# ---------------------------------------------------------------------

def plot_skewt(
    profile,
    pressure,
    temperature,
    wind_pressure,
    u,
    v,
):

    fig = plt.figure(
        figsize=(9, 9)
    )

    skew = SkewT(
        fig,
        rotation=45
    )

    # --------------------------------------------------------------
    # Temperature profile
    # --------------------------------------------------------------

    skew.plot(
        pressure,
        temperature,
        color="red",
        linewidth=2.5,
        marker="o",
        markersize=7,
        label="Observed Temperature",
    )

    # --------------------------------------------------------------
    # Wind barbs
    # --------------------------------------------------------------

    if wind_pressure is not None:

        skew.plot_barbs(
            wind_pressure,
            u,
            v,
            xloc=1.0,
        )

    # --------------------------------------------------------------
    # Background
    # --------------------------------------------------------------

    skew.plot_dry_adiabats(
        alpha=0.25
    )

    skew.plot_moist_adiabats(
        alpha=0.20
    )

    skew.plot_mixing_lines(
        alpha=0.15
    )

    # --------------------------------------------------------------
    # Highlight 0 C
    # --------------------------------------------------------------

    skew.ax.axvline(
        0,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )

    # --------------------------------------------------------------
    # Plot limits
    # --------------------------------------------------------------

    bottom_pressure = max(
        1020,
        pressure.max().m + 10
    )

    top_pressure = min(
        840,
        pressure.min().m - 20
    )

    skew.ax.set_ylim(
        bottom_pressure,
        top_pressure,
    )

    skew.ax.set_xlim(
        -30,
        20,
    )

    # --------------------------------------------------------------
    # Station labels
    # --------------------------------------------------------------

    for station, pp, tt in zip(
        profile,
        pressure,
        temperature,
    ):

        skew.ax.annotate(
            station["stid"],
            xy=(
                tt.to("degC").m,
                pp.to("hPa").m,
            ),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            va="center",
        )

    # --------------------------------------------------------------
    # Titles
    # --------------------------------------------------------------

    skew.ax.set_title(
        "Mount Mansfield Observed Slope Profile",
        fontsize=15,
        fontweight="bold",
        loc="left",
    )

    skew.ax.set_title(
        "Temperature + Wind",
        fontsize=10,
        loc="right",
    )

    skew.ax.legend(
        loc="upper left"
    )

    plt.tight_layout()

    output_file = "vt_pseudo_sounding.png"

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    print(
        f"\nSaved Skew-T to: {output_file}"
    )

    plt.show()


# ---------------------------------------------------------------------
# 10. MAIN
# ---------------------------------------------------------------------

def main():

    observations = fetch_all()

    profile = build_profile(
        observations
    )

    profile = calculate_pressures(
        profile
    )

    print_profile(
        profile
    )

    (
        pressure,
        temperature,
        wind_pressure,
        u,
        v,
    ) = make_metpy_arrays(
        profile
    )

    plot_skewt(
        profile,
        pressure,
        temperature,
        wind_pressure,
        u,
        v,
    )


if __name__ == "__main__":
    main()
