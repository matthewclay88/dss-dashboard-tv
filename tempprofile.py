"""
Mount Mansfield Observed Slope Profile - Proof of Concept
----------------------------------------------------------

Builds a temperature + wind pseudo-sounding from surface observations
between Burlington and the Mount Mansfield summit.

Data sources
------------
NWS API:
    KBTV
    D0383
    E6664
    A3150
    UVM05
    UVM06

MMNV1:
    Temperature -> NWS BTV RR2
    Wind        -> IEM archived RRSBTV SHEF product

The profile:
    1. Retrieves the latest usable temperature/wind observations.
    2. Anchors pressure to observed KBTV station pressure when available.
    3. Uses the hypsometric equation to estimate pressure at each elevation.
    4. Plots temperature and wind on a MetPy Skew-T.

V1 intentionally does NOT calculate:
    dewpoint
    wet bulb
    LCL
    precipitation type
    freezing level
    inversion diagnostics

Requirements:
    pip install requests numpy matplotlib metpy
"""

import re
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
import matplotlib

# GitHub Actions is headless.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import metpy.calc as mpcalc
from metpy.units import units
from metpy.plots import SkewT


# =====================================================================
# 1. CONFIGURATION
# =====================================================================

STATIONS = {
    "KBTV": 330,
    "D0383": 781,
    "E6664": 958,
    "A3150": 1293,
    "UVM05": 1309,
    "UVM06": 2877,
    "MMNV1": 3891,
}

NWS_STATIONS = [
    "KBTV",
    "D0383",
    "E6664",
    "A3150",
    "UVM05",
    "UVM06",
]

API_BASE = "https://api.weather.gov"

RR2_URL = (
    "https://forecast.weather.gov/product.php"
    "?site=NWS&product=RR2&issuedby=BTV"
)

RRS_URL = (
    "https://mesonet.agron.iastate.edu/wx/afos/p.php"
    "?pil=RRSBTV&e={timestamp}"
)

HEADERS = {
    "User-Agent": (
        "MountMansfieldPseudoSounding/1.1 "
        "(weather research/visualization)"
    ),
    "Accept": "application/geo+json",
}

TEXT_HEADERS = {
    "User-Agent": (
        "MountMansfieldPseudoSounding/1.1 "
        "(weather research/visualization)"
    ),
}

# Number of recent NWS observations to inspect.
NWS_OB_LIMIT = 20

# Number of hourly RRS products to search backward.
RRS_LOOKBACK_HOURS = 6

OUTPUT_FILE = "vt_pseudo_sounding.png"


# =====================================================================
# 2. GENERAL HELPERS
# =====================================================================

def qv_value(properties, name):
    """
    Extract numeric value from an api.weather.gov QuantitativeValue.
    """

    item = properties.get(name)

    if not isinstance(item, dict):
        return None

    return item.get("value")


def parse_iso_time(value):
    """
    Convert ISO timestamp to datetime.
    """

    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError:
        return None


def observation_age_minutes(timestamp):
    """
    Return observation age in minutes.
    """

    dt = parse_iso_time(timestamp)

    if dt is None:
        return None

    now = datetime.now(timezone.utc)

    return (now - dt).total_seconds() / 60.0


def fmt(value, decimals=1):
    """
    Safe formatting helper.
    """

    if value is None:
        return "MISSING"

    return f"{value:.{decimals}f}"


# =====================================================================
# 3. NWS API OBSERVATIONS
# =====================================================================

def fetch_nws_recent(stid):
    """
    Retrieve recent observations for one NWS/MADIS station.

    Instead of /latest, inspect several observations so an incomplete
    newest record does not hide a valid temperature/wind observation.
    """

    url = f"{API_BASE}/stations/{stid}/observations"

    params = {
        "limit": NWS_OB_LIMIT,
    }

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        return response.json().get("features", [])

    except requests.RequestException as exc:

        print(
            f"ERROR retrieving NWS observations for {stid}: {exc}"
        )

        return []


def extract_latest_nws_values(stid):
    """
    Search recent NWS observations.

    Temperature and wind do NOT have to come from exactly the same
    observation record.

    Returns the newest available:
        temperature
        wind speed
        wind direction
        pressure

    while preserving timestamps.
    """

    features = fetch_nws_recent(stid)

    if not features:
        return None

    temperature = None
    temperature_time = None

    wind_speed = None
    wind_direction = None
    wind_time = None

    pressure_pa = None
    pressure_time = None

    newest_time = None

    for feature in features:

        p = feature.get("properties", {})

        timestamp = p.get("timestamp")

        if newest_time is None and timestamp:
            newest_time = timestamp

        # -------------------------------------------------------------
        # Temperature
        # -------------------------------------------------------------

        temp = qv_value(p, "temperature")

        if temperature is None and temp is not None:

            temperature = temp
            temperature_time = timestamp

        # -------------------------------------------------------------
        # Wind
        # -------------------------------------------------------------

        speed = qv_value(p, "windSpeed")
        direction = qv_value(p, "windDirection")

        if (
            wind_speed is None
            and speed is not None
            and direction is not None
        ):

            wind_speed = speed
            wind_direction = direction
            wind_time = timestamp

        # -------------------------------------------------------------
        # Station pressure
        # -------------------------------------------------------------

        pressure = qv_value(p, "barometricPressure")

        if pressure_pa is None and pressure is not None:

            pressure_pa = pressure
            pressure_time = timestamp

        # Stop once everything has been found.

        if (
            temperature is not None
            and wind_speed is not None
            and pressure_pa is not None
        ):
            break

    return {
        "stid": stid,
        "temperature_C": temperature,
        "temperature_time": temperature_time,
        "wind_speed_kmh": wind_speed,
        "wind_direction_deg": wind_direction,
        "wind_time": wind_time,
        "barometric_pressure_Pa": pressure_pa,
        "pressure_time": pressure_time,
        "timestamp": temperature_time or newest_time,
        "source": "NWS API",
    }


# =====================================================================
# 4. MMNV1 TEMPERATURE - RR2
# =====================================================================

def fetch_mmvn1_temperature():
    """
    Retrieve MMNV1 temperature from the current BTV RR2 product.

    Expected SHEF form resembles:

        .A MMNV1 260724 Z DH1803/TAIRGZZ 63.7

    TAIR is air temperature in Fahrenheit.
    """

    try:

        response = requests.get(
            RR2_URL,
            headers=TEXT_HEADERS,
            timeout=30,
        )

        response.raise_for_status()

        text = response.text

    except requests.RequestException as exc:

        print(f"ERROR retrieving RR2: {exc}")

        return None, None

    # Remove HTML tags if product page was returned as HTML.

    text_clean = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text_clean = (
        text_clean
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
    )

    # Example:
    #
    # .A MMNV1 260724 Z DH1803/TAIRGZZ 63.7

    pattern = re.compile(
        r"\.A\s+MMNV1\s+"
        r"(\d{6})\s+Z\s+"
        r"DH(\d{4})"
        r".*?"
        r"TAIR\w*\s+"
        r"(-?\d+(?:\.\d+)?)",
        re.IGNORECASE | re.DOTALL,
    )

    matches = pattern.findall(text_clean)

    if not matches:

        print(
            "WARNING: Could not find MMNV1 temperature in RR2."
        )

        return None, None

    date_code, hhmm, temp_f = matches[-1]

    try:

        dt = datetime.strptime(
            date_code + hhmm,
            "%y%m%d%H%M"
        ).replace(tzinfo=timezone.utc)

    except ValueError:

        dt = None

    temp_f = float(temp_f)

    temp_c = (temp_f - 32.0) * 5.0 / 9.0

    timestamp = (
        dt.isoformat()
        if dt
        else None
    )

    return temp_c, timestamp


# =====================================================================
# 5. MMNV1 WIND - RRSBTV
# =====================================================================

def candidate_rrs_times():
    """
    Generate candidate IEM RRS archive timestamps.

    RRSBTV normally updates around HH:12.

    We search backward rather than assuming the current hour's product
    has already arrived.
    """

    now = datetime.now(timezone.utc)

    candidates = []

    # Start at current hour :12.

    base = now.replace(
        minute=12,
        second=0,
        microsecond=0,
    )

    # If current time is before :12, start with previous hour.

    if now.minute < 12:
        base -= timedelta(hours=1)

    for hours_back in range(RRS_LOOKBACK_HOURS):

        dt = base - timedelta(hours=hours_back)

        candidates.append(
            dt.strftime("%Y%m%d%H%M")
        )

    return candidates


def fetch_rrs_product(timestamp):
    """
    Fetch one archived RRSBTV product from IEM.
    """

    url = RRS_URL.format(
        timestamp=timestamp
    )

    try:

        response = requests.get(
            url,
            headers=TEXT_HEADERS,
            timeout=30,
        )

        response.raise_for_status()

        return response.text

    except requests.RequestException:

        return None


def extract_shef_series(text, parameter):
    """
    Extract an MMNV1 SHEF .E series from RRSBTV.

    Example conceptually:

        .E MMNV1 20260724 DH1610/USIRG/DIN5/
        4.1/4.5/5.0/...

    parameter:
        USIRG -> wind speed
        UDIRG -> wind direction

    Returns:
        [(datetime, value), ...]
    """

    if not text:
        return []

    # Strip HTML because the IEM page can wrap the AFOS text.

    clean = re.sub(
        r"<[^>]+>",
        "\n",
        text
    )

    clean = (
        clean
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
    )

    # Normalize whitespace without destroying line structure.

    lines = [
        line.strip()
        for line in clean.splitlines()
        if line.strip()
    ]

    series = []

    i = 0

    while i < len(lines):

        line = lines[i]

        if (
            line.startswith(".E MMNV1")
            and parameter in line
        ):

            # Header example:
            #
            # .E MMNV1 20260724 DH1610/USIRG/DIN5/

            header = line

            # Some SHEF values may continue onto following lines.

            value_lines = []

            # Capture anything after DIN5/ on the header.

            match = re.search(
                rf"\.E\s+MMNV1\s+"
                rf"(\d{{8}})\s+"
                rf"DH(\d{{4}})/"
                rf"{parameter}/DIN(\d+)/?(.*)",
                header,
                re.IGNORECASE,
            )

            if not match:

                i += 1
                continue

            date_code = match.group(1)
            hhmm = match.group(2)
            interval_minutes = int(
                match.group(3)
            )

            remainder = match.group(4)

            if remainder:
                value_lines.append(remainder)

            # Continue until next SHEF record.

            j = i + 1

            while j < len(lines):

                next_line = lines[j]

                if next_line.startswith("."):
                    break

                value_lines.append(next_line)

                j += 1

            values_text = "/".join(value_lines)

            raw_values = [
                x.strip()
                for x in values_text.split("/")
                if x.strip()
            ]

            try:

                start = datetime.strptime(
                    date_code + hhmm,
                    "%Y%m%d%H%M"
                ).replace(
                    tzinfo=timezone.utc
                )

            except ValueError:

                i = j
                continue

            for n, raw_value in enumerate(raw_values):

                # SHEF missing values may appear as M.

                if raw_value.upper() in {
                    "M",
                    "MM",
                    "MSG",
                }:
                    continue

                # Keep first numeric token.

                numeric = re.match(
                    r"(-?\d+(?:\.\d+)?)",
                    raw_value
                )

                if not numeric:
                    continue

                value = float(
                    numeric.group(1)
                )

                obs_time = (
                    start
                    + timedelta(
                        minutes=n * interval_minutes
                    )
                )

                series.append(
                    (
                        obs_time,
                        value
                    )
                )

            i = j
            continue

        i += 1

    return series


def fetch_mmvn1_wind():
    """
    Search recent RRSBTV products and retrieve the latest MMNV1 wind
    observation containing both speed and direction.
    """

    for product_time in candidate_rrs_times():

        text = fetch_rrs_product(
            product_time
        )

        if not text:
            continue

        speed_series = extract_shef_series(
            text,
            "USIRG"
        )

        direction_series = extract_shef_series(
            text,
            "UDIRG"
        )

        if not speed_series or not direction_series:
            continue

        # Convert to timestamp dictionaries.

        speeds = {
            dt: value
            for dt, value in speed_series
        }

        directions = {
            dt: value
            for dt, value in direction_series
        }

        common_times = sorted(
            set(speeds)
            & set(directions),
            reverse=True,
        )

        if not common_times:
            continue

        latest_time = common_times[0]

        speed = speeds[latest_time]
        direction = directions[latest_time]

        # -------------------------------------------------------------
        # IMPORTANT:
        #
        # SHEF USIRG is treated here as wind speed in mph.
        #
        # Convert mph -> km/h so the normalized observation object
        # matches the NWS API station data.
        # -------------------------------------------------------------

        speed_kmh = speed * 1.609344

        return (
            speed_kmh,
            direction,
            latest_time.isoformat(),
            product_time,
        )

    print(
        "WARNING: Could not find MMNV1 wind in recent RRS products."
    )

    return None, None, None, None


# =====================================================================
# 6. BUILD MMNV1 OBSERVATION
# =====================================================================

def fetch_mmvn1():
    """
    Assemble MMNV1 observation from RR2 + RRS.
    """

    print(
        "\nRetrieving MMNV1 from RR2/RRS..."
    )

    temp_c, temp_time = (
        fetch_mmvn1_temperature()
    )

    (
        wind_speed_kmh,
        wind_direction,
        wind_time,
        rrs_product,
    ) = fetch_mmvn1_wind()

    if temp_c is None:

        print(
            "  MMNV1 temperature: MISSING"
        )

    else:

        print(
            f"  MMNV1 temperature: "
            f"{temp_c:.1f} C "
            f"({temp_time})"
        )

    if wind_speed_kmh is None:

        print(
            "  MMNV1 wind: MISSING"
        )

    else:

        wind_kt = (
            wind_speed_kmh
            * units("km/hour")
        ).to("knots").m

        print(
            f"  MMNV1 wind: "
            f"{wind_direction:.0f}/"
            f"{wind_kt:.0f} kt "
            f"({wind_time})"
        )

        print(
            f"  RRS product used: "
            f"{rrs_product}"
        )

    return {
        "stid": "MMNV1",
        "temperature_C": temp_c,
        "temperature_time": temp_time,
        "wind_speed_kmh": wind_speed_kmh,
        "wind_direction_deg": wind_direction,
        "wind_time": wind_time,
        "barometric_pressure_Pa": None,
        "pressure_time": None,
        "timestamp": temp_time,
        "source": "RR2 + RRSBTV",
    }


# =====================================================================
# 7. FETCH ALL OBSERVATIONS
# =====================================================================

def fetch_all():
    """
    Retrieve every station and normalize into a common dictionary.
    """

    observations = {}

    print()
    print("=" * 70)
    print("FETCHING LATEST OBSERVATIONS")
    print("=" * 70)

    for stid in NWS_STATIONS:

        print(
            f"\nFetching {stid}..."
        )

        obs = extract_latest_nws_values(
            stid
        )

        if obs is None:

            print(
                f"  {stid}: NO DATA"
            )

            continue

        observations[stid] = obs

        temp = obs["temperature_C"]
        speed = obs["wind_speed_kmh"]
        direction = obs[
            "wind_direction_deg"
        ]

        print(
            f"  Temperature: "
            f"{fmt(temp)} C"
        )

        if (
            speed is not None
            and direction is not None
        ):

            speed_kt = (
                speed
                * units("km/hour")
            ).to("knots").m

            print(
                f"  Wind: "
                f"{direction:.0f}/"
                f"{speed_kt:.0f} kt"
            )

        else:

            print(
                "  Wind: MISSING"
            )

        print(
            f"  Temp time: "
            f"{obs['temperature_time']}"
        )

        print(
            f"  Wind time: "
            f"{obs['wind_time']}"
        )

    observations["MMNV1"] = (
        fetch_mmvn1()
    )

    return observations


# =====================================================================
# 8. BUILD PROFILE
# =====================================================================

def build_profile(observations):
    """
    Convert normalized observations into elevation-sorted profile.
    """

    profile = []

    for stid, elevation_ft in STATIONS.items():

        obs = observations.get(stid)

        if not obs:

            print(
                f"Skipping {stid}: "
                f"no observation."
            )

            continue

        temperature = obs.get(
            "temperature_C"
        )

        if temperature is None:

            print(
                f"Skipping {stid}: "
                f"no usable temperature."
            )

            continue

        profile.append({
            "stid": stid,
            "elevation_ft": elevation_ft,
            "temperature_C": temperature,
            "temperature_time":
                obs.get("temperature_time"),
            "wind_speed_kmh":
                obs.get("wind_speed_kmh"),
            "wind_direction_deg":
                obs.get("wind_direction_deg"),
            "wind_time":
                obs.get("wind_time"),
            "barometric_pressure_Pa":
                obs.get(
                    "barometric_pressure_Pa"
                ),
            "source":
                obs.get("source"),
        })

    profile.sort(
        key=lambda x:
        x["elevation_ft"]
    )

    if not profile:

        raise RuntimeError(
            "No valid observations available."
        )

    return profile


# =====================================================================
# 9. PRESSURE PROFILE
# =====================================================================

def calculate_pressures(profile):
    """
    Anchor pressure to KBTV and integrate upward with the
    hypsometric equation.

    Uses mean observed layer temperature.
    """

    Rd = 287.05
    g = 9.80665

    kbtv = next(
        (
            x
            for x in profile
            if x["stid"] == "KBTV"
        ),
        None,
    )

    if kbtv is None:

        raise RuntimeError(
            "KBTV is required as "
            "the pressure anchor."
        )

    pressure_pa = (
        kbtv.get(
            "barometric_pressure_Pa"
        )
    )

    if pressure_pa is not None:

        p0_hpa = (
            pressure_pa / 100.0
        )

        print(
            "\nUsing observed KBTV "
            f"station pressure: "
            f"{p0_hpa:.1f} hPa"
        )

    else:

        z_m = (
            kbtv["elevation_ft"]
            * 0.3048
        )

        p0_hpa = (
            1013.25
            * (
                1
                - 2.25577e-5
                * z_m
            )
            ** 5.25588
        )

        print(
            "\nWARNING: KBTV observed "
            "station pressure unavailable."
        )

        print(
            "Using standard-atmosphere "
            f"pressure: {p0_hpa:.1f} hPa"
        )

    # KBTV should be lowest profile station.

    profile[0][
        "pressure_hPa"
    ] = p0_hpa

    for i in range(
        1,
        len(profile)
    ):

        lower = profile[i - 1]
        upper = profile[i]

        z1 = (
            lower["elevation_ft"]
            * 0.3048
        )

        z2 = (
            upper["elevation_ft"]
            * 0.3048
        )

        dz = z2 - z1

        T1_K = (
            lower["temperature_C"]
            + 273.15
        )

        T2_K = (
            upper["temperature_C"]
            + 273.15
        )

        Tmean = (
            T1_K + T2_K
        ) / 2.0

        p1 = lower[
            "pressure_hPa"
        ]

        p2 = (
            p1
            * np.exp(
                -(g * dz)
                / (Rd * Tmean)
            )
        )

        upper[
            "pressure_hPa"
        ] = p2

    return profile


# =====================================================================
# 10. METPY ARRAYS
# =====================================================================

def make_metpy_arrays(profile):

    pressure = np.array([
        x["pressure_hPa"]
        for x in profile
    ]) * units.hPa

    temperature = np.array([
        x["temperature_C"]
        for x in profile
    ]) * units.degC

    wind_pressure = []
    wind_speed = []
    wind_direction = []

    for x in profile:

        speed = x[
            "wind_speed_kmh"
        ]

        direction = x[
            "wind_direction_deg"
        ]

        if (
            speed is None
            or direction is None
        ):
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
            np.array(wind_speed)
            * units("km/hour")
        ).to("knots")

        wdir = (
            np.array(
                wind_direction
            )
            * units.degree
        )

        u, v = (
            mpcalc.wind_components(
                wspd,
                wdir,
            )
        )

        wind_pressure = (
            np.array(
                wind_pressure
            )
            * units.hPa
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


# =====================================================================
# 11. QC TABLE
# =====================================================================

def print_profile(profile):

    print()
    print("=" * 112)

    print(
        f"{'STATION':<8}"
        f"{'ELEV':>8}"
        f"{'PRES':>10}"
        f"{'TEMP':>9}"
        f"{'WIND':>13}"
        f"{'TEMP AGE':>11}"
        f"{'WIND AGE':>11}"
        f"   SOURCE"
    )

    print("=" * 112)

    for x in profile:

        speed = x[
            "wind_speed_kmh"
        ]

        direction = x[
            "wind_direction_deg"
        ]

        if (
            speed is not None
            and direction is not None
        ):

            speed_kt = (
                speed
                * units("km/hour")
            ).to("knots").m

            wind_text = (
                f"{direction:03.0f}/"
                f"{speed_kt:.0f}kt"
            )

        else:

            wind_text = "MISSING"

        temp_age = (
            observation_age_minutes(
                x["temperature_time"]
            )
        )

        wind_age = (
            observation_age_minutes(
                x["wind_time"]
            )
        )

        temp_age_text = (
            f"{temp_age:.0f}m"
            if temp_age is not None
            else "--"
        )

        wind_age_text = (
            f"{wind_age:.0f}m"
            if wind_age is not None
            else "--"
        )

        print(
            f"{x['stid']:<8}"
            f"{x['elevation_ft']:>7.0f}'"
            f"{x['pressure_hPa']:>9.1f}"
            f"{x['temperature_C']:>9.1f}"
            f"{wind_text:>13}"
            f"{temp_age_text:>11}"
            f"{wind_age_text:>11}"
            f"   {x['source']}"
        )

    print("=" * 112)

# =====================================================================
# 12. PLOT
# =====================================================================

def plot_skewt(
    profile,
    pressure,
    temperature,
    wind_pressure,
    u,
    v,
):

    # --------------------------------------------------------------
    # Square figure
    # --------------------------------------------------------------

    fig = plt.figure(
        figsize=(9, 9)
    )

    # Give the Skew-T a controlled, nearly square plotting area.
    #
    # [left, bottom, width, height]
    skew = SkewT(
        fig,
        rotation=45,
        rect=(0.10, 0.10, 0.78, 0.82)
    )

    # --------------------------------------------------------------
    # Determine tight plotting limits from actual observations
    # --------------------------------------------------------------

    p_max = pressure.max().to("hPa").m
    p_min = pressure.min().to("hPa").m

    t_max = temperature.max().to("degC").m
    t_min = temperature.min().to("degC").m

    # Small amount of vertical padding around profile.
    pressure_padding_bottom = 10
    pressure_padding_top = 15

    bottom_pressure = p_max + pressure_padding_bottom
    top_pressure = p_min - pressure_padding_top

    # Temperature padding.
    #
    # This is deliberately fairly tight so small temperature
    # changes/inversions are easy to see.
    temp_padding_left = 2.0
    temp_padding_right = 2.0

    left_temperature = t_min - temp_padding_left
    right_temperature = t_max + temp_padding_right

    # Guarantee a reasonable minimum temperature width.
    #
    # On days when all stations have almost identical temperatures,
    # we still want enough room for the Skew-T background.
    minimum_temp_width = 8.0

    current_width = (
        right_temperature
        - left_temperature
    )

    if current_width < minimum_temp_width:

        midpoint = (
            t_max + t_min
        ) / 2

        left_temperature = (
            midpoint
            - minimum_temp_width / 2
        )

        right_temperature = (
            midpoint
            + minimum_temp_width / 2
        )

    # --------------------------------------------------------------
    # Apply limits
    # --------------------------------------------------------------

    skew.ax.set_ylim(
        bottom_pressure,
        top_pressure
    )

    skew.ax.set_xlim(
        left_temperature,
        right_temperature
    )

    # --------------------------------------------------------------
    # Temperature trace
    # --------------------------------------------------------------

    skew.plot(
        pressure,
        temperature,
        color="red",
        linewidth=3,
        marker="o",
        markersize=8,
        zorder=10,
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
            sizes={
                "emptybarb": 0.15,
                "spacing": 0.2,
                "height": 0.4,
            },
            linewidth=1.2,
        )

    # --------------------------------------------------------------
    # Station / elevation labels beside wind barbs
    # --------------------------------------------------------------

    for station in profile:

        pp = station["pressure_hPa"]

        label = (
            f"{station['stid']}  "
            f"{station['elevation_ft']:.0f} ft"
        )

        # Position just inside the right side of the plot.
        skew.ax.annotate(
            label,
            xy=(
                0.985,
                pp,
            ),
            xycoords=(
                "axes fraction",
                "data",
            ),
            xytext=(-38, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=20,
        )
    
    # --------------------------------------------------------------
    # Skew-T background
    # --------------------------------------------------------------

    skew.plot_dry_adiabats(
        alpha=0.20
    )

    skew.plot_moist_adiabats(
        alpha=0.15
    )

    skew.plot_mixing_lines(
        alpha=0.12
    )

    # --------------------------------------------------------------
    # Zero-degree isotherm
    # --------------------------------------------------------------
    #
    # Only draw it if 0 C is actually reasonably close to the
    # displayed temperature range.
    # --------------------------------------------------------------

    if (
        left_temperature <= 0
        <= right_temperature
    ):

        skew.ax.axvline(
            0,
            color="blue",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            zorder=5,
        )

    # --------------------------------------------------------------
    # Titles
    # --------------------------------------------------------------

    skew.ax.set_title(
        "Mount Mansfield Observed Slope Profile",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=12,
    )

    skew.ax.set_title(
        "Temperature + Wind",
        fontsize=11,
        loc="right",
        pad=12,
    )

    # --------------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------------

    skew.ax.set_xlabel(
        "Temperature (°C)",
        fontsize=11,
    )

    skew.ax.set_ylabel(
        "Pressure (hPa)",
        fontsize=11,
    )

    # --------------------------------------------------------------
    # Observation time footer
    # --------------------------------------------------------------

    latest_times = []

    for station in profile:

        dt = parse_iso_time(
            station.get(
                "temperature_time"
            )
        )

        if dt is not None:
            latest_times.append(dt)

    if latest_times:

        newest = max(
            latest_times
        )

        time_text = newest.strftime(
            "%d %b %Y %H:%M UTC"
        )

        fig.text(
            0.49,
            0.035,
            time_text,
            ha="center",
            fontsize=10,
        )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    plt.savefig(
        OUTPUT_FILE,
        dpi=175,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(
        f"Saved Skew-T to: "
        f"{OUTPUT_FILE}"
    )

# =====================================================================
# 13. MAIN
# =====================================================================

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
