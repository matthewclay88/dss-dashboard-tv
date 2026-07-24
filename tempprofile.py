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
    5. Derives a "Winter Profile Diagnostics" panel (freezing level,
       wet-bulb zero, lapse rate, inversions, a KBTV-to-MMNV1 bulk
       shear, and a bulk Froude number / flow regime for the ridge).

V1 intentionally does NOT calculate:
    LCL
    full moist-adiabatic parcel ascent
    precipitation type

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

# Physical constants used by the diagnostics module below.
RD = 287.05      # J/(kg*K), dry air gas constant
G = 9.80665       # m/s^2
CP = 1004.0       # J/(kg*K), dry air specific heat at constant pressure

# Stations used for the low-level bulk shear diagnostic. KBTV (330 ft)
# to MMNV1 (3891 ft) spans about 3560 ft (~1.1 km), i.e. roughly the
# 0-1 km AGL layer, and both are always part of the fixed station set.
SHEAR_BASE_STID = "KBTV"
SHEAR_TOP_STID = "MMNV1"

# Bulk Froude number thresholds used to label flow regime around the
# ridge. Fr >= 1 flow tends to go over the barrier; Fr < ~0.5 flow
# tends to be blocked and diverts around/backs up upstream; the band
# in between is a partially-blocked regime. These are conventional
# rule-of-thumb cutoffs from mountain-flow literature, not a fitted
# threshold for Mansfield specifically.
FROUDE_UNBLOCKED = 1.0
FROUDE_BLOCKED = 0.5

# Approximate compass bearing (degrees) of the Mansfield ridge line
# itself - the summit ridge trends SSW-NNE, roughly 020-030 deg
# azimuth along the spine of the Green Mountains. Used to resolve the
# cross-barrier (ridge-normal) wind component for the Froude number,
# since along-ridge flow doesn't force air over or around the
# barrier the way cross-ridge flow does.
RIDGE_ORIENTATION_DEG = 25.0


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


def ft_to_m(feet):
    """
    Feet -> meters.
    """

    return feet * 0.3048


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
    Search recent NWS observations for the newest usable values.

    Temperature, dewpoint, wind, and pressure do NOT have to come
    from the same observation record.
    """

    features = fetch_nws_recent(stid)

    if not features:
        return None

    temperature = None
    temperature_time = None

    dewpoint = None
    dewpoint_time = None

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

        # ---------------------------------------------------------
        # Temperature
        # ---------------------------------------------------------

        temp = qv_value(p, "temperature")

        if temperature is None and temp is not None:
            temperature = temp
            temperature_time = timestamp

        # ---------------------------------------------------------
        # Dewpoint
        # ---------------------------------------------------------

        td = qv_value(p, "dewpoint")

        if dewpoint is None and td is not None:
            dewpoint = td
            dewpoint_time = timestamp

        # ---------------------------------------------------------
        # Wind
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Station pressure
        # ---------------------------------------------------------

        pressure = qv_value(p, "barometricPressure")

        if pressure_pa is None and pressure is not None:
            pressure_pa = pressure
            pressure_time = timestamp

        # ---------------------------------------------------------
        # Stop once everything has been found
        # ---------------------------------------------------------

        if (
            temperature is not None
            and dewpoint is not None
            and wind_speed is not None
            and pressure_pa is not None
        ):
            break

    return {
        "stid": stid,

        "temperature_C": temperature,
        "temperature_time": temperature_time,

        "dewpoint_C": dewpoint,
        "dewpoint_time": dewpoint_time,

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
        "dewpoint_C": None,
        "dewpoint_time": None,
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
    Convert normalized observations into an elevation-sorted profile.

    Temperature is required for a station to enter the thermal profile.
    Dewpoint and wind are optional.
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

            "temperature_C":
                temperature,

            "temperature_time":
                obs.get("temperature_time"),

            "dewpoint_C":
                obs.get("dewpoint_C"),

            "dewpoint_time":
                obs.get("dewpoint_time"),

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
# 11B. WINTER PROFILE DIAGNOSTICS
# =====================================================================
#
# This section derives a compact set of winter-weather diagnostics
# from the same station profile used for the Skew-T. With only 6-7
# low-level points instead of a full sounding, everything here is a
# bulk / coarse estimate intended for situational awareness, not a
# validated forecast product. Assumptions are called out inline.
#

def interp_crossing(z1, v1, z2, v2, target=0.0):
    """
    Linear interpolation for the elevation where a variable crosses
    `target`, given two bracketing (elevation, value) points.
    """

    if v2 == v1:
        return None

    frac = (target - v1) / (v2 - v1)

    if frac < 0 or frac > 1:
        return None

    return z1 + frac * (z2 - z1)


def find_zero_level(points):
    """
    Lowest elevation (ft) at which a series crosses 0, given a list
    of (elevation_ft, value) tuples with value already screened for
    None. Returns None if no crossing exists in the observed layer.
    """

    points = [p for p in points if p[1] is not None]

    if len(points) < 2:
        return None

    for (z1, v1), (z2, v2) in zip(points, points[1:]):

        if v1 == 0:
            return z1

        if (v1 > 0 > v2) or (v1 < 0 < v2):
            return interp_crossing(z1, v1, z2, v2)

    return None


def mean_lapse_rate(profile):
    """
    Bulk lapse rate between the lowest and highest observed stations,
    in C/km, positive when temperature falls with height (the
    conventional sign).
    """

    bottom = profile[0]
    top = profile[-1]

    dz_km = ft_to_m(
        top["elevation_ft"] - bottom["elevation_ft"]
    ) / 1000.0

    if dz_km == 0:
        return None

    return (
        bottom["temperature_C"] - top["temperature_C"]
    ) / dz_km


def max_inversion(profile):
    """
    The single layer with the largest temperature increase with
    height (an inversion). Returns None if every layer cools with
    height.
    """

    best = None

    for lower, upper in zip(profile, profile[1:]):

        dT = upper["temperature_C"] - lower["temperature_C"]

        if dT > 0 and (best is None or dT > best["dT"]):

            best = {
                "dT": dT,
                "z1": lower["elevation_ft"],
                "z2": upper["elevation_ft"],
            }

    return best


def compute_wetbulb_series(profile):
    """
    Wet-bulb temperature (C) at each station that has a dewpoint.
    MMNV1 has no dewpoint sensor in this feed, so it is excluded
    here, same as the rest of the moisture-dependent diagnostics.
    """

    series = []

    for x in profile:

        td = x.get("dewpoint_C")
        p = x.get("pressure_hPa")

        if td is None or p is None:
            continue

        try:

            tw = mpcalc.wet_bulb_temperature(
                p * units.hPa,
                x["temperature_C"] * units.degC,
                td * units.degC,
            ).to("degC").m

        except Exception:
            continue

        series.append({
            "elevation_ft": x["elevation_ft"],
            "wetbulb_C": tw,
        })

    return series


def bulk_shear(profile, base_stid=SHEAR_BASE_STID, top_stid=SHEAR_TOP_STID):
    """
    Vector wind difference (kt) between two named stations - by
    default KBTV (330 ft, valley floor) and MMNV1 (3891 ft, summit),
    which together span roughly the lowest 1 km AGL of this profile.

    Returns None if either station is missing from the profile or is
    missing a wind observation.
    """

    base = next((x for x in profile if x["stid"] == base_stid), None)
    top = next((x for x in profile if x["stid"] == top_stid), None)

    if base is None or top is None:
        return None

    if (
        base.get("wind_speed_kmh") is None
        or base.get("wind_direction_deg") is None
        or top.get("wind_speed_kmh") is None
        or top.get("wind_direction_deg") is None
    ):
        return None

    speeds_kt = (
        np.array([base["wind_speed_kmh"], top["wind_speed_kmh"]])
        * units("km/hour")
    ).to("knots").m

    directions = np.array(
        [base["wind_direction_deg"], top["wind_direction_deg"]]
    )

    u, v = mpcalc.wind_components(
        speeds_kt * units.knots,
        directions * units.degree,
    )

    u = u.m
    v = v.m

    return float(np.hypot(u[1] - u[0], v[1] - v[0]))


def potential_temperature_profile(profile):
    """
    (height_m, theta_K) pairs for every station with both pressure
    and temperature, sorted by height.
    """

    points = []

    for x in profile:

        p = x.get("pressure_hPa")
        t = x.get("temperature_C")

        if p is None or t is None:
            continue

        theta = (t + 273.15) * (1000.0 / p) ** (RD / CP)

        points.append((ft_to_m(x["elevation_ft"]), theta))

    points.sort(key=lambda pt: pt[0])

    return points


def cross_barrier_component(speed, direction_from_deg, ridge_orientation_deg=RIDGE_ORIENTATION_DEG):
    """
    Magnitude of the wind component oriented perpendicular to the
    ridge line (cross-barrier flow), which is the component relevant
    to blocking/Froude-number behavior - a wind blowing along the
    ridge axis doesn't get forced up and over it the way cross-ridge
    flow does.

    `direction_from_deg` is the meteorological "wind from" direction.
    Returns an unsigned magnitude, since blocking behaves the same
    whether the flow approaches from the east or west side of the
    ridge.
    """

    toward_deg = (direction_from_deg + 180.0) % 360.0
    ridge_normal_deg = (ridge_orientation_deg + 90.0) % 360.0

    angle = np.radians(toward_deg - ridge_normal_deg)

    return abs(speed * np.cos(angle))


def brunt_vaisala_and_froude(profile, wind_stid=SHEAR_BASE_STID):
    """
    Dry Brunt-Vaisala frequency (N, 1/s) from a least-squares fit of
    potential temperature against height across every station in the
    profile, and a bulk Froude number Fr = U / (N * h) for the ridge,
    where U is the cross-barrier wind component at `wind_stid` (KBTV
    by default) and h is the profile's elevation span (base to
    summit).

    Fitting N through all stations rather than differencing just the
    top and bottom two is deliberate: with sub-degree temperature
    differences over a shallow layer, a two-point estimate is very
    sensitive to noise in either endpoint. A least-squares fit through
    every station is more stable, though still a coarse, order-of-
    magnitude estimate - not a substitute for an actual upstream
    sounding.

    Returns (None, None, None) if there are too few points, or if the
    fitted layer is neutral/unstable (dtheta/dz <= 0), since N is
    undefined there.
    """

    theta_points = potential_temperature_profile(profile)

    if len(theta_points) < 3:
        # A 2-point fit degenerates back into the noise-prone
        # difference this function is meant to avoid.
        return None, None, None

    heights = np.array([pt[0] for pt in theta_points])
    thetas = np.array([pt[1] for pt in theta_points])

    dtheta_dz, _intercept = np.polyfit(heights, thetas, 1)

    if dtheta_dz <= 0:
        # Neutral or unstable bulk layer - N (and therefore Fr) is
        # not meaningfully defined.
        return None, None, None

    theta_mean = float(np.mean(thetas))

    N = np.sqrt((G / theta_mean) * dtheta_dz)

    wind_station = next(
        (x for x in profile if x["stid"] == wind_stid), None
    )

    if (
        wind_station is None
        or wind_station.get("wind_speed_kmh") is None
        or wind_station.get("wind_direction_deg") is None
    ):
        return float(N), None, None

    speed_ms = (
        wind_station["wind_speed_kmh"] * units("km/hour")
    ).to("m/s").m

    U = cross_barrier_component(
        speed_ms, wind_station["wind_direction_deg"]
    )

    mountain_height_m = ft_to_m(
        profile[-1]["elevation_ft"] - profile[0]["elevation_ft"]
    )

    if mountain_height_m <= 0:
        return float(N), float(U), None

    Fr = U / (N * mountain_height_m)

    return float(N), float(U), float(Fr)


def classify_flow_regime(Fr):

    if Fr is None:
        return "Unknown"

    if Fr >= FROUDE_UNBLOCKED:
        return "Unblocked / flow-over"

    if Fr >= FROUDE_BLOCKED:
        return "Partially blocked"

    return "Blocked"


def build_diagnostics(profile):
    """
    Assemble the full winter-profile diagnostics dictionary from an
    elevation-sorted, pressure-populated profile.
    """

    diagnostics = {}

    # ---------------- THERMAL ----------------

    freezing_points = [
        (x["elevation_ft"], x["temperature_C"]) for x in profile
    ]

    diagnostics["freezing_level_ft"] = find_zero_level(freezing_points)

    wetbulb_series = compute_wetbulb_series(profile)

    if len(wetbulb_series) >= 2:

        wb_points = [
            (w["elevation_ft"], w["wetbulb_C"]) for w in wetbulb_series
        ]

        diagnostics["wet_bulb_zero_ft"] = find_zero_level(wb_points)

        wb_values = [w["wetbulb_C"] for w in wetbulb_series]
        diagnostics["wet_bulb_range_C"] = (min(wb_values), max(wb_values))

    else:

        diagnostics["wet_bulb_zero_ft"] = None
        diagnostics["wet_bulb_range_C"] = None

    diagnostics["mean_lapse_rate_C_km"] = mean_lapse_rate(profile)
    diagnostics["max_inversion"] = max_inversion(profile)

    # ---------------- WIND / TERRAIN ----------------

    diagnostics["bulk_shear_kt"] = bulk_shear(profile)

    N, U, Fr = brunt_vaisala_and_froude(profile)

    diagnostics["brunt_vaisala_N"] = N
    diagnostics["froude_number"] = Fr
    diagnostics["flow_regime"] = classify_flow_regime(Fr)

    return diagnostics


def diagnostic_display_rows(diagnostics):
    """
    Build the (label, value, is_section_header) rows shared by the
    console printout and the figure table.
    """

    def val(x, suffix="", decimals=1, missing="None in observed layer"):
        if x is None:
            return missing
        return f"{x:.{decimals}f}{suffix}"

    inv = diagnostics["max_inversion"]

    if inv:
        inv_text = f"+{inv['dT']:.1f} C ({inv['z1']:.0f}-{inv['z2']:.0f} ft)"
    else:
        inv_text = "None"

    wb_range = diagnostics["wet_bulb_range_C"]

    if wb_range:
        wb_range_text = f"{wb_range[0]:.1f} -> {wb_range[1]:.1f} C"
    else:
        wb_range_text = "Insufficient data"

    shear_val = diagnostics["bulk_shear_kt"]
    shear_text = (
        val(shear_val, " kt", 0)
        if shear_val is not None
        else "Insufficient data"
    )

    shear_depth_ft = STATIONS[SHEAR_TOP_STID] - STATIONS[SHEAR_BASE_STID]
    shear_label = (
        f"{SHEAR_BASE_STID}\u2192{SHEAR_TOP_STID} Shear "
        f"({shear_depth_ft/1000.0:.1f} kft)"
    )

    froude_val = diagnostics["froude_number"]
    froude_text = (
        val(froude_val, "", 2)
        if froude_val is not None
        else "N/A"
    )

    rows = [
        ("THERMAL", "", True),
        ("Freezing Level", val(diagnostics["freezing_level_ft"], " ft", 0), False),
        ("Wet-Bulb Zero", val(diagnostics["wet_bulb_zero_ft"], " ft", 0), False),
        ("Mean Lapse Rate", val(diagnostics["mean_lapse_rate_C_km"], " C/km"), False),
        ("Max Inversion", inv_text, False),
        ("Wet-Bulb Range", wb_range_text, False),
        ("WIND / TERRAIN", "", True),
        (shear_label, shear_text, False),
        ("Froude Number", froude_text, False),
        ("Flow Regime", diagnostics["flow_regime"], False),
    ]

    return rows


def print_diagnostics(diagnostics):

    rows = diagnostic_display_rows(diagnostics)

    print()
    print("=" * 60)
    print("WINTER PROFILE DIAGNOSTICS")
    print("=" * 60)

    for label, value, is_header in rows:

        if is_header:
            print(f"\n{label}")
        else:
            print(f"  {label:<22}: {value}")

    print("=" * 60)


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
    diagnostics,
):

    # ==============================================================
    # FIGURE
    # ==============================================================

    fig = plt.figure(
        figsize=(11, 11)
    )

    # Skew-T occupies the upper portion.
    # Bottom portion is reserved for the observation + diagnostics tables.

    skew = SkewT(
        fig,
        rotation=45,
        rect=(0.10, 0.40, 0.80, 0.53)
    )

    # ==============================================================
    # TIGHT PLOT LIMITS
    # ==============================================================

    p_max = pressure.max().to("hPa").m
    p_min = pressure.min().to("hPa").m

    t_max = temperature.max().to("degC").m
    t_min = temperature.min().to("degC").m

    # Include dewpoint when determining the visible temperature range.
    dewpoints = [
        station["dewpoint_C"]
        for station in profile
        if station.get("dewpoint_C") is not None
    ]

    if dewpoints:
        t_min = min(
            t_min,
            min(dewpoints)
        )
    
    # Small pressure padding above/below observations.

    bottom_pressure = p_max + 10
    top_pressure = p_min - 15

    # Very tight temperature zoom.
    #
    # This product is intentionally focused on small temperature
    # differences in the shallow low-level profile.

    temp_padding_left = 2.0
    temp_padding_right = 2.0

    left_temperature = (
        t_min - temp_padding_left
    )

    right_temperature = (
        t_max + temp_padding_right
    )

    # Maintain a minimum 8 C window if temperatures are clustered.

    minimum_temp_width = 8.0

    current_width = (
        right_temperature
        - left_temperature
    )

    if current_width < minimum_temp_width:

        midpoint = (
            t_max + t_min
        ) / 2.0

        left_temperature = (
            midpoint
            - minimum_temp_width / 2.0
        )

        right_temperature = (
            midpoint
            + minimum_temp_width / 2.0
        )

    skew.ax.set_ylim(
        bottom_pressure,
        top_pressure
    )

    skew.ax.set_xlim(
        left_temperature,
        right_temperature
    )

    # ==============================================================
    # SKEW-T BACKGROUND
    # ==============================================================

    skew.plot_dry_adiabats(
        alpha=0.20
    )

    skew.plot_moist_adiabats(
        alpha=0.15
    )

    skew.plot_mixing_lines(
        alpha=0.12
    )

    # ==============================================================
    # TEMPERATURE
    # ==============================================================

    skew.plot(
        pressure,
        temperature,
        color="red",
        linewidth=3,
        marker="o",
        markersize=8,
        zorder=10,
    )

    # ==============================================================
    # DEWPOINT
    # ==============================================================

    dewpoint_pressure = []
    dewpoint_temperature = []

    for station in profile:

        td = station.get("dewpoint_C")

        if td is None:
            continue

        dewpoint_pressure.append(
            station["pressure_hPa"]
        )

        dewpoint_temperature.append(
            td
        )

    if len(dewpoint_temperature) >= 2:

        td_pressure = (
            np.array(dewpoint_pressure)
            * units.hPa
        )

        td_temperature = (
            np.array(dewpoint_temperature)
            * units.degC
        )

        skew.plot(
            td_pressure,
            td_temperature,
            color="green",
            linewidth=3,
            marker="o",
            markersize=7,
            zorder=10,
        )

    # ==============================================================
    # WIND BARBS
    # ==============================================================

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

    # ==============================================================
    # 0 C ISOTHERM
    # ==============================================================

    if (
        left_temperature <= 0
        <= right_temperature
    ):

        skew.ax.axvline(
            0,
            color="blue",
            linestyle="--",
            linewidth=1.5,
            alpha=0.75,
            zorder=5,
        )

    # ==============================================================
    # TITLE / AXES
    # ==============================================================

    skew.ax.set_title(
        "Mount Mansfield Observed Slope Profile",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=12,
    )

    skew.ax.set_xlabel(
        "Temperature (°C)",
        fontsize=11,
    )

    skew.ax.set_ylabel(
        "Pressure (hPa)",
        fontsize=11,
    )

    # ==============================================================
    # OBSERVATION TABLE (left)
    # ==============================================================

    table_rows = []

    # Highest elevation first so the table follows the vertical
    # orientation of the sounding.

    for station in reversed(profile):

        # ----------------------------------------------------------
        # Temperature
        # ----------------------------------------------------------

        temp = station.get(
            "temperature_C"
        )

        if temp is not None:

            temp_text = (
                f"{temp:.1f}°C"
            )

        else:

            temp_text = "--"

        # ----------------------------------------------------------
        # Dewpoint
        # ----------------------------------------------------------

        dewpoint = station.get(
            "dewpoint_C"
        )

        if dewpoint is not None:

            dewpoint_text = (
                f"{dewpoint:.1f}°C"
            )

        else:

            dewpoint_text = "--"

        # ----------------------------------------------------------
        # Wind
        # ----------------------------------------------------------

        speed = station.get(
            "wind_speed_kmh"
        )

        direction = station.get(
            "wind_direction_deg"
        )

        if (
            speed is not None
            and direction is not None
        ):

            speed_kt = (
                speed
                * units("km/hour")
            ).to("knots").m

            if speed_kt < 0.5:

                wind_text = "Calm"

            else:

                wind_text = (
                    f"{direction:03.0f}° / "
                    f"{speed_kt:.0f} kt"
                )

        else:

            wind_text = "--"

        # ----------------------------------------------------------
        # Observation time
        # ----------------------------------------------------------

        obs_time = parse_iso_time(
            station.get(
                "temperature_time"
            )
        )

        if obs_time:

            time_text = (
                obs_time.strftime(
                    "%H:%MZ"
                )
            )

        else:

            time_text = "--"

        # ----------------------------------------------------------
        # Add row
        # ----------------------------------------------------------

        table_rows.append([
            station["stid"],
            f"{station['elevation_ft']:.0f} ft",
            f"{station['pressure_hPa']:.1f}",
            temp_text,
            dewpoint_text,
            wind_text,
            time_text,
        ])

    # Dedicated axes for the observation table (left half).

    table_ax = fig.add_axes([
        0.06,
        0.06,
        0.52,
        0.28,
    ])

    table_ax.axis(
        "off"
    )

    table_ax.set_title(
        "Station Observations",
        fontsize=10,
        fontweight="bold",
        loc="left",
        pad=4,
    )

    table = table_ax.table(
        cellText=table_rows,
        colLabels=[
            "Station",
            "Elevation",
            "Pressure",
            "Temp",
            "Dewpoint",
            "Wind",
            "Time",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(
        False
    )

    table.set_fontsize(
        9
    )

    table.scale(
        1.0,
        1.4,
    )

    # Make header bold.

    for column in range(7):

        table[
            (0, column)
        ].set_text_props(
            weight="bold"
        )

    # ==============================================================
    # WINTER PROFILE DIAGNOSTICS TABLE (right)
    # ==============================================================

    diag_rows = diagnostic_display_rows(diagnostics)

    diag_ax = fig.add_axes([
        0.62,
        0.14,
        0.34,
        0.20,
    ])

    diag_ax.axis("off")

    diag_ax.set_title(
        "Winter Profile Diagnostics",
        fontsize=10,
        fontweight="bold",
        loc="left",
        pad=4,
    )

    diag_table = diag_ax.table(
        cellText=[[label, value] for label, value, _ in diag_rows],
        colWidths=[0.55, 0.45],
        cellLoc="left",
        loc="center",
    )

    diag_table.auto_set_font_size(False)
    diag_table.set_fontsize(8.5)
    diag_table.scale(1.0, 1.3)

    for row_index, (label, value, is_header) in enumerate(diag_rows):

        for col in range(2):

            cell = diag_table[(row_index, col)]
            cell.set_edgecolor("none")

            if is_header:

                cell.set_text_props(weight="bold")
                cell.set_facecolor("0.90")

                # Blank out the value column on section-header rows.
                if col == 1:
                    cell.get_text().set_text("")

    # ==============================================================
    # PRODUCT TIME
    # ==============================================================

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
            0.50,
            0.015,
            time_text,
            ha="center",
            fontsize=10,
        )

    # ==============================================================
    # SAVE
    # ==============================================================

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

    diagnostics = build_diagnostics(
        profile
    )

    print_diagnostics(
        diagnostics
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
        diagnostics,
    )


if __name__ == "__main__":
    main()
