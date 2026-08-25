"""General utilities."""

from typing import Literal

import numpy as np


def to_deg_mn_sec(x, /, decimals: int = 6) -> tuple:
    """Convert from decimal degrees to degree, minutes, seconds.

    Parameters
    ----------
    x:
        Value to convert in degrees.
    decimals:
        Number of decimals to round integer values to. Seconds are rounded to
        ``decimals-3``.
    """
    x_ = np.asarray(x, "d")
    degs = np.round(x_, decimals).astype("i")
    y = (x_ - degs) * 60
    mins = np.round(y, decimals).astype("i")
    y = (y - mins) * 60
    secs = np.round(y, decimals - 3)
    return degs, mins, secs


def format_latlon(
    value: float,
    kind: Literal["lat", "lon"] = "lon",
    /,
    *,
    decimal: bool = True,
    direction_label: bool = True,
    number_format: str = "g",
    degree_symbol: str = "°",
    minute_symbol: str = "'",
    second_symbol: str = '"',
) -> str:
    """Format a latitude/longitude value.

    Parameters
    ----------
    value:
        Value to format.
    kind:
        Either 'lat' or 'lon'.
    decimal:
        If True, format the degrees with a decimal point fraction. If False, format as
        degrees-minutes-seconds.
    direction_label:
        If True, append cardinal direction. If False, use -/+ sign.
    number_format:
        String-format to use to format the value when in decimal mode, or the seconds
        in degree-minute-seconds mode.
    degree_symbol, minute_symbol, second_symbol:
        Symbols to use for degrees, minutes and seconds.
    """
    hemisphere = {"lon": "WE", "lat": "SN"}[kind][value > 0]

    if direction_label:
        value = abs(value)

    if decimal:
        label = f"{{:{number_format}}}{degree_symbol}".format(value)
    else:
        deg, mn, sec = to_deg_mn_sec(value)
        label = (
            f"{deg:d}{degree_symbol}{mn:d}{minute_symbol}"
            + f"{{:{number_format}}}{second_symbol}".format(sec)
        )

    if direction_label:
        label += hemisphere

    return label


def get_cell_size(level: int, /, *, radius: float = 6371e3) -> float:
    """Return Healpix cell average size for a given level.

    Parameters
    ----------
    level
        Healpix grid level/depth.
    radius
        Earth radius in meters.

    Returns
    -------
    cell_size
        Cell average size in meters.
    """
    return float(np.sqrt(4 * np.pi / (12 * 4**level)) * radius)
