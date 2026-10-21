"""General utilities."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import FancyArrow
from matplotlib.text import Text
from matplotlib.ticker import ScalarFormatter


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


class LatLonFormatter(ScalarFormatter):
    """Ticker formatter for latitude and longitude."""

    CARDINALS = ("WE", "NS")

    def __init__(
        self,
        kind: Literal["lon", "lat"],
        number_format: str | None = None,
        fmt: str = "{value}°{NSWE}",
    ) -> None:
        self.kind = kind
        self.number_format = number_format
        self.fmt = fmt
        super().__init__(useOffset=False, useLocale=False)

    def __call__(self, x: float, pos: int | None = None) -> str:

        nswe = self.CARDINALS[self.kind == "lat"][int(x > 0.0)]
        if self.number_format is not None:
            value_fmt = f"{{:{self.number_format}}}"
            value = value_fmt.format(abs(x))
        else:
            value = super().__call__(abs(x), pos)

        return self.fmt.format(value=value, NSWE=nswe)


def set_latlon_formatters(ax: Axes, **kwargs) -> None:
    """Set latitude and longitude tick formatters."""
    ax.xaxis.set_major_formatter(LatLonFormatter("lon", **kwargs))
    ax.yaxis.set_major_formatter(LatLonFormatter("lat", **kwargs))


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


def make_arrow(
    xy: tuple[float, float],
    angle: float,
    length: float,
    label: str | None = None,
    *,
    label_offset: tuple[float, float] | None = None,
    label_kwargs: Mapping[str, Any] | None = None,
    **kwargs,
) -> tuple[FancyArrow, Text | None]:  # type: ignore
    start = np.asarray(xy)
    dr = length * np.array([np.cos(angle), np.sin(angle)])
    end = start + dr

    arrow = FancyArrow(*start, *dr, **kwargs)  # type: ignore[arg-type]

    if label is not None:
        if label_offset is None:
            label_offset = (0, 0)
        if label_kwargs is None:
            label_kwargs = dict()
        else:
            label_kwargs = dict(label_kwargs)
        if angle > np.pi / 2:
            label_kwargs.setdefault("ha", "right")
        text = Text(*(end + np.asarray(label_offset)), label, **label_kwargs)  # type: ignore[call-arg]
    else:
        text = None

    return arrow, text


def add_look_arrow(
    ax: Axes,
    angle: float,
    label: str,
    *,
    box_size: float = 0.1,
    box_loc: str = "upper left",
    arrow_kwargs: Mapping[str, Any] | None = None,
    box_kwargs: Mapping[str, Any] | None = None,
    label_kwargs: Mapping[str, Any] | None = None,
    label_offset: tuple[float, float] | None = None,
) -> AnchoredOffsetbox:
    arrow_kwargs = dict() if arrow_kwargs is None else dict(arrow_kwargs)
    box_kwargs = dict() if box_kwargs is None else dict(box_kwargs)
    label_kwargs = dict() if label_kwargs is None else dict(label_kwargs)
    if label_offset is None:
        label_offset = (0, 0)

    box_size_pixels = ax.bbox.width * box_size
    da = DrawingArea(box_size_pixels, box_size_pixels)

    arrow_kwargs = (
        dict(width=2, ec="none", fc="k", length_includes_head=True) | arrow_kwargs
    )

    start = np.full(2, box_size_pixels / 2.0)
    arrow, text = make_arrow(
        tuple(start),
        angle,
        length=box_size_pixels * 0.45,
        label=label,
        label_kwargs=label_kwargs,
        label_offset=label_offset,
        **arrow_kwargs,
    )
    da.add_artist(arrow)
    if text is not None:
        da.add_artist(text)

    box_kwargs = dict(frameon=False) | box_kwargs
    box = AnchoredOffsetbox(box_loc, child=da, **box_kwargs)
    ax.add_artist(box)
    return box


def add_look_arrows(
    ax: Axes,
    angle_north: float,
    angle_vsat: float,
    angle_look: float,
    *,
    box_size: float = 0.1,
    box_loc: str = "upper left",
    arrows_to_plot: Sequence[str] | None = None,
    arrows_kwargs: Mapping[str, Any] | None = None,
    arrows_specific_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    arrows_base_kwargs: Mapping[str, Any] | None = None,
    label_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    label_offsets: Mapping[str, tuple[float, float] | None] | None = None,
    box_kwargs: Mapping[str, Any] | None = None,
) -> AnchoredOffsetbox:  # type: ignore
    if arrows_to_plot is None:
        arrows_to_plot = ["north", "vsat", "look"]
    arrows_kwargs = dict() if arrows_kwargs is None else dict(arrows_kwargs)
    arrows_specific_kwargs = (
        dict() if arrows_specific_kwargs is None else dict(arrows_specific_kwargs)
    )
    arrows_base_kwargs = (
        dict() if arrows_base_kwargs is None else dict(arrows_base_kwargs)
    )
    box_kwargs = dict() if box_kwargs is None else dict(box_kwargs)
    label_kwargs = dict() if label_kwargs is None else dict(label_kwargs)
    label_offsets = dict() if label_offsets is None else dict(label_offsets)

    box_size_pixels = ax.bbox.width * box_size
    da = DrawingArea(box_size_pixels, box_size_pixels)

    arrows_kwargs = (
        dict(width=2, ec="none", fc="k", length_includes_head=True) | arrows_kwargs
    )

    angles = dict(north=angle_north, vsat=angle_vsat, look=angle_look)
    labels = dict(north="N", vsat="$v_{sat}$", look="look")
    start = np.full(2, box_size_pixels / 2.0)
    for name in arrows_to_plot:
        arrow, text = make_arrow(
            tuple(start),
            angles[name],
            length=box_size_pixels * 0.45,
            label=labels[name],
            label_kwargs=label_kwargs.get(name, {}),
            label_offset=label_offsets.get(name),
            **(arrows_kwargs | dict(arrows_specific_kwargs.get(name, {}))),  # type: ignore[arg-type]
        )
        da.add_artist(arrow)
        if text is not None:
            da.add_artist(text)

    for k in ["ec", "fc"]:
        if k in arrows_kwargs:
            arrows_base_kwargs.setdefault(k, arrows_kwargs[k])
    da.add_artist(
        plt.Circle(
            tuple(start),
            radius=arrows_kwargs["width"] / 2,  # type: ignore[operator]
            **arrows_base_kwargs,
        )
    )

    box_kwargs = dict(frameon=False) | box_kwargs
    box = AnchoredOffsetbox(box_loc, child=da, **box_kwargs)
    ax.add_artist(box)
    return box
