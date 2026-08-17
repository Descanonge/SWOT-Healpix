"""Compute spectra over a full pass.

Loops over every 40km x 40km box in both swaths (left and right).
Save the result to a netcdf file.
"""

import datetime as dt
import logging

import numpy as np
import xarray as xr

from swot_healpix.analysis.spectra import FFTComputer, FFTWelchComputer
from swot_healpix.data import SWOT, check_target

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def compute_spectra_swath(
    swot: xr.Dataset, side: str, fft_computer: FFTComputer
) -> xr.Dataset:
    """Compute the spectra for one swath.

    Parameters
    ----------
    swot:
        Dataset of L2 SWOT data.
    side:
        "left" or "right" swath.
    moving_box:
        Allow to select boxes along swath.
    welch:
        Compute PSD inside boxes.
    """
    box_increment_y = fft_computer.box_size_y
    box_nadir_side_x = 20
    """Index of the box side closer to nadir, in cross-track direction."""

    box_indices_y = np.arange(
        0, swot.num_lines.size - fft_computer.box_size_y, box_increment_y
    )
    n_box = len(box_indices_y)

    # Initialize
    box_psd_ssh = np.zeros(
        (n_box, fft_computer.n_spectral_y, fft_computer.n_spectral_x)
    )
    box_extent = np.zeros((n_box, 4), dtype=int)
    box_lon = np.zeros(n_box)
    box_lat = np.zeros(n_box)
    box_time = np.zeros(n_box, dtype=swot.time.dtype)

    # Loop over all boxes along track
    main_loop = enumerate(box_indices_y)
    if _HAS_TQDM:
        main_loop = tqdm(main_loop, total=len(box_indices_y))
    for i_box, i_box_y in main_loop:
        box = swot.isel(
            num_lines=slice(i_box_y, i_box_y + fft_computer.box_size_y),
            num_pixels=slice(
                box_nadir_side_x, box_nadir_side_x + fft_computer.box_size_x
            ),
        )

        # Flip the left swath (num_pixels increases away from nadir)
        if side == "left":
            box = box.isel(num_pixels=slice(None, None, -1))
        # Flip to have north up
        if track.pass_type == "descending":
            box = box.isel(num_lines=slice(None, None, -1))

        box_psd_ssh[i_box] = fft_computer.psd(box.ssh_karin_2.values)

        box_extent[i_box, :] = [
            box_nadir_side_x,
            i_box_y,
            box_nadir_side_x + fft_computer.box_size_x,
            i_box_y + fft_computer.box_size_y,
        ]

        # Center position
        center = dict(
            num_lines=fft_computer.box_size_y // 2,
            num_pixels=fft_computer.box_size_x // 2,
        )
        box_lon[i_box] = box.longitude.isel(**center).values
        box_lat[i_box] = box.latitude.isel(**center).values
        box_time[i_box] = box.time.isel(num_lines=center["num_lines"]).values

    fx, fy = fft_computer.get_frequencies()

    psd_ds = xr.Dataset(
        {
            "psd_ssh": (("box", "fy", "fx"), box_psd_ssh),
            "box_extent": (("box", "box_extent"), box_extent),
            "box_lon": (("box"), box_lon),
            "box_lat": (("box"), box_lat),
            "box_time": (("box"), box_time),
            "box_swath": (("box"), np.full((n_box), side)),
        },
        coords={
            "box": np.arange(0, n_box),
            "fx": fx,
            "fy": fy,
        },
        attrs=fft_computer.asdict(),
    )

    psd_ds.psd_ssh.attrs.update(
        {"long_name": "PSD of surface elevation", "units": "m^4"}
    )

    psd_ds.box_lon.attrs["long_name"] = "Longitude at box center"
    psd_ds.box_lat.attrs["long_name"] = "Latitude at box center"
    psd_ds.box_time.attrs.update(swot.time.attrs, long_name="Time at box center")

    psd_ds.fx.attrs.update(
        {"long_name": "Spatial frequency cross-track", "units": "cycles/m"}
    )
    psd_ds.fy.attrs.update(
        {"long_name": "Spatial frequency along-track", "units": "cycles/m"}
    )

    return psd_ds


if __name__ == "__main__":
    track = SWOT(cycle=546, track=19, version="PGC0")

    # x is cross-track, y is along-track
    # Boxes of size 40x40km, with no overlap
    # Tiles of size 5x5km, overlap 50%
    fft_computer = FFTWelchComputer(
        box_size_x=160,
        box_size_y=168,
        dx=250,
        dy=235,
        tile_size_x=20,
        tile_size_y=21,
        tile_increment_x=10,
        tile_increment_y=10,
    )

    # Find corresponding SWOT file
    swot_file = track.get_file()
    log.info("Found file %s", swot_file)

    # Load SWOT data
    log.info("Loading data")
    swot = xr.open_datatree(swot_file)

    log.info("Computing PSD for left swat")
    psd_l = compute_spectra_swath(swot.left, "left", fft_computer)
    log.info("Computing PSD for right swat")
    psd_r = compute_spectra_swath(swot.right, "right", fft_computer)

    psd_l = psd_l.assign_coords({"box": psd_l.box * 2})
    psd_r = psd_r.assign_coords({"box": psd_r.box * 2 + 1})
    psd = xr.concat([psd_l, psd_r], "box").sortby("box")

    # Output
    coverage_start = dt.datetime.fromisoformat(swot.time_coverage_start)
    coverage_end = dt.datetime.fromisoformat(swot.time_coverage_end)
    outfile = track.make_filename(coverage_start, coverage_end, product="L2_LR_PSD")

    log.info("Writing to file %s", outfile)
    check_target(outfile)

    psd.to_netcdf(outfile)
