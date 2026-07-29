"""Compute spectra from a SWOT trace."""

import datetime as dt
import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from swell_swot.data import SWOT, check_target

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class SpectraSpecs:
    """Container for spectra-related variables."""

    dx: int = 250
    """Cross-track resolution in meters."""
    dy: int = 235
    """Along-track resolution in meters."""
    # Those are approximations, these are not true constants

    n_spectral_x: int = 20
    """Number of cross-track spectral points."""
    n_spectral_y: int = 21
    """Number of along-track spectral points."""

    # Box of size 40km x 40km, one per swath
    box_size_x: int = 160
    """Box size in pixels in cross-track direction."""
    box_size_y: int = 168
    """Box size in pixels in along-track direction."""
    box_increment_y: int = 168

    box_nadir_side_x: int = 20
    """Index of the box side closer to nadir, in cross-track direction."""

    # Tiles of size 5km x 5km, with 50% overlap
    tile_size_x: int = 20
    """Cross-track tile size for Welch's method."""
    tile_size_y: int = 21
    """Along-track tile size for Welch's method."""

    tile_increment_x: int = 10
    """Increment between tiles in cross-track, 50% overlap."""
    tile_increment_y: int = 10
    """Increment between tiles in along-track, 50% overlap."""

    def get_box_indices_y(self, n_lines: int):
        """Return indices of boxes (lower-left corner) in along-track direction."""
        return np.arange(0, n_lines - self.box_size_y, self.box_increment_y)

    def get_tile_indices_x(self) -> NDArray:
        """Return indices of tiles in cross-track direction."""
        return np.arange(
            0, self.box_size_x - self.tile_size_x + 1, self.tile_increment_x
        )

    def get_tile_indices_y(self) -> NDArray:
        """Return indices of tiles in along-track direction."""
        return np.arange(
            0, self.box_size_y - self.tile_size_y + 1, self.tile_increment_y
        )

    def get_frequencies(self) -> tuple[NDArray, NDArray]:
        """Return frequencies.

        Returns
        -------
        fx, fy:
            Spectral frequencies, centered around zero.
        """
        fx = np.fft.fftshift(np.fft.fftfreq(self.tile_size_x, self.dx))
        fy = np.fft.fftshift(np.fft.fftfreq(self.tile_size_y, self.dy))
        return fx, fy

    def get_attrs(self) -> dict:
        """Return specs as a dictionary for Dataset attributes."""
        return {attr: getattr(self, attr) for attr in self.__dataclass_fields__.keys()}


class PSD2:
    """Container for PSD of 2 arrays."""

    def __init__(self, psd_a: NDArray, psd_b: NDArray, phase: NDArray, n_tiles: int):
        self.psd_a = psd_a / n_tiles
        """PSD of the first array"""
        self.psd_b = psd_b / n_tiles
        """PSD of the second array"""

        self.n_tiles = n_tiles
        """Number of valid tiles used."""

        self.phase = phase / n_tiles
        """Cross-spectra phase"""
        self.angle = np.angle(phase)
        """Cross-spectra angle."""
        self.cross = np.real(phase) / n_tiles
        """Cross-spectra (real part)."""
        self.coherence = np.abs(self.phase**2) / (self.psd_a * self.psd_b)
        """Spectral coherence between SSHA and backscatter."""


def fft_tile(box: NDArray, hanning: NDArray, slice_x: slice, slice_y: slice) -> NDArray:
    """Compute FFT for a single tile."""
    tile = box[slice_y, slice_x]
    tile = tile - np.mean(tile)
    tile = tile * hanning

    fft = np.fft.fft2(tile, norm="forward")
    fft = np.fft.fftshift(fft)
    return fft


def psd_2d(a: NDArray, specs: SpectraSpecs):
    """Compute PSD of a single 2D array.

    Parameters
    ----------
    a:
        Array to FFT. Must be of dimensions (y, x), north-up and east-right.

    Returns
    -------
    psd:
        Array of PSD.
    """
    assert a.shape == (specs.box_size_y, specs.box_size_x)

    tile_indices_x = specs.get_tile_indices_x()
    tile_indices_y = specs.get_tile_indices_y()
    n_tile_x = len(tile_indices_x)
    n_tile_y = len(tile_indices_y)

    # Wavenumber resolution
    dk_x = 1 / (specs.dx * specs.tile_size_x)
    dk_y = 1 / (specs.dy * specs.tile_size_y)

    # Hanning windows
    hanning_x = np.hanning(specs.tile_size_x)
    hanning_y = np.hanning(specs.tile_size_y)
    hanning = hanning_x[None, :] * hanning_y[:, None]
    # Window correction factors
    window_correction_x = 1 / np.mean(hanning_x**2)
    window_correction_y = 1 / np.mean(hanning_y**2)
    normalization = (window_correction_x * window_correction_y) / (dk_x * dk_y)

    # Initialize mean PSD
    psd = np.zeros((specs.tile_size_y, specs.tile_size_x))

    for i_tile_y in tile_indices_y:
        slice_y = slice(i_tile_y, i_tile_y + specs.tile_size_y)
        for i_tile_x in tile_indices_x:
            slice_x = slice(i_tile_x, i_tile_x + specs.tile_size_x)
            tile_fft = fft_tile(a, hanning, slice_x, slice_y)
            psd[:, :] = psd[:, :] + np.abs(tile_fft) ** 2 * normalization

    psd = psd / (n_tile_x * n_tile_y)

    return psd


def cross_psd_2d(a: NDArray, b: np.ndarray, specs: SpectraSpecs) -> PSD2:
    """Compute PSD of two 2D array (with cross-correlation).

    Parameters
    ----------
    a:
        First array to FFT. Must be of dimensions (y, x), north-up and east-right.
    b:
        Second array to FFT. Must be of same dimensions and orientation as `a`.

    Returns
    -------
    psd2:
        PSD object containing spectras and other results.
    """
    assert a.shape == (specs.box_size_y, specs.box_size_x)
    assert b.shape == (specs.box_size_y, specs.box_size_x)

    tile_indices_x = specs.get_tile_indices_x()
    tile_indices_y = specs.get_tile_indices_y()
    n_tile_x = len(tile_indices_x)
    n_tile_y = len(tile_indices_y)

    # Wavenumber resolution
    dk_x = 1 / (specs.dx * specs.tile_size_x)
    dk_y = 1 / (specs.dy * specs.tile_size_y)

    # Hanning windows
    hanning_x = np.hanning(specs.tile_size_x)
    hanning_y = np.hanning(specs.tile_size_y)
    hanning = hanning_x[None, :] * hanning_y[:, None]
    # Window correction factors
    window_correction_x = 1 / np.mean(hanning_x**2)
    window_correction_y = 1 / np.mean(hanning_y**2)
    normalization = (window_correction_x * window_correction_y) / (dk_x * dk_y)

    # Initialize mean PSD
    spectra_shape = (specs.tile_size_y, specs.tile_size_x)
    psd_a = np.zeros(spectra_shape)
    psd_b = np.zeros(spectra_shape)
    phase = np.zeros(spectra_shape, dtype=complex)

    for i_tile_y in tile_indices_y:
        slice_y = slice(i_tile_y, i_tile_y + specs.tile_size_y)
        for i_tile_x in tile_indices_x:
            slice_x = slice(i_tile_x, i_tile_x + specs.tile_size_x)

            a_tile_fft = fft_tile(a, hanning, slice_x, slice_y)
            b_tile_fft = fft_tile(b, hanning, slice_x, slice_y)

            psd_a[:, :] += np.abs(a_tile_fft) ** 2 * normalization
            psd_b[:, :] += np.abs(b_tile_fft) ** 2 * normalization
            phase[:, :] += b_tile_fft * np.conj(a_tile_fft) * normalization

    # the PSD2 object will do the normalization by n_tiles
    psd2 = PSD2(psd_a, psd_b, phase, n_tiles=n_tile_x * n_tile_y)

    return psd2


def compute_spectra_swath(
    specs: SpectraSpecs, swot: xr.Dataset, side: str
) -> xr.Dataset:
    """Compute the spectra for one swath.

    Parameters
    ----------
    specs:
        Set of variables.
    swot:
        Dataset of L2 SWOT data.
    side:
        "left" or "right" swath.
    """
    box_indices_y = specs.get_box_indices_y(swot.num_lines.size)
    n_box = len(box_indices_y)

    # Initialize
    box_psd_ssha = np.zeros((n_box, specs.tile_size_y, specs.tile_size_x))
    box_psd_sig0 = np.zeros((n_box, specs.tile_size_y, specs.tile_size_x))
    box_psd_coherence = np.zeros((n_box, specs.tile_size_y, specs.tile_size_x))
    box_psd_angle = np.zeros((n_box, specs.tile_size_y, specs.tile_size_x))
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
            num_lines=slice(i_box_y, i_box_y + specs.box_size_y),
            num_pixels=slice(
                specs.box_nadir_side_x, specs.box_nadir_side_x + specs.box_size_x
            ),
        )

        # Flip the left swath (num_pixels increases away from nadir)
        if side == "left":
            box = box.isel(num_pixels=slice(None, None, -1))
        # Flip to have north up
        if track.pass_type == "descending":
            box = box.isel(num_lines=slice(None, None, -1))

        # psd2 = cross_psd_2d(box.ssh_karin_2.values, box.sig0_karin_2, specs)
        psd2 = cross_psd_2d(box.sig0_karin_2, box.ssh_karin_2.values, specs)
        box_psd_ssha[i_box] = psd2.psd_b
        box_psd_sig0[i_box] = psd2.psd_a
        box_psd_angle[i_box] = psd2.angle
        box_psd_coherence[i_box] = psd2.coherence

        box_extent[i_box, :] = [
            specs.box_nadir_side_x,
            i_box_y,
            specs.box_nadir_side_x + specs.box_size_x,
            i_box_y + specs.box_size_y,
        ]

        # Center position
        box_lon[i_box] = box.longitude.isel(
            num_lines=specs.box_size_y // 2, num_pixels=specs.box_size_x // 2
        ).values
        box_lat[i_box] = box.latitude.isel(
            num_lines=specs.box_size_y // 2, num_pixels=specs.box_size_x // 2
        ).values
        box_time[i_box] = box.time.isel(num_lines=specs.box_size_y // 2).values

    fx, fy = specs.get_frequencies()

    psd_ds = xr.Dataset(
        {
            "psd_ssha": (("box", "fy", "fx"), box_psd_ssha),
            "psd_sig0": (("box", "fy", "fx"), box_psd_sig0),
            "psd_coherence": (("box", "fy", "fx"), box_psd_coherence),
            "psd_angle": (("box", "fy", "fx"), box_psd_angle),
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
        attrs=specs.get_attrs(),
    )

    psd_ds.psd_ssha.attrs["long_name"] = "PSD of surface elevation"
    psd_ds.psd_ssha.attrs["units"] = "m^4"
    psd_ds.psd_sig0.attrs["long_name"] = "PSD of backscatter"
    psd_ds.psd_ssha.attrs["units"] = "m^4"
    psd_ds.psd_coherence.attrs["long_name"] = "Coherence SSHA-sig0"

    psd_ds.box_lon.attrs["long_name"] = "Longitude at box center"
    psd_ds.box_lat.attrs["long_name"] = "Latitude at box center"
    psd_ds.box_time.attrs.update(swot.time.attrs)
    psd_ds.box_time.attrs["long_name"] = "Time at box center"

    psd_ds.fx.attrs["long_name"] = "Spatial frequency cross-track"
    psd_ds.fy.attrs["long_name"] = "Spatial frequency along-track"
    psd_ds.fx.attrs["units"] = "cycles/m"
    psd_ds.fx.attrs["units"] = "cycles/m"

    return psd_ds


if __name__ == "__main__":
    track = SWOT(cycle=546, track=19, version="PGC0")
    specs = SpectraSpecs()

    # Find corresponding SWOT file
    swot_file = track.get_file()
    log.info("Found file %s", swot_file)

    # Load SWOT data
    log.info("Loading data")
    swot = xr.open_datatree(swot_file)

    log.info("Computing PSD for left swat")
    psd_l = compute_spectra_swath(specs, swot.left, "left")
    log.info("Computing PSD for right swat")
    psd_r = compute_spectra_swath(specs, swot.right, "right")

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
