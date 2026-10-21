"""Compute spectra."""

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import hann, tukey


def fft_2d(a: NDArray, window: NDArray, norm: str = "ortho") -> NDArray:
    """Compute 2D FFT of array with hanning window."""
    a = a - np.mean(a)
    a = a * window

    fft = np.fft.fft2(a, norm=norm)  # type: ignore[call-overload]
    fft = np.fft.fftshift(fft)
    return fft


def average_psd_azimuth(
    psd: NDArray, fx: NDArray, fy: NDArray
) -> tuple[NDArray, NDArray]:
    """Average azimuthally a 2D PSD to obtain an isotropic 1D power spectra.

    Parameters
    ----------
    psd:
        The 2D PSD. Array of dimensions [y, x].
    fx, fy:
        The frequencies in x and y directions.

    Returns
    -------
    freqs:
        Array of 1D frequencies.
    psd:
        1D power spectra.
    """
    ny, nx = psd.shape
    fx2d, fy2d = np.meshgrid(fx, fy, indexing="xy")
    fr = np.sqrt(fx2d**2 + fy2d**2)  # radial spatial frequency (cycles per unit length)

    # Radial binning
    nbins = min(nx, ny) // 2
    f_bins = np.linspace(0.0, fr.max(), nbins + 1)

    # Vectorized bin average of P2D over annuli
    fr_flat = fr.ravel()
    psd_flat = psd.ravel()
    bin_idx = np.digitize(fr_flat, f_bins) - 1  # -> [0, nbins-1]
    valid = (bin_idx >= 0) & (bin_idx < nbins)

    # Sum and count per bin, then mean
    sum_per_bin = np.bincount(bin_idx[valid], weights=psd_flat[valid], minlength=nbins)
    cnt_per_bin = np.bincount(bin_idx[valid], minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        psd_k = sum_per_bin / cnt_per_bin
    psd_k[cnt_per_bin == 0] = np.nan  # empty bins

    # Bin centers
    f_centers = 0.5 * (f_bins[1:] + f_bins[:-1])

    return f_centers, psd_k / (nx * ny)


@dataclass
class PSD2:
    """Container for cross-PSD of 2 arrays."""

    psd_a: NDArray
    """PSD of the first array"""
    psd_b: NDArray
    """PSD of the second array"""
    phase: NDArray
    """Cross-spectra phase"""

    def __post_init__(self) -> None:
        self.angle = np.angle(self.phase)
        """Cross-spectra angle."""
        self.cross = np.real(self.phase)
        """Cross-spectra (real part)."""
        self.coherence = np.abs(self.phase**2) / (self.psd_a * self.psd_b)
        """Spectral coherence between SSHA and backscatter."""


@dataclass(kw_only=True)
class FFTComputer:
    """Parameters of box moving along the swath."""

    box_size_x: int
    """Box size in pixels."""
    box_size_y: int
    """Box size in pixels."""

    dx: float
    """Resolution in meters."""
    dy: float
    """Resolution in meters."""

    apodizing_kind: Literal["tukey", "hanning"] = "hanning"

    @property
    def n_spectral_x(self) -> int:
        """Number of frequencies."""
        return self.box_size_x

    @property
    def n_spectral_y(self) -> int:
        """Number of frequencies."""
        return self.box_size_y

    def get_frequencies(self) -> tuple[NDArray, NDArray]:
        """Return frequencies for directions x and y (in that order)."""
        fx = np.fft.fftshift(np.fft.fftfreq(self.n_spectral_x, self.dx))
        fy = np.fft.fftshift(np.fft.fftfreq(self.n_spectral_y, self.dy))
        return fx, fy

    def get_apodizing_window(
        self,
        kind: Literal["tukey", "hanning"] | None = None,
        *,
        sym: bool = True,
        **kwargs: Any,
    ) -> tuple[NDArray, float]:
        """Compute apodizing window.

        Parameters
        ----------
        kind:
            Either 'tukey' or 'hanning'. If None, defaults to attribute
            :attr:`apodizing_kind`.
        kwargs:
            Passed to the function creating the 1D apodizing window.

        Returns
        -------
        window:
            2D apodizing window.
        normalization:
            Window correction factor.
        """
        if kind is None:
            kind = self.apodizing_kind
        if kind == "tukey":
            win_x = tukey(self.n_spectral_x, sym=sym, **kwargs)
            win_y = tukey(self.n_spectral_y, sym=sym, **kwargs)
        elif kind == "hanning":
            win_x = hann(self.n_spectral_x, sym=sym, **kwargs)
            win_y = hann(self.n_spectral_y, sym=sym, **kwargs)
        else:
            raise KeyError(f"Unsupported apodizing window kind '{kind}'")

        win = win_x[None, :] * win_y[:, None]

        # Window correction factors
        dk_x = 1 / (self.dx * self.n_spectral_x)
        dk_y = 1 / (self.dy * self.n_spectral_y)
        win_correction_x = 1 / np.mean(win_x**2)
        win_correction_y = 1 / np.mean(win_y**2)
        normalization = (win_correction_x * win_correction_y) / (dk_x * dk_y)

        return win, normalization

    def asdict(self) -> dict[str, Any]:
        """Return self as a dictionnary."""
        return asdict(self)

    def psd(self, box: NDArray, norm: str = "ortho") -> NDArray:
        """Compute 2D PSD."""
        window, normalization = self.get_apodizing_window("hanning")
        fft = fft_2d(box, window, norm=norm)
        psd = np.abs(fft) ** 2 * normalization
        return psd

    def cross_psd(self, a: NDArray, b: np.ndarray) -> PSD2:
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
        assert a.shape == (self.box_size_y, self.box_size_x)
        assert b.shape == (self.box_size_y, self.box_size_x)

        window, normalization = self.get_apodizing_window("hanning")

        fft_a = fft_2d(a, window)
        fft_b = fft_2d(b, window)

        psd_a = np.abs(fft_a) ** 2 * normalization
        psd_b = np.abs(fft_b) ** 2 * normalization
        phase = fft_b * np.conj(fft_a) * normalization

        psd2 = PSD2(psd_a, psd_b, phase)

        return psd2


@dataclass(kw_only=True)
class FFTWelchComputer(FFTComputer):
    """Compute power-spectra using Welch's method."""

    tile_size_x: int
    """Tile size in pixels."""
    tile_size_y: int
    """Tile size in pixels."""

    tile_increment_x: int
    """Shift of tile in pixels."""
    tile_increment_y: int
    """Shift of tile in pixels."""

    @property
    def n_spectral_x(self) -> int:
        """Number of frequencies."""
        return self.tile_size_x

    @property
    def n_spectral_y(self) -> int:
        """Number of frequencies."""
        return self.tile_size_y

    @property
    def _spectra_shape(self) -> tuple[int, int]:
        return (self.n_spectral_y, self.n_spectral_x)

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

    def psd(self, box: NDArray, norm: str = "ortho") -> NDArray:
        """Compute FFT for a box."""
        assert box.shape == (self.box_size_y, self.box_size_x)

        tile_indices_x = self.get_tile_indices_x()
        tile_indices_y = self.get_tile_indices_y()
        n_tile_x = len(tile_indices_x)
        n_tile_y = len(tile_indices_y)

        window, normalization = self.get_apodizing_window("hanning")

        # Initialize mean PSD
        psd = np.zeros(self._spectra_shape)

        for i_tile_y in tile_indices_y:
            slice_y = slice(i_tile_y, i_tile_y + self.tile_size_y)
            for i_tile_x in tile_indices_x:
                slice_x = slice(i_tile_x, i_tile_x + self.tile_size_x)
                tile_fft = fft_2d(box[slice_y, slice_x], window, norm=norm)
                psd[:, :] = psd[:, :] + np.abs(tile_fft) ** 2 * normalization

        psd = psd / (n_tile_x * n_tile_y)

        return psd

    def cross_psd(self, a: NDArray, b: np.ndarray) -> PSD2:
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
        assert a.shape == (self.box_size_y, self.box_size_x)
        assert b.shape == (self.box_size_y, self.box_size_x)

        tile_indices_x = self.get_tile_indices_x()
        tile_indices_y = self.get_tile_indices_y()
        n_tile_x = len(tile_indices_x)
        n_tile_y = len(tile_indices_y)

        window, normalization = self.get_apodizing_window("hanning")

        # Initialize mean PSD
        psd_a = np.zeros(self._spectra_shape)
        psd_b = np.zeros(self._spectra_shape)
        phase = np.zeros(self._spectra_shape, dtype=complex)

        for i_tile_y in tile_indices_y:
            slice_y = slice(i_tile_y, i_tile_y + self.tile_size_y)
            for i_tile_x in tile_indices_x:
                slice_x = slice(i_tile_x, i_tile_x + self.tile_size_x)

                a_tile_fft = fft_2d(a[slice_y, slice_x], window)
                b_tile_fft = fft_2d(b[slice_y, slice_x], window)

                psd_a[:, :] += np.abs(a_tile_fft) ** 2 * normalization
                psd_b[:, :] += np.abs(b_tile_fft) ** 2 * normalization
                phase[:, :] += b_tile_fft * np.conj(a_tile_fft) * normalization
        n_tiles = n_tile_x * n_tile_y
        psd_a /= n_tiles
        psd_b /= n_tiles
        phase /= n_tiles

        psd2 = PSD2(psd_a, psd_b, phase)

        return psd2
