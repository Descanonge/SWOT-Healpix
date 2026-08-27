#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import healpix_geo
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from cdshealpix import from_ring
from healpix_plot.healpix import HealpixGrid
from matplotlib.collections import QuadMesh
from matplotlib.colorizer import Colorizer
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from numpy.typing import NDArray


def _find_tiles(
    cell_ids: NDArray, level: int, level_tile: int
) -> tuple[NDArray, NDArray]:
    """Find tile ids and splits to select them."""
    cell_tile_ids = healpix_geo.nested.zoom_to(cell_ids, level, level_tile)
    tile_ids, tile_splits = np.unique(cell_tile_ids, return_index=True)
    tile_splits = np.append(tile_splits, [None])
    return tile_ids, tile_splits


def _extract_tile(
    cell_ids: NDArray, cell_data: NDArray, splits: NDArray, i_tile: int
) -> tuple[NDArray, NDArray]:
    """Extract tile ids and data."""
    slc = slice(splits[i_tile], splits[i_tile + 1])
    return cell_ids[slc], cell_data[slc]


def _fill_tile(
    tile_id: np.uint64,
    tile_cell_ids_partial: NDArray,
    tile_cell_data_partial: NDArray,
    level: int,
    level_tile: int,
) -> tuple[NDArray, NDArray]:
    """Fill tile if necessary.

    The parent tile may have missing values (not present in input data). Return 1D array
    of the tile with missing values as NaNs.
    """
    n_cells = 4 ** (level - level_tile)
    if tile_cell_ids_partial.size == n_cells:
        tile_cell_ids = tile_cell_ids_partial
        tile_cell_data = tile_cell_data_partial
    else:
        # We need to fill missing cells in tile. We need the index of each cell,
        # relative to the parent tile
        # Starting id for tile (at fine level)
        tile_id0 = np.left_shift(tile_id, 2 * (level - level_tile))
        tile_cell_ids_relative = tile_cell_ids_partial - tile_id0

        tile_cell_ids = np.arange(n_cells, dtype=tile_cell_ids_partial.dtype) + tile_id0
        tile_cell_data = np.full(n_cells, np.nan, dtype=tile_cell_data_partial.dtype)
        tile_cell_data[tile_cell_ids_relative] = tile_cell_data_partial
    return tile_cell_ids, tile_cell_data


def _build_2d_indices(size: int) -> NDArray:
    """Build indices used to reshape tiles to 2d.

    Parameters
    ----------
    size:
        Size of tiles as a healpix level.
    """
    n_side = 2 ** (size)
    _, ix, iy = healpix_geo.nested.pix2xyf(np.arange(4 ** (size)), depth=size)
    return ix + n_side * iy


def _reindex_tile_2d(
    tile_cell_ids: NDArray,
    tile_cell_data: NDArray,
    n_side: int,
    nested2d_indices: NDArray,
) -> tuple[NDArray, NDArray]:
    """Reindex tile from nested to 2D."""
    tile_cell_data_2d = np.zeros((n_side, n_side), dtype=tile_cell_data.dtype)
    np.put(tile_cell_data_2d, nested2d_indices, tile_cell_data)
    tile_cell_ids_2d = np.zeros((n_side, n_side), dtype="u8")
    np.put(tile_cell_ids_2d, nested2d_indices, tile_cell_ids)

    return tile_cell_ids_2d, tile_cell_data_2d


def wrap_longitude(lon: NDArray, base_cell: int) -> NDArray:
    lon = (lon + 180) % 360 - 180
    if base_cell in [1, 9]:
        lon[lon == -180] = 180
    return lon


def _compute_grid(
    tile_cell_ids_2d: NDArray, n_side: int, healpix_grid: HealpixGrid
) -> tuple[NDArray, NDArray]:
    """Compute the grid coordinates for pcolormesh.

    Returns
    -------
    lon, lat:
        Coordinates as 2D arrays each of size `(n_side+1, n_side+1)`.
    """
    vertices = np.stack(
        healpix_geo.nested.vertices(
            tile_cell_ids_2d, depth=healpix_grid.level, ellipsoid=healpix_grid.ellipsoid
        ),
        axis=0,
    )

    grid = np.zeros((2, n_side + 1, n_side + 1))

    # Grid "top" is towards north-west, x is fast index.
    # Bottom vertex for bottom-right part of grid
    grid[:, :-1, :-1] = vertices[:, :, :, 0]
    # Right vertex for last column
    grid[:, :-1, -1] = vertices[:, :, -1, 1]
    # Left vertex for top row
    grid[:, -1, :-1] = vertices[:, -1, :, 3]
    # Top vertex for top-right cell
    grid[:, -1, -1] = vertices[:, -1, -1, 2]

    lon, lat = grid[0], grid[1]
    # Map longitude to [-180, 180]
    # lon = wrap_longitude(lon)

    base_cell = healpix_geo.nested.zoom_to(
        tile_cell_ids_2d[0, 0], healpix_grid.level, 0
    )
    if base_cell != 6:
        lon = wrap_longitude(lon, base_cell)

    return lon, lat


def find_extent(cell_ids: NDArray, healpix_grid: HealpixGrid) -> Bbox:
    """Find extend of given cells."""
    # Use partial (ie valid) cells for scaling data view
    lon, lat = np.stack(
        healpix_geo.nested.healpix_to_lonlat(
            cell_ids,
            depth=healpix_grid.level,
            ellipsoid=healpix_grid.ellipsoid,
        ),
        axis=0,
    )
    base_cell = healpix_geo.nested.zoom_to(cell_ids[0], healpix_grid.level, 0)
    # central cell, need to map longitude in [-180, 180]
    if base_cell == 4:
        lon = wrap_longitude(lon, base_cell)

    i_xmin = np.argmin(lon)
    i_xmax = np.argmax(lon)
    i_ymin = np.argmin(lat)
    i_ymax = np.argmax(lat)

    vlon, vlat = healpix_geo.nested.vertices(
        cell_ids[[i_xmin, i_ymin, i_xmax, i_ymax]],
        depth=healpix_grid.level,
        ellipsoid=healpix_grid.ellipsoid,
    )
    if base_cell != 6:
        vlon = wrap_longitude(vlon, base_cell)

    return Bbox(
        [
            [vlon[0, 3], vlat[1, 0]],
            [vlon[2, 1], vlat[3, 2]],
        ]
    )


def _pcolormesh(
    ax: Axes,
    x: NDArray,
    y: NDArray,
    c: NDArray,
    /,
    *,
    extent: tuple[float, float, float, float],
    healpix_grid: HealpixGrid,
    colorizer: Colorizer,
    **kwargs,
) -> QuadMesh:
    """Pcolormesh that deals with data limits appropriately.

    pcolormesh scales the view to fit the data, even if invalid. Because our tiles can
    contain a lot of invalid data this can create a lot of white space.

    This functions is a copy of pcolormesh that scales the view correctly.

    Since we supply a Colorizer, no need for some norm-related code.
    """
    shading = kwargs.pop("shading", None)
    shading = mpl._val_or_rc(shading, "pcolor.shading").lower()  # type: ignore[attr-defined]
    kwargs.setdefault("edgecolors", "none")
    kwargs.setdefault("antialiased", None)

    X, Y, C, shading = ax._pcolorargs(  # type: ignore[attr-defined]
        "pcolormesh", x, y, c, shading=shading, kwargs=kwargs
    )
    coords = np.stack([X, Y], axis=-1)

    kwargs.setdefault("snap", mpl.rcParams["pcolormesh.snap"])

    collection = QuadMesh(
        coords,
        array=C,
        colorizer=colorizer,
        shading=shading,
        **kwargs,
    )

    t = collection._transform  # type: ignore[attr-defined]
    if not isinstance(t, mtransforms.Transform) and hasattr(t, "_as_mpl_transform"):
        t = t._as_mpl_transform(ax.axes)

    if t and any(t.contains_branch_separately(ax.transData)):
        trans_to_data = t - ax.transData
        coords = trans_to_data.transform(coords)

    xmin, xmax, ymin, ymax = extent
    ax.update_datalim([[xmin, ymin], [xmax, ymax]])
    collection.sticky_edges.x[:] = [xmin, xmax]
    collection.sticky_edges.y[:] = [ymin, ymax]

    ax.add_collection(collection, autolim=False)

    return collection


def _to_nested_sorted(
    cell_ids: NDArray, cell_data: NDArray, healpix_grid: HealpixGrid
) -> tuple[NDArray, NDArray]:
    """Ensure data is indexed nested and cells are sorted by id."""
    if healpix_grid.indexing_scheme == "ring":
        # Need to reindex and sort by id
        cell_ids = from_ring(cell_ids, healpix_grid.level)
    if not np.all(cell_ids[:-1] < cell_ids[1:]):
        sort_indices = np.argsort(cell_ids)
        cell_ids = cell_ids[sort_indices]
        cell_data = cell_data[sort_indices]

    return cell_ids, cell_data


def healpix_pcolormesh(
    cell_ids: NDArray,
    cell_data: NDArray,
    healpix_grid: HealpixGrid | Mapping,
    level_tile: int | None = None,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: Colormap | str | None = None,
    norm: Normalize | str | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> list[QuadMesh]:
    """Plot data using pcolormesh in tiles.

    In nested indexing, data from the same parent cell can easily be reindexed in 2D.
    This function takes advantage of this to chunk data into larger tiles and plot each
    one of them with a :func:`~matplotlib.pyplot.pcolormesh` call.

    Parameters
    ----------
    cell_ids:
        The cell ids describing the spatial position of the data.
    cell_data:
        The data to plot as a 1D array.
    healpix_grid:
        Object or dictionnary giving parameters of the input data Healpix grid.
    level_tile:
        Healpix level/depth of tiles. If left to None, will default to `level - 11`,
        giving tiles of size 2048 x 2048.
    vmin, vmax:
        When no explicit norm is passed, `vmin` and `vmax` will be used to define the
        data range that the colormap covers.
    cmap:
       The Colormap instance or registered colormap name used to map scalar data to
       colors.
    norm:
        Normalization method used to scale data to the [0, 1] range before mapping to
        colors. See documentation of :func:`~matplotlib.pyplot.pcolormesh` for details.
    kwargs:
        Other arguments passed to `pcolormesh`.
    """
    if isinstance(healpix_grid, Mapping):
        healpix_grid = HealpixGrid(**healpix_grid)

    level = healpix_grid.level

    if level_tile is None:
        level_tile = max(0, level - 11)
    if level_tile >= level:
        raise ValueError(
            f"Tile level ({level_tile}) should be coarser than data level ({level})"
        )

    if ax is None:
        ax = plt.gca()

    # We need to ensure vmin is set (either via kwargs or via the norm) otherwise
    # each tile would have a different range
    if isinstance(norm, Normalize):
        if norm.vmin is None:
            norm.vmin = np.min(cell_data)
        if norm.vmax is None:
            norm.vmax = np.max(cell_data)
    else:
        if vmin is None:
            vmin = np.min(cell_data)
        if vmax is None:
            vmax = np.max(cell_data)

    if isinstance(ax, GeoAxes):
        kwargs.setdefault("transform", ccrs.PlateCarree())

    # Size of tiles
    n_side = 2 ** (level - level_tile)
    # Build indices to reindex a tile from nested scheme to 2d
    nested2d_indices = _build_2d_indices(level - level_tile)

    cell_ids, cell_data = _to_nested_sorted(cell_ids, cell_data, healpix_grid)

    # We do some indexing work that will fail if dtypes are mismatched
    if cell_ids.dtype.kind == "i":
        cell_ids = cell_ids.astype("u8")

    # Find the parent tile and the indices to select each tile from data
    tile_ids, tile_splits = _find_tiles(cell_ids, level, level_tile)

    quadmeshes = []

    ax_extent = ax.dataLim.frozen()
    any_partial = False
    for i_tile, tile_id in enumerate(tile_ids):
        tile_cell_ids_partial, tile_cell_data_partial = _extract_tile(
            cell_ids, cell_data, tile_splits, i_tile
        )
        tile_cell_ids, tile_cell_data = _fill_tile(
            tile_id, tile_cell_ids_partial, tile_cell_data_partial, level, level_tile
        )
        tile_cell_ids_2d, tile_cell_data_2d = _reindex_tile_2d(
            tile_cell_ids, tile_cell_data, n_side, nested2d_indices
        )
        x, y = _compute_grid(tile_cell_ids_2d, n_side, healpix_grid)

        # This will be changed by pcolormesh, save it
        ignore_existing_data_limits = ax.ignore_existing_data_limits

        quadmesh = ax.pcolormesh(
            x,
            y,
            tile_cell_data_2d,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            cmap=cmap,
            **kwargs,
        )

        if tile_cell_ids_partial.size < tile_cell_ids.size:
            any_partial = True
            tile_extent = find_extent(tile_cell_ids_partial, healpix_grid)
            ax_extent.update_from_data_xy(
                tile_extent.get_points(), ignore=ignore_existing_data_limits
            )
            ax.ignore_existing_data_limits = False
            quadmesh.sticky_edges.x[:] = tile_extent.intervalx
            quadmesh.sticky_edges.y[:] = tile_extent.intervaly

        quadmeshes.append(quadmesh)

    if any_partial:
        if isinstance(ax, GeoAxes):
            ax.set_extent((*ax_extent.intervalx, *ax_extent.intervaly))
        else:
            ax.dataLim.set_points(ax_extent.get_points())
            ax.autoscale(enable=None)  # type: ignore[arg-type]

    return quadmeshes
