#!/usr/bin/env python3

from collections.abc import Mapping
from typing import TYPE_CHECKING

import healpix_geo
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from cdshealpix import from_ring
from healpix_plot.healpix import HealpixGrid
from matplotlib.collections import QuadMesh
from matplotlib.colorizer import Colorizer

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Norm
    from numpy.typing import NDArray


def reindex_ring_to_2d(level: int) -> np.ndarray:
    """Return indices to reorder cells from ring order to 2d.

    This uses the fact that cells in the ring indexing scheme are arranged from left
    to right and top to bottom.

    To use those indices: Be sure to use 'ring' indexing scheme and that cells are
    sorted by id:
    ```
    cell_ids = to_ring(cell_ids, depth=level)
    sort_indices = np.argsort(cell_ids)
    cell_ids = cell_ids[sort_indices]
    cell_data = cell_data[sort_indices]
    ```
    Then to wrap data into a 2d array:
    ```
    cell_data_2d = np.zeros((2**level, 2**level))
    np.put(cell_data_2d, reindex_ring_to_2d(level), cell_data)
    ```

    Example (level 2)
    -----------------
    Left is ring-order, right is 2D (flattened) order.
    ```
              0                        15
           1     2                  14    11
        3     4     5            13    10     7
     6     7     8     9  =>  12     9     6     3
       10    11    12             8     5     2
          13    14                   4     1
             15                         0
    ```
    """
    n_side = 2**level
    indices = np.zeros(n_side**2, dtype="i8")
    i_cell_start = 0
    n_cell_ring = 1
    for _ in range(n_side):
        ix = np.arange(0, n_cell_ring)
        iy = np.arange(n_cell_ring - 1, -1, -1)
        indices[i_cell_start : i_cell_start + n_cell_ring] = iy * n_side + ix

        i_cell_start += n_cell_ring
        n_cell_ring += 1

    n_cell_ring -= 2

    for i_ring in range(n_side - 1):
        ix = np.arange(0, n_cell_ring) + i_ring + 1
        iy = np.arange(n_cell_ring - 1, -1, -1) + i_ring + 1
        indices[i_cell_start : i_cell_start + n_cell_ring] = iy * n_side + ix

        i_cell_start += n_cell_ring
        n_cell_ring -= 1

    indices = np.asarray(indices)
    return indices


def reindex_nested_to_2d(level: int) -> NDArray:
    """Return indices to reorder cells from nested order to 2d.

    To use those indices: Be sure to use 'nested' indexing scheme and that cells are
    sorted by id:
    ```
    cell_ids = from_ring(cell_ids, depth=level)
    sort_indices = np.argsort(cell_ids)
    cell_ids = cell_ids[sort_indices]
    cell_data = cell_data[sort_indices]
    ```
    Then to wrap data into a 2d array:
    ```
    cell_data_2d = np.zeroes((2**level, 2**level))
    cell_data_2d[reindex_nested_to_2d(level)] = cell_data
    ```

    Example (level 2)
    -----------------
    Left is nested-order, right is 2D (flattened) order.
    ```
              15                        15
           14    13                  14    11
        11    12     7            13    10     7
     10     9     6     5  =>  12     9     6     3
         8     3     4             8     5     2
            2     1                   4     1
               0                         0
    ```
    """
    n_side = 2**level
    cell_ids = np.arange(n_side**2)

    # When considering nested indexing:
    # - 0 (bottom cell): shift by 0
    # - 1 (right cell): shift by 1 cell
    # - 2 (left cell): shift by 1 row
    # - 3 (top cell): shift by 1 row + 1 cell
    shift = np.array([0, 1, n_side, n_side + 1])

    reindex = np.zeros(n_side**2, dtype="i8")
    for i_level in range(level):
        reindex += 2**i_level * shift[cell_ids % 4]
        cell_ids = np.right_shift(cell_ids, 2)

    return reindex


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
    if tile_cell_ids_partial.size == 4 ** (level - level_tile):
        tile_cell_ids = tile_cell_ids_partial
        tile_cell_data = tile_cell_data_partial
    else:
        # We need to fill missing cells in tile. We need the index of each cell,
        # relative to the parent tile
        # Starting id for tile (at fine level)
        tile_id0 = np.left_shift(tile_id, 2 * (level - level_tile))
        tile_cell_ids_relative = tile_cell_ids_partial - tile_id0

        tile_cell_ids = np.arange(4 ** (level - level_tile), dtype="u8") + tile_id0
        tile_cell_data = np.full(4 ** (level - level_tile), np.nan)
        tile_cell_data[tile_cell_ids_relative] = tile_cell_data_partial
    return tile_cell_ids, tile_cell_data


def _reindex_tile_2d(
    tile_cell_ids: NDArray,
    tile_cell_data: NDArray,
    n_side: int,
    nested2d_indices: NDArray,
) -> tuple[NDArray, NDArray]:
    """Reindex tile from nested to 2D."""
    tile_cell_data_2d = np.zeros((n_side, n_side))
    np.put(tile_cell_data_2d, nested2d_indices, tile_cell_data)
    tile_cell_ids_2d = np.zeros((n_side, n_side), dtype="u8")
    np.put(tile_cell_ids_2d, nested2d_indices, tile_cell_ids)

    return tile_cell_ids_2d, tile_cell_data_2d


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

    return grid[0], grid[1]


def _pcolormesh(
    ax: Axes,
    x: NDArray,
    y: NDArray,
    c: NDArray,
    *,
    tile_cell_ids_partial,
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

    ax.add_collection(collection, autolim=False)

    # Use partial (ie valid) cells for scaling data view
    tile_xy = np.stack(
        healpix_geo.nested.healpix_to_lonlat(
            tile_cell_ids_partial,
            depth=healpix_grid.level,
            ellipsoid=healpix_grid.ellipsoid,
        ),
        axis=-1,
    )
    minx, miny = np.min(tile_xy, axis=0)
    maxx, maxy = np.max(tile_xy, axis=0)
    collection.sticky_edges.x[:] = [minx, maxx]
    collection.sticky_edges.y[:] = [miny, maxy]
    ax.update_datalim([[minx, miny], [maxx, maxy]])

    return collection


def healpix_pcolormesh(
    cell_ids: NDArray,
    cell_data: NDArray,
    healpix_grid: HealpixGrid | Mapping,
    level_tile: int | None = None,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: Colormap | str | None = None,
    norm: Norm | str | None = None,
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

    # Size of tiles
    n_side = 2 ** (level - level_tile)

    if ax is None:
        ax = plt.gca()

    # We need to enseure that the norm has vmin and vmax specified,
    # otherwise each tile will have different ranges
    colorizer = Colorizer(cmap=cmap, norm=norm)
    c_vmin, c_vmax = colorizer.get_clim()
    if c_vmin is None:
        c_vmin = vmin if vmin is not None else np.min(cell_data)
    if c_vmax is None:
        c_vmax = vmax if vmax is not None else np.max(cell_data)
    colorizer.set_clim(c_vmin, c_vmax)

    # Build indices to reindex a tile from nested scheme to 2d
    n_side = 2 ** (level - level_tile)
    nested2d_indices = reindex_nested_to_2d(level - level_tile)

    if healpix_grid.indexing_scheme == "ring":
        # Need to reindex and sort by id
        cell_ids = from_ring(cell_ids, level)
    if not np.all(cell_ids[:-1] < cell_ids[1:]):
        sort_indices = np.argsort(cell_ids)
        cell_ids = cell_ids[sort_indices]
        cell_data = cell_data[sort_indices]

    # We do some indexing work that will fail if dtypes are mismatched
    if cell_ids.dtype.kind == "i":
        cell_ids = cell_ids.astype("u8")

    # Find the parent tile and the indices to select each tile from data
    cell_tile_ids = healpix_geo.nested.zoom_to(cell_ids, level, level_tile)
    tile_ids, tile_sections = np.unique(cell_tile_ids, return_index=True)
    tile_indices = np.split(np.arange(cell_ids.size), tile_sections)[1:]

    quadmeshes = []

    for i_tile, tile_id in enumerate(tile_ids):
        tile_cell_ids_partial = cell_ids[tile_indices[i_tile]]
        tile_cell_data_partial = cell_data[tile_indices[i_tile]]

        tile_cell_ids, tile_cell_data = _fill_tile(
            tile_id, tile_cell_ids_partial, tile_cell_data_partial, level, level_tile
        )
        tile_cell_ids_2d, tile_cell_data_2d = _reindex_tile_2d(
            tile_cell_ids, tile_cell_data, n_side, nested2d_indices
        )

        x, y = _compute_grid(tile_cell_ids_2d, n_side, healpix_grid)

        quadmeshes.append(
            _pcolormesh(
                ax,
                x,
                y,
                tile_cell_data_2d,
                tile_cell_ids_partial=tile_cell_ids_partial,
                healpix_grid=healpix_grid,
                colorizer=colorizer,
                **kwargs,
            )
        )
    ax._request_autoscale_view()  # type: ignore[attr-defined]

    return quadmeshes
