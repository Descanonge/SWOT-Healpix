#!/usr/bin/env python3

from typing import TYPE_CHECKING

import healpix_geo
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from cdshealpix import from_ring, to_ring
from matplotlib.collections import QuadMesh
from matplotlib.colorizer import Colorizer

if TYPE_CHECKING:
    from healpix_plot import EllipsoidLike
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Norm
    from numpy.typing import NDArray


def reindex_ring_to_2d(n_side: int) -> np.ndarray:
    """Return indices to reorder cells from ring order to 2d.

    For Healpix cells arranged in a square (rotated 45°). They can be obtained from a
    parent cell or from a center cell expanded with `healpix_geo.*.kth_neighborhood`.

    This cannot work if the tile crosses wrapping longitude.

    To use those indices: Be sure to use 'ring' indexing scheme. Sort cells by id:
    ```
    cell_ids = to_ring(cell_ids, depth=level)
    sort_indices = np.argsort(cell_ids)
    cell_ids = cell_ids[sort_indices]
    cell_data = cell_data[sort_indices]
    ```
    Then to wrap data into a 2d array:
    ```
    cell_data_2d = np.zeroes((n_side, n_side))
    np.put(cell_data_2d, argsort_ring_to_2d(n_side), cell_data)
    ```
    """
    indices = np.zeros(n_side**2, dtype="i8")
    i_cell_start = 0
    n_cell_ring = 1
    for i_ring in range(n_side):
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


def _pcolormesh(
    ax: Axes,
    x: NDArray,
    y: NDArray,
    c: NDArray,
    tile_cell_ids_partial,
    *,
    level: int,
    colorizer: Colorizer,
    **kwargs,
) -> QuadMesh:
    """Pcolormesh that deals with data limits appropriately.

    pcolormesh scales the view to fit the data, even if invalid. Because our tiles can
    contain a lot of invalid data this can create a lot of white space around the data.

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
        healpix_geo.nested.healpix_to_lonlat(tile_cell_ids_partial, level), axis=-1
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
    level: int,
    level_tile: int | None = None,
    *,
    indexing_scheme: str = "nested",
    ellipsoid: EllipsoidLike = "sphere",
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
        The cell ids describing the spatial position of the data. Cells must be indexed
        as nested (and be sorted in correct order).
    data:
        The data to plot. If 1D, will be color-coded using the standard
        matplotlib mechanisms. If 2D, the last axis must have a size of 3 (for
        RGB) or 4 (for RGBA).
    healpix_grid:
        Object or dictionnary giving parameters of the input data Healpix grid.
    level:
        Healpix level/depth of data.
    level_tile:
        Healpix level/depth of tiles. If left to None, will default to `level - 11`,
        giving tiles of size 2048 x 2048.
    ellipsoid:
        Ellipsoid descriptor. Defaults to "sphere".
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
    if level_tile is None:
        level_tile = max(0, level - 11)
    if level_tile >= level:
        raise ValueError(
            f"Tile level ({level_tile}) should be coarser than data level ({level})"
        )

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

    # Build indices to reindex a tile from ring scheme to 2d
    n_side = 2 ** (level - level_tile)
    ring2d_indices = reindex_ring_to_2d(n_side)

    if indexing_scheme == "ring":
        # Need to reindex and sort by id
        cell_ids = from_ring(cell_ids, level)
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

        # Reindex tile in 2d. This is fairly easy to do from a ring-scheme id.
        tile_cell_ids = to_ring(tile_cell_ids, depth=level)
        sort_indices = np.argsort(tile_cell_ids)
        tile_cell_ids = tile_cell_ids[sort_indices]
        tile_cell_data = tile_cell_data[sort_indices]

        tile_data_2d = np.zeros((n_side, n_side))
        np.put(tile_data_2d, ring2d_indices, tile_cell_data)
        tile_cell_ids_2d = np.zeros(n_side**2, dtype="u8")
        np.put(tile_cell_ids_2d, ring2d_indices, tile_cell_ids)

        # Get cell coordinates. We need to a grid of side N+1
        # (to get the rightmost edge)
        vertices = np.stack(
            healpix_geo.ring.vertices(
                tile_cell_ids_2d, depth=level, ellipsoid=ellipsoid
            ),
            axis=-1,
        )
        vertices = vertices.reshape(n_side, n_side, 4, 2)

        # Start with top vertex (of the healpix diamond cell)
        grid = vertices[:, :, 2]
        # Add rightmost column (right vertex)
        grid = np.concatenate([grid, vertices[:, [-1], 1]], axis=1)
        # Take last line (left vertex)
        last_line = vertices[-1, :, 3]
        # Add rightmost colum to last line (bottom vertex)
        last_line = np.concatenate([last_line, vertices[-1, [-1], 0]], axis=0)
        # Add last line
        grid = np.concatenate([grid, last_line[None, :]], axis=0)
        x = grid[..., 0]
        y = grid[..., 1]

        # quadmeshes.append(
        #     ax.pcolormesh(x, y, tile_data_2d, colorizer=colorizer, **kwargs)
        # )
        quadmeshes.append(
            _pcolormesh(
                ax,
                x,
                y,
                tile_data_2d,
                tile_cell_ids_partial,
                level=level,
                colorizer=colorizer,
                **kwargs,
            )
        )
    ax._request_autoscale_view()  # type: ignore[attr-defined]

    return quadmeshes
