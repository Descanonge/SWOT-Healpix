"""Define datasets."""

import datetime as dt
import glob
import logging
import os
from dataclasses import dataclass
from typing import Literal, Self

log = logging.getLogger(__name__)


def check_target(
    target: str,
    is_dir: bool = False,
    create_dir: bool = True,
    allow_overwrite: bool = True,
):
    """Check status of target file/directory.

    Parameters
    ----------
    target:
        Target file or directory to be written to.
    is_dir:
        True if the target is a directory.
    create_dir:
        If True, create missing directories leading to the target.
    clobber:
        If False and the target is a file that already exist, raise an exception.
    """
    if is_dir:
        target_dir = target
    else:
        target_dir = os.path.dirname(target)

    if not os.path.isdir(target_dir):
        log.info("Creating directories %s", target_dir)
        os.makedirs(target_dir, exist_ok=True)

    if not is_dir and not allow_overwrite and os.path.isfile(target):
        raise RuntimeError(f"Output file {target} already exists.")


@dataclass
class SWOT:
    """Defines SWOT data location."""

    cycle: int
    """Number of repeat cycle."""
    track: int
    """Number of pass in repeat cycle."""

    dir_data: str = "/home/chaeck/Data"
    """Root directory containing data."""

    product: str = "L2_LR_SSH"
    """Product name.

    Constructed as `<level>_<origin>[_HP]_<variable>`.
    - level: L2 or L3
    - origin: LR or HR
    - HP: if data is resampled on Healpix
    - variable: type of product (SSH, PSD)
    """

    version: str = "PGC0"
    """Reprocessing version.

    This is different between L2 and L3.
    """

    variant: str = "Unsmoothed"
    """Name of product variant.

    Basic, Expert, Unsmoothed.
    """

    @property
    def pass_type(self) -> Literal["ascending", "descending"]:
        """Type of pass (ascending or descending)."""
        if self.track % 2 == 0:
            return "descending"
        return "ascending"

    def get_fields(self, **kwargs) -> Self:
        """Return dataclasses fields, overwriting them with `kwargs`."""
        current = {f: getattr(self, f) for f in self.__dataclass_fields__}
        merged = current | kwargs
        return type(self)(**merged)

    def get_directory(self, **kwargs) -> str:
        """Path to SWOT data (relative to data root directory).

        :param kwargs: Overwrite current field values for this function.
        """
        fields = self.get_fields(**kwargs)
        return os.path.join(
            fields.dir_data,
            "SWOT",
            fields.product,
            fields.version,
            f"{fields.cycle:03d}",
            f"{fields.track:03d}",
        )

    def get_file(self, **kwargs) -> str:
        """Return corresponding filename on disk.

        :param kwargs: Overwrite current field values for this function.
        """
        fields = self.get_fields(**kwargs)
        root = fields.get_directory()
        pattern = (
            f"SWOT_{fields.product}_{fields.variant}"
            f"_{fields.cycle:03d}_{fields.track:03d}_*"
            f"_{fields.version}*.nc"
        )
        files = glob.glob(pattern, root_dir=str(root))
        if len(files) == 0:
            raise RuntimeError(
                f"No files found on disk with pattern '{root}/{pattern}'"
            )
        if len(files) > 1:
            raise RuntimeError(
                f"More than one file found with pattern '{root}/{pattern}'"
            )

        return os.path.join(root, files[0])

    def make_filename(
        self,
        coverage_start: dt.datetime | None = None,
        coverage_end: dt.datetime | None = None,
        id: str | None = None,
        **kwargs,
    ) -> str:
        """Create filename from scratch.

        If `coverage_start` and `coverage_end` are supplied, there are specified in the
        same format as SWOT filenames.
        If `id` is specified, it is added at the end of the filename.

        :param kwargs: Overwrite current field values for this function.
        """
        fields = self.get_fields(**kwargs)
        filename = (
            f"SWOT_{fields.product}_{fields.variant}"
            f"_{fields.cycle:03d}_{fields.track:03d}"
        )

        if coverage_start is not None and coverage_end is not None:
            filename += "_{}_{}".format(
                *[c.strftime("%Y%m%dT%H%M%S") for c in [coverage_start, coverage_end]]
            )

        filename += f"_{fields.version}"

        if id is not None:
            filename += f"_{id}"

        filename += ".nc"

        return os.path.join(fields.get_directory(), filename)
