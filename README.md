
# SWOT-Healpix

Resample SWOT data onto Healpix.

## Setup

### Environment 
``` sh
git clone https://github.com/Descanonge/SWOT-Healpix
cd SWOT-Healpix

micromamba env create -f environment.yml

micromamba activate swot-healpix
pip install -e .
```

### Data

SWOT data is either retrieved from CNES object storage, or from local disk. Data
is retrieved by the class `swot_healpix.data.SWOT`, which specifies a root
directory in attribute `dir_data`. It is then arranged in folders:
`SWOT/<product>/<version>/cycle/track`. See the class documentation for more
details.
