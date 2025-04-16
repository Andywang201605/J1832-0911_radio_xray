### VLBA related code

Code and some intermediate results for VLBA analysis

`vlba_astrometry.ipynb` is the notebook used for is the notebook used for plotting an example VLBA image and the astrometry fit results.

The astrometry fit was performed using `pmpar` (https://github.com/walterfb/pmpar). `input.pmpar` is the input file for `pmpar`, which contains raw measurement of the source position (with 1 mas added to each coordinate to account for systematic error). `pmpar_e` and `pmpar_t` are the output files from `pmpar`

> raw VLBA data are available in https://data.nrao.edu