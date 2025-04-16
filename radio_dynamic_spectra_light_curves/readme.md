### Radio Dynamic Spectra and Related Analysis

This directory contains two python scripts that are useful in loading and analyzing dynamic spectra files (those pickle files under `data` folder or in zenodo).
- `vis_ds.py` - functions to load dynamic spectra (Credit: Emil Lenc)
- `burst.py` - functions to fold dynamic spectra (and also light curves)

> raw ASKAP data are available in https://research.csiro.au/casda/; raw GMRT data are available in https://naps.ncra.tifr.res.in/goa/data/search, raw MeerKAT data are avaiable in https://archive.sarao.ac.za/

#### Source Discovery Plot - `discovery_plot.ipynb`

Code for Figure 1 (full polarisation light curves and CRACO dynamic spectrum for the discovery observation) and rotation measure synthesis using `RMTools`.

#### Radio and X-ray Folded Light Curves - `radio_xray_fold.ipynb`

Code for Figure 2 (radio folded light curves for long observations and Chandra observations in 2024 Feburary, with exposure time in each phase bin) and radio and X-ray burst phase delay estimations.

#### Folded ASKAP Light Curves - `askap_folded_fullpol.ipynb`

Code for plotting full polarisation light curves for four selected ASKAP observations (long observations).

> folded light curves in total intensity, linear polarisation, circular polarisation and position angle (and its uncertainty) are in `data/askap_folded`

#### Simultaneous Wide Frequency Dynamic Spectra - `widefreq_radio_dynamic_spectra.ipynb`

Code for Extended Data Figure <font color="red">TODO</font> (dynamic spectra for MeerKAT observation (at UHF and S0 band), and GMRT observation (at B3, B4, and B5))

#### PTUSE Position Angle Plot - `ptuse_data.ipynb`

Code for Supplemetary Information Figure <font color="red">TODO</font> (polarisation position angle evolution within a pulse)