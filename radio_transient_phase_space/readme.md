### Transient Phase Space Plot

Code adapted based on https://github.com/FRBs/Transient_Phase_Space but implemented in python. `python_plot.ipynb` is the code for plotting the radio transient space plot (but focus on long period radio transient population).

- `gach_rud` - original folder containing basic information such as flux density, pulse duration etc.
- `lpt` - folder containing x and y values for long period radio transients
- `plotconfig.csv` - source type and plotted color
- `plottext.csv` - source type (full text) and text position
- `load_plot_data.py` - functions used to load data under `gach_rud` directory

---
Original `readmd.md` file

This is some code to make a plot that some people have asked me
for. If you use this code, it would be nice if you gave me an
acknowledgemen. You could cite something appropriate like, for
example:

https://ui.adsabs.harvard.edu/abs/2015MNRAS.446.3687P/

OR 

https://ui.adsabs.harvard.edu/abs/2018NatAs...2..865K/

but you don't have to.

For the C version just compile with something like:

gcc phase_space.c -o phase_space
