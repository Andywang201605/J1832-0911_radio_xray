import matplotlib.pyplot as plt
import numpy as np

from astropy import units as u, constants as c 
from astropy.table import Table

# nominal frequency
freq0 = 1.4 * u.GHz
# x-axis units
xunits = u.mJy * u.kpc**2

# colors for different source types
colors = {
    "wd": "m",
    "ulp": "k",
    "gcrt": "brown",
    "sgr": "r",
    "magnetar": "r",
    "pulsar": "gray",
    "cv": "darkturquoise",
    "mcv": "purple",
    "nsxrb": "b",
    "bhxrb": "g",
    "star": "orange",
    "amxp": "purple",
}

### pulsars
def plot_pulsars(ax, ratio=False):
    """Helper function to plot pulsars

    Args:
        ax (matplotlib.axes.Axes, optional): Axes object to plot. Defaults to None.
        ratio (bool, optional): Plot radio/X-ray ob y-axis?. Defaults to False.

    Returns:
        matplotlib.axes.Axes: Plotted axes object
    """

    # In this table radio and X-ray luminosities are estimated as flux times the
    # distance squared, when the flux is the mean integrated flux for radio and the
    # point source flux in X-ray (so mean flux) and hence they are period averaged.
    # All of the radio fluxes
    # Since there are pulars, here the 4pi factor is omitted so add it here. All of the radio fluxes
    # are at 1400 MHz.
    tab = Table.read("data/pulsars.ecsv")
    rad = tab["Lradio"].to(xunits)
    xray = tab["Lxray"] * 4 * np.pi
    mask = tab["upp_lim"] == 1.0

    if ratio:
        yval = (rad * freq0 / xray).to(u.dimensionless_unscaled)
    else:
        yval = xray
    ax.scatter(rad[~mask], yval[~mask], s=6, c=colors["pulsar"], alpha=0.5)

    ### add J0901 here...
    tab = Table.read("data/J0901-4046.txt", format="ascii")
    xray = tab["xray"] * u.erg / u.second / u.cm**2
    radio = tab["radio"] * u.mJy
    d = tab["d"] * u.kpc

    xray = (xray * 4 * np.pi * d**2).to(u.erg / u.second)
    radio = (radio * d**2).to(xunits)
    if ratio:
        yval = (radio * freq0 / xray).to(u.dimensionless_unscaled)
        plot, caps, bars = ax.errorbar(
            radio,
            yval,
            yerr=0.75 * yval,
            lolims=True,
            fmt="s",
            markersize=4,
            color=colors["pulsar"],
            barsabove=True,
            # capsize=2,
            # label="PSR J0901",
            zorder=10,
        )
    else:
        yval = xray
        plot, caps, bars = ax.errorbar(
            radio,
            yval,
            yerr=0.75 * yval,
            uplims=True,
            fmt="s",
            markersize=4,
            color=colors["pulsar"],
            barsabove=True,
            # capsize=2,
            # label="PSR J0901",
            zorder=10,
        )

    caps[1].set_visible(False)

    return ax

### magnetars
def plot_magnetars(ax, ratio=False):
    """Helper function to plot magnetars

    Args:
        ax (matplotlib.axes.Axes, optional): Axes object to plot. Defaults to None.
        ratio (bool, optional): Plot radio/X-ray ob y-axis?. Defaults to False.

    Returns:
        matplotlib.axes.Axes: Plotted axes object
    """

    # In this table radio flux density is in mJy, X-ray flux is in erg/cm^2/s
    # distance is in kpc, radio flux density is mean (so period averaged)
    # X-ray flux is of point source, so period averaged again. The radio fluxes are
    # scaled to 1400 MHz.

    tab = Table.read("data/magnetars.txt", format="ascii")
    sgr = tab["Name"] == "SGR1935+2154"
    radio = tab["Fradio"].data * u.mJy
    xray = tab["Fxray"].data * u.erg / u.cm**2 / u.second
    d = tab["d"].data * u.kpc

    radio = (radio * d**2).to(xunits)
    xray = (4 * np.pi * xray * d**2).to(u.erg / u.second)
    uplims = tab["upp"].data.astype(bool)

    if ratio:
        yval = (radio * freq0 / xray).to(u.dimensionless_unscaled)
    else:
        yval = xray

    err = np.zeros(len(radio))
    err[uplims] = 0.5 * radio[uplims]
    ax.errorbar(
        radio[sgr].to_value(xunits),
        yval[sgr].value,
        xerr=err[sgr],
        xuplims=uplims[sgr],
        fmt="D",
        markersize=3,
        # capsize=1.5,
        # marker="D",
        c=colors["sgr"],
    )
    ax.errorbar(
        radio.to_value(xunits)[~sgr],
        xray.value[~sgr],
        xerr=err[~sgr],
        xuplims=uplims[~sgr],
        fmt="d",
        c=colors["magnetar"],
        markersize=3,
        # capsize=1.5,
    )

    return ax

### X-ray binaries
def plot_xrbs(ax, ratio=False):
    """Helper function to plot X-ray binaries

    Args:
        ax (matplotlib.axes.Axes, optional): Axes object to plot. Defaults to None.
        ratio (bool, optional): Plot radio/X-ray ob y-axis?. Defaults to False.

    Returns:
        matplotlib.axes.Axes: Plotted axes object
    """

    # In these files radio luminosity is in erg/s, X-ray luminosity is in erg/s
    # distance is in kpc. A flat sepctrum is assumed to convert it from 5 to 1.4GHz.
    # The 4pi factor is included here in the Lx
    fac = (u.erg / u.second / (5 * u.GHz)).to(xunits)

    # Plot AMXPs
    amxps = Table.read("data/xrbs/lrlx_data_AMXPs.csv")
    radiomask = np.array([True if "Lr" in i else False for i in amxps["uplim"]])
    radio = amxps["Lr"] * fac.value / 4 / np.pi
    xray = amxps["Lx"] * u.erg / u.second

    if ratio:
        yval = (radio * xunits * freq0 / xray).to(u.dimensionless_unscaled)
    else:
        yval = xray.value

    err = np.zeros(len(amxps))
    err[radiomask] = 0.6 * radio[radiomask]
    _, caps, _ = ax.errorbar(
        radio,
        yval,
        xerr=err,
        xuplims=radiomask,
        fmt=".",
        markersize=6,
        c=colors["amxp"],
        alpha=0.4,
        capsize=1.2,
    )
    caps[3].set_visible(False)

    # Plot BHXRBs
    bhxrbs = Table.read("data/xrbs/lrlx_data_BHs.csv")
    radiomask = np.array([True if "Lr" in i else False for i in bhxrbs["uplim"]])
    radio = bhxrbs["Lr"] * fac.value / 4 / np.pi
    bhxrbs = bhxrbs[~radiomask]
    radiomask = np.array([True if "Lr" in i else False for i in bhxrbs["uplim"]])
    radio = bhxrbs["Lr"] * fac.value / 4 / np.pi
    xray = bhxrbs["Lx"] * u.erg / u.second

    if ratio:
        yval = (radio * xunits * freq0 / xray).to(u.dimensionless_unscaled)
    else:
        yval = xray.value

    err = np.zeros(len(bhxrbs))
    err[radiomask] = 0.6 * radio[radiomask]
    plot, caps, bars = ax.errorbar(
        radio,
        yval,
        xerr=err,
        xuplims=radiomask,
        fmt=".",
        markersize=5,
        c=colors["bhxrb"],
        alpha=0.2,
        capsize=0.,
    )

    # Plot NSXRBs
    nsxrbs = Table.read("data/xrbs/lrlx_data_NSs.csv")
    radiomask = np.array([True if "Lr" in i else False for i in nsxrbs["uplim"]])
    radio = nsxrbs["Lr"] * fac.value / 4 / np.pi
    nsxrbs = nsxrbs[~radiomask]
    radiomask = np.array([True if "Lr" in i else False for i in nsxrbs["uplim"]])
    radio = nsxrbs["Lr"] * fac.value / 4 / np.pi
    xray = nsxrbs["Lx"] * u.erg / u.second

    if ratio:
        yval = (radio * xunits * freq0 / xray).to(u.dimensionless_unscaled)
    else:
        yval = xray.value

    err = np.zeros(len(nsxrbs))
    err[radiomask] = 0.5 * radio[radiomask]
    ax.errorbar(
        radio,
        yval,
        xerr=err,
        xuplims=radiomask,
        fmt=".",
        markersize=5,
        c=colors["nsxrb"],
        alpha=0.2,
        capsize=0.,
    )
    # caps[1].set_visible(False)

    return ax

### stars
def plot_stars(ax, ratio=False):
    """Helper function to plot stars

    Args:
        ax (matplotlib.axes.Axes, optional): Axes object to plot. Defaults to None.
        ratio (bool, optional): Plot radio/X-ray ob y-axis?. Defaults to False.

    Returns:
        matplotlib.axes.Axes: Plotted axes object
    """
    tab = Table.read("data/stars.ecsv")

    # Radio luminosity in these files is in mJy * kpc^2, with a factor of 4pi
    # remove it from the radio
    radio = tab["radio"].quantity / 4 / np.pi
    xray = tab["xray"].quantity

    yval = (radio * freq0 / xray).to(u.dimensionless_unscaled) if ratio else xray

    ax.scatter(
        radio.to_value(xunits),
        yval.value,
        s=6,
        c=colors["star"],
        alpha=0.2,
    )
    return ax

### cvs
def plot_cvs(ax, ratio=False):
    """Helper function to plot CVs/MCVs

    Args:
        ax (matplotlib.axes.Axes, optional): Axes object to plot. Defaults to None.
        ratio (bool, optional): Plot radio/X-ray ob y-axis?. Defaults to False.

    Returns:
        matplotlib.axes.Axes: Plotted axes object
    """

    tab = Table.read("data/Ridder_CV_Fluxes_Multi-Filled.csv")
    xray_upp_lim_mask = tab["0.1-10 keV flux [erg  s^-1 cm^-2]"].data == "…"
    tab = tab[~xray_upp_lim_mask]

    mcvs = tab["MCV"] == "True"
    cvs = tab["MCV"] == "False"

    radio_freq = np.array(
        [np.mean(np.array(i.split("-")).astype(float)) for i in tab["Radio Band [GHz]"]]
    )
    radio_upp_lim_mask = np.array(
        [True if "<" in i else False for i in tab["Radio Flux [uJy]"]]
    )
    radio_fluxes = np.array(
        [float(i.replace("<", "")) for i in tab["Radio Flux [uJy]"]]
    ).astype(float)

    # Do a spectral correction
    radio_fluxes *= radio_freq / 1.4
    radio_lum = (radio_fluxes * u.uJy * (tab["Distance [pc]"] * u.pc) ** 2).to(xunits)

    xray_lum = (
        tab["0.1-10 keV flux [erg  s^-1 cm^-2]"].astype(float)
        * u.erg
        / u.second
        / u.cm**2
        * 4
        * np.pi
        * (tab["Distance [pc]"] * u.pc) ** 2
    ).to(u.erg / u.second)

    # First plot CVs
    if not ratio:
        ax.errorbar(
            radio_lum[cvs],
            xray_lum[cvs],
            fmt="d",
            xuplims=radio_upp_lim_mask[cvs],
            # marker="d",
            markersize=3,
            color=colors["cv"],
            alpha=0.5,
        )

        ax.errorbar(
            radio_lum[mcvs],
            xray_lum[mcvs],
            fmt="s",
            xuplims=radio_upp_lim_mask[mcvs],
            # marker="s",
            markersize=3,
            color=colors["mcv"],
            alpha=0.5,
        )
    else:
        ax.errorbar(
            radio_lum[cvs],
            (radio_lum[cvs] * u.GHz / xray_lum[cvs]).to(u.dimensionless_unscaled),
            fmt="d",
            xuplims=radio_upp_lim_mask[cvs],
            # marker="d",
            markersize=3,
            color=colors["cv"],
            alpha=0.5,
        )

        ax.errorbar(
            radio_lum[mcvs],
            (radio_lum[mcvs] * u.GHz / xray_lum[mcvs]).to(u.dimensionless_unscaled),
            fmt="s",
            xuplims=radio_upp_lim_mask[mcvs],
            # marker="s",
            markersize=3,
            color=colors["mcv"],
            alpha=0.5,
        )
    return ax

### LPRTs - note: this part of code is more complicated, cuz it is currently source based
def plot_gcrts(ax, ratio=False):
    """Helper function to plot gcrts

    Args:
        ax (matplotlib.axes.Axes, optional): Axes object to plot. Defaults to None.
        ratio (bool, optional): Plot radio/X-ray ob y-axis?. Defaults to False.

    Returns:
        matplotlib.axes.Axes: Plotted axes object
    """

    # In this files radio flux density is in mJy, X-ray flux is in erg/cm^2/s
    # distance is in kpc, radio flux density is mean (so period averaged)
    # X-ray flux is of point source, so period averaged again. The radio fluxes are
    # not scaled to 1400 MHz. Add 4 pi factor as needed.
    
    # Plot J1745
    tab = Table.read("data/J1745.txt", format="ascii")
    xray = tab["xray"] * u.erg / u.second / u.cm**2
    radio = tab["radio"] * u.mJy
    d = tab["d"] * u.kpc

    xray = (xray * 4 * np.pi * d**2).to(u.erg / u.second)
    radio = (radio * d**2).to(xunits)

    if ratio:
        yval = (radio * freq0 / xray).to(u.dimensionless_unscaled)
        ax.errorbar(
            radio,
            yval,
            yerr=0.75 * yval,
            lolims=True,
            fmt="h",
            markersize=4,
            # color=colors["gcrt"],
            color="k",
            barsabove=True,
            capsize=2,
            # label="GCRT J1745",
        )
    else:
        yval = xray
        ax.errorbar(
            radio,
            yval,
            yerr=0.75 * yval,
            uplims=True,
            fmt="h",
            markersize=4,
            # color=colors["gcrt"],
            color="k",
            barsabove=True,
            capsize=2,
            # label="GCRT J1745",
        )

    # Plot J1912
    tab = Table.read("data/J1912.txt", format="ascii")
    xray = tab["xray"] * u.erg / u.second / u.cm**2
    radio = tab["radio"] * u.mJy
    d = tab["d"] * u.kpc

    xray = (xray * 4 * np.pi * d**2).to(u.erg / u.second)
    radio = (radio * d**2).to(xunits)

    if ratio:
        yval = (radio * freq0 / xray).to(u.dimensionless_unscaled)
    else:
        yval = xray

    ax.errorbar(
        radio,
        yval,
        fmt="8",
        markersize=4,
        color=colors["wd"],
        barsabove=True,
        capsize=2,
        # label="WD J1912$-$4410",
        zorder=10,
    )

    # Plot Ar SCo
    tab = Table.read("data/ar_sco.txt", format="ascii")
    xray = tab["xray"] * u.erg / u.second / u.cm**2
    radio = tab["radio"] * u.mJy
    d = tab["d"] * u.kpc

    xray = (xray * 4 * np.pi * d**2).to(u.erg / u.second)
    radio = (radio * d**2).to(xunits)

    if ratio:
        yval = (radio * freq0 / xray).to(u.dimensionless_unscaled)
    else:
        yval = xray

    ax.errorbar(
        radio,
        yval,
        fmt="X",
        markersize=4,
        color=colors["wd"],
        barsabove=True,
        capsize=2,
        # label="WD Ar Sco",
        zorder=20,
    )

    # Plot GLEAM1627
    tab = Table.read("data/GLEAM1627.txt", format="ascii")
    xray = tab["xray"] * u.erg / u.second / u.cm**2
    radio = tab["radio"] * u.mJy
    d = tab["d"] * u.kpc

    xray = (xray * 4 * np.pi * d**2).to(u.erg / u.second)
    radio = (radio * d**2).to(xunits)

    if ratio:
        yval = (radio * freq0 / xray).to(u.dimensionless_unscaled)
        ax.errorbar(
            radio,
            yval,
            yerr=0.75 * yval,
            lolims=True,
            fmt="*",
            markersize=6,
            color=colors["ulp"],
            barsabove=True,
            capsize=2,
            # label="ULP GLEAM-X J1627",
            zorder=10,
        )
    else:
        yval = xray
        ax.errorbar(
            radio,
            yval,
            yerr=0.75 * yval,
            uplims=True,
            fmt="*",
            markersize=6,
            color=colors["ulp"],
            barsabove=True,
            capsize=2,
            # label="ULP GLEAM-X J1627",
            zorder=10,
        )

    ### plot ILT J1101+5521
    ax.errorbar(
        1e-4, 1e30, marker="H", color=colors["wd"],
        yerr=1e30 * 0.8, uplims=True, markersize=4,
        # label="ILT J1101+5521"
    )

    # Plot GPMJ1839
    tab = Table.read("data/GPMJ1839.txt", format="ascii")
    xray = tab["xray"] * u.erg / u.second / u.cm**2
    radio = tab["radio"] * u.mJy
    d = tab["d"] * u.kpc

    xray = (xray * 4 * np.pi * d**2).to(u.erg / u.second)
    radio = (radio * d**2).to(xunits)

    if ratio:
        yval = (radio * freq0 / xray).to(u.dimensionless_unscaled)
        ax.errorbar(
            radio,
            yval,
            yerr=0.75 * yval,
            lolims=True,
            fmt="P",
            markersize=4,
            color=colors["ulp"],
            barsabove=True,
            capsize=2,
            # label="ULP GPM J1839",
            zorder=10,
        )
    else:
        yval = xray
        ax.errorbar(
            radio,
            yval,
            yerr=0.75 * yval,
            uplims=True,
            fmt="P",
            markersize=4,
            color=colors["ulp"],
            barsabove=True,
            capsize=2,
            # label="ULP GPM J1839",
            zorder=10,
        )

    # Plot CHIME J0630
    # Dong et al. 2024
    # mean that they quote, averaged over the period but then divided by the duty cycle
    radio = (0.6 * u.mJy) / 9.5e-3
    d = 170 * u.pc
    # take Swift CR of unknown sources, 6e-4
    # model with NH=2e21, Gamma=2
    xray = 4.5e-14 * u.erg / u.s / u.cm**2
    # BB(0.3 keV) gives factor of 2 lower
    radio = (radio * d**2).to(xunits)
    xray = (4 * np.pi * xray * d**2).cgs
    ax.errorbar(
        radio,
        xray,
        yerr=0.75 * xray,
        uplims=True,
        fmt="s",
        markersize=4,
        color=colors["ulp"],
        barsabove=True,
        capsize=2,
        # label="CHIME J0630+25",
        zorder=10,
    )

    # Plot J1935
    tab = Table.read("data/J1935+2148.txt", format="ascii")
    xray = tab["xray"] * u.erg / u.second / u.cm**2
    radio = tab["radio"] * u.mJy
    d = tab["d"] * u.kpc

    xray = (xray * 4 * np.pi * d**2).to(u.erg / u.second)
    radio = (radio * d**2).to(xunits)

    if ratio:
        yval = (radio * freq0 / xray).to(u.dimensionless_unscaled)
        ax.errorbar(
            radio,
            yval,
            fmt="H",
            yerr=0.75 * yval,
            lolims=True,
            markersize=4,
            color=colors["ulp"],
            barsabove=True,
            capsize=2,
            label="ULP J1935+2148",
            zorder=10,
        )
    else:
        yval = xray
        ax.errorbar(
            radio,
            yval,
            fmt="H",
            yerr=0.75 * yval,
            uplims=True,
            markersize=4,
            color=colors["ulp"],
            barsabove=True,
            capsize=2,
            # label="ULP J1935+2148",
            zorder=10,
        )

    ### Plot GLEAM-X J0704
    ax.errorbar(
        (1.2 * u.mJy * (1.5 * u.kpc) ** 2).to(u.mJy * u.kpc**2),
        1.1e32, # in Chandra
        yerr = 1.1e32 * 0.8, uplims=True,
        marker="D", color="m", markersize=4,
    )

    #################### Add J1627 deep limit
    ax.errorbar(
        x=0.083, y=1e29, xerr=0.083*0.8, yerr=1e29*0.8,
        marker='*', color="k", uplims=True, xuplims=True,
        markersize=6,
    )

    return ax

### add text...
def add_text(ax, ratio=False):
    ax.text(8e0, 1.5e29, "Pulsars",   color=colors["pulsar"], alpha=0.75)
    ax.text(0.5e-2, 1e36, "AMXPs",   color=colors["amxp"], alpha=0.75)
    ax.text(1e3, 1e36, "BH XRBs",   color=colors["bhxrb"], alpha=0.75)
    ax.text(1e1, 1e38, "NS XRBs",   color=colors["nsxrb"], alpha=0.75)
    ax.text(1e-3, 3e34, "Magnetars",   color=colors["magnetar"], alpha=0.75)
    ax.text(1e7, 1e35, "SGR 1935",   color=colors["sgr"], alpha=0.75)
    # ax.text(1e14, 1e41, "AGNs/QSOs", size=20, color="blue", alpha=0.75)
    # ax.text(1e7, 1e46, "Supernovae", size=20, color="green", alpha=0.75)
    # ax.text(1e16, 1e47, "FRBs", size=20, color="gray")
    ax.text(1e-1, 1e28, "Stars",   color=colors["star"])
    t0 = ax.text(
        3e-5,
        3e32,
        "MCV",
         
        color=colors["mcv"],
        alpha=0.75,
        # fontweight="bold",
    )
    ax.annotate(
        " / CV",
         
        color=colors["cv"],
        alpha=0.75,
        # fontweight="bold",
        xycoords=t0,
        xy=(1, 0),
        verticalalignment="bottom",
    )

    return ax

### for plotting the second axis
fac = 4 * np.pi * xunits.to(u.erg / u.second / u.Hz)

def forward(x):
    return x * fac


def inverse(x):
    return x / fac


### plot nuLnu/Lx ratio
def plot_ratio(ax, ratio=False):
    lr = np.logspace(-8, 12)
    lx = np.logspace(25, 39, 1000)
    Lr, Lx = np.meshgrid(lr, lx)
    # rat = np.array([1e-14, 1e-11, 1e-8, 1e-5, 1e-2, 1e1, 1e4, 1e7])
    # manual_x = [1e-10, 1e-2, 1e-1, 1e5, 1e8, 7e3, 1e9, 1e12]
    rat = np.array([1e-11, 1e-8, 1e-5, 1e-2, 1e1, 1e4,])
    manual_x = [1e-2, 1e-1, 1e5, 1e8, 7e3, 1e9,]
    manual_locations = [
        (x, (x * xunits * freq0).to_value(u.erg / u.s) / r)
        for x, r in zip(manual_x, rat)
    ]

    cs = ax.contour(
        Lr,
        Lx,
        np.log10(4*np.pi*(Lr * xunits * freq0).to_value(u.erg / u.s) / Lx),
        levels=np.int8(np.log10(rat)),
        colors="gray",
        alpha=0.5,
        linestyles="-.",
    )
    ax.clabel(
        cs,
        inline=True,
        fmt=lambda x: "$\\nu L_{{\\nu,\\mathrm{{radio}}}}/L_X=10^{{{0}}}$".format(
            int(x)
        ),
        manual=manual_locations,
    )
    return ax