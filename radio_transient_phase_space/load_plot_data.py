import pandas as pd
import numpy as np

# functions to load gach_rud data
def simple_file_loader(fpath, skiprow=None):
    with open(fpath, "r") as fp:
        lines = fp.readlines()
    if skiprow is not None: lines = lines[skiprow+1:]
    lines = [l.strip().split() for l in lines]
    if "misc" in fpath:
        lines = [[*l[:2], " ".join(l[2:])] for l in lines]
    return np.array(lines)

def make_data_frame(xdata, ydata, label):
    df = pd.DataFrame(dict(xdata=xdata, ydata=ydata))
    df["label"] = label
    return df

def load_data():
    data_list = []
    datapath = "./gach_rud"
    ### start to load data...
    # pulsar
    data = simple_file_loader(f"{datapath}/psrs_2")
    df = make_data_frame(
        xdata = data[:, 4].astype(float),
        ydata = data[:, 5].astype(float),
        label = "pulsar"
    )
    data_list.append(df)

    # crab nano giant pulse
    data = simple_file_loader(f"{datapath}/crab_nanogiant")
    df = make_data_frame(
        xdata = data[:, 0].astype(float),
        ydata = data[:, 1].astype(float),
        label = "crabnano"
    )
    data_list.append(df)

    # crab giant pulse
    data = simple_file_loader(f"{datapath}/crab_GRP")
    df = make_data_frame(
        xdata = data[:, 5].astype(float),
        ydata = data[:, 4].astype(float),
        label = "crabgrp"
    )
    data_list.append(df)

    # pulsar giant radio pulse
    data = simple_file_loader(f"{datapath}/GRPs_vals", skiprow=0)
    df = make_data_frame(
        xdata = data[:, 6].astype(float),
        ydata = data[:, 7].astype(float),
        label = "psrgrp"
    )
    data_list.append(df)

    # rrat
    data = simple_file_loader(f"{datapath}/rrats_nohead")
    df = make_data_frame(
        xdata = data[:, 4].astype(float),
        ydata = data[:, 5].astype(float),
        label = "rrat"
    )
    data_list.append(df)

    # frb
    data = simple_file_loader(f"{datapath}/frbs_vals_to_plot")
    df = make_data_frame(
        xdata = data[:, 1].astype(float),
        ydata = data[:, 0].astype(float),
        label = "frb"
    )
    data_list.append(df)

    # solar bursts
    data = simple_file_loader(f"{datapath}/solar_vals", skiprow=0)
    df = make_data_frame(
        xdata = data[:, 4].astype(float),
        ydata = data[:, 5].astype(float),
        label = "solar"
    )
    data_list.append(df)

    # SGR 1935
    data = simple_file_loader(f"{datapath}/SGR1935+2154")
    df = make_data_frame(
        xdata = data[:, 2].astype(float) * data[:, 3].astype(float),
        ydata = data[:, 0].astype(float) * (data[:, 1].astype(float)) ** 2,
        label = "sgr1935"
    )
    data_list.append(df)

    ### slow variables
    data = simple_file_loader(f"{datapath}/solar_vals", skiprow=0)
    df = make_data_frame(
        xdata = data[:, 4].astype(float),
        ydata = data[:, 5].astype(float),
        label = "solar"
    )
    data_list.append(df)

    # several objects, including Jupiter DAM etc.
    data = simple_file_loader(f"{datapath}/misc")
    df = make_data_frame(
        xdata = data[:, 0].astype(float),
        ydata = data[:, 1].astype(float),
        label = data[:, 2]
    )
    data_list.append(df)

    # AGN QSO
    data = simple_file_loader(f"{datapath}/Gosia_AGN_QSO_Blazar_TDE2", skiprow=0)
    df = make_data_frame(
        xdata = data[:, 1].astype(float) * data[:, 8].astype(float) * 86400,
        ydata = data[:, 6].astype(float) * 1.05026e-20,
        label = "agn"
    )
    data_list.append(df)

    # XRB
    data = simple_file_loader(f"{datapath}/Gosia_XRB2")
    df = make_data_frame(
        xdata = data[:, 1].astype(float) * data[:, 8].astype(float) * 86400,
        ydata = data[:, 6].astype(float) * 1.05026e-20,
        label = "xrb"
    )
    data_list.append(df)

    # GRB
    data = simple_file_loader(f"{datapath}/Gosia_GRB2")
    df = make_data_frame(
        xdata = data[:, 1].astype(float) * data[:, 8].astype(float) * 86400,
        ydata = data[:, 6].astype(float) * 1.05026e-20,
        label = "grb"
    )
    data_list.append(df)

    # SN
    data = simple_file_loader(f"{datapath}/Gosia_SN2")
    df = make_data_frame(
        xdata = data[:, 1].astype(float) * data[:, 8].astype(float) * 86400,
        ydata = data[:, 6].astype(float) * 1.05026e-20,
        label = "sn"
    )
    data_list.append(df)

    # rscvn
    data = simple_file_loader(f"{datapath}/Gosia_RSCVn2")
    df = make_data_frame(
        xdata = data[:, 1].astype(float) * data[:, 8].astype(float) * 86400,
        ydata = data[:, 6].astype(float) * 1.05026e-20,
        label = "rscvn"
    )
    data_list.append(df)

    # flare star
    data = simple_file_loader(f"{datapath}/Gosia_flare_stars2")
    df = make_data_frame(
        xdata = data[:, 1].astype(float) * data[:, 8].astype(float) * 86400,
        ydata = data[:, 6].astype(float) * 1.05026e-20,
        label = "star"
    )
    data_list.append(df)

    # novae
    data = simple_file_loader(f"{datapath}/Gosia_Novae2")
    df = make_data_frame(
        xdata = data[:, 1].astype(float) * data[:, 8].astype(float) * 86400,
        ydata = data[:, 6].astype(float) * 1.05026e-20,
        label = "novae"
    )
    data_list.append(df)

    # mcv
    data = simple_file_loader(f"{datapath}/Gosia_MagCV2", skiprow=0)
    df = make_data_frame(
        xdata = data[:, 1].astype(float) * data[:, 8].astype(float) * 86400,
        ydata = data[:, 6].astype(float) * 1.05026e-20,
        label = "mcv"
    )
    data_list.append(df)

    # gw170817
    data = simple_file_loader(f"{datapath}/gw170817")
    df = make_data_frame(
        xdata = data[:, 0].astype(float),
        ydata = data[:, 1].astype(float),
        label = "gw170817"
    )
    data_list.append(df)

    #lpt
    data = simple_file_loader(f"{datapath}/lpt")
    df = make_data_frame(
        xdata = data[:, 1].astype(float),
        ydata = data[:, 2].astype(float),
        label = "lpt"
    )
    data_list.append(df)

    #### combine all data together
    pdf = pd.concat(data_list)

    return pdf