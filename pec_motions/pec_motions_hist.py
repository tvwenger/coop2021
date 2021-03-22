"""
pec_motions_hist.py

Plots histograms of the peculiar motions

Isaac Cheng - March 2021
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

from calc_hpd import calc_hpd


def hist_mc_data():
    mc_type = "HPDmode"
    datafile = Path(__file__).parent / Path(f"csvfiles/alldata_{mc_type}.csv")
    data = pd.read_csv(datafile)

    # Only choose sources that have R > 4 kpc & R_halfhpd < 1.4
    # data = data[(data["is_tooclose"].values == 0) & (data["R_halfhpd"].values < 1.4)]
    data = data[data["is_tooclose"].values == 0]
    num_sources = len(data)
    print("Number of sources", num_sources)

    R = data["R_mode"].values
    R_halfhpd = data["R_halfhpd"].values
    Upec = data["Upec_mode"].values
    Vpec = data["Vpec_mode"].values
    Wpec = data["Wpec_mode"].values
    print("Calculating HPD...")
    # Upec_hpd = np.array([calc_hpd(Upec, "scipy") for idx in range(num_sources)])
    # Upec_hpd_mode, Upec_hpd_low, Upec_hpd_high = (
    #     Upec_hpd[:, 1],
    #     Upec_hpd[:, 2],
    #     Upec_hpd[:, 3],
    # )
    # Vpec_hpd = np.array([calc_hpd(Vpec, "scipy") for idx in range(num_sources)])
    # Vpec_hpd_mode, Vpec_hpd_low, Vpec_hpd_high = (
    #     Vpec_hpd[:, 1],
    #     Vpec_hpd[:, 2],
    #     Vpec_hpd[:, 3],
    # )
    _, Upec_hpd_mode, Upec_hpd_low, Upec_hpd_high = calc_hpd(Upec, "scipy")
    _, Vpec_hpd_mode, Vpec_hpd_low, Vpec_hpd_high = calc_hpd(Vpec, "scipy")
    _, Wpec_hpd_mode, Wpec_hpd_low, Wpec_hpd_high = calc_hpd(Wpec, "scipy")
    #
    tot = np.sqrt(Upec ** 2 + Vpec ** 2 + Wpec ** 2)
    tot_xy = np.sqrt(Upec ** 2 + Vpec ** 2)
    Upec_halfhpd = data["Upec_halfhpd"].values
    Vpec_halfhpd = data["Vpec_halfhpd"].values
    Wpec_halfhpd = data["Wpec_halfhpd"].values
    tot_std = np.sqrt(Upec_halfhpd ** 2 + Vpec_halfhpd ** 2 + Wpec_halfhpd ** 2)
    tot_xy_std = np.sqrt(Upec_halfhpd ** 2 + Vpec_halfhpd ** 2)
    print("Means:", np.mean(Upec), np.mean(Vpec), np.mean(Wpec), np.mean(tot))
    print("Medians:", np.median(Upec), np.median(Vpec), np.median(Wpec), np.median(tot))
    print("STDs:", np.std(Upec), np.std(Vpec), np.std(Wpec), np.std(tot))
    print(
        "Upec: mode, mode-lower, upper-mode:",
        Upec_hpd_mode,
        Upec_hpd_mode - Upec_hpd_low,
        Upec_hpd_high - Upec_hpd_mode,
    )
    print(
        "Vpec: mode, mode-lower, upper-mode:",
        Vpec_hpd_mode,
        Vpec_hpd_mode - Vpec_hpd_low,
        Vpec_hpd_high - Vpec_hpd_mode,
    )
    print(
        "Wpec: mode, mode-lower, upper-mode:",
        Wpec_hpd_mode,
        Wpec_hpd_mode - Wpec_hpd_low,
        Wpec_hpd_high - Wpec_hpd_mode,
    )

    fig, ax = plt.subplots(6, figsize=plt.figaspect(4))
    bins = 20
    ax[0].hist(Upec, bins=bins)
    ax[0].set_title("Upec")
    ax[1].hist(Vpec, bins=bins)
    ax[1].set_title("Vpec")
    ax[2].hist(Wpec, bins=bins)
    ax[2].set_title("Wpec")
    ax[3].hist(tot, bins=bins)
    ax[3].set_title("Total Magnitude")
    ax[4].hist(tot_xy, bins=bins)
    ax[4].set_title("Total xy-Magnitude")
    ax[5].hist(R, bins=bins)
    ax[5].set_title("Radius")
    fig.tight_layout()
    fig.savefig(
        Path(__file__).parent / f"pec_motions_hist_{mc_type}_Rhpd1.4.pdf",
        bbox_inches="tight",
    )
    plt.show()

    fig, ax = plt.subplots(6, figsize=plt.figaspect(4))
    bins = 20
    ax[0].hist(Upec_halfhpd, bins=bins)
    ax[0].set_title("Half HPD of Upec")
    ax[1].hist(Vpec_halfhpd, bins=bins)
    ax[1].set_title("Half HPD of Vpec")
    ax[2].hist(Wpec_halfhpd, bins=bins)
    ax[2].set_title("Half HPD of Wpec")
    ax[3].hist(tot_std, bins=bins)
    ax[3].set_title("Half HPD of Total Magnitude")
    ax[4].hist(tot_xy_std, bins=bins)
    ax[4].set_title("Half HPD of Total xy-Magnitude")
    ax[5].hist(R_halfhpd, bins=bins)
    ax[5].set_title("Half HPD of Radius")
    fig.tight_layout()
    fig.savefig(
        Path(__file__).parent / f"pec_motions_halfhpd_hist_{mc_type}_Rhpd1.4.pdf",
        bbox_inches="tight",
    )
    plt.show()


def hist_nonmc_peakDist_data():
    datafile = Path(__file__).parent / Path(
        "csvfiles/100dist_meanUpecVpec_cauchyOutlierRejection_peakDist.csv"
    )
    data = pd.read_csv(datafile)

    # Only choose sources that have R > 4 kpc & R_halfhpd < 1
    data = data[(data["is_tooclose"].values == 0)]
    num_sources = len(data)
    print("Number of sources", num_sources)

    Upec = data["Upec"].values
    Vpec = data["Vpec"].values
    Wpec = data["Wpec"].values
    tot = np.sqrt(Upec ** 2 + Vpec ** 2 + Wpec ** 2)
    tot_xy = np.sqrt(Upec ** 2 + Vpec ** 2)
    print(np.mean(Upec), np.mean(Vpec), np.mean(Wpec), np.mean(tot))
    print(np.median(Upec), np.median(Vpec), np.median(Wpec), np.median(tot))
    print(np.std(Upec), np.std(Vpec), np.std(Wpec), np.std(tot))

    fig, ax = plt.subplots(5, figsize=plt.figaspect(3.33))
    bins = 20
    ax[0].hist(Upec, bins=bins)
    ax[0].set_title("Upec")
    ax[1].hist(Vpec, bins=bins)
    ax[1].set_title("Vpec")
    ax[2].hist(Wpec, bins=bins)
    ax[2].set_title("Wpec")
    ax[3].hist(tot, bins=bins)
    ax[3].set_title("Total Magnitude")
    ax[4].hist(tot_xy, bins=bins)
    ax[4].set_title("Total xy-Magnitude")
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "pec_motions_hist_nonmc.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    hist_mc_data()
