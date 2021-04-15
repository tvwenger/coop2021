"""
plt_pec_mot_no_colour.py

Plots the peculiar (non-circular) motions of the sources
without distinguishing points (i.e., for Isaac's end-of-term presentation)

Isaac Cheng - April 2021
"""
import sys
from pathlib import Path
import numpy as np
import dill
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans
from universal_rotcurve import urc

_DEG_TO_RAD = 0.017453292519943295  # pi/180
_RAD_TO_DEG = 57.29577951308232  # 180/pi (Don't forget to % 360 after)
# Roll angle between galactic midplane and galactocentric frame
_ROLL = 0.0  # deg (Anderson et al. 2019)
# Sun's height above galactic midplane (Reid et al. 2019)
_ZSUN = 5.5  # pc


def main(csvfile, tracefile, plot_tooclose=False, plot_large_uncer=True):
    data = pd.read_csv(csvfile)
    figname_append = "_noColours"

    if not plot_large_uncer:
        # Only plot sources with vector uncertainties < 20 km/s
        Upec_halfhpd = data["Upec_halfhpd"].values
        Vpec_halfhpd = data["Vpec_halfhpd"].values
        tot_halfhpd = np.sqrt(Upec_halfhpd**2 + Vpec_halfhpd**2)
        data = data[tot_halfhpd < 20]
        figname_append += "_smallErrs"

    is_tooclose = data["is_tooclose"].values
    data_tooclose = data[is_tooclose == 1]  # 19 sources
    data = data[is_tooclose == 0]  # 185 sources

    Upec = data["Upec_mode"].values
    Vpec = data["Vpec_mode"].values
    Wpec = data["Wpec_mode"].values
    x = data["x_mode"].values
    y = data["y_mode"].values
    Upec_tooclose = data_tooclose["Upec_mode"].values
    Vpec_tooclose = data_tooclose["Vpec_mode"].values
    x_tooclose = data_tooclose["x_mode"].values
    y_tooclose = data_tooclose["y_mode"].values

    good = data["is_outlier"].values == 0
    is_outlier = ~good
    print("Num R < 4 kpc:", sum(is_tooclose))
    print("Num outliers:", sum(is_outlier))
    print("Num good:", sum(good))

    # Mode of 100 distances, mean Upec/Vpec + peak everything
    R0_mode = 8.174602364395952
    Zsun_mode = 5.398550615892994
    Usun_mode = 10.878914326160878
    Vsun_mode = 10.696801784160257
    Wsun_mode = 8.087892505141708
    Upec_mode = 4.9071771802606285
    Vpec_mode = -4.521832904300172
    roll_mode = -0.010742182667190958
    a2_mode = 0.9768982857793898
    a3_mode = 1.626400628724733

    # Rotate 90 deg CCW to galactocentric convention in mytransforms
    x, y = -y, x
    x_tooclose, y_tooclose = -y_tooclose, x_tooclose

    # Get vx & vy in Cartesian coordinates
    azimuth = (np.arctan2(y, -x) * _RAD_TO_DEG) % 360
    cos_az = np.cos(azimuth * _DEG_TO_RAD)
    sin_az = np.sin(azimuth * _DEG_TO_RAD)
    vx = Upec * cos_az + Vpec * sin_az  # km/s
    vy = -Upec * sin_az + Vpec * cos_az  # km/s

    # Get vx & vy in Cartesian coordinates
    azimuth_tooclose = (np.arctan2(y_tooclose, -x_tooclose) * _RAD_TO_DEG) % 360
    cos_az_tooclose = np.cos(azimuth_tooclose * _DEG_TO_RAD)
    sin_az_tooclose = np.sin(azimuth_tooclose * _DEG_TO_RAD)
    vx_tooclose = (
        Upec_tooclose * cos_az_tooclose + Vpec_tooclose * sin_az_tooclose
    )  # km/s
    vy_tooclose = (
        -Upec_tooclose * sin_az_tooclose + Vpec_tooclose * cos_az_tooclose
    )  # km/s

    # Rotate 90 deg CW to Reid convention
    x, y = y, -x
    vx, vy = vy, -vx
    x_tooclose, y_tooclose = y_tooclose, -x_tooclose
    vx_tooclose, vy_tooclose = vy_tooclose, -vx_tooclose
    #
    # print("Recall Upec = -v_rad")
    print("Mean Upec & Vpec:", np.mean(Upec), np.mean(Vpec))
    print("Max & min Upec:", np.max(Upec), np.min(Upec))
    print("Max & min Vpec:", np.max(Vpec), np.min(Vpec))
    print("---")
    print("Mean Good Upec & Vpec:", np.mean(Upec[good]), np.mean(Vpec[good]))
    print("Max & min Good Upec:", np.max(Upec[good]), np.min(Upec[good]))
    print("Max & min Good Vpec:", np.max(Vpec[good]), np.min(Vpec[good]))
    print("Max & min Good Wpec:", np.max(Wpec[good]), np.min(Wpec[good]))
    print("---")
    print("Mean vx & vy:", np.mean(vx), np.mean(vy))
    print("Max & min vx:", np.max(vx), np.min(vx))
    print("Max & min vy:", np.max(vy), np.min(vy))
    v_tot = np.sqrt(vx * vx + vy * vy)
    print("---")
    print("Mean magnitude (w/out Wpec):", np.mean(v_tot))
    print("Max & min magnitude (w/out Wpec):", np.max(v_tot), np.min(v_tot))
    print("---")
    v_tot_good = np.sqrt(vx[good] * vx[good] + vy[good] * vy[good])
    print("Mean good magnitude (w/out Wpec):", np.mean(v_tot_good))
    print(
        "Max & min good magnitude (w/out Wpec):", np.max(v_tot_good), np.min(v_tot_good)
    )
    print("---")
    v_tot_tooclose = np.sqrt(vx_tooclose ** 2 + vy_tooclose ** 2)
    print("Mean magnitude (of sources R < 4 kpc) (w/out Wpec):", np.mean(v_tot_tooclose))
    print(
        "Max & min magnitude (of sources R < 4 kpc) (w/out Wpec):",
        np.max(v_tot_tooclose),
        np.min(v_tot_tooclose),
    )
    #
    # Plot
    #
    scattersize = 1
    fig, ax = plt.subplots()
    # Plot sources
    ax.scatter(x, y, marker="o", c="k", s=scattersize)
    if plot_tooclose:
        ax.scatter(x_tooclose, y_tooclose, marker="o", c="k", s=scattersize)
    # Plot Sun
    ax.scatter(0, 8.181, marker="*", c="gold", s=30, zorder=100)
    # Plot residual motions
    vectors = ax.quiver(x, y, vx, vy, color="k",scale=600, width=0.002)
    if plot_tooclose:
        ax.quiver(x_tooclose, y_tooclose, vx_tooclose, vy_tooclose,
                  color="k", scale=600, width=0.002)
    ax.quiverkey(
        vectors,
        X=0.25,
        Y=0.1,
        U=-50,
        label="50 km s$^{-1}$",
        labelpos="N",
        fontproperties={"size": 10},
    )
    # Other plot parameters
    ax.set_xlabel("$x$ (kpc)")
    ax.set_ylabel("$y$ (kpc)")
    ax.axhline(y=0, linewidth=0.5, linestyle="--", color="k", zorder=0)  # horizontal line
    ax.axvline(x=0, linewidth=0.5, linestyle="--", color="k", zorder=0)  # vertical line
    ax.set_xlim(-8, 12)
    ax.set_xticks([-5, 0, 5, 10])
    ax.set_ylim(-5, 15)
    ax.set_yticks([-5, 0, 5, 10, 15])
    ax.grid(False)
    ax.set_aspect("equal")
    # Change legend properties
    figname = "pec_motions_all" if plot_tooclose else "pec_motions"
    figname += figname_append + ".pdf"
    fig.savefig(
        Path(__file__).parent / figname,
        format="pdf",
        # dpi=300,
        bbox_inches="tight",
    )
    print("--- Showing plot ---")
    plt.show()


if __name__ == "__main__":
    # csvfilepath = input("Enter filepath of csv file: ")
    # tracefilepath = input("Enter filepath of pickle file (trace): ")
    # csvfilepath = (
    #     Path(__file__).parent
    #     / "csvfiles/100dist_meanUpecVpec_cauchyOutlierRejection_peakEverything.csv"
    # )
    csvfilepath = (
        Path(__file__).parent
        / "csvfiles/alldata_HPDmode_NEW2.csv"
    )
    tracefilepath = (
        Path(__file__).parent.parent
        / "bayesian_mcmc_rot_curve/mcmc_outfile_A5_102dist_6.pkl"
    )

    main(csvfilepath, tracefilepath, plot_tooclose=True, plot_large_uncer=False)
