"""
plt_pec_mot_csv_mcmcoutliers.py

Plots the peculiar (non-circular) motions of the sources
from a csv file and colour-codes by ratio of v_radial to v_tangential.

Outliers are those with R > 4 kpc & not used in MCMC fit.

Isaac Cheng - February 2021
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


def main(csvfile, tracefile):
    data = pd.read_csv(csvfile)

    glon = data["glong"].values
    glat = data["glat"].values
    plx = data["plx"].values
    Upec = data["Upec"].values
    Vpec = data["Vpec"].values
    Wpec = data["Wpec"].values

    is_tooclose = data["is_tooclose"].values
    is_outlier = data["is_outlier"].values
    good = (is_outlier==0) & (is_tooclose==0)
    print("Num R < 4 kpc:", sum(is_tooclose))
    print("Num outliers:", sum(is_outlier))
    print("Num good:", sum(good))

    # with open(tracefile, "rb") as f:
    #     file = dill.load(f)
    #     trace = file["trace"]
    #     free_Zsun = file["free_Zsun"]
    #     free_roll = file["free_roll"]
    # R0 = np.median(trace["R0"])  # kpc
    # Zsun = np.median(trace["Zsun"]) if free_Zsun else _ZSUN  # pc
    # roll = np.median(trace["roll"]) if free_roll else _ROLL  # deg
    # a2 = np.median(trace["a2"])
    # a3 = np.median(trace["a3"])
    # print(R0, Zsun, roll)
    # print(a2, a3)

    # # Mean values from 100 distance, mean Upec/Vpec trace, Cauchy outlier rejection file
    R0, Zsun, roll = 8.181364, 5.5833244, 0.009740928
    a2, a3 = 0.97133905, 1.6247351

    dist = trans.parallax_to_dist(plx)
    # Galactic to barycentric Cartesian coordinates
    bary_x, bary_y, bary_z = trans.gal_to_bary(glon, glat, dist)
    # Barycentric Cartesian to galactocentric Cartesian coodinates
    x, y, z = trans.bary_to_gcen(bary_x, bary_y, bary_z, R0=R0, Zsun=Zsun, roll=roll)
    # Galactocentric cylindrical coordinates to get predicted circular velocity
    gcen_cyl_dist = np.sqrt(x * x + y * y)  # kpc
    v_circ_pred = urc(gcen_cyl_dist, a2=a2, a3=a3, R0=R0) + Vpec  # km/s
    # Get vx & vy in Cartesian coordinates
    azimuth = (np.arctan2(y, -x) * _RAD_TO_DEG) % 360
    cos_az = np.cos(azimuth * _DEG_TO_RAD)
    sin_az = np.sin(azimuth * _DEG_TO_RAD)
    vx = Upec * cos_az + Vpec * sin_az  # km/s
    vy = -Upec * sin_az + Vpec * cos_az  # km/s

    # Rotate 90 deg CW to Reid convention
    x, y = y, -x
    vx, vy = vy, -vx
    print("Recall Upec = -v_rad")
    print("Mean Upec & Vpec:", np.mean(Upec), np.mean(Vpec))
    print("Max & min Upec:", np.max(Upec), np.min(Upec))
    print("Max & min Vpec:", np.max(Vpec), np.min(Vpec))
    print("---")
    print("Mean vx & vy:", np.mean(vx), np.mean(vy))
    print("Max & min vx:", np.max(vx), np.min(vx))
    print("Max & min vy:", np.max(vy), np.min(vy))
    v_tot = np.sqrt(vx * vx + vy * vy)
    print("---")
    print("Mean magnitude:", np.mean(v_tot))
    print("Max & min magnitude:", np.max(v_tot), np.min(v_tot))

    # plt.rcParams["text.latex.preamble"] = r"\usepackage{newtxtext}\usepackage{newtxmath}"
    vrad_vcirc = -Upec / v_circ_pred
    cmap = "viridis"  # "coolwarm" is another option
    scattersize = 1
    fig, ax = plt.subplots()
    # Colorbar
    cmap_min = (
        np.floor(100 * np.min(vrad_vcirc)) / 100
    )
    cmap_max = (
        np.ceil(100 * np.max(vrad_vcirc)) / 100
    )
    norm = mpl.colors.Normalize(vmin=cmap_min, vmax=cmap_max)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, format="%.1f")
    cbar.ax.set_ylabel(r"$v_{\rm radial}/v_{\rm circular}$", rotation=270)
    cbar.ax.get_yaxis().labelpad = 20
    # Plot sources
    ax.scatter(
        x[good],
        y[good],
        marker="o",
        c="darkorange",
        # c=vrad_vcirc[good],
        # cmap=cmap,
        # norm=norm,
        s=scattersize,
        label="Good",
    )
    ax.scatter(
        x[is_outlier == 1],
        y[is_outlier == 1],
        marker="s",
        c="r",
        # c=vrad_vcirc[is_outlier == 1],
        # cmap=cmap,
        # norm=norm,
        s=scattersize,
        label="Outlier",
    )
    ax.scatter(
        x[is_tooclose==1],
        y[is_tooclose==1],
        marker="v",
        c="k",
        # c=vrad_vcirc[is_tooclose == 1],
        # cmap=cmap,
        # norm=norm,
        s=scattersize,
        label="$R < 4$ kpc",
    )
    # Plot residual motions
    vectors = ax.quiver(
        x, y, vx, vy, vrad_vcirc, cmap=cmap, norm=norm, scale=600, width=0.002
    )
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
    ax.axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax.axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax.set_xlim(-8, 12)
    ax.set_xticks([-5, 0, 5, 10])
    ax.set_ylim(-5, 15)
    ax.set_yticks([-5, 0, 5, 10, 15])
    ax.grid(False)
    # Change legend properties
    ax.legend(loc="best")
    for element in ax.get_legend().legendHandles:
        # element.set_color("k")
        element._sizes = [20]
    fig.savefig(
        Path(__file__).parent / "pec_motions_mcmcoutliers.pdf",
        format="pdf",
        # dpi=300,
        bbox_inches="tight",
    )
    # print("--- Showing plot ---")
    plt.show()

    # Plot only good sources
    fig, ax = plt.subplots()
    # Colorbar
    cmap_min = np.floor(100 * np.min(vrad_vcirc[good])) / 100
    cmap_max = np.ceil(100 * np.max(vrad_vcirc[good])) / 100
    norm = mpl.colors.Normalize(vmin=cmap_min, vmax=cmap_max)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, format="%.2f")
    cbar.ax.set_ylabel(r"$v_{\rm radial}/v_{\rm circular}$", rotation=270)
    cbar.ax.get_yaxis().labelpad = 20
    # Plot residual motions
    vectors = ax.quiver(
        x[good],
        y[good],
        vx[good],
        vy[good],
        vrad_vcirc[good],
        cmap=cmap,
        norm=norm,
        scale=600,
        width=0.002,
    )
    ax.quiverkey(
        vectors,
        X=0.25,
        Y=0.1,
        U=-50,
        label="50 km s$^{-1}$",
        labelpos="N",
        fontproperties={"size": 10},
    )
    # Plot sources
    ax.scatter(
        x[good],
        y[good],
        marker="o",
        c=vrad_vcirc[good],
        cmap=cmap,
        norm=norm,
        s=scattersize,
        label="Good",
    )
    # Other plot parameters
    ax.set_xlabel("$x$ (kpc)")
    ax.set_ylabel("$y$ (kpc)")
    ax.axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax.axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax.set_xlim(-8, 12)
    ax.set_xticks([-5, 0, 5, 10])
    ax.set_ylim(-5, 15)
    ax.set_yticks([-5, 0, 5, 10, 15])
    ax.grid(False)
    fig.savefig(
        Path(__file__).parent / "pec_motions_onlymcmcgood.pdf",
        format="pdf",
        # dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    # csvfilepath = input("Enter filepath of csv file: ")
    # tracefilepath = input("Enter filepath of pickle file (trace): ")
    csvfilepath = (
        Path(__file__).parent / "100dist_meanUpecVpec_cauchyOutlierRejection.csv"
    )
    tracefilepath = (
        Path(__file__).parent.parent
        / "bayesian_mcmc_rot_curve/mcmc_outfile_A1_100dist_5.pkl"
    )

    main(csvfilepath, tracefilepath)
