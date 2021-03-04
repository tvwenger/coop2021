"""
plt_pec_mot_csv.py

Plots the peculiar (non-circular) motions of the sources
from a csv file and colour-codes by ratio of v_radial to v_tangential.

Outliers are those with R > 4 kpc & have any motion component in outer 10 percentile.

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
    is_tooclose = data["is_tooclose"].values
    data_tooclose = data[is_tooclose == 1]  # 19 sources
    data = data[is_tooclose == 0]  # 183 sources

    glon = data["glong"].values
    glat = data["glat"].values
    plx = data["plx"].values
    Upec = data["Upec"].values
    Vpec = data["Vpec"].values
    Wpec = data["Wpec"].values

    glon_tooclose = data_tooclose["glong"].values
    glat_tooclose = data_tooclose["glat"].values
    plx_tooclose = data_tooclose["plx"].values
    Upec_tooclose = data_tooclose["Upec"].values
    Vpec_tooclose = data_tooclose["Vpec"].values

    # is_outlier = data["is_outlier"].values
    percentile = 90
    lower = 0.5 * (100 - percentile)
    upper = 0.5 * (100 + percentile)
    good = (
        (Upec > np.percentile(Upec, lower))
        & (Upec < np.percentile(Upec, upper))
        & (Vpec > np.percentile(Vpec, lower))
        & (Vpec < np.percentile(Vpec, upper))
        & (Wpec > np.percentile(Wpec, lower))
        & (Wpec < np.percentile(Wpec, upper))
    )
    is_outlier = ~good
    print("Num R < 4 kpc:", sum(is_tooclose))
    print("Num outliers:", sum(is_outlier))
    print("Num good:", sum(good))
    print("=== Sources that are outliers in MCMC but not outliers here/for kriging ===")
    print(data["gname"][((good) & (data["is_outlier"].values == 1))])
    print("=== Sources that are not outliers in MCMC but outliers here/for kriging ===")
    print(data["gname"][((is_outlier) & (data["is_outlier"].values == 0))])

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

    dist_tooclose = trans.parallax_to_dist(plx_tooclose)
    # Galactic to barycentric Cartesian coordinates
    bary_x_tooclose, bary_y_tooclose, bary_z_tooclose = trans.gal_to_bary(
        glon_tooclose, glat_tooclose, dist_tooclose
    )
    # Barycentric Cartesian to galactocentric Cartesian coodinates
    x_tooclose, y_tooclose, z_tooclose = trans.bary_to_gcen(
        bary_x_tooclose, bary_y_tooclose, bary_z_tooclose, R0=R0, Zsun=Zsun, roll=roll
    )
    # Galactocentric cylindrical coordinates to get predicted circular velocity
    gcen_cyl_dist_tooclose = np.sqrt(
        x_tooclose * x_tooclose + y_tooclose * y_tooclose
    )  # kpc
    v_circ_pred_tooclose = (
        urc(gcen_cyl_dist_tooclose, a2=a2, a3=a3, R0=R0) + Vpec_tooclose
    )  # km/s
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
    v_tot_tooclose = np.sqrt(vx_tooclose ** 2 + vy_tooclose ** 2)
    print("Mean magnitude (of sources R < 4 kpc):", np.mean(v_tot_tooclose))
    print(
        "Max & min magnitude (of sources R < 4 kpc):",
        np.max(v_tot_tooclose),
        np.min(v_tot_tooclose),
    )

    # plt.rcParams["text.latex.preamble"] = r"\usepackage{newtxtext}\usepackage{newtxmath}"
    vrad_vcirc = -Upec / v_circ_pred
    vrad_vcirc_tooclose = -Upec_tooclose / v_circ_pred_tooclose
    cmap = "viridis"  # "coolwarm" is another option
    scattersize = 1
    fig, ax = plt.subplots()
    # Colorbar
    cmap_min = (
        np.floor(100 * np.min([np.min(vrad_vcirc), np.min(vrad_vcirc_tooclose)])) / 100
    )
    cmap_max = (
        np.ceil(100 * np.max([np.max(vrad_vcirc), np.max(vrad_vcirc_tooclose)])) / 100
    )
    # print(cmap_min, cmap_max)
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
        x_tooclose,
        y_tooclose,
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
    vectors_tooclose = ax.quiver(
        x_tooclose,
        y_tooclose,
        vx_tooclose,
        vy_tooclose,
        vrad_vcirc_tooclose,
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
    # fig.savefig(
    #     Path(__file__).parent / "pec_motions.pdf",
    #     format="pdf",
    #     # dpi=300,
    #     bbox_inches="tight",
    # )
    # print("--- Showing plot ---")
    plt.show()

    # Plot only good sources
    fig, ax = plt.subplots()
    # Colorbar
    cmap_min = np.floor(100 * np.min(vrad_vcirc[good])) / 100
    cmap_max = np.ceil(100 * np.max(vrad_vcirc[good])) / 100
    print(cmap_min, cmap_max)
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
    # fig.savefig(
    #     Path(__file__).parent / "pec_motions_onlygood.pdf",
    #     format="pdf",
    #     # dpi=300,
    #     bbox_inches="tight",
    # )
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
