"""
fit_pec_motions_krige_csv.py

Fits the vector peculiar motions of sources using kriging. Data from a csv file.

Isaac Cheng - February 2021
"""

import sys
from pathlib import Path
import dill
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from kriging import kriging

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans

# Roll angle between galactic midplane and galactocentric frame
_ROLL = 0.0  # deg (Anderson et al. 2019)
# Sun's height above galactic midplane (Reid et al. 2019)
_ZSUN = 5.5  # pc


def get_coords(data, tracefile):
    # with open(tracefile, "rb") as f:
    #     file = dill.load(f)
    #     trace = file["trace"]
    #     free_Zsun = file["free_Zsun"]
    #     free_roll = file["free_roll"]

    # === Get optimal parameters from MCMC trace ===
    # R0 = np.mean(trace["R0"])  # kpc
    # Zsun = np.mean(trace["Zsun"]) if free_Zsun else _ZSUN  # pc
    # roll = np.mean(trace["roll"]) if free_roll else _ROLL  # deg
    # a2 = np.mean(trace["a2"])
    # a3 = np.mean(trace["a3"])
    # print(R0, Zsun, roll)
    # print(a2, a3)

    # # Median values from 100 distance, mean Upec/Vpec trace, Gaussian outlier rejection file
    # R0 = 8.17744864142579
    # Zsun = 5.244659085536849
    # roll = 0.004996518973221375

    # # Median values from 100 distance, mean Upec/Vpec trace, Cauchy outlier rejection file
    # R0, Zsun, roll = 8.181488, 5.691242, 0.009044973
    # a2, a3 = 0.9727373, 1.6251589

    # # Mean values from 100 distance, mean Upec/Vpec trace, Cauchy outlier rejection file
    R0, Zsun, roll = 8.181364, 5.5833244, 0.009740928
    a2, a3 = 0.97133905, 1.6247351

    # # Median values from allfree pickle file
    # R0, Zsun, roll = 8.176646, 5.5956845, -0.00089795317

    # === Get data ===
    glon = data["glong"].values  # deg
    glat = data["glat"].values  # deg
    plx = data["plx"].values  # mas

    # === Calculate predicted values from optimal parameters ===
    # Parallax to distance
    dist = trans.parallax_to_dist(plx)

    # Galactic to barycentric Cartesian coordinates
    bary_x, bary_y, bary_z = trans.gal_to_bary(glon, glat, dist)

    # Barycentric Cartesian to galactocentric Cartesian coodinates
    gcen_x, gcen_y, gcen_z = trans.bary_to_gcen(
        bary_x, bary_y, bary_z, R0=R0, Zsun=Zsun, roll=roll
    )
    # Rotate 90deg CW to Reid convention
    gcen_x, gcen_y = gcen_y, -gcen_x

    return gcen_x, gcen_y, gcen_z


def main():
    # tracefile = Path(
    #     "/home/chengi/Documents/coop2021/bayesian_mcmc_rot_curve/"
    #     "mcmc_outfile_A5_100dist_5.pkl"
    # )
    # datafile = Path(
    #     "/home/chengi/Documents/coop2021/pec_motions/100dist_meanUpecVpec.csv"
    # )
    datafile = Path(__file__).parent / Path("100dist_meanUpecVpec_cauchyOutlierRejection.csv")
    tracefile = Path(__file__).parent.parent / Path("bayesian_mcmc_rot_curve/mcmc_outfile_A1_100dist_5.pkl")
    data100plx = pd.read_csv(datafile)

    # Only choose sources that have R > 4 kpc
    data100plx = data100plx[data100plx["is_tooclose"].values == 0]

    x, y, z = get_coords(data100plx, tracefile)

    Upec = data100plx["Upec"].values
    Vpec = data100plx["Vpec"].values
    Wpec = data100plx["Wpec"].values

    # Only choose good data for kriging
    percentile = 90
    lower = 0.5 * (100 - percentile)
    upper = 0.5 * (100 + percentile)
    is_good = (
        (Upec > np.percentile(Upec, lower))
        & (Upec < np.percentile(Upec, upper))
        & (Vpec > np.percentile(Vpec, lower))
        & (Vpec < np.percentile(Vpec, upper))
        & (Wpec > np.percentile(Wpec, lower))
        & (Wpec < np.percentile(Wpec, upper))
    )
    # percentile = "nonoutlier"
    # is_good = (data100plx["is_outlier"].values == 0) & (data100plx["is_tooclose"].values == 0)
    num_good = sum(is_good)
    print("# sources used in kriging:", num_good)
    # print(Upec[is_good].mean(), Vpec[is_good].mean(), Wpec[is_good].mean())
    # print(np.median(Upec[is_good]), np.median(Vpec[is_good]), np.median(Wpec[is_good]))
    # print(np.std(Upec[is_good]), np.std(Vpec[is_good]), np.std(Wpec[is_good]))
    # v_tot = np.sqrt(Upec[is_good]**2 + Vpec[is_good]**2 + Wpec[is_good]**2)
    # print(np.mean(v_tot), np.median(v_tot), np.std(v_tot))

    # Krig good data
    coord_obs = np.vstack((x[is_good], y[is_good])).T

    xlow, xhigh = -8, 12
    ylow, yhigh = -5, 15
    gridx, gridy = np.mgrid[xlow:xhigh:500j, ylow:yhigh:500j]
    coord_interp = np.vstack((gridx.flatten(), gridy.flatten())).T

    variogram_model = "gaussian"  # "gaussian", "spherical", or "exponential"
    print("Variogram Model:", variogram_model)

    Upec_interp, Upec_interp_var = kriging.kriging(
        coord_obs,
        Upec[is_good],
        coord_interp,
        model=variogram_model,
        deg=1,
        nbins=10,
        bin_number=True,
        plot=Path(__file__).parent / f"semivariogram_Upec_{num_good}good_{percentile}ptile_{variogram_model}.pdf",
    )
    Vpec_interp, Vpec_interp_var = kriging.kriging(
        coord_obs,
        Vpec[is_good],
        coord_interp,
        model=variogram_model,
        deg=1,
        nbins=10,
        bin_number=True,
        plot=Path(__file__).parent / f"semivariogram_Vpec_{num_good}good_{percentile}ptile_{variogram_model}.pdf",
    )
    Upec_interp = Upec_interp.reshape(500, 500)
    Upec_interp_sd = np.sqrt(Upec_interp_var).reshape(500, 500)
    Vpec_interp = Vpec_interp.reshape(500, 500)
    Vpec_interp_sd = np.sqrt(Vpec_interp_var).reshape(500, 500)
    print("Min & Max of interpolated Upec:", np.min(Upec_interp), np.max(Upec_interp))
    print("Min & Max of interpolated Vpec:", np.min(Vpec_interp), np.max(Vpec_interp))
    # print(np.mean(Upec_interp), np.mean(Vpec_interp))
    # print(np.median(Upec_interp), np.median(Vpec_interp))
    # print(np.mean(Upec_interp_sd), np.mean(Vpec_interp_sd))
    # print(np.median(Upec_interp_sd), np.median(Vpec_interp_sd))
    # v_tot_interp = np.sqrt(Upec_interp**2 + Vpec_interp**2)
    # print(np.mean(v_tot_interp), np.median(v_tot_interp), np.std(v_tot_interp))

    fig, ax = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    cmap = "viridis"
    extent = (xlow, xhigh, ylow, yhigh)

    # Plot interpolated Upec
    # norm_Upec = mpl.colors.Normalize(vmin=-15., vmax=15.)
    norm_Upec = mpl.colors.Normalize(vmin=np.min(Upec_interp), vmax=np.max(Upec_interp))
    ax[0].imshow(Upec_interp.T, origin="lower", extent=extent, norm=norm_Upec)
    cbar_Upec = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_Upec, cmap=cmap), ax=ax[0], format="%.0f"
    )
    ax[0].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[0].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[0].set_xlabel("$x$ (kpc)")
    ax[0].set_ylabel("$y$ (kpc)")
    ax[0].set_title("$\overline{U_s}$")
    cbar_Upec.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar_Upec.ax.get_yaxis().labelpad = 15
    ax[0].set_aspect("equal")
    ax[0].grid(False)

    # Plot interpolated Vpec
    # norm_Vpec = mpl.colors.Normalize(vmin=-20., vmax=10.)
    norm_Vpec = mpl.colors.Normalize(vmin=np.min(Vpec_interp), vmax=np.max(Vpec_interp))
    ax[1].imshow(Vpec_interp.T, origin="lower", extent=extent, norm=norm_Vpec)
    cbar_Vpec = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_Vpec, cmap=cmap), ax=ax[1], format="%.0f"
    )
    ax[1].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[1].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[1].set_xlabel("$x$ (kpc)")
    ax[1].set_ylabel("$y$ (kpc)")
    ax[1].set_title("$\overline{V_s}$")
    cbar_Vpec.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar_Vpec.ax.get_yaxis().labelpad = 15
    ax[1].set_aspect("equal")
    ax[1].grid(False)

    # Plot actual Upec & Vpec
    ax[0].scatter(
        x[is_good],
        y[is_good],
        c=Upec[is_good],
        norm=norm_Upec,
        cmap=cmap,
        s=10,
        edgecolors="k",
        marker="o",
        label="Good Masers",
    )
    ax[0].scatter(
        x[~is_good],
        y[~is_good],
        c=Upec[~is_good],
        norm=norm_Upec,
        cmap=cmap,
        s=10,
        edgecolors="k",
        marker="s",
        label="Outlier Masers",
    )
    ax[0].set_xlim(xlow, xhigh)
    # ax[0].set_xticks([-5, 0, 5, 10])
    ax[0].set_ylim(ylow, yhigh)
    # ax[0].set_yticks([-5, 0, 5, 10, 15])
    ax[0].legend(loc="lower left", fontsize=9)
    ax[1].scatter(
        x[is_good],
        y[is_good],
        c=Vpec[is_good],
        norm=norm_Vpec,
        cmap=cmap,
        s=10,
        edgecolors="k",
        marker="o",
        label="Good Masers",
    )
    ax[1].scatter(
        x[~is_good],
        y[~is_good],
        c=Vpec[~is_good],
        norm=norm_Vpec,
        cmap=cmap,
        s=10,
        edgecolors="k",
        marker="s",
        label="Outlier Masers",
    )
    ax[1].set_xlim(xlow, xhigh)
    # ax[1].set_xticks([-5, 0, 5, 10])
    ax[1].set_ylim(ylow, yhigh)
    # ax[1].set_yticks([-5, 0, 5, 10, 15])
    ax[1].legend(loc="lower left", fontsize=9)

    # fig.suptitle(
    #     f"Interpolated Peculiar Motions ({num_good} Good Masers)\n"
    #     fr"(Universal Kriging, \texttt{{variogram\_model={variogram_model}}})"
    # )
    fig.tight_layout()
    filename = f"krige_{num_good}good_{percentile}ptile_{variogram_model}.pdf"
    fig.savefig(
        Path(__file__).parent / filename, format="pdf", dpi=300, bbox_inches="tight",
    )
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    # Plot interpolated Upec's standard deviation
    norm_Upec = mpl.colors.Normalize(vmin=np.min(Upec_interp_sd), vmax=np.max(Upec_interp_sd))
    ax[0].imshow(Upec_interp_sd.T, origin="lower", extent=extent, norm=norm_Upec)
    cbar_Upec = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_Upec, cmap=cmap), ax=ax[0], format="%.0f"
    )
    ax[0].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[0].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[0].set_xlabel("$x$ (kpc)")
    ax[0].set_ylabel("$y$ (kpc)")
    ax[0].set_title("$\overline{U_s}$ Standard Deviation")
    cbar_Upec.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar_Upec.ax.get_yaxis().labelpad = 15
    ax[0].set_aspect("equal")
    ax[0].grid(False)

    # Plot interpolated Vpec's standard deviation
    norm_Vpec = mpl.colors.Normalize(vmin=np.min(Vpec_interp_sd), vmax=np.max(Vpec_interp_sd))
    ax[1].imshow(Vpec_interp_sd.T, origin="lower", extent=extent, norm=norm_Vpec)
    cbar_Vpec = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_Vpec, cmap=cmap), ax=ax[1], format="%.0f"
    )
    ax[1].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[1].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[1].set_xlabel("$x$ (kpc)")
    ax[1].set_ylabel("$y$ (kpc)")
    ax[1].set_title("$\overline{V_s}$ Standard Deviation")
    cbar_Vpec.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar_Vpec.ax.get_yaxis().labelpad = 15
    ax[1].set_aspect("equal")
    ax[1].grid(False)

    # fig.suptitle(
    #     f"Standard Deviations of Interpolated Peculiar Motions ({num_good} Good Masers)\n"
    #     fr"(Universal Kriging, \texttt{{variogram\_model={variogram_model}}})"
    # )
    fig.tight_layout()
    filename = f"krige_sd_{num_good}good_{percentile}ptile_{variogram_model}.pdf"
    fig.savefig(
        Path(__file__).parent / filename, format="pdf", dpi=300, bbox_inches="tight",
    )
    plt.show()

if __name__ == "__main__":
    main()
