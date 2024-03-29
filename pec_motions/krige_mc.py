"""
krige_mc.py

Universal kriging of Upec and Vpec from data in .csv file.
Monte Carlo generates 1000 kriging maps

Isaac Cheng - March 2021
"""

import sys
from pathlib import Path
import dill
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from kriging import kriging
from numba import jit

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans
from universal_rotcurve import urc


# def get_coords(data):
#     # # Mean values from 100 distance, mean Upec/Vpec trace, peak everything file
#     # R0_mean, Zsun_mean, roll_mean = 8.17845, 5.0649223, 0.0014527875
#     # Usun_mean, Vsun_mean, Wsun_mean = 10.879447, 10.540543, 8.1168785
#     # Upec_mean, Vpec_mean = 4.912622, -4.588946
#     # a2_mean, a3_mean = 0.96717525, 1.624953

#     # Mode of 100 distances, mean Upec/Vpec + peak everything
#     R0_mode = 8.174602364395952
#     Zsun_mode = 5.398550615892994
#     Usun_mode = 10.878914326160878
#     Vsun_mode = 10.696801784160257
#     Wsun_mode = 8.087892505141708
#     Upec_mode = 4.9071771802606285
#     Vpec_mode = -4.521832904300172
#     roll_mode = -0.010742182667190958
#     a2_mode = 0.9768982857793898
#     a3_mode = 1.626400628724733

#     # === Get data ===
#     glon = data["glong"].values  # deg
#     glat = data["glat"].values  # deg
#     plx = data["plx"].values  # mas
#     e_plx = data["e_plx"].values

#     # === Calculate predicted values from optimal parameters ===
#     # Parallax to distance
#     dist = trans.parallax_to_dist(plx, e_parallax=e_plx)

#     # Galactic to barycentric Cartesian coordinates
#     bary_x, bary_y, bary_z = trans.gal_to_bary(glon, glat, dist)

#     # Barycentric Cartesian to galactocentric Cartesian coodinates
#     gcen_x, gcen_y, gcen_z = trans.bary_to_gcen(
#         bary_x, bary_y, bary_z, R0=R0_mode, Zsun=Zsun_mode, roll=roll_mode
#     )
#     # Rotate 90deg CW to Reid convention
#     gcen_x, gcen_y = gcen_y, -gcen_x

#     return gcen_x, gcen_y, gcen_z

def main():
    # * CONCLUSION:
    # * range of the semivariance points is much larger than the variation from small lags
    # * to large lags. Basically there isn't much structure in the semivariograms.
    mc_type = "HPDmode_NEW2"
    # datafile = Path(__file__).parent / Path("csvfiles/alldata_{}.csv".format(mc_type))
    datafile = Path(__file__).parent / Path(f"csvfiles/alldata_{mc_type}.csv")
    pearsonrfile = Path(__file__).parent / "pearsonr_cov.pkl"
    # datafile = "csvfiles/alldata_HPDmode_NEW2.csv"
    # pearsonrfile = "pearsonr_cov.pkl"
    data = pd.read_csv(datafile)
    with open(pearsonrfile, "rb") as f:
        file = dill.load(f)
        cov_Upec = file["cov_Upec"]
        cov_Vpec = file["cov_Vpec"]

    # Only choose sources that have R > 4 kpc
    is_tooclose = data["is_tooclose"].values == 1
    data = data[~is_tooclose]
    cov_Upec = cov_Upec[:, ~is_tooclose][~is_tooclose]
    cov_Vpec = cov_Vpec[:, ~is_tooclose][~is_tooclose]

    R = data["R_mode"].values
    R_halfhpd = data["R_halfhpd"].values
    Upec = data["Upec_mode"].values
    Upec_halfhpd = data["Upec_halfhpd"].values
    Vpec = data["Vpec_mode"].values
    Vpec_halfhpd = data["Vpec_halfhpd"].values
    Wpec = data["Wpec_mode"].values
    Wpec_halfhpd = data["Wpec_halfhpd"].values
    tot = np.sqrt(Upec ** 2 + Vpec ** 2 + Wpec ** 2)
    tot_xy = np.sqrt(Upec ** 2 + Vpec ** 2)
    Upec_halfhpd = data["Upec_halfhpd"].values
    Vpec_halfhpd = data["Vpec_halfhpd"].values
    Wpec_halfhpd = data["Wpec_halfhpd"].values
    tot_halfhpd = np.sqrt(Upec_halfhpd ** 2 + Vpec_halfhpd ** 2 + Wpec_halfhpd ** 2)
    tot_xy_halfhpd = np.sqrt(Upec_halfhpd ** 2 + Vpec_halfhpd ** 2)
    print(np.mean(Upec), np.mean(Vpec), np.mean(Wpec))
    print(np.median(Upec), np.median(Vpec), np.median(Wpec))
    print(np.mean(tot), np.median(tot))
    print(np.mean(Upec_halfhpd), np.mean(Vpec_halfhpd), np.mean(Wpec_halfhpd))
    print(np.median(Upec_halfhpd), np.median(Vpec_halfhpd), np.median(Wpec_halfhpd))
    print(np.mean(tot_halfhpd), np.median(tot_halfhpd))

    x = data["x_mode"].values
    y = data["y_mode"].values

    # # Only choose good data for kriging
    Upec_err_high = data["Upec_hpdhigh"].values -  data["Upec_mode"].values
    Upec_err_low = data["Upec_mode"].values - data["Upec_hpdlow"].values
    Vpec_err_high = data["Vpec_hpdhigh"].values -  data["Vpec_mode"].values
    Vpec_err_low = data["Vpec_mode"].values - data["Vpec_hpdlow"].values
    condition = "noOutliersGaussianErrs"
    is_good = (
        (abs(Upec_err_high - Upec_err_low) < 0.33 * Upec_halfhpd)
        & (abs(Vpec_err_high - Vpec_err_low) < 0.33 * Vpec_halfhpd)
        & (data["is_outlier"].values == 0)
    )

    print("--- MC GOOD DATA STATS ---")
    print(np.mean(Upec[is_good]), np.mean(Vpec[is_good]), np.mean(Wpec[is_good]))
    print(np.median(Upec[is_good]), np.median(Vpec[is_good]), np.median(Wpec[is_good]))
    print(np.mean(tot[is_good]), np.median(tot[is_good]))
    print(
        np.mean(Upec_halfhpd[is_good]),
        np.mean(Vpec_halfhpd[is_good]),
        np.mean(Wpec_halfhpd[is_good]),
    )
    print(
        np.median(Upec_halfhpd[is_good]),
        np.median(Vpec_halfhpd[is_good]),
        np.median(Wpec_halfhpd[is_good]),
    )
    print(np.mean(tot_halfhpd[is_good]), np.median(tot_halfhpd[is_good]))
    num_good = sum(is_good)
    print("# sources used in kriging:", num_good)
    #
    # Krig only good data
    #
    coord_obs = np.vstack((x[is_good], y[is_good])).T
    #
    # Initialize kriging object
    #
    Upec_krig = kriging.Kriging(
        coord_obs,
        Upec[is_good],
        obs_data_cov=cov_Upec[:, is_good][is_good],
    )
    Vpec_krig = kriging.Kriging(
        coord_obs,
        Vpec[is_good],
        obs_data_cov=cov_Vpec[:, is_good][is_good],
    )
    #
    # Fit semivariogram model
    #
    variogram_model = "gaussian"
    nbins = 10
    bin_number = False
    deg = 1
    if bin_number: condition += "-binnumberTrue"
    lag_cutoff = 0.5
    print("Semivariogram Model:", variogram_model)
    Upec_semivar = Upec_krig.fit(
        model=variogram_model,
        deg=deg,
        nbins=nbins,
        bin_number=bin_number,
        lag_cutoff=lag_cutoff,
    )
    Vpec_semivar = Vpec_krig.fit(
        model=variogram_model,
        deg=deg,
        nbins=nbins,
        bin_number=bin_number,
        lag_cutoff=lag_cutoff,
    )
    #
    # Interpolate data
    #
    resample = True  # resample data for kriging
    num_mc = 1000  # number of Monte Carlo samples
    xlow, xhigh = -8, 12
    ylow, yhigh = -5, 15
    # gridx, gridy = np.mgrid[xlow:xhigh:500j, ylow:yhigh:500j]
    # coord_interp = np.vstack((gridx.flatten(), gridy.flatten())).T
    # Upec_arr = np.array([gridx,] * num_mc) * np.nan  # shape = (num_mc, 500, 500)
    # Vpec_arr = np.array([gridx,] * num_mc) * np.nan  # shape = (num_mc, 500, 500)
    # for i in range(num_mc):
    #     Upec_interp, _ = Upec_krig.interp(coord_interp, resample=resample)
    #     Vpec_interp, _ = Vpec_krig.interp(coord_interp, resample=resample)
    #     # Reshape
    #     Upec_interp = Upec_interp.reshape(gridx.shape)
    #     Vpec_interp = Vpec_interp.reshape(gridx.shape)
    #     # Populate arrays
    #     Upec_arr[i] = Upec_interp
    #     Vpec_arr[i] = Vpec_interp
    # Upec_interp_sd = Upec_arr.std(axis=0)
    # Vpec_interp_sd = Vpec_arr.std(axis=0)
    # #
    # # Save to pickle file
    # #
    # pkl = Path(__file__).parent / f"krige_sd_{num_mc}samples.pkl"
    # with open(pkl, "wb") as f:
    #     dill.dump(
    #         {
    #             "Upec_mean_sd": Upec_interp_sd,
    #             "Vpec_mean_sd": Vpec_interp_sd,
    #         }, f
    #     )
    # #
    # # Show semivariograms
    # #
    # Upec_semivar.savefig(
    #     Path(__file__).parent
    #     / f"Upec_semivar_{num_mc}samples.pdf",
    #     bbox_inches="tight",
    # )
    # Vpec_semivar.savefig(
    #     Path(__file__).parent
    #     / f"Vpec_semivar_{num_mc}samples.pdf",
    #     bbox_inches="tight",
    # )
    # Upec_semivar.show()
    # Vpec_semivar.show()
    # plt.show()
    #
    # Load data from pickle file
    #
    pkl = Path(__file__).parent / f"krige_sd_{num_mc}samples.pkl"
    with open(pkl, "rb") as f:
        file = dill.load(f)
        Upec_interp_sd = file["Upec_mean_sd"]
        Vpec_interp_sd = file["Vpec_mean_sd"]
    #
    # Plot interpolated kriging results
    #
    # fig, ax = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    cmap = "viridis"
    extent = (xlow, xhigh, ylow, yhigh)

    # # Plot interpolated Upec
    # # norm_Upec = mpl.colors.Normalize(vmin=-10., vmax=20.)
    # # norm_Upec = mpl.colors.Normalize(vmin=np.min(Upec[is_good]), vmax=np.max(Upec[is_good]))
    # norm_Upec = mpl.colors.Normalize(vmin=np.min(Upec_interp), vmax=np.max(Upec_interp))
    # ax[0].imshow(Upec_interp.T, origin="lower", extent=extent, norm=norm_Upec)
    # cbar_Upec = fig.colorbar(
    #     mpl.cm.ScalarMappable(norm=norm_Upec, cmap=cmap), ax=ax[0], format="%.0f"
    # )
    # ax[0].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    # ax[0].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    # ax[0].set_xlabel("$x$ (kpc)")
    # ax[0].set_ylabel("$y$ (kpc)")
    # ax[0].set_title("$U_s$")
    # cbar_Upec.ax.set_ylabel("km s$^{-1}$", rotation=270)
    # cbar_Upec.ax.get_yaxis().labelpad = 15
    # ax[0].set_aspect("equal")
    # ax[0].grid(False)

    # # Plot interpolated Vpec
    # # norm_Vpec = mpl.colors.Normalize(vmin=-20., vmax=10.)
    # norm_Vpec = mpl.colors.Normalize(vmin=np.min(Vpec_interp), vmax=np.max(Vpec_interp))
    # ax[1].imshow(Vpec_interp.T, origin="lower", extent=extent, norm=norm_Vpec)
    # cbar_Vpec = fig.colorbar(
    #     mpl.cm.ScalarMappable(norm=norm_Vpec, cmap=cmap), ax=ax[1], format="%.0f"
    # )
    # ax[1].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    # ax[1].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    # ax[1].set_xlabel("$x$ (kpc)")
    # ax[1].set_ylabel("$y$ (kpc)")
    # ax[1].set_title("$V_s$")
    # cbar_Vpec.ax.set_ylabel("km s$^{-1}$", rotation=270)
    # cbar_Vpec.ax.get_yaxis().labelpad = 15
    # ax[1].set_aspect("equal")
    # ax[1].grid(False)

    # # Plot actual Upec & Vpec
    # ax[0].scatter(
    #     x[is_good],
    #     y[is_good],
    #     c=Upec[is_good],
    #     norm=norm_Upec,
    #     cmap=cmap,
    #     s=10,
    #     edgecolors="k",
    #     marker="o",
    #     label="Good Masers",
    # )
    # ax[0].scatter(
    #     x[~is_good],
    #     y[~is_good],
    #     c=Upec[~is_good],
    #     norm=norm_Upec,
    #     cmap=cmap,
    #     s=10,
    #     edgecolors="k",
    #     marker="s",
    #     label="Outlier Masers",
    # )
    # ax[0].set_xlim(xlow, xhigh)
    # # ax[0].set_xticks([-5, 0, 5, 10])
    # ax[0].set_ylim(ylow, yhigh)
    # # ax[0].set_yticks([-5, 0, 5, 10, 15])
    # ax[0].legend(loc="lower left", fontsize=9)
    # ax[1].scatter(
    #     x[is_good],
    #     y[is_good],
    #     c=Vpec[is_good],
    #     norm=norm_Vpec,
    #     cmap=cmap,
    #     s=10,
    #     edgecolors="k",
    #     marker="o",
    #     label="Good Masers",
    # )
    # ax[1].scatter(
    #     x[~is_good],
    #     y[~is_good],
    #     c=Vpec[~is_good],
    #     norm=norm_Vpec,
    #     cmap=cmap,
    #     s=10,
    #     edgecolors="k",
    #     marker="s",
    #     label="Outlier Masers",
    # )
    # ax[1].set_xlim(xlow, xhigh)
    # # ax[1].set_xticks([-5, 0, 5, 10])
    # ax[1].set_ylim(ylow, yhigh)
    # # ax[1].set_yticks([-5, 0, 5, 10, 15])
    # ax[1].legend(loc="lower left", fontsize=9)

    # # fig.suptitle(
    # #     f"Interpolated Peculiar Motions ({num_good} Good Masers)\n"
    # #     fr"(Universal Kriging, \texttt{{variogram\_model={variogram_model}}})"
    # # )
    # fig.tight_layout()
    # filename = f"krige_{num_good}good_{condition}_{variogram_model}_{mc_type}.pdf"
    # fig.savefig(
    #     Path(__file__).parent / filename, format="pdf", dpi=300, bbox_inches="tight",
    # )
    # plt.show()

    fig, ax = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    # Plot interpolated Upec's standard deviation
    norm_Upec = mpl.colors.Normalize(
        vmin=np.min(Upec_interp_sd), vmax=np.max(Upec_interp_sd)
    )
    ax[0].imshow(Upec_interp_sd.T, origin="lower", extent=extent, norm=norm_Upec)
    cbar_Upec = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_Upec, cmap=cmap), ax=ax[0], format="%.0f"
    )
    ax[0].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[0].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[0].set_xlabel("$x$ (kpc)")
    ax[0].set_ylabel("$y$ (kpc)")
    ax[0].set_title(r"$\overline{U_s}$ Standard Deviation")
    cbar_Upec.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar_Upec.ax.get_yaxis().labelpad = 15
    ax[0].set_aspect("equal")
    ax[0].grid(False)

    # Plot interpolated Vpec's standard deviation
    norm_Vpec = mpl.colors.Normalize(
        vmin=np.min(Vpec_interp_sd), vmax=np.max(Vpec_interp_sd)
    )
    ax[1].imshow(Vpec_interp_sd.T, origin="lower", extent=extent, norm=norm_Vpec)
    cbar_Vpec = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_Vpec, cmap=cmap), ax=ax[1], format="%.0f"
    )
    ax[1].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[1].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[1].set_xlabel("$x$ (kpc)")
    ax[1].set_ylabel("$y$ (kpc)")
    ax[1].set_title(r"$\overline{V_s}$ Standard Deviation")
    cbar_Vpec.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar_Vpec.ax.get_yaxis().labelpad = 15
    ax[1].set_aspect("equal")
    ax[1].grid(False)

    fig.tight_layout()
    filename = f"krige_sd_{num_mc}samples.pdf"
    fig.savefig(
        Path(__file__).parent / filename, format="pdf", dpi=300, bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
