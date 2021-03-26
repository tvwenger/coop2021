"""
krige_xy.py

Universal kriging of x- and y- velocity components
from data in .csv file

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

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans

def get_coords(data):
    # # Mean values from 100 distance, mean Upec/Vpec trace, peak everything file
    # R0_mean, Zsun_mean, roll_mean = 8.17845, 5.0649223, 0.0014527875
    # Usun_mean, Vsun_mean, Wsun_mean = 10.879447, 10.540543, 8.1168785
    # Upec_mean, Vpec_mean = 4.912622, -4.588946
    # a2_mean, a3_mean = 0.96717525, 1.624953

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

    # === Get data ===
    glon = data["glong"].values  # deg
    glat = data["glat"].values  # deg
    plx = data["plx"].values  # mas
    e_plx = data["e_plx"].values

    # === Calculate predicted values from optimal parameters ===
    # Parallax to distance
    dist = trans.parallax_to_dist(plx, e_parallax=e_plx)

    # Galactic to barycentric Cartesian coordinates
    bary_x, bary_y, bary_z = trans.gal_to_bary(glon, glat, dist)

    # Barycentric Cartesian to galactocentric Cartesian coodinates
    gcen_x, gcen_y, gcen_z = trans.bary_to_gcen(
        bary_x, bary_y, bary_z, R0=R0_mode, Zsun=Zsun_mode, roll=roll_mode
    )
    # Rotate 90deg CW to Reid convention
    gcen_x, gcen_y = gcen_y, -gcen_x

    return gcen_x, gcen_y, gcen_z


def main():
    # * CONCLUSION:
    # * range of the semivariance points is much larger than the variation from small lags
    # * to large lags. Basically there isn't much structure in the semivariograms.
    mc_type = "HPDmode_NEW"
    datafile = Path(__file__).parent / Path(f"csvfiles/alldata_{mc_type}.csv")
    pearsonrfile = Path(__file__).parent / "pearsonr_cov.pkl"
    data = pd.read_csv(datafile)
    with open(pearsonrfile, "rb") as f:
        file = dill.load(f)
        cov_vx = file["cov_vx"]
        cov_vy = file["cov_vy"]

    # Only choose sources that have R > 4 kpc
    is_tooclose = data["is_tooclose"].values == 1
    data = data[~is_tooclose]
    cov_vx = cov_vx[:, ~is_tooclose][~is_tooclose]
    cov_vy = cov_vy[:, ~is_tooclose][~is_tooclose]

    # R = data["R_mode"].values
    # R_halfhpd = data["R_halfhpd"].values
    az = data["az_mode"].values
    Upec = data["Upec_mode"].values
    Vpec = data["Vpec_mode"].values
    vx = data["vx_mode"].values
    vy = data["vy_mode"].values
    vx_halfhpd = data["vx_halfhpd"].values
    vy_halfhpd = data["vy_halfhpd"].values
    # vz = data["vz_mode"].values
    # tot = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    tot_xy = np.sqrt(vx ** 2 + vy ** 2)
    # print("Min & max vx:", np.min(vx), np.max(vx))
    # print("Min & max vy:", np.min(vy), np.max(vy))
    # print("Min & max tot_xy:", np.min(tot_xy), np.max(tot_xy))
    # print("Mean & median vx:", np.mean(vx), np.median(vx))
    # print("Mean & median vy:", np.mean(vy), np.median(vy))
    # print("Mean & median tot_xy:", np.mean(tot_xy), np.median(tot_xy))

    x = data["x_mode"].values
    y = data["y_mode"].values

    # # Azimuthal angle
    # x, y = -y, x  # rotate 90 deg CCW
    # az = np.arctan2(y, -x)
    # cos_az = np.cos(az)
    # sin_az = np.sin(az)
    # # Upec & Vpec to vx & vy
    # # (recall v_radial = -Upec & Vpec is positive CW)
    # vx = Upec * cos_az + Vpec * sin_az  # km/s
    # vy = -Upec * sin_az + Vpec * cos_az  # km/s
    # x, y = y, -x  # Rotate 90 deg CW
    # vx, vy = vy, -vx

    # # Azimuthal angle
    # # az = np.arctan2(y, x)
    # cos_az = np.cos(az)
    # sin_az = np.sin(az)
    # # # Upec & Vpec to vx & vy
    # # # (recall v_radial = -Upec & Vpec is positive CW)
    # # vx = -Upec * cos_az + Vpec * sin_az
    # # vy = -Upec * sin_az - Vpec * cos_az
    # vx = Upec * cos_az + Vpec * sin_az  # km/s
    # vy = -Upec * sin_az + Vpec * cos_az  # km/s
    # vx, vy = vy, -vx  # Rotate 90 deg CW


    tot_xy = np.sqrt(vx ** 2 + vy ** 2)
    print("Min & max vx:", np.min(vx), np.max(vx))
    print("Min & max vy:", np.min(vy), np.max(vy))
    print("Min & max tot_xy:", np.min(tot_xy), np.max(tot_xy))
    print("Mean & median vx:", np.mean(vx), np.median(vx))
    print("Mean & median vy:", np.mean(vy), np.median(vy))
    print("Mean & median tot_xy:", np.mean(tot_xy), np.median(tot_xy))

    # # Only choose good data for kriging
    # condition = "halfhpd-all-R1.4-binnumberTrue"
    # # condition = "all185-eobsdata"
    # # condition = "tot35-R1-binnumTrue"
    # # condition = "all202-tot35-R1-binnumTrue"
    # is_good = (
    #     # (R < 10000)
    #     (R_halfhpd < 0.8)
    #     & (Upec_halfhpd < 24.0)
    #     & (Vpec_halfhpd < 24.0)
    #     & (Wpec_halfhpd < 20.0)
    #     & (tot_halfhpd < 35.0)
    #     & (tot_xy_halfhpd < 32.0)
    # )
    #
    # condition = "90ptile"
    # percentile = 90
    # lower = 0.5 * (100 - percentile)
    # upper = 0.5 * (100 + percentile)
    # is_good = (
    #     (Upec > np.percentile(Upec, lower))
    #     & (Upec < np.percentile(Upec, upper))
    #     & (Vpec > np.percentile(Vpec, lower))
    #     & (Vpec < np.percentile(Vpec, upper))
    #     & (Wpec > np.percentile(Wpec, lower))
    #     & (Wpec < np.percentile(Wpec, upper))
    # )
    #
    condition = "no-mcmc-outlier"
    is_good = data["is_outlier"].values == 0

    print("--- MC GOOD DATA STATS ---")
    print("Min & max vx:", np.min(vx[is_good]), np.max(vx[is_good]))
    print("Min & max vy:", np.min(vy[is_good]), np.max(vy[is_good]))
    print("Min & max tot_xy:", np.min(tot_xy[is_good]), np.max(tot_xy[is_good]))
    print("Mean & median vx:", np.mean(vx[is_good]), np.median(vx[is_good]))
    print("Mean & median vy:", np.mean(vy[is_good]), np.median(vy[is_good]))
    print("Mean & median tot_xy:", np.mean(tot_xy[is_good]), np.median(tot_xy[is_good]))
    num_good = sum(is_good)
    print("# sources used in kriging:", num_good)
    #
    # Krig only good data
    #
    coord_obs = np.vstack((x[is_good], y[is_good])).T
    #
    # Initialize kriging object
    #
    vx_krige = kriging.Kriging(
        coord_obs,
        vx[is_good],
        # e_obs_data=vx_halfhpd[is_good],
        obs_data_cov=cov_vx[:, is_good][is_good],
    )
    vy_krige = kriging.Kriging(
        coord_obs,
        vy[is_good],
        # e_obs_data=vy_halfhpd[is_good],
        obs_data_cov=cov_vy[:, is_good][is_good],
    )
    #
    # Fit semivariogram model
    #
    variogram_model = "wave"
    nbins = 10
    bin_number = False
    if bin_number:
        condition += "-binnumberTrue"
    lag_cutoff = 1.0
    print("Semivariogram Model:", variogram_model)
    vx_semivar, vx_corner = vx_krige.fit(
        model=variogram_model,
        deg=1,
        nbins=nbins,
        bin_number=bin_number,
        lag_cutoff=lag_cutoff,
        nsims=1000,
    )
    vy_semivar, vy_corner = vy_krige.fit(
        model=variogram_model,
        deg=1,
        nbins=nbins,
        bin_number=bin_number,
        lag_cutoff=lag_cutoff,
        nsims=1000,
    )
    #
    # Interpolate data
    #
    xlow, xhigh = -8, 12
    ylow, yhigh = -5, 15
    gridx, gridy = np.mgrid[xlow:xhigh:500j, ylow:yhigh:500j]
    coord_interp = np.vstack((gridx.flatten(), gridy.flatten())).T
    vx_interp, vx_interp_var = vx_krige.interp(coord_interp)
    vy_interp, vy_interp_var = vy_krige.interp(coord_interp)
    # Reshape
    vx_interp = vx_interp.reshape(gridx.shape)
    vx_interp_sd = np.sqrt(vx_interp_var).reshape(gridx.shape)
    vy_interp = vy_interp.reshape(gridx.shape)
    vy_interp_sd = np.sqrt(vy_interp_var).reshape(gridx.shape)
    print("Min & Max of interpolated vx:", np.min(vx_interp), np.max(vx_interp))
    print("Min & Max of interpolated vy:", np.min(vy_interp), np.max(vy_interp))
    print("Mean interpolated vx & vy:", np.mean(vx_interp), np.mean(vy_interp))
    print("Median interpolated vx & vy:", np.median(vx_interp), np.median(vy_interp))
    print(
        "Mean SD of interpolated vx & vy:", np.mean(vx_interp_sd), np.mean(vy_interp_sd),
    )
    print(
        "Median SD of interpolated vx & vy:",
        np.median(vy_interp_sd),
        np.median(vy_interp_sd),
    )
    v_tot_interp = np.sqrt(vy_interp ** 2 + vy_interp ** 2)
    print(
        "Mean, median, and SD of interpolated magnitude",
        np.mean(v_tot_interp),
        np.median(v_tot_interp),
        np.std(v_tot_interp),
    )
    #
    # Show semivariograms
    #
    vx_semivar.savefig(
        Path(__file__).parent
        / f"vx_semivar_{num_good}good_{condition}_{variogram_model}_{mc_type}.pdf",
        bbox_inches="tight",
    )
    vy_semivar.savefig(
        Path(__file__).parent
        / f"vy_semivar_{num_good}good_{condition}_{variogram_model}_{mc_type}.pdf",
        bbox_inches="tight",
    )
    vx_semivar.show()
    vy_semivar.show()
    #
    # Plot corner plots
    #
    if vx_corner is not None and vy_corner is not None:
        vx_corner.savefig(
            Path(__file__).parent
            / f"vx_corner_{num_good}good_{condition}_{variogram_model}_{mc_type}.pdf",
            bbox_inches="tight",
        )
        vy_corner.savefig(
            Path(__file__).parent
            / f"vy_corner_{num_good}good_{condition}_{variogram_model}_{mc_type}.pdf",
            bbox_inches="tight",
        )
        vx_corner.show()
        vy_corner.show()
    print("Showing plots")
    plt.show()
    #
    # Plot interpolated kriging results
    #
    fig, ax = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    cmap = "viridis"
    extent = (xlow, xhigh, ylow, yhigh)

    # Plot interpolated vx
    # norm_vx = mpl.colors.Normalize(vmin=-10., vmax=20.)
    # norm_vx = mpl.colors.Normalize(vmin=np.min(vx[is_good]), vmax=np.max(vx[is_good]))
    norm_vx = mpl.colors.Normalize(vmin=np.min(vx_interp), vmax=np.max(vx_interp))
    ax[0].imshow(vx_interp.T, origin="lower", extent=extent, norm=norm_vx)
    cbar_vx = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_vx, cmap=cmap), ax=ax[0], format="%.0f"
    )
    ax[0].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[0].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[0].set_xlabel("$x$ (kpc)")
    ax[0].set_ylabel("$y$ (kpc)")
    ax[0].set_title("$v_x$")
    cbar_vx.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar_vx.ax.get_yaxis().labelpad = 15
    ax[0].set_aspect("equal")
    ax[0].grid(False)

    # Plot interpolated vy
    # norm_vy = mpl.colors.Normalize(vmin=-20., vmax=10.)
    norm_vy = mpl.colors.Normalize(vmin=np.min(vy_interp), vmax=np.max(vy_interp))
    ax[1].imshow(vy_interp.T, origin="lower", extent=extent, norm=norm_vy)
    cbar_vy = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_vy, cmap=cmap), ax=ax[1], format="%.0f"
    )
    ax[1].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[1].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[1].set_xlabel("$x$ (kpc)")
    ax[1].set_ylabel("$y$ (kpc)")
    ax[1].set_title("$v_y$")
    cbar_vy.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar_vy.ax.get_yaxis().labelpad = 15
    ax[1].set_aspect("equal")
    ax[1].grid(False)

    # Plot actual vx & vy
    ax[0].scatter(
        x[is_good],
        y[is_good],
        c=vx[is_good],
        norm=norm_vx,
        cmap=cmap,
        s=10,
        edgecolors="k",
        marker="o",
        label="Good Masers",
    )
    ax[0].scatter(
        x[~is_good],
        y[~is_good],
        c=vx[~is_good],
        norm=norm_vx,
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
        c=vy[is_good],
        norm=norm_vy,
        cmap=cmap,
        s=10,
        edgecolors="k",
        marker="o",
        label="Good Masers",
    )
    ax[1].scatter(
        x[~is_good],
        y[~is_good],
        c=vy[~is_good],
        norm=norm_vy,
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

    fig.tight_layout()
    filename = f"krige_xy_{num_good}good_{condition}_{variogram_model}_{mc_type}.pdf"
    fig.savefig(
        Path(__file__).parent / filename, format="pdf", dpi=300, bbox_inches="tight",
    )
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    # Plot interpolated vx's standard deviation
    norm_vx = mpl.colors.Normalize(vmin=np.min(vx_interp_sd), vmax=np.max(vx_interp_sd))
    ax[0].imshow(vx_interp_sd.T, origin="lower", extent=extent, norm=norm_vx)
    cbar_vx = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_vx, cmap=cmap), ax=ax[0], format="%.0f"
    )
    ax[0].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[0].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[0].set_xlabel("$x$ (kpc)")
    ax[0].set_ylabel("$y$ (kpc)")
    ax[0].set_title("$v_x$ Standard Deviation")
    cbar_vx.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar_vx.ax.get_yaxis().labelpad = 15
    ax[0].set_aspect("equal")
    ax[0].grid(False)

    # Plot interpolated Vpec's standard deviation
    norm_vy = mpl.colors.Normalize(
        vmin=np.min(vy_interp_sd), vmax=np.max(vy_interp_sd)
    )
    ax[1].imshow(vy_interp_sd.T, origin="lower", extent=extent, norm=norm_vy)
    cbar_vy = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_vy, cmap=cmap), ax=ax[1], format="%.0f"
    )
    ax[1].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[1].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[1].set_xlabel("$x$ (kpc)")
    ax[1].set_ylabel("$y$ (kpc)")
    ax[1].set_title("$v_y$ Standard Deviation")
    cbar_vy.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar_vy.ax.get_yaxis().labelpad = 15
    ax[1].set_aspect("equal")
    ax[1].grid(False)

    # fig.suptitle(
    #     f"Standard Deviations of Interpolated Peculiar Motions ({num_good} Good Masers)\n"
    #     fr"(Universal Kriging, \texttt{{variogram\_model={variogram_model}}})"
    # )
    fig.tight_layout()
    filename = f"krige_xy_sd_{num_good}good_{condition}_{variogram_model}_{mc_type}.pdf"
    fig.savefig(
        Path(__file__).parent / filename, format="pdf", dpi=300, bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
