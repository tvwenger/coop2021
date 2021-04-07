"""
krige_diff.py

Universal kriging of the difference between individual Upecs/Vpecs
and the MCMC average Upec/Vpec.

Isaac Cheng - April 2021
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
from calc_hpd import calc_hpd
from universal_rotcurve import urc

# Values from CW21 A6
_R0_MODE = 8.174602364395952
_ZSUN_MODE = 5.398550615892994
_USUN_MODE = 10.878914326160878
_VSUN_MODE = 10.696801784160257
_WSUN_MODE = 8.087892505141708
_UPEC_MODE = 4.9071771802606285
_VPEC_MODE = -4.521832904300172
_ROLL_MODE = -0.010742182667190958
_A2_MODE = 0.9768982857793898
_A3_MODE = 1.626400628724733


def main(use_bary=True):
    """
    use_bary :: boolean
      If True, use barycentric Cartesian coordinates
      If False, use galactocentric Cartesian coordinates
    """
    # * CONCLUSION:
    # * range of the semivariance points is much larger than the variation from small lags
    # * to large lags. Basically there isn't much structure in the semivariograms.
    mc_type = "HPDmode_NEW2"
    datafile = Path(__file__).parent / Path(f"csvfiles/alldata_{mc_type}.csv")
    pearsonrfile = Path(__file__).parent / "pearsonr_cov.pkl"
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

    # R = data["R_mode"].values
    # R_halfhpd = data["R_halfhpd"].values
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
    z = data["z_mode"].values

    # Only choose good data for kriging
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
    # NOTE: 0.33 => 110 sources
    #       0.35 => 114 sources (looks smoother, but hard to justify cutoff)
    if use_bary:
        # # 2 METHODS THAT PRODUCE NEARLY IDENTICAL RESULTS
        # # (differences maybe due to numerical imprecision in calculating mode)
        # # * METHOD ONE
        #
        # Convert data to barycentric Cartesian coordinates
        #
        # Calculate circular rotation speed
        theta0 = urc(_R0_MODE, a2=_A2_MODE, a3=_A3_MODE, R0=_R0_MODE)
        v_circ_noVpec = urc(np.sqrt(x * x + y * y), a2=_A2_MODE, a3=_A3_MODE, R0=_R0_MODE)
        v_circ = v_circ_noVpec + Vpec
        # Rotate 90 deg CCW
        x, y = -y, x
        # Convert peculiar motions to galactocentric Cartesian
        az = np.arctan2(y, -x)
        cos_az = np.cos(az)
        sin_az = np.sin(az)
        vx_g = Upec * cos_az + v_circ * sin_az
        vy_g = -Upec * sin_az + v_circ * cos_az
        vz_g = Wpec
        # Galactocentric Cartesian to Barycentric Cartesian
        xb, yb, zb, vxb, vyb, vzb = trans.gcen_to_bary(
            x, y, z, Vxg=vx_g, Vyg=vy_g, Vzg=vz_g,
            R0=_R0_MODE, Zsun=_ZSUN_MODE, roll=_ROLL_MODE,
            Usun=_USUN_MODE, Vsun=_VSUN_MODE, Wsun=_WSUN_MODE, Theta0=theta0)
        #
        # # * METHOD TWO
        # glong = data["glong"].values
        # glat = data["glat"].values
        # dist = data["dist_mode"].values
        # cos_glat = np.cos(np.deg2rad(glat))
        # xb = dist * cos_glat * np.cos(np.deg2rad(glong))
        # yb = dist * cos_glat * np.sin(np.deg2rad(glong))
        # zb = dist * np.sin(np.deg2rad(glat))
        #
        # # * ALWAYS DO THIS (for this plot)
        # Rotate 90 deg CW (Sun is on +y-axis)
        xb, yb = yb, -xb
        # Rename Barycentric Cartesian coordinates
        x, y, z = xb, yb, zb

    print("--- MC GOOD DATA STATS ---")
    print(np.mean(Upec[is_good]), np.mean(Upec[is_good]), np.mean(Wpec[is_good]))
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
        Upec[is_good] - _UPEC_MODE,
        # e_obs_data=Upec_halfhpd[is_good],
        obs_data_cov=cov_Upec[:, is_good][is_good],
    )
    Vpec_krig = kriging.Kriging(
        coord_obs,
        Vpec[is_good] - _VPEC_MODE,
        # e_obs_data=Vpec_halfhpd[is_good],
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
    resample = False  # resample data for kriging
    if use_bary:
        xlow, xhigh = -8, 12
        ylow, yhigh = -5 - _R0_MODE, 15 - _R0_MODE
    else:
        xlow, xhigh = -8, 12
        ylow, yhigh = -5, 15
    gridx, gridy = np.mgrid[xlow:xhigh:500j, ylow:yhigh:500j]
    coord_interp = np.vstack((gridx.flatten(), gridy.flatten())).T
    Upec_interp, Upec_interp_var = Upec_krig.interp(coord_interp, resample=resample)
    Vpec_interp, Vpec_interp_var = Vpec_krig.interp(coord_interp, resample=resample)
    #
    # Print stats
    #
    _, Upec_mode, Upec_low, Upec_high = calc_hpd(Upec_interp, "scipy")
    _, Vpec_mode, Vpec_low, Vpec_high = calc_hpd(Vpec_interp, "scipy")
    _, Upec_var_mode, Upec_var_low, Upec_var_high = calc_hpd(Upec_interp_var, "scipy")
    _, Vpec_var_mode, Vpec_var_low, Vpec_var_high = calc_hpd(Vpec_interp_var, "scipy")
    Upec_sd_mode, Upec_sd_low, Upec_sd_high = np.sqrt(Upec_var_mode), np.sqrt(Upec_var_low), np.sqrt(Upec_var_high)
    Vpec_sd_mode, Vpec_sd_low, Vpec_sd_high = np.sqrt(Vpec_var_mode), np.sqrt(Vpec_var_low), np.sqrt(Vpec_var_high)
    v_tot_interp = np.sqrt(Upec_interp ** 2 + Vpec_interp ** 2)
    _, tot_mode, tot_low, tot_high = calc_hpd(v_tot_interp, "scipy")
    print("Interpolated Upec mode, low, high:", Upec_mode, Upec_low, Upec_high)
    print("Interpolated Vpec mode, low, high:", Vpec_mode, Vpec_low, Vpec_high)
    print("Interpolated Upec_var mode, low, high:", Upec_var_mode, Upec_var_low, Upec_var_high)
    print("Interpolated Vpec_var mode, low, high:", Vpec_var_mode, Vpec_var_low, Vpec_var_high)
    print("Interpolated Upec_sd mode, low, high:", Upec_sd_mode, Upec_sd_low, Upec_sd_high)
    print("Interpolated Vpec_sd mode, low, high:", Vpec_sd_mode, Vpec_sd_low, Vpec_sd_high)
    print("Interpolated xy-magnitude mode, low, high:", tot_mode, tot_low, tot_high)
    # Reshape
    Upec_interp = Upec_interp.reshape(gridx.shape)
    Upec_interp_sd = np.sqrt(Upec_interp_var).reshape(gridx.shape)
    Vpec_interp = Vpec_interp.reshape(gridx.shape)
    Vpec_interp_sd = np.sqrt(Vpec_interp_var).reshape(gridx.shape)
    print("Min & Max of interpolated Upec:", np.min(Upec_interp), np.max(Upec_interp))
    print("Min & Max of interpolated Vpec:", np.min(Vpec_interp), np.max(Vpec_interp))
    print("Mean interpolated Upec & Vpec:", np.mean(Upec_interp), np.mean(Vpec_interp))
    print(
        "Median interpolated Upec & Vpec:", np.median(Upec_interp), np.median(Vpec_interp)
    )
    print(
        "Mean SD of interpolated Upec & Vpec:",
        np.mean(Upec_interp_sd),
        np.mean(Vpec_interp_sd),
    )
    print(
        "Median SD of interpolated Upec & Vpec:",
        np.median(Upec_interp_sd),
        np.median(Vpec_interp_sd),
    )
    v_tot_interp = np.sqrt(Upec_interp ** 2 + Vpec_interp ** 2)
    print(
        "Mean, median, and SD of interpolated magnitude",
        np.mean(v_tot_interp),
        np.median(v_tot_interp),
        np.std(v_tot_interp),
    )
    #
    # Show semivariograms
    #
    Upec_semivar.savefig(
        Path(__file__).parent
        / f"UpecDiff_semivar_{num_good}good_{condition}_{variogram_model}_{mc_type}.pdf",
        bbox_inches="tight",
    )
    Vpec_semivar.savefig(
        Path(__file__).parent
        / f"VpecDiff_semivar_{num_good}good_{condition}_{variogram_model}_{mc_type}.pdf",
        bbox_inches="tight",
    )
    Upec_semivar.show()
    Vpec_semivar.show()
    plt.show()
    #
    # Plot interpolated kriging results
    #
    fig, ax = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    cmap = "viridis"
    extent = (xlow, xhigh, ylow, yhigh)

    # Plot interpolated Upec
    norm_Upec = mpl.colors.Normalize(vmin=np.min(Upec_interp), vmax=np.max(Upec_interp))
    ax[0].imshow(Upec_interp.T, origin="lower", extent=extent, norm=norm_Upec)
    cbar_Upec = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_Upec, cmap=cmap), ax=ax[0], format="%.0f"
    )
    if use_bary:
        ax[0].scatter(0, -_R0_MODE, marker="X", c='tab:red', s=15, zorder=10) # marker at galactic centre
    ax[0].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[0].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[0].set_xlabel("$x$ (kpc)")
    ax[0].set_ylabel("$y$ (kpc)")
    ax[0].set_title(r"$U_s - \overline{U_s}$")
    cbar_Upec.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar_Upec.ax.get_yaxis().labelpad = 15
    ax[0].set_aspect("equal")
    ax[0].grid(False)

    # Plot interpolated Vpec
    norm_Vpec = mpl.colors.Normalize(vmin=np.min(Vpec_interp), vmax=np.max(Vpec_interp))
    ax[1].imshow(Vpec_interp.T, origin="lower", extent=extent, norm=norm_Vpec)
    cbar_Vpec = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_Vpec, cmap=cmap), ax=ax[1], format="%.0f"
    )
    if use_bary:
        ax[1].scatter(0, -_R0_MODE, marker="X", c='tab:red', s=15, zorder=10) # marker at galactic centre
    ax[1].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[1].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[1].set_xlabel("$x$ (kpc)")
    ax[1].set_ylabel("$y$ (kpc)")
    ax[1].set_title(r"$V_s - \overline{V_s}$")
    cbar_Vpec.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar_Vpec.ax.get_yaxis().labelpad = 15
    ax[1].set_aspect("equal")
    ax[1].grid(False)

    # Plot actual Upec & Vpec
    ax[0].scatter(
        x[is_good],
        y[is_good],
        c=Upec[is_good] - _UPEC_MODE,
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
        c=Upec[~is_good] - _UPEC_MODE,
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
        c=Vpec[is_good] - _VPEC_MODE,
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
        c=Vpec[~is_good] - _VPEC_MODE,
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
    filename = f"krigeDiff_{num_good}good_{condition}_{variogram_model}_{mc_type}.pdf"
    fig.savefig(
        Path(__file__).parent / filename, format="pdf", dpi=300, bbox_inches="tight",
    )
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    # Plot interpolated Upec's standard deviation
    norm_Upec = mpl.colors.Normalize(
        vmin=np.min(Upec_interp_sd), vmax=np.max(Upec_interp_sd)
    )
    ax[0].imshow(Upec_interp_sd.T, origin="lower", extent=extent, norm=norm_Upec)
    cbar_Upec = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_Upec, cmap=cmap), ax=ax[0], format="%.0f"
    )
    if use_bary:
        ax[0].scatter(0, -_R0_MODE, marker="X", c='tab:red', s=15, zorder=10) # marker at galactic centre
    ax[0].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[0].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[0].set_xlabel("$x$ (kpc)")
    ax[0].set_ylabel("$y$ (kpc)")
    ax[0].set_title(r"$U_s - \overline{U_s}$ Standard Deviation")
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
    if use_bary:
        ax[1].scatter(0, -_R0_MODE, marker="X", c='tab:red', s=15, zorder=10) # marker at galactic centre
    ax[1].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[1].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[1].set_xlabel("$x$ (kpc)")
    ax[1].set_ylabel("$y$ (kpc)")
    ax[1].set_title(r"$V_s - \overline{V_s}$ Standard Deviation")
    cbar_Vpec.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar_Vpec.ax.get_yaxis().labelpad = 15
    ax[1].set_aspect("equal")
    ax[1].grid(False)

    # fig.suptitle(
    #     f"Standard Deviations of Interpolated Peculiar Motions ({num_good} Good Masers)\n"
    #     fr"(Universal Kriging, \texttt{{variogram\_model={variogram_model}}})"
    # )
    fig.tight_layout()
    filename = f"krigeDiff_sd_{num_good}good_{condition}_{variogram_model}_{mc_type}.pdf"
    fig.savefig(
        Path(__file__).parent / filename, format="pdf", dpi=300, bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main(use_bary=True)
