"""
krige_vlsr.py

Universal kriging of the LSR velocity difference
between the vlsr derived using individual peculiar motions
and the vlsr derived using the average pecular motions (i.e., from MCMC).

Isaac Cheng - April 2021
"""

import sys
from pathlib import Path
import dill
from kd.cw21_rotcurve_w_mc import nominal_params
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from kriging import kriging
from kd import kd_utils, cw21_rotcurve

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans
from calc_hpd import calc_hpd
from universal_rotcurve import urc

# Values from WC21 A6
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
#
# IAU defined LSR
#
_Ustd = 10.27
_Vstd = 15.32
_Wstd = 7.74


def calc_theta(R, a2=_A2_MODE, a3=_A3_MODE, R0=_R0_MODE):
    """
    Return circular orbit speed at a given Galactocentric radius.

    Parameters:
      R :: scalar or array of scalars
        Galactocentric radius (kpc)

      a2, a3 :: scalars (optional)
        CW21 rotation curve parameters

      R0 :: scalar (optional)
        Solar Galactocentric radius (kpc)

    Returns: theta
      theta :: scalar or array of scalars
        circular orbit speed at R (km/s)
    """
    input_scalar = np.isscalar(R)
    R = np.atleast_1d(R)
    #
    # Re-production of Reid+2019 FORTRAN code
    #
    rho = R / (a2 * R0)
    lam = (a3 / 1.5) ** 5.0
    loglam = np.log10(lam)
    term1 = 200.0 * lam ** 0.41
    term2 = np.sqrt(
        0.8 + 0.49 * loglam + 0.75 * np.exp(-0.4 * lam) / (0.47 + 2.25 * lam ** 0.4)
    )
    term3 = (0.72 + 0.44 * loglam) * 1.97 * rho ** 1.22 / (rho ** 2.0 + 0.61) ** 1.43
    term4 = 1.6 * np.exp(-0.4 * lam) * rho ** 2.0 / (rho ** 2.0 + 2.25 * lam ** 0.4)
    #
    # Catch non-physical case where term3 + term4 < 0
    #
    term = term3 + term4
    term[term < 0.0] = np.nan
    #
    # Circular velocity
    #
    theta = term1 / term2 * np.sqrt(term)
    if input_scalar:
        return theta[0]
    return theta


def calc_vlsr(
    glong,
    glat,
    dist,
    R0=_R0_MODE,
    Usun=_USUN_MODE,
    Vsun=_VSUN_MODE,
    Wsun=_WSUN_MODE,
    Upec=_UPEC_MODE,
    Vpec=_VPEC_MODE,
    a2=_A2_MODE,
    a3=_A3_MODE,
    Zsun=_ZSUN_MODE,
    roll=_ROLL_MODE,
    peculiar=True,
):
    """
    Return the IAU-LSR velocity at a given Galactic longitude and
    line-of-sight distance.

    Parameters:
      glong, glat :: scalars or arrays of scalars
        Galactic longitude and latitude (deg).

      dist :: scalar or array of scalars
        line-of-sight distance (kpc).

      R0 :: scalar (optional)
        Solar Galactocentric radius (kpc)

      Usun, Vsun, Wsun, Upec, Vpec, a2, a3 :: scalars (optional)
        CW21 rotation curve parameters

      Zsun :: scalar (optional)
        Height of sun above Galactic midplane (pc)

      roll :: scalar (optional)
        Roll of Galactic midplane relative to b=0 (deg)

      peculiar :: boolean (optional)
        If True, include HMSFR peculiar motion component

    Returns: vlsr
      vlsr :: scalar or array of scalars
        LSR velocity (km/s).
    """
    is_print = False
    if is_print:
        print(
            "glong, glat, dist in calc_vlsr",
            np.shape(glong),
            np.shape(glat),
            np.shape(dist),
        )
    input_scalar = np.isscalar(glong) and np.isscalar(glat) and np.isscalar(dist)
    glong, glat, dist = np.atleast_1d(glong, glat, dist)
    cos_glong = np.cos(np.deg2rad(glong))
    sin_glong = np.sin(np.deg2rad(glong))
    cos_glat = np.cos(np.deg2rad(glat))
    sin_glat = np.sin(np.deg2rad(glat))
    #
    # Convert distance to Galactocentric, catch small Rgal
    #
    Rgal = kd_utils.calc_Rgal(
        glong, glat, dist, R0=R0, Zsun=Zsun, roll=roll, use_Zsunroll=True
    )
    Rgal[Rgal < 1.0e-6] = 1.0e-6  # Catch small Rgal
    az = kd_utils.calc_az(
        glong, glat, dist, R0=R0, Zsun=Zsun, roll=roll, use_Zsunroll=True
    )
    cos_az = np.cos(np.deg2rad(az))
    sin_az = np.sin(np.deg2rad(az))
    if is_print:
        print(
            "Rgal, cos_az, sin_az in calc_vlsr",
            np.shape(Rgal),
            np.shape(cos_az),
            np.shape(sin_az),
        )
    #
    # Rotation curve circular velocity
    #
    theta = calc_theta(Rgal, a2=a2, a3=a3, R0=R0)
    theta0 = calc_theta(R0, a2=a2, a3=a3, R0=R0)
    if is_print:
        print("Theta, theta0 in calc_vlsr", np.shape(theta), np.shape(theta0))
        print("Upec, Vpec in calc_vlsr", np.shape(Upec), np.shape(Vpec))
    #
    # Add HMSFR peculiar motion
    #
    if peculiar:
        vR = -Upec
        vAz = theta + Vpec
        vZ = 0.0
    else:
        vR = 0.0
        vAz = theta
        vZ = 0.0
    vXg = -vR * cos_az + vAz * sin_az
    vYg = vR * sin_az + vAz * cos_az
    vZg = vZ
    if is_print:
        print(
            "1st vXg, vYg, vZg in calc_vlsr", np.shape(vXg), np.shape(vYg), np.shape(vZg)
        )
    #
    # Convert to barycentric
    #
    X = dist * cos_glat * cos_glong
    Y = dist * cos_glat * sin_glong
    Z = dist * sin_glat
    if is_print:
        print("X, Y, Z in calc_vlsr", np.shape(X), np.shape(Y), np.shape(Z))
    # useful constants
    sin_tilt = Zsun / 1000.0 / R0
    cos_tilt = np.cos(np.arcsin(sin_tilt))
    sin_roll = np.sin(np.deg2rad(roll))
    cos_roll = np.cos(np.deg2rad(roll))
    # solar peculiar motion
    vXg = vXg - Usun
    vYg = vYg - theta0 - Vsun
    vZg = vZg - Wsun
    if is_print:
        print(
            "2nd vXg, vYg, vZg in calc_vlsr", np.shape(vXg), np.shape(vYg), np.shape(vZg)
        )
    # correct tilt and roll of Galactic midplane
    vXg1 = vXg * cos_tilt - vZg * sin_tilt
    vYg1 = vYg
    vZg1 = vXg * sin_tilt + vZg * cos_tilt
    vXh = vXg1
    vYh = vYg1 * cos_roll + vZg1 * sin_roll
    vZh = -vYg1 * sin_roll + vZg1 * cos_roll
    vbary = (X * vXh + Y * vYh + Z * vZh) / dist
    if is_print:
        print("vbary in calc_vlsr", np.shape(vbary))
    #
    # Convert to IAU-LSR
    #
    vlsr = vbary + (_Ustd * cos_glong + _Vstd * sin_glong) * cos_glat + _Wstd * sin_glat
    if is_print:
        print("final vlsr shape", np.shape(vlsr))
    if input_scalar:
        return vlsr[0]
    return vlsr


def calc_cov_vlsr(num_samples=10000):
    datafile = Path(__file__).parent / Path(f"csvfiles/alldata_HPDmode_NEW2.csv")
    data = pd.read_csv(datafile)
    #
    # Observed vlsr
    glong = data["glong"].values
    glat = data["glat"].values
    dist = data["dist_mode"].values
    vlsr_data = data["vlsr_med"].values
    e_vlsr_data = data["vlsr_std"].values
    num_sources = len(vlsr_data)
    vlsr = np.random.normal(loc=vlsr_data, scale=e_vlsr_data, size=(num_samples, num_sources))
    #
    kdefile = Path("kd_pkl/cw21_kde_krige.pkl")
    with open(kdefile, "rb") as f:
        kde = dill.load(f)["full"]
    R0, Zsun, Usun, Vsun, Wsun, Upec, Vpec, roll, a2, a3 = kde.resample(num_samples)
    #
    # Predicted vlsr
    #
    mc_params = {
        "R0": R0[np.newaxis].T,
        "Zsun": Zsun[np.newaxis].T,
        "Usun": Usun[np.newaxis].T,
        "Vsun": Vsun[np.newaxis].T,
        "Wsun": Wsun[np.newaxis].T,
        "Upec": Upec[np.newaxis].T,
        "Vpec": Vpec[np.newaxis].T,
        "roll": roll[np.newaxis].T,
        "a2": a2[np.newaxis].T,
        "a3": a3[np.newaxis].T,
    }
    vlsr_pred = cw21_rotcurve.calc_vlsr(glong, glat, dist, peculiar=True, **mc_params)
    vlsr_diff = vlsr - vlsr_pred
    #
    # Calculate correlation coefficient & covariance
    #
    r_vlsr_diff = np.ones((num_sources, num_sources), float) * np.nan
    cov_vlsr_diff = np.ones((num_sources, num_sources), float) * np.nan
    for i in range(num_sources):
        for j in range(num_sources):
            vlsr_diff_meani = np.mean(vlsr_diff[:, i])
            vlsr_diff_meanj = np.mean(vlsr_diff[:, j])
            vlsr_diff_diffi = vlsr_diff[:, i] - vlsr_diff_meani
            vlsr_diff_diffj = vlsr_diff[:, j] - vlsr_diff_meanj
            r_num = np.sum(vlsr_diff_diffi * vlsr_diff_diffj)
            r_den = np.sqrt(np.sum(vlsr_diff_diffi ** 2)) * np.sqrt(
                np.sum(vlsr_diff_diffj ** 2)
            )
            # Fringe cases
            r = r_num / r_den
            r = 1 if r > 1 else r
            r = -1 if r < -1 else r
            # Populate array
            r_vlsr_diff[i, j] = r
            cov_vlsr_diff[i, j] = r_num / num_samples
    print(np.max(r_vlsr_diff[r_vlsr_diff < 0.99]))
    print(np.min(r_vlsr_diff[r_vlsr_diff > -0.99]))
    outfile = Path(__file__).parent / Path("pearsonr_cov_vlsr.pkl")
    with open(outfile, "wb") as f:
        dill.dump(
            {"r_vlsr_diff": r_vlsr_diff, "cov_vlsr_diff": cov_vlsr_diff,}, f,
        )
    print("Saved to .pkl file!")


def main():
    datafile = Path(__file__).parent / Path(f"csvfiles/alldata_HPDmode_NEW2.csv")
    data = pd.read_csv(datafile)
    covfile = Path(__file__).parent / Path("pearsonr_cov_vlsr.pkl")
    with open(covfile, "rb") as f:
        cov_vlsr_diff = dill.load(f)["cov_vlsr_diff"]
        # r_vlsr_diff = dill.load(f)["r_vlsr_diff"]
    # print(r_vlsr_diff.diagonal())
    # plt.hist(r_vlsr_diff.flatten(), bins=50)
    # plt.show()
    # return None
    #
    # Database vlsr
    #
    glong = data["glong"].values
    glat = data["glat"].values
    dist = data["dist_mode"].values
    vlsr = data["vlsr_med"].values
    #
    # Filter data
    #
    is_good = np.ones_like(vlsr, dtype=bool)
    # is_good = (data["is_tooclose"].values == 0)
    # is_good = (data["is_tooclose"].values == 0) & (data["is_outlier"] == 0)
    num_good = is_good.sum()
    print("NUM GOOD MASERS:", num_good)
    #
    # Predicted vlsr from axisymmetric GRC
    #
    nom_params = {
        "R0": _R0_MODE,
        "Zsun": _ZSUN_MODE,
        "Usun": _USUN_MODE,
        "Vsun": _VSUN_MODE,
        "Wsun": _WSUN_MODE,
        "Upec": _UPEC_MODE,
        "Vpec": _VPEC_MODE,
        "roll": _ROLL_MODE,
        "a2": _A2_MODE,
        "a3": _A3_MODE,
    }
    vlsr_pred = cw21_rotcurve.calc_vlsr(glong, glat, dist, peculiar=True, **nom_params)
    #
    # Find difference
    #
    vlsr_diff = vlsr - vlsr_pred
    #
    # Change data to barycentric Cartesian coordinates
    #
    xb, yb, zb = trans.gal_to_bary(glong, glat, dist)
    xb, yb = yb, -xb  # rotate 90 deg CW (Sun is on +y-axis)
    #
    # Initialize kriging object (krige difference between observed and predicted vlsr)
    #
    obs_pos = np.vstack((xb, yb)).T
    vlsr_krige = kriging.Kriging(obs_pos[is_good], vlsr_diff[is_good],
                                 obs_data_cov=cov_vlsr_diff[:, is_good][is_good])
    #
    # Fit gammavariogram
    #
    model = "gaussian"
    deg = 1
    nbins = 6
    bin_number = False
    lag_cutoff = 0.33
    vlsr_variogram = vlsr_krige.fit(
        model=model,
        deg=deg,
        nbins=nbins,
        bin_number=bin_number,
        lag_cutoff=lag_cutoff,
        plot=True,
    )
    vlsr_variogram.savefig(
        Path(__file__).parent / f"vlsr_variogram_{num_good}good.pdf", bbox_inches="tight",
    )
    vlsr_variogram.show()
    plt.show()
    #
    # Interpolate data
    #
    xlow, xhigh = -8, 12
    ylow, yhigh = -5 - _R0_MODE, 15 - _R0_MODE
    gridx, gridy = np.mgrid[xlow:xhigh:500j, ylow:yhigh:500j]
    interp_pos = np.vstack((gridx.flatten(), gridy.flatten())).T
    vlsr_interp, vlsr_interp_var = vlsr_krige.interp(interp_pos, resample=False)
    # Reshape
    vlsr_interp = vlsr_interp.reshape(gridx.shape)
    vlsr_interp_var = vlsr_interp_var.reshape(gridx.shape)
    vlsr_interp_sd = np.sqrt(vlsr_interp_var)
    #
    # Plot interpolated vlsr differences
    #
    fig, ax = plt.subplots()
    cmap = "viridis"
    extent = (xlow, xhigh, ylow, yhigh)
    #
    norm = mpl.colors.Normalize(vmin=np.min(vlsr_interp), vmax=np.max(vlsr_interp))
    ax.imshow(vlsr_interp.T, origin="lower", extent=extent, norm=norm, cmap=cmap)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, format="%.0f")
    ax.scatter(
        0, -_R0_MODE, marker="X", c="tab:red", s=15, zorder=10
    )  # marker at Galactic centre
    ax.axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax.axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax.set_xlabel("$x$ (kpc)")
    ax.set_ylabel("$y$ (kpc)")
    ax.set_title(
        r"$v_{\scriptscriptstyle\rm LSR} - v_{\scriptscriptstyle\rm LSR\text{, pred}}$"
    )
    cbar.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar.ax.get_yaxis().labelpad = 15
    ax.set_aspect("equal")
    ax.grid(False)
    # Plot actual vlsr
    xdata, ydata, zdata = trans.gal_to_bary(glong, glat, dist)
    xdata, ydata = ydata, -xdata  # rotate 90 deg CW (Sun is on +y-axis)
    ax.scatter(
        xdata[is_good],
        ydata[is_good],
        c=vlsr[is_good] - vlsr_pred[is_good],
        norm=norm,
        cmap=cmap,
        s=10,
        edgecolors="k",
        marker="o",
        label="Good Masers",
    )
    ax.scatter(
        xdata[~is_good],
        ydata[~is_good],
        c=vlsr[~is_good] - vlsr_pred[~is_good],
        norm=norm,
        cmap=cmap,
        s=10,
        edgecolors="k",
        marker="s",
        label="Outlier Masers",
    )
    ax.set_xlim(xlow, xhigh)
    ax.set_ylim(ylow, yhigh)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    filename = f"krige_vlsr_{num_good}good.pdf"
    fig.savefig(
        Path(__file__).parent / filename, format="pdf", dpi=300, bbox_inches="tight",
    )
    plt.show()
    #
    # Plot standard deviations
    #
    fig, ax = plt.subplots()
    cmap = "viridis"
    extent = (xlow, xhigh, ylow, yhigh)
    #
    norm = mpl.colors.Normalize(vmin=np.min(vlsr_interp_sd), vmax=np.max(vlsr_interp_sd))
    ax.imshow(vlsr_interp_sd.T, origin="lower", extent=extent, norm=norm, cmap=cmap)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, format="%.1f")
    ax.scatter(
        0, -_R0_MODE, marker="X", c="tab:red", s=15, zorder=10
    )  # marker at Galactic centre
    ax.axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax.axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax.set_xlabel("$x$ (kpc)")
    ax.set_ylabel("$y$ (kpc)")
    ax.set_title(r"Standard Deviation of $v_{\scriptscriptstyle\rm LSR}$ Differences")
    cbar.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar.ax.get_yaxis().labelpad = 15
    ax.set_aspect("equal")
    ax.grid(False)
    fig.tight_layout()
    filename = f"krige_vlsr_sd_{num_good}good.pdf"
    fig.savefig(
        Path(__file__).parent / filename, format="pdf", dpi=300, bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
    # calc_cov_vlsr(num_samples=10000)
