"""
generate_mc_pec_motions.py

Creates .csv file with peculiar motions of all sources using MC sampling.

Isaac Cheng - March 2021
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import dill
from scipy.stats.kde import gaussian_kde
from pyqt_fit import kde as pyqt_kde
from pyqt_fit import kde_methods

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans
from universal_rotcurve import urc


def calc_hpd(samples, kdetype, alpha=0.683, pdf_bins=1000):
    """
    From Dr. Trey V. Wenger's kd_utils.py

    Fit a kernel density estimator (KDE) to the posterior given
    by a collection of samples. Return the mode (posterior peak)
    and the highest posterior density (HPD) determined by the minimum
    width Bayesian credible interval (BCI) containing a fraction of
    the posterior samples. The posterior should be well described by a
    single-modal distribution.

    Parameters:
      samples :: 1-D array of scalars
        The samples being fit with a KDE

      kdetype :: string
        Which KDE method to use
          'pyqt' uses pyqt_fit with boundary at 0
          'scipy' uses gaussian_kde with no boundary

      alpha :: scalar (optional)
        The fraction of samples included in the BCI.

      pdf_bins :: integer (optional)
        Number of bins used in calculating the PDF

    Returns: kde, mode, lower, upper
      kde :: scipy.gaussian_kde or pyqt_fit.1DKDE object
        The KDE calculated for this kinematic distance

      mode :: scalar
        The mode of the posterior

      lower :: scalar
        The lower bound of the BCI

      upper :: scalar
        The upper bound of the BCI
    """
    # check inputs
    if (alpha <= 0.0) or (alpha >= 1.0):
        raise ValueError("alpha should be between 0 and 1.")
    #
    # Fit KDE
    #
    nans = np.isnan(samples)
    if np.sum(~nans) < 2:
        # skip if fewer than two non-nans
        return (None, np.nan, np.nan, np.nan)
    try:
        if kdetype == "scipy":
            kde = gaussian_kde(samples[~nans])
        elif kdetype == "pyqt":
            kde = pyqt_kde.KDE1D(
                samples[~nans], lower=0, method=kde_methods.linear_combination
            )
        else:
            raise ValueError("Invalid KDE method: {0}".format(kdetype))
    except np.linalg.LinAlgError:
        # catch singular matricies (i.e. all values are the same)
        return (None, np.nan, np.nan, np.nan)
    #
    # Compute PDF
    #
    xdata = np.linspace(np.nanmin(samples), np.nanmax(samples), pdf_bins)
    pdf = kde(xdata)
    #
    # Get the location of the mode
    #
    mode = xdata[np.argmax(pdf)]
    if np.isnan(mode):
        return (None, np.nan, np.nan, np.nan)
    #
    # Reverse sort the PDF and xdata and find the BCI
    #
    sort_pdf = sorted(zip(xdata, pdf / np.sum(pdf)), key=lambda x: x[1], reverse=True)
    cum_prob = 0.0
    bci_xdata = np.empty(len(xdata), dtype=float) * np.nan
    for i, dat in enumerate(sort_pdf):
        cum_prob += dat[1]
        bci_xdata[i] = dat[0]
        if cum_prob >= alpha:
            break
    lower = np.nanmin(bci_xdata)
    upper = np.nanmax(bci_xdata)
    return kde, mode, lower, upper


def dist_prob(dist, plx, e_plx):
    """
    Evaluate the probability density of a given distance for a
    given observed parallax and parallax uncertainty.
​
    Inputs:
      dist :: Scalar (kpc)
        Distance at which to evaluate probability density
      plx, e_plx :: Scalar (mas)
        Observed parallax and (absolute value of) uncertainty
​
    Returns: prob
      prob :: scalar (kpc-1)
        Probability density at dist
    """

    mean_dist = 1 / plx
    exp_part = (
        -0.5
        * (dist - mean_dist)
        * (dist - mean_dist)
        / (dist * dist * mean_dist * mean_dist * e_plx * e_plx)
    )
    coeff = 1 / (dist * dist * e_plx * np.sqrt(2 * np.pi))

    return coeff * np.exp(exp_part)


def generate_dists(plx, e_plx, num_samples):
    """
    Generates a specified number of random distance samples
    given a parallax and its uncertainty.
    Function taken verbatim from Dr. Trey Wenger (February 2021)

    Inputs:
      plx :: Array of scalars (mas)
        Parallaxes
      e_plx :: Array of scalars (mas)
        Uncertainty in the parallaxes. Strictly non-negative
      num_samples :: Scalar
        Number of distance samples to generate per parallax

    Returns: dists
      dists :: Array of scalars (kpc)
        Galactic-frame distances sampled from the asymmetric parallax-to-distance pdf
    """

    # possible distance values (1 pc resolution)
    dist_values = np.arange(0.001, 50.001, 0.001)

    # distance probability densities
    # (shape: len(dist_values) = 50000, num_sources)
    dist_probs = dist_prob(dist_values[..., None], plx, e_plx)

    # normalize probabilities to area=1
    dist_probs = dist_probs / dist_probs.sum(axis=0)

    # sample distance
    # (shape: num_samples, num_sources)
    print("Number of distance samples:", num_samples)
    dists = np.array(
        [
            np.random.choice(dist_values, p=dist_probs[:, i], size=num_samples)
            for i in range(len(plx))
        ]
    ).T

    return dists


def mc_plx_upecvpec(data):
    _NUM_SAMPLES = 10000  # number of MC samples

    kdefile = Path("kd_pkl/cw21_kde.pkl")
    with open(kdefile, "rb") as f:
        kde = dill.load(f)["full"]

    R0, Zsun, Usun, Vsun, Wsun, Upec, Vpec, roll, a2, a3 = kde.resample(_NUM_SAMPLES)
    Upec = Vpec = 0

    # === Get data ===
    ra = data["ra"].values  # deg
    dec = data["dec"].values  # deg
    glon = data["glong"].values  # deg
    glat = data["glat"].values  # deg
    plx = data["plx"].values  # mas
    e_plx = data["e_plx"].values  # mas
    eqmux = data["mux"].values  # mas/yr (equatorial frame)
    e_eqmux = data["e_mux"].values  # mas/yr (equatorial frame)
    eqmuy = data["muy"].values  # mas/y (equatorial frame)
    e_eqmuy = data["e_muy"].values  # mas/y (equatorial frame)
    vlsr = data["vlsr"].values  # km/s
    e_vlsr = data["e_vlsr"].values  # km/s
    num_sources = len(ra)
    print("Num sources:", num_sources)

    # MC sample data
    dist = generate_dists(plx, e_plx, _NUM_SAMPLES)
    eqmux = np.random.normal(loc=eqmux, scale=e_eqmux, size=(_NUM_SAMPLES, len(ra)))
    eqmuy = np.random.normal(loc=eqmuy, scale=e_eqmuy, size=(_NUM_SAMPLES, len(ra)))
    vlsr = np.random.normal(loc=vlsr, scale=e_vlsr, size=(_NUM_SAMPLES, len(ra)))

    Theta0 = urc(R0, a2=a2, a3=a3, R0=R0)  # km/s, LSR circular rotation speed
    # radius, azimuth, height, v_radial, v_circ, v_vert = trans.eq_and_gal_to_gcen_cyl(
    #     ra,
    #     dec,
    #     plx,
    #     glon,
    #     glat,
    #     eqmux,
    #     eqmuy,
    #     vlsr,
    #     mc_dists=dist,
    #     R0=R0[np.newaxis].T,
    #     Zsun=Zsun[np.newaxis].T,
    #     roll=roll[np.newaxis].T,
    #     Usun=Usun[np.newaxis].T,
    #     Vsun=Vsun[np.newaxis].T,
    #     Wsun=Wsun[np.newaxis].T,
    #     Theta0=Theta0[np.newaxis].T,
    #     use_theano=False,
    #     return_only_r_and_theta=False,
    # )
    #
    # Transform from galactic to galactocentric Cartesian coordinates
    bary_x, bary_y, bary_z = trans.gal_to_bary(glon, glat, dist)
    gcen_x, gcen_y, gcen_z = trans.bary_to_gcen(
      bary_x, bary_y, bary_z, R0=R0[np.newaxis].T,
      Zsun=Zsun[np.newaxis].T, roll=roll[np.newaxis].T)
    # LSR velocity to barycentric velocity
    vbary = trans.vlsr_to_vbary(vlsr, glon, glat)
    # Transform equatorial proper motions to galactic proper motions
    gmul, gmub = trans.eq_to_gal(
        ra, dec, eqmux, eqmuy, return_pos=False, use_theano=False)
    # Transform galactic proper motions to barycentric Cartesian velocities
    U, V, W = trans.gal_to_bary_vel(glon, glat, dist, gmul, gmub, vbary)
    # Transform barycentric Cartesian velocities to galactocentric Cartesian velocities
    gcen_vx, gcen_vy, gcen_vz = trans.bary_to_gcen_vel(
      U, V, W,
      R0=R0[np.newaxis].T, Zsun=Zsun[np.newaxis].T, roll=roll[np.newaxis].T,
      Usun=Usun[np.newaxis].T, Vsun=Vsun[np.newaxis].T, Wsun=Wsun[np.newaxis].T,
      Theta0=Theta0[np.newaxis].T)
    radius, azimuth, height, v_radial, v_circ, v_vert = trans.gcen_cart_to_gcen_cyl(
        gcen_x, gcen_y, gcen_z, gcen_vx, gcen_vy, gcen_vz, use_theano=False)
    #
    #
    v_circ_pred = (
        urc(radius, a2=a2[np.newaxis].T, a3=a3[np.newaxis].T, R0=R0[np.newaxis].T) + Vpec
    )  # km/s
    v_circ_res = v_circ - v_circ_pred
    print("median Vpec & Upec:", np.median(v_circ_res), np.median(-v_radial))
    print("mean Vpec & Upec:", np.mean(v_circ_res), np.mean(-v_radial))
    #
    # Calculate mode and errors
    #
    print("Now calculating HPD...")
    r_hpd = np.array([calc_hpd(radius[:, idx], "scipy") for idx in range(num_sources)])
    az_hpd = np.array([calc_hpd(azimuth[:, idx], "scipy") for idx in range(num_sources)])
    Upec_hpd = np.array(
        [calc_hpd(-v_radial[:, idx], "scipy") for idx in range(num_sources)]
    )
    Vpec_hpd = np.array(
        [calc_hpd(v_circ_res[:, idx], "scipy") for idx in range(num_sources)]
    )
    Wpec_hpd = np.array(
        [calc_hpd(v_vert[:, idx], "scipy") for idx in range(num_sources)]
    )
    r_hpd_mode, r_hpd_low, r_hpd_high = r_hpd[:, 1], r_hpd[:, 2], r_hpd[:, 3]
    r_err_hpd_low = r_hpd_mode - r_hpd_low
    r_err_hpd_high = r_hpd_high - r_hpd_mode
    r_err_hpd = np.vstack((r_err_hpd_low, r_err_hpd_high))
    az_hpd_mode, az_hpd_low, az_hpd_high = az_hpd[:, 1], az_hpd[:, 2], az_hpd[:, 3]
    Upec_hpd_mode, Upec_hpd_low, Upec_hpd_high = Upec_hpd[:, 1], Upec_hpd[:, 2], Upec_hpd[:, 3]
    Upec_err_hpd_low = Upec_hpd_mode - Upec_hpd_low
    Upec_err_hpd_high = Upec_hpd_high - Upec_hpd_mode
    Upec_err_hpd = np.vstack((Upec_err_hpd_low, Upec_err_hpd_high))
    Vpec_hpd_mode, Vpec_hpd_low, Vpec_hpd_high = Vpec_hpd[:, 1], Vpec_hpd[:, 2], Vpec_hpd[:, 3]
    Vpec_err_hpd_low = Vpec_hpd_mode - Vpec_hpd_low
    Vpec_err_hpd_high = Vpec_hpd_high - Vpec_hpd_mode
    Vpec_err_hpd = np.vstack((Vpec_err_hpd_low, Vpec_err_hpd_high))
    Wpec_hpd_mode, Wpec_hpd_low, Wpec_hpd_high = Wpec_hpd[:, 1], Wpec_hpd[:, 2], Wpec_hpd[:, 3]
    print("Num sources with mode R < 0:", sum(r_hpd_mode < 0))
    if np.any(r_err_hpd) < 0 or np.any(Upec_err_hpd) < 0 or np.any(Vpec_err_hpd) < 0:
        raise ValueError("Negative HPD error found!")
    #
    # Galactocentric Cartesian positions
    #
    # x, y, z, = trans.gcen_cyl_to_gcen_cart(radius, azimuth, height)
    # Rotate 90 deg CW to Reid convention
    x, y = gcen_y, -gcen_x
    vx, vy = gcen_vy, -gcen_vx
    z = gcen_z
    vz = gcen_vz
    # Calculate mode and errors
    x_hpd = np.array([calc_hpd(x[:, idx], "scipy") for idx in range(num_sources)])
    y_hpd = np.array([calc_hpd(y[:, idx], "scipy") for idx in range(num_sources)])
    z_hpd = np.array([calc_hpd(z[:, idx], "scipy") for idx in range(num_sources)])
    vx_hpd = np.array([calc_hpd(vx[:, idx], "scipy") for idx in range(num_sources)])
    vy_hpd = np.array([calc_hpd(vy[:, idx], "scipy") for idx in range(num_sources)])
    vz_hpd = np.array([calc_hpd(vz[:, idx], "scipy") for idx in range(num_sources)])
    x_hpd_mode, x_hpd_low, x_hpd_high = x_hpd[:, 1], x_hpd[:, 2], x_hpd[:, 3]
    y_hpd_mode, y_hpd_low, y_hpd_high = y_hpd[:, 1], y_hpd[:, 2], y_hpd[:, 3]
    z_hpd_mode, z_hpd_low, z_hpd_high = z_hpd[:, 1], z_hpd[:, 2], z_hpd[:, 3]
    vx_hpd_mode, vx_hpd_low, vx_hpd_high = vx_hpd[:, 1], vx_hpd[:, 2], vx_hpd[:, 3]
    vy_hpd_mode, vy_hpd_low, vy_hpd_high = vy_hpd[:, 1], vy_hpd[:, 2], vy_hpd[:, 3]
    vz_hpd_mode, vz_hpd_low, vz_hpd_high = vz_hpd[:, 1], vz_hpd[:, 2], vz_hpd[:, 3]
    #
    # Plot posterior distributions of one source
    #
    var_lst = [dist, x, y, z, vx, vy, vz, radius, azimuth, -v_radial, v_circ_res, v_vert]
    name_lst = ["dist", "x", "y", "z", "vx", "vy", "vz", "R", "azimuth", "Upec", "Vpec", "Wpec"]
    # var_lst = [dist, eqmux, eqmuy, vlsr, x, y, z, radius, -v_radial, v_circ_res, v_vert]
    # name_lst = ["dist", "mux", "muy", "vlsr", "x", "y", "z", "R", "Upec", "Vpec", "Wpec"]
    fig, axes = plt.subplots(len(var_lst), figsize=plt.figaspect(5 / 8 * len(var_lst)))
    bins = 50
    source_to_plot = 7
    source_name = data["gname"][source_to_plot]
    print(source_name)
    for ax, var, name in zip(axes, var_lst, name_lst):
        mode = calc_hpd(var[:, source_to_plot], "scipy")[1]
        ax.hist(var[:, source_to_plot], bins=bins)
        ax.axvline(np.mean(var[:, source_to_plot]), color="k", label="mean")
        ax.axvline(np.median(var[:, source_to_plot]), color="g", label="median")
        ax.axvline(mode, color="deeppink", label="mode")
        ax.set_title(name)
        ax.legend(fontsize=9)
        print(
            name,
            np.mean(var[:, source_to_plot]),
            np.median(var[:, source_to_plot]),
            mode,
            np.std(var[:, source_to_plot]),
        )
    fig.tight_layout()
    fig.savefig(
        Path(__file__).parent / f"{source_name}_distributions.pdf", bbox_inches="tight"
    )
    plt.show()
    #
    # Plotting median distances and median peculiar motions
    #
    avg = np.median
    fig, ax = plt.subplots(2, figsize=plt.figaspect(1))
    ebUpec = ax[0].errorbar(
        x=avg(radius, axis=0),
        y=avg(-v_radial, axis=0),
        xerr=np.std(radius, axis=0),
        yerr=np.std(-v_radial, axis=0),
        fmt="k.",
        markersize=2,
        elinewidth=0.5,
        ecolor="r",
    )
    ebVpec = ax[1].errorbar(
        x=avg(radius, axis=0),
        y=avg(v_circ_res, axis=0),
        xerr=np.std(radius, axis=0),
        yerr=np.std(v_circ_res, axis=0),
        fmt="k.",
        markersize=2,
        elinewidth=0.5,
        ecolor="r",
    )
    # ax[0].set_xlabel("Galactocentric Radius (kpc)")
    ax[0].set_ylabel("$\overline{U_s}$ (km s$^{-1}$)")
    ax[1].set_xlabel("Galactocentric Radius (kpc)")
    ax[1].set_ylabel("$\overline{V_s}$ (km s$^{-1}$)")
    # Dashed errorbars
    # [-1] denotes the LineCollection objects of the errorbar lines
    ebUpec[-1][0].set_linestyle("--")  # x-errorbar linestyle
    ebUpec[-1][1].set_linestyle("--")  # y-errorbar linestyle
    ebVpec[-1][0].set_linestyle("--")
    ebVpec[-1][1].set_linestyle("--")
    ax[0].set_xlim(left=0)
    ax[1].set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(
        Path(__file__).parent / "Upec_Vpec_errors_STDmedian.pdf", bbox_inches="tight"
    )
    plt.show()
    #
    # Plotting distance modes and peculiar motions modes
    #
    fig, ax = plt.subplots(2, figsize=plt.figaspect(1))
    ebUpec = ax[0].errorbar(
        x=r_hpd_mode,
        y=Upec_hpd_mode,
        xerr=r_err_hpd,
        yerr=Upec_err_hpd,
        fmt="k.",
        markersize=2,
        elinewidth=0.5,
        ecolor="r",
    )
    ebVpec = ax[1].errorbar(
        x=r_hpd_mode,
        y=Vpec_hpd_mode,
        xerr=r_err_hpd,
        yerr=Vpec_err_hpd,
        fmt="k.",
        markersize=2,
        elinewidth=0.5,
        ecolor="r",
    )
    # ax[0].set_xlabel("Galactocentric Radius (kpc)")
    ax[0].set_ylabel("$\overline{U_s}$ (km s$^{-1}$)")
    ax[1].set_xlabel("Galactocentric Radius (kpc)")
    ax[1].set_ylabel("$\overline{V_s}$ (km s$^{-1}$)")
    # Dashed errorbars
    # [-1] denotes the LineCollection objects of the errorbar lines
    ebUpec[-1][0].set_linestyle("--")  # x-errorbar linestyle
    ebUpec[-1][1].set_linestyle("--")  # y-errorbar linestyle
    ebVpec[-1][0].set_linestyle("--")
    ebVpec[-1][1].set_linestyle("--")
    ax[0].set_xlim(left=0)
    ax[1].set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(
        Path(__file__).parent / "Upec_Vpec_errors_HPDmode.pdf", bbox_inches="tight"
    )
    plt.show()

    x_halfhpd = 0.5 * (x_hpd_high - x_hpd_low)
    y_halfhpd = 0.5 * (y_hpd_high - y_hpd_low)
    z_halfhpd = 0.5 * (z_hpd_high - z_hpd_low)
    r_halfhpd = 0.5 * (r_hpd_high - r_hpd_low)
    az_halfhpd = 0.5 * (az_hpd_high - az_hpd_low)
    Upec_halfhpd = 0.5 * (Upec_hpd_high - Upec_hpd_low)
    Vpec_halfhpd = 0.5 * (Vpec_hpd_high - Vpec_hpd_low)
    Wpec_halfhpd = 0.5 * (Wpec_hpd_high - Wpec_hpd_low)
    vx_halfhpd = 0.5 * (vx_hpd_high - vx_hpd_low)
    vy_halfhpd = 0.5 * (vy_hpd_high - vy_hpd_low)
    vz_halfhpd = 0.5 * (vz_hpd_high - vz_hpd_low)
    halfhpd_lst = [x_halfhpd, y_halfhpd, z_halfhpd, r_halfhpd, az_halfhpd, Upec_halfhpd,
                   Vpec_halfhpd, Wpec_halfhpd, vx_halfhpd, vy_halfhpd, vz_halfhpd]
    halfhpd_names = ["x_halfhpd", "y_halfhpd", "z_halfhpd", "r_halfhpd", "az_halfhpd", "Upec_halfhpd",
                     "Vpec_halfhpd", "Wpec_halfhpd", "vx_halfhpd", "vy_halfhpd", "vz_halfhpd"]
    for name, halfhpd in zip(halfhpd_names, halfhpd_lst):
        print("Negative halfhpd found in", name) if np.any(halfhpd) < 0 else None

    avg = np.median
    df = pd.DataFrame(
        {
            "gname": data["gname"],
            "alias": data["alias"],
            "ra": data["ra"],
            "dec": data["dec"],
            "glong": data["glong"],
            "glat": data["glat"],
            "plx": data["plx"],
            "e_plx": data["e_plx"],
            "mux_med": avg(eqmux, axis=0),
            "mux_std": np.std(eqmux, axis=0),
            "muy_med": avg(eqmuy, axis=0),
            "muy_std": np.std(eqmuy, axis=0),
            "vlsr_med": avg(vlsr, axis=0),
            "vlsr_std": np.std(vlsr, axis=0),
            "x_mode": x_hpd_mode,
            "x_halfhpd": x_halfhpd,
            "x_hpdlow": x_hpd_low,
            "x_hpdhigh": x_hpd_high,
            "y_mode": y_hpd_mode,
            "y_halfhpd": y_halfhpd,
            "y_hpdlow": y_hpd_low,
            "y_hpdhigh": y_hpd_high,
            "z_mode": z_hpd_mode,
            "z_halfhpd": z_halfhpd,
            "z_hpdlow": z_hpd_low,
            "z_hpdhigh": z_hpd_high,
            "vx_mode": vx_hpd_mode,
            "vx_halfhpd": vx_halfhpd,
            "vx_hpdlow": vx_hpd_low,
            "vx_hpdhigh": vx_hpd_high,
            "vy_mode": vy_hpd_mode,
            "vy_halfhpd": vy_halfhpd,
            "vy_hpdlow": vy_hpd_low,
            "vy_hpdhigh": vy_hpd_high,
            "vz_mode": vz_hpd_mode,
            "vz_halfhpd": vz_halfhpd,
            "vz_hpdlow": vz_hpd_low,
            "vz_hpdhigh": vz_hpd_high,
            "R_mode": r_hpd_mode,
            "R_halfhpd": r_halfhpd,
            "R_hpdlow": r_hpd_low,
            "R_hpdhigh": r_hpd_high,
            "az_mode": az_hpd_mode,
            "az_halfhpd": az_halfhpd,
            "az_hpdlow": az_hpd_low,
            "az_hpdhigh": az_hpd_high,
            "Upec_mode": Upec_hpd_mode,
            "Upec_halfhpd": Upec_halfhpd,
            "Upec_hpdlow": Upec_hpd_low,
            "Upec_hpdhigh": Upec_hpd_high,
            "Vpec_mode": Vpec_hpd_mode,
            "Vpec_halfhpd": Vpec_halfhpd,
            "Vpec_hpdlow": Vpec_hpd_low,
            "Vpec_hpdhigh":Vpec_hpd_high,
            "Wpec_mode": Wpec_hpd_mode,
            "Wpec_halfhpd": Wpec_halfhpd,
            "Wpec_hpdlow": Wpec_hpd_low,
            "Wpec_hpdhigh": Wpec_hpd_high,
            "is_tooclose": data["is_tooclose"],
            "is_outlier": data["is_outlier"],
        }
    )
    df.to_csv(
        path_or_buf=Path(__file__).parent / Path("alldata_HPDmode_NEW.csv"),
        sep=",",
        index=False,
        header=True,
    )
    print("Saved .csv file!")


def main():
    # tracefile = Path(
    #     "/home/chengi/Documents/coop2021/bayesian_mcmc_rot_curve/"
    #     "mcmc_outfile_A5_100dist_5.pkl"
    # )
    # datafile = Path(
    #     "/home/chengi/Documents/coop2021/pec_motions/100dist_meanUpecVpec.csv"
    # )
    # datafile = Path(__file__).parent / Path("100dist_meanUpecVpec_cauchyOutlierRejection.csv")
    datafile = Path(__file__).parent / Path(
        "csvfiles/100dist_meanUpecVpec_cauchyOutlierRejection_peakEverything.csv"
    )
    data100plx = pd.read_csv(datafile)

    # Only choose sources that have R > 4 kpc
    # data100plx = data100plx[data100plx["is_tooclose"].values == 0]
    mc_plx_upecvpec(data100plx)


if __name__ == "__main__":
    main()
