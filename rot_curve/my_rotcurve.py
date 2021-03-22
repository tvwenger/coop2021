"""
my_rotcurve.py

Plots Persic et al. (1996) universal rotation curve with uncertainty bars
derived via Monte Carlo sampling. Uses CW21 A5 parameters

Isaac Cheng - March 2021
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import dill
import arviz as az
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


def plot_rotcurve_mcerrors(data):
    _NUM_SAMPLES = 10000  # number of MC samples

    # # Mean values from 100 distance, mean Upec/Vpec trace, peak everything file
    # R0_mean, Zsun_mean, roll_mean = 8.17845, 5.0649223, 0.0014527875
    # Usun_mean, Vsun_mean, Wsun_mean = 10.879447, 10.540543, 8.1168785
    # Upec_mean, Vpec_mean = 4.912622, -4.588946
    # a2_mean, a3_mean = 0.96717525, 1.624953
    # Wpec_mean = 0.0  # km/s
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

    kdefile = Path("kd_pkl/cw21_kde.pkl")
    with open(kdefile, "rb") as f:
        kde = dill.load(f)["full"]

    R0, Zsun, Usun, Vsun, Wsun, Upec, Vpec, roll, a2, a3 = kde.resample(_NUM_SAMPLES)

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
    # MC peculiar motions
    radius, azimuth, height, v_radial, v_circ, v_vert = trans.eq_and_gal_to_gcen_cyl(
        ra,
        dec,
        plx,
        glon,
        glat,
        eqmux,
        eqmuy,
        vlsr,
        mc_dists=dist,
        R0=R0[np.newaxis].T,
        Zsun=Zsun[np.newaxis].T,
        roll=roll[np.newaxis].T,
        Usun=Usun[np.newaxis].T,
        Vsun=Vsun[np.newaxis].T,
        Wsun=Wsun[np.newaxis].T,
        Theta0=Theta0[np.newaxis].T,
        use_theano=False,
        return_only_r_and_theta=False,
    )

    # v_circ_pred = (
    #     urc(radius, a2=a2[np.newaxis].T, a3=a3[np.newaxis].T, R0=R0[np.newaxis].T)
    #     + Vpec[np.newaxis].T
    # )  # km/s

    # Peak distance
    (
        radius_peak,
        azimuth_peak,
        height_peak,
        v_radial_peak,
        v_circ_peak,
        v_vert_peak,
    ) = trans.eq_and_gal_to_gcen_cyl(
        ra,
        dec,
        plx,
        glon,
        glat,
        eqmux,
        eqmuy,
        vlsr,
        e_plx=e_plx,
        # R0=R0[np.newaxis].T,
        Zsun=Zsun[np.newaxis].T,
        roll=roll[np.newaxis].T,
        Usun=Usun[np.newaxis].T,
        Vsun=Vsun[np.newaxis].T,
        Wsun=Wsun[np.newaxis].T,
        Theta0=Theta0[np.newaxis].T,
        use_theano=False,
        return_only_r_and_theta=False,
    )
    radius_peak = np.median(radius_peak, axis=0)
    v_circ_peak = np.median(v_circ_peak, axis=0)
    # print(radius_peak[data["gname"].values == "G028.14-00.00"])
    # return None
    #
    # print(data["gname"][v_circ_peak < 0])
    # print(v_circ_peak[v_circ_peak < 0])
    # return None
    #
    # Direct calculation (i.e. dist = 1 / plx)
    #
    # radius_direct = trans.get_gcen_cyl_radius(glon, glat, plx)
    #
    # Calculate errors
    #
    radius_hdi = np.array(
        [az.hdi(radius[:, idx], hdi_prob=0.6827) for idx in range(num_sources)]
    )
    v_circ_hdi = np.array(
        [az.hdi(v_circ[:, idx], hdi_prob=0.6827) for idx in range(num_sources)]
    )
    radius_err = abs(radius_hdi.T - radius_peak).T
    v_circ_err = abs(v_circ_hdi.T - v_circ_peak).T
    # print("Should be -ve then +ve:", (radius_hdi.T - radius_peak).T[0])  # Correct
    #
    print("Now calculating HPD...")
    r_hpd = np.array([calc_hpd(radius[:, idx], "scipy") for idx in range(num_sources)])
    v_circ_hpd = np.array(
        [calc_hpd(v_circ[:, idx], "scipy") for idx in range(num_sources)]
    )
    r_hpd_mode = r_hpd[:, 1]
    r_err_hpd_low = r_hpd_mode - r_hpd[:, 2]
    r_err_hpd_high = r_hpd[:, 3] - r_hpd_mode
    r_err_hpd = np.vstack((r_err_hpd_low, r_err_hpd_high))
    v_circ_hpd_mode = v_circ_hpd[:, 1]
    v_circ_err_hpd_low = v_circ_hpd_mode - v_circ_hpd[:, 2]
    v_circ_err_hpd_high = v_circ_hpd[:, 3] - v_circ_hpd_mode
    v_circ_err_hpd = np.vstack((v_circ_err_hpd_low, v_circ_err_hpd_high))
    # print(v_circ_err_hpd.shape)
    # print("Should both be zero:", sum(r_err_hpd < 0), sum(v_circ_err_hpd < 0))  # Correct
    #
    is_tooclose = data["is_tooclose"].values == 1
    is_outlier = data["is_outlier"].values == 1
    is_good = (~is_tooclose) & (~is_outlier)
    print("num R < 4 kpc:", sum(is_tooclose))
    print("num outliers:", sum(is_outlier))
    print("num peak derived R < 4 kpc using R0=8.178:", sum(radius_peak < 4))
    print("num HPD derived R < 4 kpc using R0=8.178:", sum(r_hpd_mode < 4))
    print(
        "Non-outlier that now has mode distance < 4 kpc:",
        data["gname"][(r_hpd_mode < 4) & (~is_tooclose)].values,
        r_hpd_mode[(r_hpd_mode < 4) & (~is_tooclose)],
    )
    if np.any(v_circ_hpd_mode < 0):
        print("Source(s) with v_circ_hpd_mode < 0:")
        print(
            data["gname"][v_circ_hpd_mode < 0].values,
            data["alias"][v_circ_hpd_mode < 0].values,
            v_circ_hpd_mode[v_circ_hpd_mode < 0],
        )

    #
    # ------
    # Plot at peak distance, Vpec @ peak dist, and errors using HDI
    # ------
    #
    fig, ax = plt.subplots()
    markersize = 4
    markeredgewidth = 0.5
    # Plot curve
    r_curve = np.linspace(0, 17, 101)
    v_curve = (
        urc(r_curve, a2=a2[np.newaxis].T, a3=a3[np.newaxis].T, R0=R0[np.newaxis].T)
        + Vpec_mode
    )
    v_curve = np.median(v_curve, axis=0)
    ax.plot(r_curve, v_curve, "g--")
    # Sources within 4 kpc of GC
    eb_tooclose = ax.errorbar(
        x=radius_peak[is_tooclose],
        y=v_circ_peak[is_tooclose],
        xerr=radius_err[is_tooclose].T,
        yerr=v_circ_err[is_tooclose].T,
        fmt="r.",
        mfc="none",
        markersize=markersize,
        markeredgewidth=markeredgewidth,
        elinewidth=0.5,
        ecolor="r",
    )
    # MCMC outliers
    eb_outlier = ax.errorbar(
        x=radius_peak[is_outlier],
        y=v_circ_peak[is_outlier],
        xerr=radius_err[is_outlier].T,
        yerr=v_circ_err[is_outlier].T,
        fmt="b.",
        mfc="none",
        markersize=markersize,
        markeredgewidth=markeredgewidth,
        elinewidth=0.5,
        ecolor="b",
    )
    # Good sources
    eb_good = ax.errorbar(
        x=radius_peak[is_good],
        y=v_circ_peak[is_good],
        xerr=radius_err[is_good].T,
        yerr=v_circ_err[is_good].T,
        fmt="k.",
        # mfc="none",
        markersize=markersize,
        markeredgewidth=markeredgewidth,
        elinewidth=0.5,
        ecolor="k",
        zorder=100,
    )
    # Dashed errorbars
    for eb in [eb_tooclose, eb_outlier, eb_good]:
        eb[-1][0].set_linestyle("--")  # 1st errorbar (e.g. x-errorbar) linestyle
        eb[-1][1].set_linestyle("--")  # 2nd errorbar (e.g. y-errorbar) linestyle
    ax.set_xlabel("Galactocentric Radius (kpc)")
    ax.set_ylabel("$\Theta$ (km s$^{-1}$)")
    ax.set_xlim(0, 16)
    ax.set_ylim(-25, 300)
    ax.grid(False)
    fig.savefig(
        Path(__file__).parent / "my_rotcurve_peakEverything_HDIpeak.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()
    #
    # ------
    # Plot at distance mode, Vpec @ dist mode, and errors using HPD
    # ------
    #
    fig, ax = plt.subplots()
    markersize = 4
    markeredgewidth = 0.5
    # Plot curve
    r_curve = np.linspace(0, 17, 101)
    v_curve = (
        urc(r_curve, a2=a2[np.newaxis].T, a3=a3[np.newaxis].T, R0=R0[np.newaxis].T)
        + Vpec_mode
    )
    v_curve = np.median(v_curve, axis=0)
    ax.plot(r_curve, v_curve, "g--")
    # Sources within 4 kpc of GC
    eb_tooclose = ax.errorbar(
        x=r_hpd_mode[is_tooclose],
        y=v_circ_hpd_mode[is_tooclose],
        xerr=r_err_hpd[:, is_tooclose],
        yerr=v_circ_err_hpd[:, is_tooclose],
        fmt="r.",
        mfc="none",
        markersize=markersize,
        markeredgewidth=markeredgewidth,
        elinewidth=0.5,
        ecolor="r",
    )
    # MCMC outliers
    eb_outlier = ax.errorbar(
        x=r_hpd_mode[is_outlier],
        y=v_circ_hpd_mode[is_outlier],
        xerr=r_err_hpd[:, is_outlier],
        yerr=v_circ_err_hpd[:, is_outlier],
        fmt="b.",
        mfc="none",
        markersize=markersize,
        markeredgewidth=markeredgewidth,
        elinewidth=0.5,
        ecolor="b",
    )
    # Good sources
    eb_good = ax.errorbar(
        x=r_hpd_mode[is_good],
        y=v_circ_hpd_mode[is_good],
        xerr=r_err_hpd[:, is_good],
        yerr=v_circ_err_hpd[:, is_good],
        fmt="k.",
        # mfc="none",
        markersize=markersize,
        markeredgewidth=markeredgewidth,
        elinewidth=0.5,
        ecolor="k",
        zorder=100,
    )
    # Dashed errorbars
    for eb in [eb_tooclose, eb_outlier, eb_good]:
        eb[-1][0].set_linestyle("--")  # 1st errorbar (e.g. x-errorbar) linestyle
        eb[-1][1].set_linestyle("--")  # 2nd errorbar (e.g. y-errorbar) linestyle
    ax.set_xlabel("Galactocentric Radius (kpc)")
    ax.set_ylabel("$\Theta$ (km s$^{-1}$)")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 300)
    ax.grid(False)
    fig.savefig(
        Path(__file__).parent / "my_rotcurve_peakEverything_HPDmode.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(
        Path(__file__).parent.parent
        / Path(
            "pec_motions/csvfiles/100dist_meanUpecVpec_cauchyOutlierRejection_peakEverything.csv"
        )
    )
    plot_rotcurve_mcerrors(df)
