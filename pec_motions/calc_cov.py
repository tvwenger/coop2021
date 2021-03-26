"""
calc_cov.py

Calculates the Pearson product-moment correlation coefficient of data
as well as the covariance between two sources using data from .csv file

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
from scipy.stats import pearsonr, spearmanr

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


def calc_cc():
    _NUM_SAMPLES = 10000  # number of MC samples

    kdefile = Path("kd_pkl/cw21_kde.pkl")
    with open(kdefile, "rb") as f:
        kde = dill.load(f)["full"]

    R0, Zsun, Usun, Vsun, Wsun, Upec, Vpec, roll, a2, a3 = kde.resample(_NUM_SAMPLES)
    Upec = Vpec = 0

    # === Get data ===
    datafile = Path(__file__).parent / Path(
        "csvfiles/100dist_meanUpecVpec_cauchyOutlierRejection_peakEverything.csv"
    )
    data = pd.read_csv(datafile)
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
    v_circ_pred = (
        urc(radius, a2=a2[np.newaxis].T, a3=a3[np.newaxis].T, R0=R0[np.newaxis].T) + Vpec
    )  # km/s
    v_circ_res = v_circ - v_circ_pred
    cos_az = np.cos(np.deg2rad(azimuth))
    sin_az = np.sin(np.deg2rad(azimuth))
    vx_res = v_radial * -cos_az + v_circ_res * sin_az  # km/s
    vy_res = v_radial * sin_az + v_circ_res * cos_az  # km/s
    # Rotate 90 deg CW
    vx, vy = gcen_vy, -gcen_vx
    vx_res, vy_res = vy_res, -vx_res
    # v_radial and v_circ_res shapes: (_NUM_SAMPLES, num_sources)
    #
    # Correlation coefficient b/w Upecs
    #
    Upec = -v_radial
    Vpec = v_circ_res
    r_Upec = np.ones((num_sources, num_sources), float)
    cov_Upec = np.ones((num_sources, num_sources), float)
    for i in range(num_sources):
        for j in range(num_sources):
            Upec_meani = np.mean(Upec[:, i])
            Upec_meanj = np.mean(Upec[:, j])
            Upec_diffi = Upec[:, i] - Upec_meani
            Upec_diffj = Upec[:, j] - Upec_meanj
            r_num = np.sum(Upec_diffi * Upec_diffj)
            r_den = np.sqrt(np.sum(Upec_diffi ** 2)) * np.sqrt(np.sum(Upec_diffj ** 2))
            # Fringe cases
            r = r_num / r_den
            r = 1 if r > 1 else r
            r = -1 if r < -1 else r
            # Populate array
            r_Upec[i, j] = r
            cov_Upec[i, j] = r_num / _NUM_SAMPLES
    print(np.max(r_Upec[r_Upec < 0.999999999999999]))
    print(np.min(r_Upec[r_Upec > -0.999999999999999]))
    #
    # Correlation coefficient b/w Vpecs
    #
    r_Vpec = np.ones((num_sources, num_sources), float)
    cov_Vpec = np.ones((num_sources, num_sources), float)
    for i in range(num_sources):
        for j in range(num_sources):
            Vpec_meani = np.mean(Vpec[:, i])
            Vpec_meanj = np.mean(Vpec[:, j])
            Vpec_diffi = Vpec[:, i] - Vpec_meani
            Vpec_diffj = Vpec[:, j] - Vpec_meanj
            r_num = np.sum(Vpec_diffi * Vpec_diffj)
            r_den = np.sqrt(np.sum(Vpec_diffi ** 2)) * np.sqrt(np.sum(Vpec_diffj ** 2))
            # Fringe cases
            r = r_num / r_den
            r = 1 if r > 1 else r
            r = -1 if r < -1 else r
            # Populate array
            r_Vpec[i, j] = r
            cov_Vpec[i, j] = r_num / _NUM_SAMPLES
    print(np.max(r_Vpec[r_Vpec < 0.999999999999999]))
    print(np.min(r_Vpec[r_Vpec > -0.999999999999999]))
    #
    # Correlation coefficient b/w vx
    #
    r_vx = np.ones((num_sources, num_sources), float)
    cov_vx = np.ones((num_sources, num_sources), float)
    for i in range(num_sources):
        for j in range(num_sources):
            vx_meani = np.mean(vx[:, i])
            vx_meanj = np.mean(vx[:, j])
            vx_diffi = vx[:, i] - vx_meani
            vx_diffj = vx[:, j] - vx_meanj
            r_num = np.sum(vx_diffi * vx_diffj)
            r_den = np.sqrt(np.sum(vx_diffi ** 2 )) * np.sqrt(np.sum(vx_diffj ** 2))
            # Fringe cases
            r = r_num / r_den
            r = 1 if r > 1 else r
            r = -1 if r < -1 else r
            # Populate array
            r_vx[i, j] = r
            cov_vx[i, j] = r_num / _NUM_SAMPLES
    print(np.max(r_vx[r_vx < 0.999999999999999]))
    print(np.min(r_vx[r_vx > -0.999999999999999]))
    #
    # Correlation coefficient b/w vy
    #
    r_vy = np.ones((num_sources, num_sources), float)
    cov_vy = np.ones((num_sources, num_sources), float)
    for i in range(num_sources):
        for j in range(num_sources):
            vy_meani = np.mean(vy[:, i])
            vy_meanj = np.mean(vy[:, j])
            vy_diffi = vy[:, i] - vy_meani
            vy_diffj = vy[:, j] - vy_meanj
            r_num = np.sum(vy_diffi * vy_diffj)
            r_den = np.sqrt(np.sum(vy_diffi ** 2 )) * np.sqrt(np.sum(vy_diffj ** 2))
            # Fringe cases
            r = r_num / r_den
            r = 1 if r > 1 else r
            r = -1 if r < -1 else r
            # Populate array
            r_vy[i, j] = r
            cov_vy[i, j] = r_num / _NUM_SAMPLES
    # print(np.max(r_vy[r_vy < 0.999999999999999]))
    # print(np.min(r_vy[r_vy > -0.999999999999999]))
    print(np.max(r_vy[r_vy < 0.99]))
    print(np.min(r_vy[r_vy > -0.99]))
    #
    # Correlation coefficient b/w vx_res
    #
    r_vx_res = np.ones((num_sources, num_sources), float)
    cov_vx_res = np.ones((num_sources, num_sources), float)
    for i in range(num_sources):
        for j in range(num_sources):
            vx_res_meani = np.mean(vx_res[:, i])
            vx_res_meanj = np.mean(vx_res[:, j])
            vx_res_diffi = vx_res[:, i] - vx_res_meani
            vx_res_diffj = vx_res[:, j] - vx_res_meanj
            r_num = np.sum(vx_res_diffi * vx_res_diffj)
            r_den = np.sqrt(np.sum(vx_res_diffi ** 2 )) * np.sqrt(np.sum(vx_res_diffj ** 2))
            # Fringe cases
            r = r_num / r_den
            r = 1 if r > 1 else r
            r = -1 if r < -1 else r
            # Populate array
            r_vx_res[i, j] = r
            cov_vx_res[i, j] = r_num / _NUM_SAMPLES
    print(np.max(r_vx_res[r_vx_res < 0.999999999999999]))
    print(np.min(r_vx_res[r_vx_res > -0.999999999999999]))
    #
    # Correlation coefficient b/w vy_res
    #
    r_vy_res = np.ones((num_sources, num_sources), float)
    cov_vy_res = np.ones((num_sources, num_sources), float)
    for i in range(num_sources):
        for j in range(num_sources):
            vy_res_meani = np.mean(vy_res[:, i])
            vy_res_meanj = np.mean(vy_res[:, j])
            vy_res_diffi = vy_res[:, i] - vy_res_meani
            vy_res_diffj = vy_res[:, j] - vy_res_meanj
            r_num = np.sum(vy_res_diffi * vy_res_diffj)
            r_den = np.sqrt(np.sum(vy_res_diffi ** 2 )) * np.sqrt(np.sum(vy_res_diffj ** 2))
            # Fringe cases
            r = r_num / r_den
            r = 1 if r > 1 else r
            r = -1 if r < -1 else r
            # Populate array
            r_vy_res[i, j] = r
            cov_vy_res[i, j] = r_num / _NUM_SAMPLES
    print(np.max(r_vy_res[r_vy_res < 0.999999999999999]))
    print(np.min(r_vy_res[r_vy_res > -0.999999999999999]))


    outfile = Path(__file__).parent / Path("pearsonr_cov.pkl")
    with open(outfile, "wb") as f:
        dill.dump(
            {
                "r_Upec": r_Upec,
                "cov_Upec": cov_Upec,
                "r_Vpec": r_Vpec,
                "cov_Vpec": cov_Vpec,
                "r_vx": r_vx,
                "cov_vx": cov_vx,
                "r_vy": r_vy,
                "cov_vy": cov_vy,
                "r_vx_res": r_vx_res,
                "cov_vx_res": cov_vx_res,
                "r_vy_res": r_vy_res,
                "cov_vy_res": cov_vy_res,
            },
            f,
        )
    print("Saved to .pkl file!")


def main():
    # mc_type = "HPDmode"
    # datafile = Path(__file__).parent / Path(f"csvfiles/alldata_{mc_type}.csv")
    # data = pd.read_csv(datafile)
    calc_cc()


if __name__ == "__main__":
    main()
