"""
MCMC_w_dist_uncer.py

Bayesian MCMC with MC parallax distances using priors from Reid et al. (2019)

Isaac Cheng - February 2021
"""

import argparse
import sys
from pathlib import Path
import sqlite3
from contextlib import closing
import numpy as np
import pandas as pd
import theano.tensor as tt
import pymc3 as pm
import dill
import textwrap

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans
from universal_rotcurve import urc
import mcmc_cleanup as clean
import mcmc_bic as bic

# Roll angle between galactic midplane and galactocentric frame
_ROLL = 0.0  # deg (Anderson et al. 2019)
# Sun's height above galactic midplane (Reid et al. 2019)
_ZSUN = 5.5  # pc
# Useful constants
_RAD_TO_DEG = 57.29577951308232  # 180/pi (Don't forget to % 360 after)


def get_data(db_file):
    """
    Puts all relevant data from db_file's Parallax table into pandas DataFrame.

    Inputs:
      db_file :: pathlib Path object
        Path object to SQL database containing maser galactic longitudes, latitudes,
        right ascensions, declinations, parallaxes, equatorial proper motions,
        and LSR velocities with all associated uncertainties

    Returns: data
      data :: pandas DataFrame
        Contains db_file data in pandas DataFrame. Specifically, it includes:
        ra (deg), dec (deg), glong (deg), glat (deg), plx (mas), e_plx (mas),
        mux (mas/yr), muy (mas/yr), vlsr (km/s) + all proper motion/vlsr uncertainties
    """

    with closing(sqlite3.connect(db_file).cursor()) as cur:  # context manager, auto-close
        cur.execute("SELECT ra, dec, glong, glat, plx, e_plx, "
                    "mux, muy, vlsr, e_mux, e_muy, e_vlsr FROM Parallax")
        data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    return data


# def plx_to_peak_dist(mean_plx, e_plx):
#     """
#     Computes peak (i.e. mode) of the distance distribution given the
#     parallax & the uncertainty in the parallax (assuming the parallax is Gaussian)

#     TODO: finish docstring
#     """

#     mean_dist = 1 / mean_plx
#     sigma_sq = e_plx * e_plx
#     numerator = np.sqrt(8 * sigma_sq * mean_dist * mean_dist + 1) - 1
#     denominator = 4 * sigma_sq * mean_dist

#     return (numerator / denominator)


def filter_data(data, filter_e_plx):
    """
    Filters sources < 4 kpc from galactic centre and
    (optionally) filters sources with e_plx/plx > 20%

    Inputs:
      data :: pandas DataFrame
        Contains maser galactic longitudes, latitudes, right ascensions, declinations,
        parallaxes, equatorial proper motions, and LSR velocities
        with all associated uncertainties
      filter_e_plx :: boolean
        If False, only filter sources closer than 4 kpc tp galactic centre
        If True, also filter sources with parallax uncertainties > 20% of the parallax

    Returns: filtered_data
      filtered_data :: pandas DataFrame
        Contains same data as input data DataFrame except with some sources removed
    """
    # Calculate galactocentric cylindrical radius
    #   N.B. We assume R0=8.15 kpc. This ensures we are rejecting the same set
    #   of sources each iteration. Also R0 is fairly well-constrained bc of Sgr A*
    all_radii = trans.get_gcen_cyl_radius(data["glong"], data["glat"], data["plx"])

    # Bad data criteria (N.B. casting to array prevents "+" not supported warnings)
    if filter_e_plx:  # Filtering used by Reid et al. (2019)
        print("Filter sources with R < 4 kpc & e_plx/plx > 20%")
        bad = (np.array(all_radii) < 4.0) + \
              (np.array(data["e_plx"] / data["plx"]) > 0.2)
    else:  # Only filter sources closer than 4 kpc to galactic centre
        print("Only filter sources with R < 4 kpc")
        bad = (np.array(all_radii) < 4.0)

    # Slice data into components (using np.asarray to prevent PyMC3 error with pandas)
    ra = data["ra"][~bad]  # deg
    dec = data["dec"][~bad]  # deg
    glon = data["glong"][~bad]  # deg
    glat = data["glat"][~bad]  # deg
    plx_orig = data["plx"][~bad]  # mas
    e_plx = data["e_plx"][~bad]  # mas
    eqmux = data["mux"][~bad]  # mas/yr (equatorial frame)
    e_eqmux = data["e_mux"][~bad]  # mas/y (equatorial frame)
    eqmuy = data["muy"][~bad]  # mas/y (equatorial frame)
    e_eqmuy = data["e_muy"][~bad]  # mas/y (equatorial frame)
    vlsr = data["vlsr"][~bad]  # km/s
    e_vlsr = data["e_vlsr"][~bad]  # km/s

    # Store filtered data in DataFrame
    filtered_data = pd.DataFrame(
        {
            "ra": ra,
            "dec": dec,
            "glong": glon,
            "glat": glat,
            "plx": plx_orig,
            "e_plx": e_plx,
            "mux": eqmux,
            "e_mux": e_eqmux,
            "muy": eqmuy,
            "e_muy": e_eqmuy,
            "vlsr": vlsr,
            "e_vlsr": e_vlsr,
        }
    )

    return filtered_data


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
    exp_part = (-0.5 * (dist - mean_dist) * (dist - mean_dist)
               / (dist * dist * mean_dist * mean_dist * e_plx * e_plx))
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
        ]).T

    return dists


def get_weights(dist, e_mux, e_muy, e_vlsr):
    """
    Calculates uncertainties in proper motions and LSR velocity
    using Reid et al. (2014) weights (i.e. accounts for 1D virial dispersion)

    Inputs:
      dist :: Array of scalars (kpc)
        Distances to sources (in the galactic frame). e.g., dist = 1 / parallax
      e_mux :: Array of scalars (mas/yr)
        Uncertainty in RA proper motion with cos(Declination) correction
      e_eqmuy :: Array of scalars (mas/yr)
        Uncertainty in declination proper motion
      e_vlsr :: Array of scalars (km/s)
        Uncertainty in LSR velocity

    Returns: weight_mux, weight_muy, weight_vlsr
      weight_mux, weight_muy :: Array of scalars (mas/yr)
        Reid et al. weights for the RA (w/ cos(dec) correction) & dec proper motions
      weight_vlsr :: Array of scalars (km/s)
        Reid et al. weights for the LSR velocity
    """

    km_kpc_s_to_mas_yr = 0.21094952656969873  # (mas/yr) / (km/kpc/s)

    # 1D Virial dispersion for stars in HMSFR w/ mass ~ 10^4 Msun w/in radius of ~ 1 pc
    sigma_vir_sq = 25.0  # km/s

    # Parallax to reciprocal of distance^2 (i.e. 1 / distance^2)
    reciprocal_dist_sq = km_kpc_s_to_mas_yr * km_kpc_s_to_mas_yr / (dist * dist)

    weight_mux = tt.sqrt(e_mux * e_mux + sigma_vir_sq * reciprocal_dist_sq)
    weight_muy = tt.sqrt(e_muy * e_muy + sigma_vir_sq * reciprocal_dist_sq)
    weight_vlsr = tt.sqrt(e_vlsr * e_vlsr + sigma_vir_sq)

    return weight_mux, weight_muy, weight_vlsr


# def ln_cauchy(x, peak, weight):
#     """
#     Returns natural log of the cauchy distribution. Here, peak = mean

#     TODO: finish docstring
#     """

#     hwhm = 1.177410023  # half width at half maximum == sqrt(2 * ln(2))
#     ln_hwhm_pi = 1.308047016  # ln(half width at half max * pi) = ln(sqrt(2*ln(2)) * pi)
#     coeff = - ln_hwhm_pi - tt.log(weight)
#     frac = (x - peak) / (hwhm * weight)

#     return coeff - tt.log(1 + frac * frac)


def ln_siviaskilling(x, mean, weight):
    """
    Returns the natural log of Sivia & Skilling's (2006) "Lorentzian-like" PDF.
    N.B. That the PDF is _not_ normalized. Peak is at 0.5

    Inputs:
      (x, mean, & weight must have the same units)
      x :: Array of scalars
        Data
      mean :: Array of scalars
        Arithmetic mean of the data
      weight :: Array of scalars
        Reid et al. (2014) weights for the data (i.e. data uncertaintites)

    Returns: lnlike
      lnlike :: Theano TensorVariable (typically dmatrix)
        Object representing the ln-likelihood of the data using this pdf
    """

    residual = abs((x - mean) / weight)
    lnlike = tt.log(1 - tt.exp(-0.5 * residual * residual)) - 2 * tt.log(residual)

    # Replace residuals near zero (i.e. near peak of ln(likelihood))
    # with value at peak of ln(likelihood) to prevent nans. Peak = ln(0.5) = -0.69314718
    lnlike = tt.switch(residual < 1e-7, -0.69314718, lnlike)

    return lnlike


def run_MCMC(
    data, num_iters, num_tune, num_cores, num_chains, num_samples,
    prior_set, like_type, is_database_data, this_round, filter_parallax, reject_method,
    free_Wpec=False, free_Zsun=False, free_roll=False, return_num_sources=False):
    """
    Runs Bayesian MCMC. Returns trace & number of sources used in fit.

    Inputs:
      data :: pandas DataFrame
        DataFrame with all relevant data (expand later)
      num_iters :: scalar (int)
        Number of MCMC iterations for sampler to run
      num_tune :: scalar (int)
        Number of MCMC tuning iterations
      num_cores :: scalar (int)
        Number of computer cores to allocate to PyMC3
      num_chains :: scalar (int)
        Number of parallel MCMC chains to run
      num_samples :: scalar (int)
        Number of distance samples per source
      prior_set :: string
        Prior set from Reid et al. (2019). (A1, A5, B, C, or D)
        N.B. the R0 prior has been updated to the value given by GRAVITY Collaboration
      like_type :: string
        Likelihood pdf. (gauss, cauchy, or sivia)
          - gauss = Gaussian pdf
          - cauchy = Cauchy (aka Lorentzian) pdf
          - sivia = Sivia & Skilling's 2006 pdf
      is_database_data :: boolean
        If True, will filter data (i.e. using a new set of priors)
        If False, will not filter data (i.e. reading filtered data from pickle file)
      this_round :: scalar (int)
        Number of times the MCMC algorithm has run
      filter_parallax :: boolean
        If True, will remove sources w/ R < 4kpc to galactic centre & e_plx/plx > 20%
        If False, will only remove sources w/ R < 4kpc to galactic centre
      reject_method :: string
        Outlier rejection method (lnlike or sigma)
          - lnlike = compares median ln-likelihood of each source to threshold value
          - sigma = rejects sources with residual motions > 3 sigma
      free_Wpec :: boolean (default: False)
        If True, the Sun's vertical velocity toward NGP is a free parameter
      free_Zsun :: boolean (defualt: False)
        If True, the Sun's height above galactic midplane is a free parameter
      free_roll :: boolean (defualt: False)
        If True, the roll angle between the galactic midplane and
        the galactocentric frame is a free parameter
      return_num_sources :: boolean (default: False)
        If True, return the number of sources used in MCMC fitting

    Returns: num_sources (optional)
      num_sources :: scalar (int), optional
        The number of sources used in MCMC fitting
        Returned iff return_num_sources == True
    """

    # New binary file to store MCMC output
    outfile = Path(__file__).parent / f"mcmc_outfile_{prior_set}_{this_round}.pkl"

    # Extract data components from database & filter if necessary
    if is_database_data:  # data is from database, need to filter data
        print("===\nStarting with fresh data from database")
        # Filter data
        data = filter_data(data, filter_parallax)
    else:  # do not need to filter data (data has been filtered)
        print("===\nUsing data from pickle file")

    # Slice data into components
    # (using np.asarray to prevent PyMC3 error with pandas)
    glon = np.asarray(data["glong"].values)  # deg
    glat = np.asarray(data["glat"].values)  # deg
    plx = np.asarray(data["plx"].values)  # mas
    e_plx = np.asarray(data["e_plx"].values)  # mas
    eqmux = np.asarray(data["mux"].values)  # mas/yr (equatorial frame)
    e_eqmux = np.asarray(data["e_mux"].values)  # mas/y (equatorial frame)
    eqmuy = np.asarray(data["muy"].values)  # mas/y (equatorial frame)
    e_eqmuy = np.asarray(data["e_muy"].values)  # mas/y (equatorial frame)
    vlsr = np.asarray(data["vlsr"].values)  # km/s
    e_vlsr = np.asarray(data["e_vlsr"].values)  # km/s
    # Calculate number of sources used in fit
    num_sources = len(eqmux)
    print("Number of data points used:", num_sources)

    # Sample random distances from parallaxes
    dist = generate_dists(plx, e_plx, num_samples)

    # # Making array of random parallaxes. Columns are samples of the same source
    # print("===\nNumber of plx samples:", num_samples)
    # plx = np.random.normal(loc=plx, scale=e_plx, size=(num_samples, num_sources))
    # # Replace non-positive parallax with small positive epsilon
    # print("Number of plx <= 0 replaced:", np.sum(plx<=0))
    # plx[plx<=0] = 1e-9
    # dist = 1 / plx

    # Galactic to galactocentric Cartesian coordinates
    bary_x, bary_y, bary_z = trans.gal_to_bary(glon, glat, dist)

    # 8 parameters from Reid et al. (2019): (see Section 4 & Table 3)
    #   R0, Usun, Vsun, Wsun, Upec, Vpec, a2, a3, (optional: Zsun, roll, Wpec)
    with pm.Model() as model:
        # === Define priors ===
        # R0 = pm.Uniform("R0", lower=7.0, upper=10.0)  # kpc
        R0 = pm.Normal("R0", mu=8.178, sigma=0.026)  # kpc (from GRAVITY Collaboration)
        a2 = pm.Uniform("a2", lower=0.7, upper=1.5)  # dimensionless
        a3 = pm.Uniform("a3", lower=1.5, upper=1.8)  # dimensionless
        if prior_set == "A1" or prior_set == "A5":
            Usun = pm.Normal("Usun", mu=11.1, sigma=1.2)  # km/s
            Vsun = pm.Normal("Vsun", mu=15.0, sigma=10.0)  # km/s
            Wsun = pm.Normal("Wsun", mu=7.2, sigma=1.1)  # km/s
            Upec = pm.Normal("Upec", mu=3.0, sigma=10.0)  # km/s
            Vpec = pm.Normal("Vpec", mu=-3.0, sigma=10.0)  # km/s
        elif prior_set == "B":
            Usun = pm.Normal("Usun", mu=11.1, sigma=1.2)  # km/s
            Vsun = pm.Normal("Vsun", mu=12.2, sigma=2.1)  # km/s
            Wsun = pm.Normal("Wsun", mu=7.2, sigma=1.1)  # km/s
            Upec = pm.Uniform("Upec", lower=-500.0, upper=500.0)  # km/s
            Vpec = pm.Uniform("Vpec", lower=-500.0, upper=500.0)  # km/s
        elif prior_set == "C":
            Usun = pm.Uniform("Usun", lower=-500.0, upper=500.0)  # km/s
            Vsun = pm.Uniform("Vsun", lower=-500.0, upper=500.0)  # km/s
            Wsun = pm.Uniform("Wsun", lower=-500.0, upper=500.0)  # km/s
            Upec = pm.Normal("Upec", mu=3.0, sigma=5.0)  # km/s
            Vpec = pm.Normal("Vpec", mu=-3.0, sigma=5.0)  # km/s
        elif prior_set == "D":
            Usun = pm.Uniform("Usun", lower=-500.0, upper=500.0)  # km/s
            Vsun = pm.Uniform("Vsun", lower=-5.0, upper=35.0)  # km/s
            Wsun = pm.Uniform("Wsun", lower=-500.0, upper=500.0)  # km/s
            Upec = pm.Uniform("Upec", lower=-500.0, upper=500.0)  # kpc
            Vpec = pm.Uniform("Vpec", lower=-23.0, upper=17.0)  # km/s
        else:
            raise ValueError("Illegal prior_set. Choose 'A1', 'A5', 'B', 'C', or 'D'.")
        print("===\nUsing prior set", prior_set)

        # === Predicted values (using data) ===
        # Barycentric Cartesian to galactocentric Cartesian coodinates
        if free_Zsun:
            print("+ free Zsun parameter", end=" ", flush=True)
            # Conservative prior based on Reid et al. 2019 & Anderson et al. 2019
            Zsun = pm.Normal("Zsun", mu=_ZSUN, sigma=10.)  # pc
        else:
            Zsun = _ZSUN  # pc
        if free_roll:
            print("+ free roll parameter", end=" ", flush=True)
            # Prior based on Reid et al. 2019
            roll = pm.Normal("roll", mu=_ROLL, sigma=0.1)  # deg
        else:
            roll = _ROLL  # deg

        gcen_x, gcen_y, gcen_z = trans.bary_to_gcen(
            bary_x, bary_y, bary_z, R0=R0, Zsun=Zsun, roll=roll)

        # Galactocentric Cartesian frame to galactocentric cylindrical frame
        gcen_cyl_dist = tt.sqrt(gcen_x * gcen_x + gcen_y * gcen_y)  # kpc
        azimuth = (tt.arctan2(gcen_y, -gcen_x) * _RAD_TO_DEG) % 360  # deg in [0,360)

        # Predicted galactocentric cylindrical velocity components
        v_circ = urc(gcen_cyl_dist, a2=a2, a3=a3, R0=R0) + Vpec  # km/s
        v_rad = -1 * Upec  # km/s, negative bc toward GC
        if free_Wpec:
            print("+ free Wpec parameter", end="", flush=True)
            # If nonzero, means masers moving out of galactic plane on average
            Wpec = pm.Normal("Wpec", mu=0., sigma=15.)  # km/s
        else:
            Wpec = 0.0  # km/s, zero vertical velocity in URC

        Theta0 = urc(R0, a2=a2, a3=a3, R0=R0)  # km/s, circular rotation speed of Sun

        # Go in reverse!
        # Galactocentric cylindrical to equatorial proper motions & LSR velocity
        eqmux_pred, eqmuy_pred, vlsr_pred = trans.gcen_cyl_to_pm_and_vlsr(
            gcen_cyl_dist, azimuth, gcen_z, v_rad, v_circ, Wpec,
            R0=R0, Zsun=Zsun, roll=roll,
            Usun=Usun, Vsun=Vsun, Wsun=Wsun, Theta0=Theta0,
            use_theano=True
        )

        # === Likelihood components (sigma values are from observed data) ===
        # Calculate uncertainties for likelihood function
        # (using Reid et al. (2014) weights as uncertainties)
        weight_eqmux, weight_eqmuy, weight_vlsr = get_weights(
            dist, e_eqmux, e_eqmuy, e_vlsr)

        # Make array of likelihood values evaluated at data points
        if free_Zsun or free_roll or free_Wpec:
            print()  # start a new line
        if like_type == "gauss":
            # GAUSSIAN MIXTURE PDF
            print("Using Gaussian PDF")
            lnlike_eqmux = pm.Normal.dist(mu=eqmux_pred, sigma=weight_eqmux).logp(eqmux)
            lnlike_eqmuy = pm.Normal.dist(mu=eqmuy_pred, sigma=weight_eqmuy).logp(eqmuy)
            lnlike_vlsr = pm.Normal.dist(mu=vlsr_pred, sigma=weight_vlsr).logp(vlsr)
            if reject_method == 'lnlike':
                ln_sqrt_2pi = 0.918938533  # ln(sqrt(2*pi))
                # Normalize Gaussian to peak of 1 & save to trace
                lnlike_eqmux_norm = pm.Deterministic(
                    "lnlike_eqmux_norm", lnlike_eqmux + tt.log(e_eqmux) + ln_sqrt_2pi)
                lnlike_eqmuy_norm = pm.Deterministic(
                    "lnlike_eqmuy_norm", lnlike_eqmuy + tt.log(e_eqmuy) + ln_sqrt_2pi)
                lnlike_vlsr_norm = pm.Deterministic(
                    "lnlike_vlsr_norm", lnlike_vlsr + tt.log(e_vlsr) + ln_sqrt_2pi)
        elif like_type == "cauchy":
            # CAUCHY PDF
            print("Using Cauchy PDF")
            hwhm = 1.177410023  # half width at half maximum = sqrt(2 * ln(2))
            lnlike_eqmux = pm.Cauchy.dist(
                alpha=eqmux_pred, beta=hwhm * weight_eqmux).logp(eqmux)
            lnlike_eqmuy = pm.Cauchy.dist(
                alpha=eqmuy_pred, beta=hwhm * weight_eqmuy).logp(eqmuy)
            lnlike_vlsr = pm.Cauchy.dist(
                alpha=vlsr_pred, beta=hwhm * weight_vlsr).logp(vlsr)
            if reject_method == 'lnlike':
                # ln(half width at half maximum * pi) = ln(sqrt(2*ln(2)) * pi)
                ln_hwhm_pi = 1.308047016
                # Normalize Cauchy to peak of 1 & save to trace
                lnlike_eqmux_norm = pm.Deterministic(
                    'lnlike_eqmux_norm', lnlike_eqmux + ln_hwhm_pi + tt.log(weight_eqmux))
                lnlike_eqmuy_norm = pm.Deterministic(
                    'lnlike_eqmuy_norm', lnlike_eqmuy + ln_hwhm_pi + tt.log(weight_eqmuy))
                lnlike_vlsr_norm = pm.Deterministic(
                    'lnlike_vlsr_norm', lnlike_vlsr + ln_hwhm_pi + tt.log(weight_vlsr))
        elif like_type == "sivia":
            # SIVIA & SKILLING (2006) "LORENTZIAN-LIKE" CONSERVATIVE PDF
            print("Using Sivia & Skilling (2006) PDF")
            lnlike_eqmux = ln_siviaskilling(eqmux, eqmux_pred, weight_eqmux)
            lnlike_eqmuy = ln_siviaskilling(eqmuy, eqmuy_pred, weight_eqmuy)
            lnlike_vlsr = ln_siviaskilling(vlsr, vlsr_pred, weight_vlsr)
            if reject_method == 'lnlike':
                # Save log-likelihood distributions to trace w/out impacting model
                # (Note that these are not normalized!)
                lnlike_eqmux_norm = pm.Deterministic("lnlike_eqmux_norm", lnlike_eqmux)
                lnlike_eqmuy_norm = pm.Deterministic("lnlike_eqmuy_norm", lnlike_eqmuy)
                lnlike_vlsr_norm = pm.Deterministic("lnlike_vlsr_norm", lnlike_vlsr)
        else:
            raise ValueError(
                "Invalid like_type. Please input 'gauss', 'cauchy', or 'sivia'.")

        # === Full likelihood function (specified by log-probability) ===
        # Joint likelihood
        lnlike_tot = lnlike_eqmux + lnlike_eqmuy + lnlike_vlsr
        # Marginalize over each distance samples
        lnlike_sum = pm.logsumexp(lnlike_tot, axis=0)
        lnlike_avg = lnlike_sum - tt.log(num_samples)
        # Sum over sources
        lnlike_final = lnlike_avg.sum()
        # Likelihood function
        likelihood = pm.Potential("likelihood", lnlike_final)

        # # === Check model ===
        # print(textwrap.fill("test_point: " + str(model.test_point),
        #                     width=90, initial_indent='', subsequent_indent='    '))
        # print("logp(test_point):", model.logp(model.test_point))

        # === Run MCMC ===
        print(f"Using {num_cores} cores, {num_chains} chains, "
            f"{num_tune} tunings, and {num_iters} iterations.\n===")
        trace = pm.sample(
            num_iters,
            init="advi",
            tune=num_tune,
            cores=num_cores,
            chains=num_chains,
            return_inferencedata=False)

        # === See results (calling within model to prevent FutureWarning) ===
        # if reject_method == "lnlike":
        #     print(
        #         pm.summary(
        #             trace,
        #             var_names=[
        #                 "~lnlike_eqmux_norm",
        #                 "~lnlike_eqmuy_norm",
        #                 "~lnlike_vlsr_norm",
        #             ],
        #         ).to_string()
        #     )
        # else:  # reject_method == "sigma"
        #     print(pm.summary(trace).to_string())

        # Varnames order: [R0, Zsun, Usun, Vsun, Wsun, Upec, Vpec, Wpec, roll, a2, a3]
        varnames = ['R0', 'Usun', 'Vsun', 'Wsun', 'Upec', 'Vpec', 'a2', 'a3']
        varnames.insert(6, "roll") if free_roll else None
        varnames.insert(6, "Wpec") if free_Wpec else None
        varnames.insert(1, "Zsun") if free_Zsun else None
        print(pm.summary(trace, var_names=varnames).to_string())

        # === Save results to pickle file ===
        with open(outfile, "wb") as f:
            dill.dump(
                {
                    "data": data,
                    "model": model,
                    "trace": trace,
                    "prior_set": prior_set,
                    "like_type": like_type,
                    "num_sources": num_sources,
                    "num_samples": num_samples,
                    "this_round": this_round,
                    "reject_method": reject_method,
                    "free_Zsun": free_Zsun,
                    "free_roll": free_roll,
                    "free_Wpec": free_Wpec,
                }, f)

        if return_num_sources:
            return num_sources


def main(infile, num_cores=None, num_chains=None, num_tune=2000, num_iters=5000,
        num_samples=100, prior_set="A1", like_type="gauss", num_rounds=1,
        reject_method="sigma", this_round=1, filter_plx=False,
        free_Wpec=False, free_Zsun=False, free_roll=False, auto_run=False):
    """
    Inputs:
      infile :: string
        Absolute path of the .db or .pkl file containing the data
      num_cores :: scalar (int), optional
        Number of computer cores to allocate to PyMC3
      num_chains :: scalar (int), optional
        Number of parallel MCMC chains to run
      num_tune :: scalar (int), optional
        Number of MCMC tuning iterations
      num_iters :: scalar (int), optional
        Number of MCMC iterations for sampler to run
      num_samples :: scalar (int), optional
        Number of distance samples per source
      prior_set :: string, optional
        Prior set from Reid et al. (2019). (A1, A5, B, C, or D)
        N.B. the R0 prior has been updated to the value given by GRAVITY Collaboration
      like_type :: string, optional
        Likelihood pdf. (gauss, cauchy, or sivia)
          - gauss = Gaussian pdf
          - cauchy = Cauchy (aka Lorentzian) pdf
          - sivia = Sivia & Skilling's 2006 pdf
      num_rounds :: scalar (int), optional
        Total number of times to run MCMC algorithm. Overridden if auto_run == True
      reject_method :: string, optional
        Outlier rejection method (lnlike or sigma)
          - lnlike = compares median ln-likelihood of each source to threshold value
          - sigma = rejects sources with residual motions > 3 sigma
      this_round :: scalar (int), optional
        Number of times the MCMC algorithm has run
      filter_plx :: boolean, optional
        If False, will only remove sources closer than 4 kpc to galactic centre
        If True, will also remove sources with parallax uncertainty > 20% of parallax
      free_Wpec :: boolean (default: False)
        If True, the Sun's vertical velocity toward NGP is a free parameter
      free_Zsun :: boolean (defualt: False)
        If True, the Sun's height above galactic midplane is a free parameter
      free_roll :: boolean (defualt: False)
        If True, the roll angle between the galactic midplane and
        the galactocentric frame is a free parameter
      auto_run :: boolean, optional
        If True, will run MCMC until no more outliers are rejected
        If False, will run MCMC as specified by num_rounds
    """

    if num_cores is None:
        num_cores = 2
    if num_chains is None:
        num_chains = num_cores

    if num_rounds < 1:
        raise ValueError("num_rounds must be an integer greater than or equal to 1.")
    if this_round < 1:
        raise ValueError("this_round must be an integer greater than or equal to 1.")

    # Run simulation for num_rounds times or until no more outliers removed
    if auto_run:
        print(f"=========\nRunning Bayesian MCMC until convergence w/ "
              f"{prior_set} priors, {like_type} PDF", end="", flush=True)
    else:
        print(f"=========\nQueueing {num_rounds} Bayesian MCMC rounds w/ "
            f"{prior_set} priors, {like_type} PDF", end="", flush=True)

    if num_rounds == 1 and not auto_run:
        print()
        if reject_method != "sigma":
            # Override reject_method since no outlier cleaning will be done
            # "sigma" is faster than "lnlike" since it does not require pm.Deterministic()
            print("(Background task: overriding reject_method to 'sigma' "
                "since no outlier cleaning will be done)")
            reject_method = "sigma"
    else:
        print(f", & reject_method = {reject_method}")

    while this_round <= num_rounds or auto_run:
        print(f"===\nRunning round {this_round}")
        if infile[-3:] == ".db":
            if this_round != 1:
                raise ValueError("You should only load .db file if this_round == 1")
            # Load database file
            load_database = True
            db = Path(infile)
            data = get_data(db)
        else:
            # Load cleaned pickle file
            load_database = False
            with open(infile, "rb") as f:
                data = dill.load(f)["data"]
            if this_round != 1:
                # Override like_type to "gauss"
                like_type = "gauss"

        num_sources = run_MCMC(
            data,
            num_iters, num_tune, num_cores, num_chains, num_samples,
            prior_set, like_type, load_database, this_round, filter_plx, reject_method,
            free_Wpec=free_Wpec, free_Zsun=free_Zsun, free_roll=free_roll,
            return_num_sources=True
        )

        # Calculate Bayesian information criterion
        bic.main(prior_set, this_round)

        # Seeing if outlier rejection is necessary
        if this_round == num_rounds and not auto_run:
            break  # No need to clean data after final MCMC run
        # Else: do outlier rejection
        num_sources_cleaned = clean.main(
            prior_set, this_round, return_num_sources_cleaned=True)
        if like_type == "gauss" and num_sources == num_sources_cleaned:
            # Even if no outliers rejected, if like_type == "sivia" or "cauchy",
            # do at least one iteration with "gauss" (since num_rounds > 1)
            print(f"No more outliers rejected after round {this_round}. Exiting")
            break

        # Set auto-generated cleaned pickle file as next infile
        filename = f"mcmc_outfile_{prior_set}_{this_round}_clean.pkl"
        infile = str(Path(__file__).parent / filename)
        this_round += 1

    print(f"===\n{this_round} Bayesian MCMC runs complete\n=========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCMC stuff",
        prog="MCMC_w_dist_uncer.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "infile",
        type=str,
        help="Either database file or pickle file containing all relevant data")
    parser.add_argument(
        "--num_cores",
        type=int,
        default=None,
        help="Maximum number of CPU cores to use")
    parser.add_argument(
        "--num_chains",
        type=int,
        default=None,
        help="Number of parallel MCMC chains")
    parser.add_argument(
        "--num_tune",
        type=int,
        default=2000,
        help="Number of tuning iterations")
    parser.add_argument(
        "--num_iters",
        type=int,
        default=5000,
        help="Number of actual MCMC iterations")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of times to sample each parallax")
    parser.add_argument(
        "--prior_set",
        type=str,
        default="A5",
        help="Prior set to use from Reid et al. 2019 (A1, A5, B, C, or D)")
    parser.add_argument(
        "--like_type",
        type=str,
        default="gauss",
        help="Likelihood PDF (gauss, cauchy, or sivia)")
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=1,
        help="Number of times to run Bayesian MCMC")
    parser.add_argument(
        "--reject_method",
        type=str,
        default="sigma",
        help="Outlier rejection method for mcmc_cleanup.py (sigma or lnlike)")
    parser.add_argument(
        "--this_round",
        type=int,
        default=1,
        help="Overrides default starting point for number of rounds. Useful if previous run failed & want to load from pickle file")
    parser.add_argument(
        "--filter_plx",
        action="store_true",
        default=False,
        help="Filter sources with e_plx/plx > 0.2")
    parser.add_argument(
        "--free_Zsun",
        action="store_true",
        default=False,
        help="Let the Sun's height above galactic midplane be a free parameter")
    parser.add_argument(
        "--free_Wpec",
        action="store_true",
        default=False,
        help="Let the Sun's vertical velocity toward NGP be a free parameter")
    parser.add_argument(
        "--free_roll",
        action="store_true",
        default=False,
        help="Let the roll angle between galactic midplane and galactocentric frame be a free parameter")
    parser.add_argument(
        "--auto_run",
        action="store_true",
        default=False,
        help="Run the MCMC program until no more outliers are rejected. Will override num_rounds")
    args = vars(parser.parse_args())

    main(
        args["infile"],
        num_cores=args["num_cores"],
        num_chains=args["num_chains"],
        num_tune=args["num_tune"],
        num_iters=args["num_iters"],
        num_samples=args["num_samples"],
        prior_set=args["prior_set"],
        like_type=args["like_type"],
        num_rounds=args["num_rounds"],
        reject_method=args["reject_method"],
        this_round=args["this_round"],
        filter_plx=args["filter_plx"],
        free_Zsun=args["free_Zsun"],
        free_Wpec=args["free_Wpec"],
        free_roll=args["free_roll"],
        auto_run=args["auto_run"],
    )
