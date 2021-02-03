"""
Bayesian MCMC using priors from Reid et al. (2019)
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
import arviz as az

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans
from universal_rotcurve import urc
import mcmc_cleanup as clean

# Universal rotation curve parameters (Persic et al. 1996)
_A_TWO = 0.96  # (Reid et al. 2019)
_A_THREE = 1.62  # (Reid et al. 2019)

# Sun's distance from galactic centre
_RSUN = 8.15  # kpc (Reid et al. 2019)

# Useful constants
_DEG_TO_RAD = 0.017453292519943295  # pi/180
_RAD_TO_DEG = 57.29577951308232  # 180/pi (Don't forget to % 360 after)
_AU_PER_YR_TO_KM_PER_S = 4.740470463533348  # from astropy (uses tropical year)
_KM_KPC_S_TO_MAS_YR = 0.21094952656969873  # (mas/yr) / (km/kpc/s)
_KPC_TO_KM = 3.085677581e16
_KM_TO_KPC = 3.24077929e-17
_LN_SQRT_2PI = 0.918938533
_LN_HALF = -0.69314718


# def str2bool(string):
#     """
#     Parses a string into a boolean value
#     """

#     if isinstance(string, bool):
#         return string
#     if string.lower() in ['yes', 'true', 't', 'y', '1']:
#         return True
#     elif string.lower() in ['no', 'false', 'f', 'n', '0']:
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')


def get_data(db_file):
    """
    Retrieves all relevant data in Parallax table
    from database connection specified by db_file

    Returns DataFrame with:
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


def get_weights_sq(plx, e_mux, e_muy, e_vlsr):
    """
    Calculates sigma values for proper motions and LSR velocity
    using Reid et al. (2014) weights

    Returns: sqaure of the weights (sigma_mux_sq, sigma_muy_sq, sigma_vlsr_sq)

    TODO: finish docstring
    """

    # 1D Virial dispersion for stars in HMSFR w/ mass ~ 10^4 Msun w/in radius of ~ 1 pc
    sigma_vir_sq = 25.0  # km/s

    # Parallax to reciprocal of distance^2 (i.e. 1 / distance^2)
    reciprocal_dist_sq = plx * plx * _KM_KPC_S_TO_MAS_YR * _KM_KPC_S_TO_MAS_YR

    weight_mux_sq = e_mux * e_mux + sigma_vir_sq * reciprocal_dist_sq
    weight_muy_sq = e_muy * e_muy + sigma_vir_sq * reciprocal_dist_sq 
    weight_vlsr_sq = e_vlsr * e_vlsr + sigma_vir_sq

    return weight_mux_sq, weight_muy_sq, weight_vlsr_sq


def ln_gaussian(x, mean, weight_sq):
    """
    Returns the natural log of a Normal distribution given
    data, mean of data, and square of the standard deviation

    TODO: finish docstring
    """

    residual_sq = (x - mean) * (x - mean) / weight_sq
    lnlike = -0.5 * (residual_sq + tt.log(weight_sq) + _LN_SQRT_2PI)
    return lnlike


def ln_siviaskilling(x, mean, weight_sq):
    """
    Returns the natural log of Sivia & Skilling's (2006) "Lorentzian-like" PDF.
    N.B. That the PDF is _not_ normalized.

    TODO: Finish docstring
    """

    residual_sq = (x - mean) * (x - mean) / weight_sq
    lnlike = tt.log(1 - tt.exp(-0.5 * residual_sq)) - tt.log(residual_sq)

    # Replace residuals near zero (i.e. near peak of ln(likelihood)
    # with value at peak of ln(likelihood) to prevent nans
    # is_nan2 = tt.isnan(lnlike)
    # lnlike2 = tt.switch(is_nan2, -np.inf, lnlike)
    lnlike = tt.switch(residual_sq < 1e-12, _LN_HALF, lnlike)

    return lnlike


def run_MCMC(
    data, num_iters, num_tune, num_cores, num_chains, num_samples,
    prior_set, like_type, is_database_data, filter_parallax, num_round
):
    """
    Runs Bayesian MCMC. Returns trace & number of sources used in fit.

    Inputs:
      data : pandas DataFrame
        DataFrame with all relevant data (expand later)
      is_database_data : boolean
        If True, will filter data (i.e. using a new set of priors)
        If False, will not filter data (i.e. reading filtered data from pickle file)

    TODO: finish docstring
    """

    # New binary file to store MCMC output
    outfile = Path(__file__).parent / f"mcmc_outfile_{prior_set}_{num_round}.pkl"

    if is_database_data:  # data is from database, need to filter data
        print("===\nStarting with fresh data from database")
        # Create condition to filter data
        # N.B. We assume R0=8.15 kpc. This ensures we are rejecting the same set
        # of sources each iteration. Also R0 is fairly well-constrained bc of Sgr A*
        all_radii = trans.get_gcen_cyl_radius(data["glong"], data["glat"], data["plx"])

        # Bad data criteria (N.B. casting to array prevents "+" not supported warnings)
        if filter_parallax:  # Standard filtering used by Reid et al. (2019)
            print("Filter sources w/ R < 4 kpc & e_plx/plx > 20%")
            bad = (np.array(all_radii) < 4.0) + (np.array(data["e_plx"] / data["plx"]) > 0.2)
        else:  # Only filter sources closer than 4 kpc to galactic centre
            print("Only filter sources w/ R < 4 kpc")
            bad = (np.array(all_radii) < 4.0)

        # Slice data into components (using np.asarray to prevent PyMC3 error with pandas)
        ra = np.asarray(data["ra"][~bad].values)  # deg
        dec = np.asarray(data["dec"][~bad].values)  # deg
        glon = np.asarray(data["glong"][~bad].values)  # deg
        glat = np.asarray(data["glat"][~bad].values)  # deg
        plx_orig = np.asarray(data["plx"][~bad].values)  # mas
        e_plx = np.asarray(data["e_plx"][~bad].values)  # mas
        eqmux = np.asarray(data["mux"][~bad].values)  # mas/yr (equatorial frame)
        e_eqmux = np.asarray(data["e_mux"][~bad].values)  # mas/y (equatorial frame)
        eqmuy = np.asarray(data["muy"][~bad].values)  # mas/y (equatorial frame)
        e_eqmuy = np.asarray(data["e_muy"][~bad].values)  # mas/y (equatorial frame)
        vlsr = np.asarray(data["vlsr"][~bad].values)  # km/s
        e_vlsr = np.asarray(data["e_vlsr"][~bad].values)  # km/s

        # Store filtered data in DataFrame
        data = pd.DataFrame(
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
    else:  # do not need to filter data (data has been filtered)
        print("===\nUsing data from pickle file")
        # Slice data into components (using np.asarray to prevent PyMC3 error with pandas)
        # ra = np.asarray(data["ra"].values)  # deg
        # dec = np.asarray(data["dec"].values)  # deg
        glon = np.asarray(data["glong"].values)  # deg
        glat = np.asarray(data["glat"].values)  # deg
        plx_orig = np.asarray(data["plx"].values)  # mas
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

    # Making array of random parallaxes. Columns are samples of the same source
    print("===\nNumber of plx samples:", num_samples)
    # plx = np.array([plx_orig, ] * num_samples)
    plx = np.random.normal(loc=plx_orig, scale=e_plx, size=(num_samples, num_sources))
    # Replace non-positive parallax with nan
    print("Number of plx <= 0 replaced:", np.sum(plx<=0))
    plx[plx<=0] = np.nan
    # plx[0, 0] = np.nan
    # print(np.sum(np.isnan(plx)))
    # print(np.sum(~np.isfinite(plx)))

    # e_plx = np.array([e_plx,] * num_samples)
    # glon = np.array([glon,] * num_samples)  # num_samples by num_sources
    # glat = np.array([glat,] * num_samples)  # num_samples by num_sources
    # eqmux = np.array([eqmux,] * num_samples)
    # eqmuy = np.array([eqmuy,] * num_samples)
    # vlsr = np.array([vlsr,] * num_samples)
    # e_eqmux = np.array([e_eqmux,] * num_samples)
    # e_eqmuy = np.array([e_eqmuy,] * num_samples)
    # e_vlsr = np.array([e_vlsr,] * num_samples)

    # Parallax to distance
    gdist = trans.parallax_to_dist(plx)
    # Galactic to galactocentric Cartesian coordinates
    bary_x, bary_y, bary_z = trans.gal_to_bary(glon, glat, gdist)

    # 8 parameters from Reid et al. (2019): (see Section 4 & Table 3)
    #   R0, Usun, Vsun, Wsun, Upec, Vpec, a2, a3
    model_obj = pm.Model()
    with model_obj:
        if prior_set == "A1" or prior_set == "A5":  # Make model with SET A PRIORS
            # R0 = pm.Uniform("R0", lower=0, upper=500.)  # kpc
            R0 = pm.Uniform("R0", lower=7.0, upper=10.0)  # kpc
            Usun = pm.Normal("Usun", mu=11.1, sigma=1.2)  # km/s
            Vsun = pm.Normal("Vsun", mu=15.0, sigma=10.0)  # km/s
            Wsun = pm.Normal("Wsun", mu=7.2, sigma=1.1)  # km/s
            Upec = pm.Normal("Upec", mu=3.0, sigma=10.0)  # km/s
            Vpec = pm.Normal("Vpec", mu=-3.0, sigma=10.0)  # km/s
            a2 = pm.Uniform("a2", lower=0.8, upper=1.3)  # dimensionless
            a3 = pm.Uniform("a3", lower=1.5, upper=1.8)  # dimensionless
        elif prior_set == "B":
            # R0 = pm.Uniform("R0", lower=0, upper=500.)  # kpc
            R0 = pm.Uniform("R0", lower=7.0, upper=10.0)  # kpc
            Usun = pm.Normal("Usun", mu=11.1, sigma=1.2)  # km/s
            Vsun = pm.Normal("Vsun", mu=12.2, sigma=2.1)  # km/s
            Wsun = pm.Normal("Wsun", mu=7.2, sigma=1.1)  # km/s
            Upec = pm.Uniform("Upec", lower=-500.0, upper=500.0)  # km/s
            Vpec = pm.Uniform("Vpec", lower=-500.0, upper=500.0)  # km/s
            a2 = pm.Uniform("a2", lower=0.5, upper=1.5)  # dimensionless
            a3 = pm.Uniform("a3", lower=1.5, upper=1.8)  # dimensionless
        elif prior_set == "C":
            # R0 = pm.Uniform("R0", lower=0, upper=500.0)  # kpc
            R0 = pm.Uniform("R0", lower=7.0, upper=10.0)  # kpc
            Usun = pm.Uniform("Usun", lower=-500.0, upper=500.0)  # km/s
            Vsun = pm.Uniform("Vsun", lower=-500.0, upper=500.0)  # km/s
            Wsun = pm.Uniform("Wsun", lower=-500.0, upper=500.0)  # km/s
            Upec = pm.Normal("Upec", mu=3.0, sigma=5.0)  # km/s
            Vpec = pm.Normal("Vpec", mu=-3.0, sigma=5.0)  # km/s
            a2 = pm.Uniform("a2", lower=0.5, upper=1.5)  # dimensionless
            a3 = pm.Uniform("a3", lower=1.5, upper=1.8)  # dimensionless
        elif prior_set == "D":
            # R0 = pm.Uniform("R0", lower=0, upper=500.0)  # kpc
            R0 = pm.Uniform("R0", lower=7.0, upper=10.0)  # kpc
            Usun = pm.Uniform("Usun", lower=-500.0, upper=500.0)  # km/s
            Vsun = pm.Uniform("Vsun", lower=-5.0, upper=35.0)  # km/s
            Wsun = pm.Uniform("Wsun", lower=-500.0, upper=500.0)  # km/s
            Upec = pm.Uniform("Upec", lower=-500.0, upper=500.0)  # kpc
            Vpec = pm.Uniform("Vpec", lower=-23.0, upper=17.0)  # km/s
            a2 = pm.Uniform("a2", lower=0.5, upper=1.5)  # dimensionless
            a3 = pm.Uniform("a3", lower=1.5, upper=1.8)  # dimensionless
        else:
            raise ValueError("Illegal prior_set. Choose 'A1', 'A5', 'B', 'C', or 'D'.")
        print("===\nUsing prior set", prior_set)

        # === Predicted values (using data) ===
        # Barycentric Cartesian to galactocentric Cartesian coodinates
        gcen_x, gcen_y, gcen_z = trans.bary_to_gcen(bary_x, bary_y, bary_z, R0=R0)

        # Galactocentric Cartesian frame to galactocentric cylindrical frame
        gcen_cyl_dist = tt.sqrt(gcen_x * gcen_x + gcen_y * gcen_y)  # kpc
        azimuth = (tt.arctan2(gcen_y, -gcen_x) * _RAD_TO_DEG) % 360  # deg in [0,360)
        v_circ_pred = urc(gcen_cyl_dist, a2=a2, a3=a3, R0=R0) + Vpec  # km/s
        v_rad = -1 * Upec  # km/s, negative bc toward GC
        v_vert = 0.0  # Zero vertical velocity in URC

        Theta0 = urc(R0, a2=a2, a3=a3, R0=R0)  # km/s, circular rotation speed of Sun

        # Go in reverse!
        # Galactocentric cylindrical to equatorial proper motions & LSR velocity
        eqmux_pred, eqmuy_pred, vlsr_pred = trans.gcen_cyl_to_pm_and_vlsr(
            gcen_cyl_dist, azimuth, gcen_z, v_rad, v_circ_pred, v_vert,
            R0=R0, Usun=Usun, Vsun=Vsun, Wsun=Wsun, Theta0=Theta0,
            use_theano=True,
        )

        # === Likelihood components (sigma values are from observed data) ===
        # Calculating uncertainties for likelihood function
        # Using Reid et al. (2014) weights as uncertainties
        # sigma_vir = 5.0  # km/s
        # weight_eqmux_sq = (
        #     e_eqmux * e_eqmux
        #     + sigma_vir * sigma_vir
        #     * plx * plx
        #     * _KM_KPC_S_TO_MAS_YR * _KM_KPC_S_TO_MAS_YR
        # )
        # weight_eqmuy_sq = (
        #     e_eqmuy * e_eqmuy
        #     + sigma_vir * sigma_vir
        #     * plx * plx
        #     * _KM_KPC_S_TO_MAS_YR * _KM_KPC_S_TO_MAS_YR
        # )
        # weight_vlsr_sq = e_vlsr * e_vlsr + sigma_vir * sigma_vir
        weight_eqmux_sq, weight_eqmuy_sq, weight_vlsr_sq = get_weights_sq(
            plx, e_eqmux, e_eqmuy, e_vlsr
        )

        # Making array of likelihood values evaluated at data points
        if like_type == "gauss":
            # GAUSSIAN MIXTURE PDF
            print("Using Gaussian PDF")
            lnlike_eqmux = ln_gaussian(eqmux, eqmux_pred, weight_eqmux_sq)
            lnlike_eqmuy = ln_gaussian(eqmuy, eqmuy_pred, weight_eqmuy_sq)
            lnlike_vlsr = ln_gaussian(vlsr, vlsr_pred, weight_vlsr_sq)
            # lnlike_eqmux = pm.Normal.dist(mu=eqmux_pred, sigma=weight_eqmux).logp(eqmux)
            # lnlike_eqmuy = pm.Normal.dist(mu=eqmuy_pred, sigma=weight_eqmuy).logp(eqmuy)
            # lnlike_vlsr = pm.Normal.dist(mu=vlsr_pred, sigma=weight_vlsr).logp(vlsr)
            # # Save exponential part of log-likelihood distributions to trace
            # lnlike_eqmux_dist = pm.Deterministic(
            #     "lnlike_eqmux_dist", lnlike_eqmux + tt.log(e_eqmux) + _LN_SQRT_2PI
            # )
            # lnlike_eqmuy_dist = pm.Deterministic(
            #     "lnlike_eqmuy_dist", lnlike_eqmuy + tt.log(e_eqmuy) + _LN_SQRT_2PI
            # )
            # lnlike_vlsr_dist = pm.Deterministic(
            #     "lnlike_vlsr_dist", lnlike_vlsr + tt.log(e_vlsr) + _LN_SQRT_2PI
            # )
        # elif like_type == "cauchy":
        #     # CAUCHY PDF
        #     print("Using Cauchy PDF")
        #     hwhm = tt.sqrt(2 * tt.log(2))  # half width at half maximum
        #     lnlike_eqmux = pm.Cauchy.dist(
        #         alpha=eqmux_pred, beta=hwhm * weight_eqmux).logp(eqmux)
        #     lnlike_eqmuy = pm.Cauchy.dist(
        #         alpha=eqmuy_pred, beta=hwhm * weight_eqmuy).logp(eqmuy)
        #     lnlike_vlsr = pm.Cauchy.dist(
        #         alpha=vlsr_pred, beta=hwhm * weight_vlsr).logp(vlsr)
        elif like_type == "sivia":
            # SIVIA & SKILLING (2006) "LORENTZIAN-LIKE" CONSERVATIVE PDF
            print("Using Sivia & Skilling (2006) PDF")
            lnlike_eqmux = ln_siviaskilling(eqmux, eqmux_pred, weight_eqmux_sq)
            lnlike_eqmuy = ln_siviaskilling(eqmuy, eqmuy_pred, weight_eqmuy_sq)
            lnlike_vlsr = ln_siviaskilling(vlsr, vlsr_pred, weight_vlsr_sq)
            # # Save log-likelihood distributions to trace w/out impacting model
            # lnlike_eqmux_dist = pm.Deterministic("lnlike_eqmux_dist", lnlike_eqmux)
            # lnlike_eqmuy_dist = pm.Deterministic("lnlike_eqmuy_dist", lnlike_eqmuy)
            # lnlike_vlsr_dist = pm.Deterministic("lnlike_vlsr_dist", lnlike_vlsr)
        else:
            raise ValueError(
                "Invalid like_type. Please input 'gauss', 'cauchy', or 'sivia'."
            )

        # Joint likelihood
        lnlike_tot = lnlike_eqmux + lnlike_eqmuy + lnlike_vlsr
        # Marginalize over parallax
        is_nan = tt.isnan(lnlike_tot)
        lnlike_tot = tt.switch(is_nan, -np.inf, lnlike_tot)
        # Calculate likelihood of each source
        lnlike_sum = pm.logsumexp(lnlike_tot, axis=0)
        lnlike_avg = lnlike_sum - tt.log(tt.invert(is_nan).sum(axis=0))
        # Sum over data
        lnlike_final = lnlike_avg.sum()

        # === Full likelihood function (specified by log-probability) ===
        # N.B. pm.Potential expects values instead of functions
        likelihood = pm.Potential("likelihood", lnlike_final)

        # Run MCMC
        print(f"Using {num_cores} cores, {num_chains} chains, "
              f"{num_tune} tunings, and {num_iters} iterations.\n===")
        trace = pm.sample(
            num_iters,
            init="advi",
            tune=num_tune,
            cores=num_cores,
            chains=num_chains,
            return_inferencedata=False,
        )  # walker

        # See results (calling within model to prevent FutureWarning)
        print(
            pm.summary(
                trace,
                # var_names=[
                #     "~lnlike_eqmux_dist",
                #     "~lnlike_eqmuy_dist",
                #     "~lnlike_vlsr_dist",
                # ],
            ).to_string()
        )
        # Save results to pickle file
        with open(outfile, "wb") as f:
            dill.dump(
                {
                    "data": data,
                    "model_obj": model_obj,
                    "trace": trace,
                    "prior_set": prior_set,
                    "like_type": like_type,
                    "num_sources": num_sources,
                    "num_samples": num_samples,
                    "num_round": num_round,
                },
                f,
            )


def main(infile, num_cores=None, num_chains=None, num_tune=2000, num_iters=5000,
        num_samples=100, prior_set="A1", like_type="gauss",
        filter_plx=False, num_rounds=1, filter_method="sigma", this_round=1
):
    if num_cores is None:
        num_cores = 2
    if num_chains is None:
        num_chains = num_cores

    # Run simulation for num_rounds times
    if num_rounds < 1:
        raise ValueError("num_rounds must be an integer greater than or equal to 1.")

    print(f"=========\nQueueing {num_rounds} Bayesian MCMC rounds")
    while this_round <= num_rounds:
        print(f"===\nRunning round {this_round}")
        if this_round == 1 and infile[-3:] == ".db":
            # Load database file
            load_database = True
            db = Path(infile)
            data = get_data(db)
        else:
            # TODO: implement way to load custom pickle file when not starting from database
            # Load cleaned pickle file
            infile = Path(__file__).parent / f"mcmc_outfile_{prior_set}_{this_round-1}_clean.pkl"
            load_database = False
            with open(infile, "rb") as f:
                data = dill.load(f)["data"]
            # Override like_type to "gauss"
            like_type = "gauss"

        run_MCMC(
            data,
            num_iters, num_tune, num_cores, num_chains, num_samples,
            prior_set, like_type, load_database, filter_plx, this_round
        )

        # Seeing if outlier rejection is necessary
        if this_round == num_rounds:
            break  # No need to clean data after final MCMC run
        # Else: do outlier rejection
        clean.main(prior_set, this_round, filter_method)
        this_round += 1
        
    print(f"===\n{num_rounds} Bayesian MCMC runs complete\n=========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCMC stuff",
        prog="MCMC_w_dist_uncer.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "infile",
        type=str,
        help="Either database file or pickle file containing all relevant data",
    )
    parser.add_argument(
        "--num_cores", type=int, default=None, help="Maximum number of CPU cores to use"
    )
    parser.add_argument(
        "--num_chains", type=int, default=None, help="Number of parallel MCMC chains"
    )
    parser.add_argument(
        "--num_tune", type=int, default=2000, help="Number of tuning iterations"
    )
    parser.add_argument(
        "--num_iters", type=int, default=5000, help="Number of actual MCMC iterations"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of times to sample each parallax",
    )
    parser.add_argument(
        "--prior_set",
        type=str,
        default="A5",
        help="Prior set to use from Reid et al. 2019 (A1, A5, B, C, or D)",
    )
    parser.add_argument(
        "--like_type", type=str, default="gauss", help="Likelihood PDF (sivia or gauss)"
    )
    parser.add_argument(
        "--filter_plx",
        action="store_true",
        default=False,
        help="Filter sources with e_plx/plx > 0.2",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=1,
        help="Number of times to run Bayesian MCMC",
    )
    parser.add_argument(
        "--filter_method",
        type=str,
        default="sigma",
        help="Outlier rejection method for mcmc_cleanup.py (sigma or lnlike)",
    )
    parser.add_argument(
        "--this_round",
        type=int,
        default=1,
        help="Overrides default starting point for number of rounds. Useful if previous run failed & want to load from pickle file",
    )
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
        filter_plx=args["filter_plx"],
        num_rounds=args["num_rounds"],
        filter_method=args["filter_method"],
        this_round=args["this_round"],
    )