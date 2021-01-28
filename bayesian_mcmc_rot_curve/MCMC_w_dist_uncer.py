"""
Bayesian MCMC using priors from Reid et al. (2019)
"""
import sys
from pathlib import Path
import sqlite3
from contextlib import closing
import numpy as np
import pandas as pd
import theano.tensor as tt
import pymc3 as pm
import dill

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans
from universal_rotcurve import urc

# Universal rotation curve parameters (Persic et al. 1996)
_A_TWO = 0.96  # (Reid et al. 2019)
_A_THREE = 1.62  # (Reid et al. 2019)

# Sun's distance from galactic centre
_RSUN = 8.15  # kpc (Reid et al. 2019)

# Useful constants
_DEG_TO_RAD = 0.017453292519943295  # pi/180
_RAD_TO_DEG = 57.29577951308232  # 180/pi (Don't forget to % 360 after)
_AU_PER_YR_TO_KM_PER_S = 4.740470463533348  # from astropy (uses tropical year)
_KM_PER_KPC_S_TO_MAS_PER_YR = 0.21094952656969873  # (mas/yr) / (km/kpc/s)
_KPC_TO_KM = 3.085677581e16
_KM_TO_KPC = 3.24077929e-17


def get_data(db_file):
    """
    Retrieves all relevant data in Parallax table
    from database connection specified by db_file

    Returns DataFrame with:
        ra (deg), dec (deg), glong (deg), glat (deg), plx (mas), e_plx (mas),
        mux (mas/yr), muy (mas/yr), vlsr (km/s) + all proper motion/vlsr uncertainties
    """

    with closing(sqlite3.connect(db_file).cursor()) as cur:  # context manager, auto-close
        cur.execute(
            "SELECT ra, dec, glong, glat, plx, e_plx, mux, muy, vlsr, e_mux, e_muy, e_vlsr FROM Parallax"
        )
        data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    return data


def plx_to_peak_dist(plx, e_plx):
    """
    Computes peak (i.e. mode) of the distance distribution given the
    parallax & the uncertainty in the parallax (assuming the parallax is Gaussian)

    TODO: finish docstring
    """

    mean_dist = 1 / plx
    sigma_sq = e_plx * e_plx
    numerator = np.sqrt(8 * sigma_sq * mean_dist * mean_dist + 1) - 1
    denominator = 4 * sigma_sq * mean_dist

    return (numerator / denominator)


def get_weights(plx, e_plx, e_mux, e_muy, e_vlsr):
    """
    Calculates sigma values for proper motions and LSR velocity
    using Reid et al. (2014) weights

    Returns: sigma_mux, sigma_muy, sigma_vlsr

    TODO: finish docstring
    """

    # 1D Virial dispersion for stars in HMSFR w/ mass ~ 10^4 Msun w/in radius of ~ 1 pc
    sigma_vir = 5.0  # km/s

    # Parallax to peak distance
    dist = plx_to_peak_dist(plx, e_plx)

    weight_mux = tt.sqrt(
        e_mux * e_mux
        + sigma_vir * sigma_vir
        / (dist * dist)
        * _KM_PER_KPC_S_TO_MAS_PER_YR * _KM_PER_KPC_S_TO_MAS_PER_YR
    )
    weight_muy = tt.sqrt(
        e_muy * e_muy
        + sigma_vir * sigma_vir
        / (dist * dist)
        * _KM_PER_KPC_S_TO_MAS_PER_YR * _KM_PER_KPC_S_TO_MAS_PER_YR
    )
    weight_vlsr = tt.sqrt(e_vlsr * e_vlsr + sigma_vir * sigma_vir)

    return weight_mux, weight_muy, weight_vlsr


def ln_siviaskilling(x, mean, weight):
    """
    Returns the natural log of Sivia & Skilling's (2006) "Lorentzian-like" PDF.
    N.B. That the PDF is _not_ normalized.

    TODO: Finish docstring
    """

    residual = (x - mean) / weight
    return tt.log((1 - tt.exp(-0.5 * residual * residual)) / (residual * residual))


def run_MCMC(
    data, num_iters, num_tune, num_cores, num_chains,
    prior_set, like_type, num_samples, is_database_data, filter_parallax
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

    # Binary file to store MCMC output
    outfile = Path(__file__).parent / "MCMC_w_dist_uncer_outfile.pkl"

    if is_database_data:  # data is from database, need to filter data
        print("Starting with fresh data from database")
        # Create condition to filter data
        all_radii = trans.get_gcen_cyl_radius(data["glong"], data["glat"], data["plx"])
        # Bad data criteria (N.B. casting to array prevents "+" not supported warnings)
        if filter_parallax:  # Standard filtering used by Reid et al. (2019)
            print("Filter sources w/ R < 4 kpc & e_plx > 20%")
            bad = (np.array(all_radii) < 4.0) + (np.array(data["e_plx"] / data["plx"]) > 0.2)
        else:  # Only filter sources closer than 4 kpc to galactic centre
            print("Only filter sources w/ R < 4 kpc")
            bad = (np.array(all_radii) < 4.0)

        # Slice data into components (using np.asarray to prevent PyMC3 error with pandas)
        ra = data["ra"][~bad]  # deg
        dec = data["dec"][~bad]  # deg
        glon = data["glong"][~bad]  # deg
        glat = data["glat"][~bad]  # deg
        plx = data["plx"][~bad]  # mas
        e_plx = data["e_plx"][~bad]  # mas
        eqmux = np.asarray(data["mux"][~bad])  # mas/yr (equatorial frame)
        e_eqmux = np.asarray(data["e_mux"][~bad])  # mas/y (equatorial frame)
        eqmuy = np.asarray(data["muy"][~bad])  # mas/y (equatorial frame)
        e_eqmuy = np.asarray(data["e_muy"][~bad])  # mas/y (equatorial frame)
        vlsr = np.asarray(data["vlsr"][~bad])  # km/s
        e_vlsr = np.asarray(data["e_vlsr"][~bad])  # km/s

        # Store filtered data in DataFrame
        data = pd.DataFrame(
            {
                "ra": ra,
                "dec": dec,
                "glong": glon,
                "glat": glat,
                "plx": plx,
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
        # Slice data into components (using np.asarray to prevent PyMC3 error with pandas)
        # ra = data["ra"]  # deg
        # dec = data["dec"]  # deg
        glon = data["glong"]  # deg
        glat = data["glat"]  # deg
        plx = data["plx"]  # mas
        e_plx = data["e_plx"]  # mas
        eqmux = np.asarray(data["mux"])  # mas/yr (equatorial frame)
        e_eqmux = np.asarray(data["e_mux"])  # mas/y (equatorial frame)
        eqmuy = np.asarray(data["muy"])  # mas/y (equatorial frame)
        e_eqmuy = np.asarray(data["e_muy"])  # mas/y (equatorial frame)
        vlsr = np.asarray(data["vlsr"])  # km/s
        e_vlsr = np.asarray(data["e_vlsr"])  # km/s
    # Calculate number of sources used in fit
    num_sources = len(eqmux)
    print("Number of data points used:", num_sources)

    # Making array of random parallaxes. Columns are samples of the same source
    # plx = np.array([plx, ] * num_samples)
    plx = np.random.normal(loc=plx, scale=e_plx, size=(num_samples, num_sources))
    e_plx = np.array([e_plx, ] * num_samples)
    glon = np.array([glon,] * num_samples)  # num_samples by num_sources
    glat = np.array([glat,] * num_samples)  # num_samples by num_sources
    eqmux = np.array([eqmux,] * num_samples)
    eqmuy = np.array([eqmuy,] * num_samples)
    vlsr = np.array([vlsr,] * num_samples)
    e_eqmux = np.array([e_eqmux,] * num_samples)
    e_eqmuy = np.array([e_eqmuy,] * num_samples)
    e_vlsr = np.array([e_vlsr,] * num_samples)

    # Parallax to distance
    gdist = trans.parallax_to_dist(plx)
    # Galactic to galactocentric Cartesian coordinates
    bary_x, bary_y, bary_z = trans.gal_to_bary(glon, glat, gdist)
    # Zero vertical velocity in URC
    v_vert = 0.0

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
            a2 = pm.Uniform("a2", lower=0.8, upper=1.2)  # dimensionless
            a3 = pm.Uniform("a3", lower=1.5, upper=1.7)  # dimensionless
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
        print("Using prior set", prior_set)

        # === Predicted values (using data) ===
        # Barycentric Cartesian to galactocentric Cartesian coodinates
        gcen_x, gcen_y, gcen_z = trans.bary_to_gcen(bary_x, bary_y, bary_z, R0=R0)
        # Galactocentric Cartesian frame to galactocentric cylindrical frame
        gcen_cyl_dist = tt.sqrt(gcen_x * gcen_x + gcen_y * gcen_y)  # kpc
        azimuth = (tt.arctan2(gcen_y, -gcen_x) * _RAD_TO_DEG) % 360  # deg in [0,360)
        v_circ_pred = urc(gcen_cyl_dist, a2=a2, a3=a3, R0=R0) + Vpec  # km/s
        v_rad = -1 * Upec  # km/s, negative bc toward GC
        Theta0 = urc(R0, a2=a2, a3=a3, R0=R0)  # km/s, LSR circular rotation speed

        # Go in reverse!
        # Galactocentric cylindrical to equatorial proper motions & LSR velocity
        (eqmux_pred, eqmuy_pred, vlsr_pred) = trans.gcen_cyl_to_pm_and_vlsr(
            gcen_cyl_dist,
            azimuth,
            gcen_z,
            v_rad,
            v_circ_pred,
            v_vert,
            R0=R0,
            Usun=Usun,
            Vsun=Vsun,
            Wsun=Wsun,
            Theta0=Theta0,
            use_theano=True,
        )

        # === Likelihood components (sigma values are from observed data) ===
        # Calculating uncertainties for likelihood function
        # Using Reid et al. (2014) weights as uncertainties
        # sigma_vir = 5.0  # km/s
        # weight_eqmux = tt.sqrt(
        #     e_eqmux * e_eqmux
        #     + sigma_vir * sigma_vir
        #     * plx * plx
        #     * _KM_PER_KPC_S_TO_MAS_PER_YR * _KM_PER_KPC_S_TO_MAS_PER_YR
        # )
        # weight_eqmuy = tt.sqrt(
        #     e_eqmuy * e_eqmuy
        #     + sigma_vir * sigma_vir
        #     * plx * plx
        #     * _KM_PER_KPC_S_TO_MAS_PER_YR * _KM_PER_KPC_S_TO_MAS_PER_YR
        # )
        # weight_vlsr = tt.sqrt(e_vlsr * e_vlsr + sigma_vir * sigma_vir)
        weight_eqmux, weight_eqmuy, weight_vlsr = get_weights(
            plx, e_plx, e_eqmux, e_eqmuy, e_vlsr
        )

        # Making array of likelihood values evaluated at data points
        if like_type == "gauss":
            # GAUSSIAN MIXTURE PDF
            print("Using Gaussian PDF")
            lnlike_eqmux = pm.Normal.dist(mu=eqmux_pred, sigma=weight_eqmux).logp(eqmux)
            lnlike_eqmuy = pm.Normal.dist(mu=eqmuy_pred, sigma=weight_eqmuy).logp(eqmuy)
            lnlike_vlsr = pm.Normal.dist(mu=vlsr_pred, sigma=weight_vlsr).logp(vlsr)
            # Save exponential part of log-likelihood distributions to trace
            lnlike_eqmux_dist = pm.Deterministic(
                "lnlike_eqmux_dist", lnlike_eqmux * e_eqmux * tt.sqrt(2 * np.pi)
            )
            lnlike_eqmuy_dist = pm.Deterministic(
                "lnlike_eqmuy_dist", lnlike_eqmuy * e_eqmuy * tt.sqrt(2 * np.pi)
            )
            lnlike_vlsr_dist = pm.Deterministic(
                "lnlike_vlsr_dist", lnlike_vlsr * e_vlsr * tt.sqrt(2 * np.pi)
            )
        elif like_type == "cauchy":
            # CAUCHY PDF
            print("Using Cauchy PDF")
            hwhm = tt.sqrt(2 * tt.log(2))  # half width at half maximum
            lnlike_eqmux = pm.Cauchy.dist(
                alpha=eqmux_pred, beta=hwhm * weight_eqmux
            ).logp(eqmux)
            lnlike_eqmuy = pm.Cauchy.dist(
                alpha=eqmuy_pred, beta=hwhm * weight_eqmuy
            ).logp(eqmuy)
            lnlike_vlsr = pm.Cauchy.dist(alpha=vlsr_pred, beta=hwhm * weight_vlsr).logp(
                vlsr
            )
        elif like_type == "sivia":
            # SIVIA & SKILLING (2006) "LORENTZIAN-LIKE" CONSERVATIVE PDF
            print("Using Sivia & Skilling (2006) PDF")
            lnlike_eqmux = ln_siviaskilling(eqmux, eqmux_pred, weight_eqmux)
            lnlike_eqmuy = ln_siviaskilling(eqmuy, eqmuy_pred, weight_eqmuy)
            lnlike_vlsr = ln_siviaskilling(vlsr, vlsr_pred, weight_vlsr)
            # Save log-likelihood distributions to trace w/out impacting model
            lnlike_eqmux_dist = pm.Deterministic("lnlike_eqmux_dist", lnlike_eqmux)
            lnlike_eqmuy_dist = pm.Deterministic("lnlike_eqmuy_dist", lnlike_eqmuy)
            lnlike_vlsr_dist = pm.Deterministic("lnlike_vlsr_dist", lnlike_vlsr)
        else:
            raise ValueError(
                "Invalid like_type. Please input 'gauss', 'cauchy', or 'sivia'."
            )

        # lnlike_tot = pm.Deterministic("lnlike_tot", lnlike_eqmux + lnlike_eqmuy + lnlike_vlsr)

        # === Full likelihood function (specified by log-probability) ===
        likelihood = pm.Potential(
            "likelihood",
            (lnlike_eqmux + lnlike_eqmuy + lnlike_vlsr).mean(
                axis=0
            ),  # Take avg of all samples per source
        )  # expects values instead of function

        # Run MCMC
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
                var_names=[
                    "~lnlike_eqmux_dist",
                    "~lnlike_eqmuy_dist",
                    "~lnlike_vlsr_dist",
                ],
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
                },
                f,
            )


def main():
    # Specify Bayesian MCMC parameters
    _NUM_ITERS = 2000  # number of iterations per chain
    _NUM_TUNE = 2000  # number of tuning iterations (will be thrown away)
    _NUM_CORES = 10  # number of CPU cores to use for MCMC
    _NUM_CHAINS = 10  # number of parallel chains to run
    _PRIOR_SET = "A5"  # Prior set from Reid et al. (2019)
    _LIKELIHOOD_TYPE = "sivia"  # "gauss", "cauchy", or "sivia"
    _NUM_SAMPLES = 1000  # number of times to sample each parallax
    _FILTER_PARALLAX = False  # only matters if _LIKELIHOOD_TYPE == "sivia" or "cauchy"
                            # If False, only remove database sources w/ R < 4 kpc

    # If data has already been filtered & using same prior set
    if _LIKELIHOOD_TYPE == "gauss":
        _LOAD_DATABASE = False  # Use data from pickle file
    # If data has not been filtered & using new prior set
    elif _LIKELIHOOD_TYPE == "cauchy" or _LIKELIHOOD_TYPE == "sivia":
        _LOAD_DATABASE = True  # Use data from database
    else:
        raise ValueError(
            "Invalid _LIKELIHOOD_TYPE. Please choose 'gauss', 'cauchy', or 'sivia'."
        )

    if _LOAD_DATABASE:
        # # Specifying database file name & folder
        # filename = Path("data/hii_v2_20201203.db")
        # # Database folder in parent directory of this script (call .parent twice)
        # db = Path(__file__).parent.parent / filename

        # Specifying absolute file path instead
        # (allows file to be run in multiple locations as long as database location does not move)
        db = Path("/home/chengi/Documents/coop2021/data/hii_v2_20201203.db")
        data = get_data(db)  # all data from Parallax table
    else:
        # Load data from pickle file
        # infile = Path(__file__).parent / "MCMC_w_dist_uncer_outfile.pkl"
        infile = Path(
            "/home/chengi/Documents/coop2021/bayesian_mcmc_rot_curve/MCMC_w_dist_uncer_outfile.pkl"
        )

        with open(infile, "rb") as f:
            data = dill.load(f)["data"]

    # Run simulation
    run_MCMC(
        data,
        _NUM_ITERS, _NUM_TUNE, _NUM_CORES, _NUM_CHAINS,
        _PRIOR_SET, _LIKELIHOOD_TYPE, _NUM_SAMPLES, _LOAD_DATABASE, _FILTER_PARALLAX,
    )


if __name__ == "__main__":
    main()
