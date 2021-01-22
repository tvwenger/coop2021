"""
Bayesian MCMC using priors from Reid et al. (2019)
"""
from inspect import Attribute
import sys
from pathlib import Path
import sqlite3
from contextlib import closing
import numpy as np
import matplotlib.pyplot as plt
import corner
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
_KM_PER_S_TO_AU_PER_YR = 0.21094952656969873  # from astropy (uses tropical year)
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


def run_MCMC(data, num_iters, num_tune, num_chains, prior_set, like_type):
    """
    Runs Bayesian MCMC. Returns trace
    
    TODO: finish docstring
    """

    # Binary file to store MCMC output
    outfile = Path(__file__).parent / "prior_reid_outfile.pkl"

    # Create condition to filter data
    all_radii = trans.get_gcen_cyl_radius(data["glong"], data["glat"], data["plx"])
    # Bad data criteria (N.B. casting to array prevents "+" not supported warnings)
    bad = (np.array(all_radii) < 4.0) + (np.array(data["e_plx"] / data["plx"]) > 0.2)

    # Slice data into components (using np.asarray to prevent PyMC3 error with pandas)
    # ra = data["ra"][~bad]  # deg
    # dec = data["dec"][~bad]  # deg
    glon = data["glong"][~bad]  # deg
    glat = data["glat"][~bad]  # deg
    plx = data["plx"][~bad]  # mas
    # e_plx = data["e_plx"][~bad]  # mas
    eqmux = np.asarray(data["mux"][~bad])  # mas/yr (equatorial frame)
    e_eqmux = np.asarray(data["e_mux"][~bad])  # mas/y (equatorial frame)
    eqmuy = np.asarray(data["muy"][~bad])  # mas/y (equatorial frame)
    e_eqmuy = np.asarray(data["e_muy"][~bad])  # mas/y (equatorial frame)
    vlsr = np.asarray(data["vlsr"][~bad])  # km/s
    e_vlsr = np.asarray(data["e_vlsr"][~bad])  # km/s
    num_sources = len(eqmux)
    print("Number of data points used:", num_sources)

    # Parallax to distance
    gdist = trans.parallax_to_dist(plx)
    # Galactic to galactocentric Cartesian coordinates
    bary_x, bary_y, bary_z = trans.gal_to_bary(glon, glat, gdist)
    # Zero vertical velocity in URC
    # ? Maybe make this Gaussian in future?
    # v_vert = np.zeros(num_sources, float)  # km/s
    v_vert = 0.0

    # 8 parameters from Reid et al. (2019): (see Section 4 & Table 3)
    #   R0, Usun, Vsun, Wsun, Upec, Vpec, a2, a3
    model_obj = pm.Model()
    with model_obj:
        if prior_set == "A":  # Make model with SET A PRIORS
            # R0 = pm.Uniform("R0", lower=0, upper=np.inf)  # kpc
            R0 = pm.Uniform("R0", lower=7.0, upper=10.0)  # kpc
            Usun = pm.Normal("Usun", mu=11.1, sigma=1.2)  # km/s
            Vsun = pm.Normal("Vsun", mu=15.0, sigma=10.0)  # km/s
            Wsun = pm.Normal("Wsun", mu=7.2, sigma=1.1)  # km/s
            Upec = pm.Normal("Upec", mu=3.0, sigma=10.0)  # km/s
            Vpec = pm.Normal("Vpec", mu=-3.0, sigma=10.0)  # km/s
            a2 = pm.Uniform("a2", lower=0.5, upper=1.5)  # dimensionless
            a3 = pm.Uniform("a3", lower=1.5, upper=1.7)  # dimensionless
        else:
            raise ValueError("Illegal prior_set. Only supports 'A' so far.")

        # === Predicted values (using data) ===
        gcen_x, gcen_y, gcen_z = trans.bary_to_gcen(bary_x, bary_y, bary_z, R0=R0)
        # Galactocentric Cartesian coordinates to galactocentric cylindrical distance
        gcen_cyl_dist = tt.sqrt(gcen_x * gcen_x + gcen_y * gcen_y)  # kpc
        azimuth = (tt.arctan2(gcen_y, -gcen_x) * _RAD_TO_DEG) % 360  # deg in [0,360)
        # height = gcen_z  # kpc
        v_circ_pred = urc(gcen_cyl_dist, a2=a2, a3=a3, R0=R0) + Upec  # km/s
        v_rad = Vpec  # km/s

        # Go in reverse!
        # Galactocentric cylindrical to galactocentric Cartesian
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
        )

        # "Lorenztian-like" conservative PDF by Sivia and Skilling (2006)
        # ? sigma_eqmux = ((eqmux - eqmux_pred) / e_eqmux) * ((eqmux - eqmux_pred) / e_eqmux)
        #
        # === Likelihood function (only works in PyMC v3.7) ===
        # likelihood_eqmux = pm.Normal(
        #     "likelihood_eqmux", mu=eqmux_pred, sigma=e_eqmux, observed=eqmux
        # )
        # likelihood_eqmuy = pm.Normal(
        #     "likelihood_eqmux", mu=eqmuy_pred, sigma=e_eqmuy, observed=eqmuy
        # )
        # likelihood_vlsr = pm.Normal(
        #     "likelihood_eqmux", mu=vlsr_pred, sigma=e_vlsr, observed=vlsr
        # )
        #
        # # Joint likelihood (Gaussian mixture model)
        # joint_lnlike = (
        #     lambda lnlike_eqmux, lnlike_eqmuy, lnlike_vlsr: lnlike_eqmux
        #     + lnlike_eqmuy
        #     + lnlike_vlsr
        # )
        # # Observed values ?
        # observed = {
        #     "lnlike_eqmux": lnlike_eqmux,
        #     "lnlike_eqmuy": lnlike_eqmuy,
        #     "lnlike_vlsr": lnlike_vlsr,
        # }
        # Full likelihood function
        # likelihood = pm.DensityDist('likelihood', logp=joint_lnlike, observed=observed)

        # === Likelihood components (sigma values are from observed data) ===
        # Returns array of likelihood values evaluated at data points
        if like_type == "gaussian":
            # GAUSSIAN MIXTURE MODEL
            print("Using Gaussian PDF")
            lnlike_eqmux = pm.Normal.dist(mu=eqmux_pred, sigma=e_eqmux).logp(eqmux)
            lnlike_eqmuy = pm.Normal.dist(mu=eqmuy_pred, sigma=e_eqmuy).logp(eqmuy)
            lnlike_vlsr = pm.Normal.dist(mu=vlsr_pred, sigma=e_vlsr).logp(vlsr)
        elif like_type == "cauchy":
            # LORENTZIAN-LIKE CONSERVATIVE PDF
            print("Using Lorentzian-like PDF")
            hwhm = tt.sqrt(2 * tt.log(2))  # half width at half maximum
            lnlike_eqmux = pm.Cauchy.dist(alpha=eqmux_pred, beta=hwhm * e_eqmux).logp(
                eqmux
            )
            lnlike_eqmuy = pm.Cauchy.dist(alpha=eqmuy_pred, beta=hwhm * e_eqmuy).logp(
                eqmuy
            )
            lnlike_vlsr = pm.Cauchy.dist(alpha=vlsr_pred, beta=hwhm * e_vlsr).logp(vlsr)
        else:
            raise ValueError("Invalid like_type. Please input 'gaussian' or 'cauchy'.")

        # === Full likelihood function (specified by log-probability) ===
        likelihood = pm.Potential(
            "likelihood", lnlike_eqmux + lnlike_eqmuy + lnlike_vlsr
        )  # expects values instead of function

        # Run MCMC
        trace = pm.sample(
            num_iters,
            init="advi",
            tune=num_tune,
            cores=2,
            chains=num_chains,
            return_inferencedata=False,
        )  # walker

        # See results (calling within model to prevent FutureWarning)
        print(pm.summary(trace))
        # Save results to pickle file
        with open(outfile, "wb") as f:
            dill.dump(
                {
                    "data": data,
                    "model_obj": model_obj,
                    "trace": trace,
                    "like_type": like_type,
                },
                f,
            )

    return trace


def plot_MCMC(trace, like_type):
    """
    Plots walkers and corner plot from trace. Returns None
    """
    if like_type != "gaussian" and like_type != "cauchy":
        raise ValueError("Invalid like_type. Please select 'gaussian' or 'cauchy'.")

    sample_lst = []
    varnames = []

    # Get names of variables & data associated with each variable
    for varname in trace.varnames:
        if "interval" in varname:
            continue  # do not want to include non user-defined parameters
        varnames.append(varname)
        sample_lst.append(trace[varname])
    samples = np.array(sample_lst)

    num_iters = len(trace)
    num_chains = len(trace.chains)
    print("varnames:", varnames)
    print("samples shape", np.shape(samples))

    # === Plot MCMC chains for each parameter ===
    # Reshape to (# params, # chains, # iter per chain)
    param_lst = [param.reshape((num_chains, num_iters)) for param in samples]
    print("param_lst shape", np.shape(param_lst))

    # Make # subplots same as # params & make figure twice as tall as is wide
    fig1, axes1 = plt.subplots(np.shape(param_lst)[0], figsize=plt.figaspect(2))

    for ax, parameter, varname in zip(axes1, param_lst, varnames):
        for chain in parameter:
            ax.plot(chain, "k-", alpha=0.5, linewidth=0.5)  # plot chains of parameter
        ax.set_title(varname, fontsize=8)  # add parameter name as title
        # Make x & y ticks smaller
        ax.tick_params(axis="both", which="major", labelsize=5)
        ax.tick_params(axis="both", which="minor", labelsize=3)

    if like_type == "gaussian":
        fig1.suptitle(
            f"MCMC walkers: {num_chains} chains each with {num_iters} iterations\n(Gaussian PDF)",
            fontsize=10,
        )
        fig1.tight_layout()  # Need this below suptitle()
        fig1.savefig(
            Path(__file__).parent / "prior_reid_chains_gauss.jpg",
            format="jpg",
            dpi=300,
            bbox_inches="tight",
        )
    else:  # like_type == "cauchy"
        fig1.suptitle(
            f"MCMC walkers: {num_chains} chains each with {num_iters} iterations\n(Lorentzian-Like PDF)",
            fontsize=10,
        )
        fig1.tight_layout()  # Need this below suptitle()
        fig1.savefig(
            Path(__file__).parent / "prior_reid_chains_lorentz.jpg",
            format="jpg",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()

    # Plot histogram of parameters
    fig2 = corner.corner(
        samples.T,
        labels=varnames,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".2f",
    )
    if like_type == "gaussian":
        fig2.savefig(
            Path(__file__).parent / "prior_reid_histogram_gauss.jpg",
            format="jpg",
            dpi=300,
            bbox_inches="tight",
        )
    else:  # like_type == "cauchy"
        fig2.savefig(
            Path(__file__).parent / "prior_reid_histogram_lorentz.jpg",
            format="jpg",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def main():
    # # Specifying database file name & folder
    # filename = Path("data/hii_v2_20201203.db")
    # # Database folder in parent directory of this script (call .parent twice)
    # db = Path(__file__).parent.parent / filename

    # Specifying absolute file path instead
    # (allows file to be run in multiple locations as long as database location does not move)
    db = Path("/home/chengi/Documents/coop2021/data/hii_v2_20201203.db")

    # Get data + put into DataFrame
    data = get_data(db)  # all data from Parallax table
    # print(data.to_markdown())

    _NUM_ITERS = 2000  # number of iterations per chain
    _NUM_TUNE = 2000  # number of tuning iterations (will be thrown away)
    _NUM_CHAINS = 2  # number of parallel chains to run
    _LIKELIHOOD_TYPE = "cauchy"  # "gaussian" or "cauchy"

    # Run simulation
    trace = run_MCMC(data, _NUM_ITERS, _NUM_TUNE, _NUM_CHAINS, "A", _LIKELIHOOD_TYPE)

    # Plot results
    plot_MCMC(trace, _LIKELIHOOD_TYPE)


if __name__ == "__main__":
    main()
