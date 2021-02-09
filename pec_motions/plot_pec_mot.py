"""
plot_pec_mot.py

Plots the peculiar (non-circular) motions of the sources,
colour-coded by ratio of radial to azimuthal velocity

Isaac Cheng - February 2021
"""
import sys
from pathlib import Path
import numpy as np
import dill
from scipy.special import logsumexp

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans
from universal_rotcurve import urc

# Roll angle between galactic midplane and galactocentric frame
_ROLL = 0.0  # deg (Anderson et al. 2019)
# Sun's height above galactic midplane (Reid et al. 2019)
_ZSUN = 5.5  # pc
# Useful constants
_RAD_TO_DEG = 57.29577951308232  # 180/pi (Don't forget to % 360 after)

def data_to_pm_and_vlsr(data, trace, free_Zsun=False, free_roll=False, free_Wpec=False):
    """
    Inputs:
      data :: pandas DataFrame
        Contains maser galactic longitudes, latitudes, right ascensions, declinations,
        parallaxes, equatorial proper motions, and LSR velocities
        with all associated uncertainties
      trace :: PyMC3 MultiTrace object
        Contains each iteration of the Bayesian MCMC algorithm for every parameter
      free_Zsun, free_roll, free_Wpec :: booleans (default: False)
        True iff Zsun, roll, or Wpec are free parameters in the model

    Returns: eqmux_pred, eqmuy_pred, vlsr_pred
      eqmux_pred :: Array of scalars (mas/yr)
        RA proper motion with cos(Declination) correction
      eqmuy_pred :: Array of scalars (mas/yr)
        Declination proper motion
      vlsr_pred :: Array of scalars (km/s)
        LSR velocity
    """
    # === Get optimal parameters from MCMC trace ===
    R0 = np.median(trace["R0"])  # kpc
    Vsun = np.median(trace["Vsun"])  # km/s
    Usun = np.median(trace["Usun"])  # km/s
    Wsun = np.median(trace["Wsun"])  # km/s
    Upec = np.median(trace["Upec"])  # km/s
    Vpec = np.median(trace["Vpec"])  # km/s
    a2 = np.median(trace["a2"])  # dimensionless
    a3 = np.median(trace["a3"])  # dimensionless
    Zsun = np.median(trace["Zsun"]) if free_Zsun else _ZSUN  # pc
    roll = np.median(trace["roll"]) if free_roll else _ROLL  # deg
    Wpec = np.median(trace["Wpec"]) if free_Wpec else 0.0  # km/s
    # # ? Set Upec = Vpec = 0 as per Reid et al. Fig. 6
    # Upec = 0.0  # km/s
    # Vpec = 0.0  # km/s

    # === Get data ===
    # Slice data into components (using np.asarray to prevent PyMC3 error with pandas)
    # ra = data["ra"].values  # deg
    # dec = data["dec"].values  # deg
    glon = data["glong"].values  # deg
    glat = data["glat"].values  # deg
    plx = data["plx"].values  # mas
    # e_plx = data["e_plx"].values  # mas

    # === Calculate predicted values from optimal parameters ===
    # Parallax to distance
    dist = trans.parallax_to_dist(plx)
    # Galactic to barycentric Cartesian coordinates
    bary_x, bary_y, bary_z = trans.gal_to_bary(glon, glat, dist)
    # Barycentric Cartesian to galactocentric Cartesian coodinates
    gcen_x, gcen_y, gcen_z = trans.bary_to_gcen(
        bary_x, bary_y, bary_z, R0=R0, Zsun=Zsun, roll=roll)
    # Galactocentric Cartesian frame to galactocentric cylindrical frame
    gcen_cyl_dist = np.sqrt(gcen_x * gcen_x + gcen_y * gcen_y)  # kpc
    azimuth = (np.arctan2(gcen_y, -gcen_x) * _RAD_TO_DEG) % 360  # deg in [0,360)
    v_circ_pred = urc(gcen_cyl_dist, a2=a2, a3=a3, R0=R0) + Vpec  # km/s
    v_rad = -Upec  # km/s
    Theta0 = urc(R0, a2=a2, a3=a3, R0=R0)  # km/s, LSR circular rotation speed

    # Go in reverse!
    # Galactocentric cylindrical to equatorial proper motions & LSR velocity
    eqmux_pred, eqmuy_pred, vlsr_pred = trans.gcen_cyl_to_pm_and_vlsr(
        gcen_cyl_dist, azimuth, gcen_z, v_rad, v_circ_pred, Wpec,
        R0=R0, Zsun=Zsun, roll=roll,
        Usun=Usun, Vsun=Vsun, Wsun=Wsun, Theta0=Theta0,
        use_theano=False)

    return eqmux_pred, eqmuy_pred, vlsr_pred


def main(prior_set, num_rounds):
    # Binary file to read
    # infile = Path(__file__).parent / "reid_MCMC_outfile.pkl"
    infile = Path(
        "/home/chengi/Documents/coop2021/bayesian_mcmc_rot_curve/"
        f"mcmc_outfile_{prior_set}_{num_rounds}.pkl"
    )

    with open(infile, "rb") as f:
        file = dill.load(f)
        data = file["data"]
        trace = file["trace"]
        # prior_set = file["prior_set"]  # "A1", "A5", "B", "C", "D"
        like_type = file["like_type"]  # "gauss", "cauchy", or "sivia"
        num_sources = file["num_sources"]
        num_samples = file["num_samples"]
        # reject_method = file["reject_method"] if num_rounds != 1 else None
        free_Zsun = file["free_Zsun"]
        free_roll = file["free_roll"]
        free_Wpec = file["free_Wpec"]

    print(f"=== Calculating BIC ({prior_set} priors & {num_rounds} MCMC rounds) ===")
    print("Number of sources:", num_sources)
    print("Likelihood function:", like_type)

    # === Extract parallax, proper motions & LSR velocity ===
    plx = data["plx"].values  # mas
    eqmux = data["mux"].values  # mas/yr (equatorial frame)
    e_eqmux = data["e_mux"].values  # mas/y (equatorial frame)
    eqmuy = data["muy"].values  # mas/y (equatorial frame)
    e_eqmuy = data["e_muy"].values  # mas/y (equatorial frame)
    vlsr = data["vlsr"].values  # km/s
    e_vlsr = data["e_vlsr"].values  # km/s
    eqmux_pred, eqmuy_pred, vlsr_pred = data_to_pm_and_vlsr(
        data, trace, free_Zsun=free_Zsun, free_roll=free_roll, free_Wpec=free_Wpec)

    # === Peculiar motions ===
    eqmux_pec = eqmux - eqmux_pred
    eqmuy_pec = eqmuy - eqmuy_pred
    vlsr_pec = vlsr - vlsr_pred

    # === Bayesian Information Criterion ===
    num_params = 8
    num_params += sum([free_Zsun, free_roll, free_Wpec])
    print("Number of parameters:", num_params)

    # sigma_eqmux, sigma_eqmuy, sigma_vlsr = get_sigmas(plx, e_eqmux, e_eqmuy, e_vlsr)

    # if like_type == "gauss":
    #     lnlike_eqmux = ln_gauss(eqmux, eqmux_pred, sigma_eqmux)
    #     lnlike_eqmuy = ln_gauss(eqmuy, eqmuy_pred, sigma_eqmuy)
    #     lnlike_vlsr = ln_gauss(vlsr, vlsr_pred, sigma_vlsr)
    # elif like_type == "cauchy":
    #     lnlike_eqmux = ln_cauchy(eqmux, eqmux_pred, sigma_eqmux)
    #     lnlike_eqmuy = ln_cauchy(eqmuy, eqmuy_pred, sigma_eqmuy)
    #     lnlike_vlsr = ln_cauchy(vlsr, vlsr_pred, sigma_vlsr)
    # else:  # like_type == "sivia"
    #     lnlike_eqmux = ln_siviaskilling(eqmux, eqmux_pred, sigma_eqmux)
    #     lnlike_eqmuy = ln_siviaskilling(eqmuy, eqmuy_pred, sigma_eqmuy)
    #     lnlike_vlsr = ln_siviaskilling(vlsr, vlsr_pred, sigma_vlsr)

    # # Joint likelihood
    # lnlike_tot = lnlike_eqmux + lnlike_eqmuy + lnlike_vlsr
    # # Marginalize over each distance samples
    # lnlike_sum = logsumexp(lnlike_tot, axis=0)
    # lnlike_avg = lnlike_sum - np.log(num_samples)
    # # Sum over sources
    # max_lnlike = lnlike_avg.sum()

    # bic = num_params * np.log(num_sources) - 2 * max_lnlike
    # print("Bayesian Information Criterion:", bic)


if __name__ == "__main__":
    prior_set_file = input("prior_set of file (A1, A5, B, C, D): ")
    num_rounds_file = int(input("round number of file to plot pec. motions (int): "))

    main(prior_set_file, num_rounds_file)