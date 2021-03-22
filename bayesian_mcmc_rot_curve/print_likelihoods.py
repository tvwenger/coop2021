"""
print_likelihoods.py

Prints a table of source names, coordinates, proper motions & vlsr,
and log-likelihoods for each source using best-fit parameters from an MCMC run.

Isaac Cheng - March 2021
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import dill
import matplotlib.pyplot as plt

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


def plx_to_peak_dist(plx, e_plx):
    """
    Computes peak of distance distribution given the
    parallax & the uncertainty in the parallax (assuming the parallax is Gaussian)

    TODO: finish docstring
    """

    mean_dist = 1 / plx
    sigma_sq = e_plx * e_plx
    return (np.sqrt(8 * sigma_sq * mean_dist * mean_dist + 1) - 1) / (
        4 * sigma_sq * mean_dist
    )


def get_sigmas(dist, e_mux, e_muy, e_vlsr):
    """
    Calculates sigma values for proper motions and LSR velocity
    using Reid et al. (2014) weights

    Returns: sigma_mux, sigma_muy, sigma_vlsr

    TODO: finish docstring
    """

    km_kps_s_to_mas_yr = 0.21094952656969873  # (mas/yr) / (km/kpc/s)

    # 1D Virial dispersion for stars in HMSFR w/ mass ~ 10^4 Msun w/in radius of ~ 1 pc
    sigma_vir_sq = 25.0  # km/s

    # Parallax to reciprocal of distance^2 (i.e. 1 / distance^2)
    reciprocal_dist_sq = km_kps_s_to_mas_yr * km_kps_s_to_mas_yr / (dist * dist)

    sigma_mux = np.sqrt(e_mux * e_mux + sigma_vir_sq * reciprocal_dist_sq)
    sigma_muy = np.sqrt(e_muy * e_muy + sigma_vir_sq * reciprocal_dist_sq)
    sigma_vlsr = np.sqrt(e_vlsr * e_vlsr + sigma_vir_sq)

    return sigma_mux, sigma_muy, sigma_vlsr


def ln_gauss_norm(x, mean, sigma):
    """
    Calculates the ln of the exponential part of a normal distribution
    i.e. returns -0.5 * (x-mean)^2 / sigma^2.
    Use this for plotting.

    TODO: finish docstring
    """

    return -0.5 * (x - mean) * (x - mean) / (sigma * sigma)


def calc_lnlikes():
    """
    Calculates log-likelihoods of all sources based on peak of distance PDF
    """
    # tracefile = Path(__file__).parent / "mcmc_outfile_A1_100dist_5.pkl"
    # with open(tracefile, "rb") as f:
    #     trace = dill.load(f)["trace"]
    datafile = (
        Path(__file__).parent.parent
        / "pec_motions/100dist_meanUpecVpec_cauchyOutlierRejection_peakDist.csv"
    )
    data = pd.read_csv(datafile)
    # # Mean values from 100 distance, mean Upec/Vpec trace, Cauchy outlier rejection file
    R0, Zsun, roll = 8.181364, 5.5833244, 0.009740928
    a2, a3 = 0.97133905, 1.6247351
    Usun, Vsun, Wsun = 10.406719, 10.212576, 8.077657
    Upec, Vpec = 4.429875, -4.8232403
    Wpec = 0.0  # km/s
    # Usun = np.mean(trace["Usun"])  # km/s
    # Vsun = np.mean(trace["Vsun"])  # km/s
    # Wsun = np.mean(trace["Wsun"])  # km/s
    # Upec = np.mean(trace["Upec"])
    # Vpec = np.mean(trace["Vpec"])
    # print(Usun, Vsun, Wsun)
    # print(Upec, Vpec)

    # === Get data ===
    # Slice data into components (using np.asarray to prevent PyMC3 error with pandas)
    # gname = data["gname"].values  # deg
    # alias = data["alias"].values  # deg
    glon = data["glong"].values  # deg
    glat = data["glat"].values  # deg
    plx = data["plx"].values  # mas
    e_plx = data["e_plx"].values  # mas
    eqmux = data["mux"].values  # mas/yr (equatorial frame)
    e_eqmux = data["e_mux"].values  # mas/y (equatorial frame)
    eqmuy = data["muy"].values  # mas/y (equatorial frame)
    e_eqmuy = data["e_muy"].values  # mas/y (equatorial frame)
    vlsr = data["vlsr"].values  # km/s
    e_vlsr = data["e_vlsr"].values  # km/s
    is_tooclose = data["is_tooclose"].values
    is_outlier = data["is_outlier"].values

    # === Calculate predicted values from optimal parameters ===
    # Parallax to distance
    gdist = trans.parallax_to_dist(plx, e_parallax=e_plx)
    # Galactic to barycentric Cartesian coordinates
    bary_x, bary_y, bary_z = trans.gal_to_bary(glon, glat, gdist)
    # Barycentric Cartesian to galactocentric Cartesian coodinates
    gcen_x, gcen_y, gcen_z = trans.bary_to_gcen(
        bary_x, bary_y, bary_z, R0=R0, Zsun=Zsun, roll=roll
    )
    # Galactocentric Cartesian frame to galactocentric cylindrical frame
    gcen_cyl_dist = np.sqrt(gcen_x * gcen_x + gcen_y * gcen_y)  # kpc
    azimuth = (np.arctan2(gcen_y, -gcen_x) * _RAD_TO_DEG) % 360  # deg in [0,360)
    v_circ_pred = urc(gcen_cyl_dist, a2=a2, a3=a3, R0=R0) + Vpec  # km/s
    v_rad = -Upec  # km/s
    Theta0 = urc(R0, a2=a2, a3=a3, R0=R0)  # km/s, LSR circular rotation speed

    # Go in reverse!
    # Galactocentric cylindrical to equatorial proper motions & LSR velocity
    eqmux_pred, eqmuy_pred, vlsr_pred = trans.gcen_cyl_to_pm_and_vlsr(
        gcen_cyl_dist,
        azimuth,
        gcen_z,
        v_rad,
        v_circ_pred,
        Wpec,
        R0=R0,
        Zsun=Zsun,
        roll=roll,
        Usun=Usun,
        Vsun=Vsun,
        Wsun=Wsun,
        Theta0=Theta0,
        use_theano=False,
    )

    sigma_eqmux, sigma_eqmuy, sigma_vlsr = get_sigmas(gdist, e_eqmux, e_eqmuy, e_vlsr)

    lnlike_eqmux = ln_gauss_norm(eqmux, eqmux_pred, sigma_eqmux)
    lnlike_eqmuy = ln_gauss_norm(eqmuy, eqmuy_pred, sigma_eqmuy)
    lnlike_vlsr = ln_gauss_norm(vlsr, vlsr_pred, sigma_vlsr)

    df = pd.DataFrame(
        {
            "gname": data["gname"],
            "alias": data["alias"],
            "glong": data["glong"],
            "glat": data["glat"],
            "plx": data["plx"],
            "e_plx": data["e_plx"],
            "x": gcen_y,
            "y": -gcen_x,
            "z": gcen_z,
            "mux": data["mux"],
            "e_mux": data["e_mux"],
            "weight_mux": sigma_eqmux,
            "mux_pred": eqmux_pred,
            "lnlike_mux": lnlike_eqmux,
            "muy": data["muy"],
            "e_muy": data["e_muy"],
            "weight_muy": sigma_eqmuy,
            "muy_pred": eqmuy_pred,
            "lnlike_muy": lnlike_eqmuy,
            "vlsr": data["vlsr"],
            "e_vlsr": data["e_vlsr"],
            "weight_vlsr": sigma_vlsr,
            "vlsr_pred": vlsr_pred,
            "lnlike_vlsr": lnlike_vlsr,
            "Upec": data["Upec"],
            "Vpec": data["Vpec"],
            "Wpec": data["Wpec"],
            "is_tooclose": is_tooclose,
            "is_outlier": is_outlier,  # MCMC outlier
        }
    )
    df.to_csv(
        path_or_buf=Path(__file__).parent / Path("log_likelihoods_normed.csv"),
        sep=",",
        index=False,
        header=True,
    )


def main():
    calc_lnlikes()


if __name__ == "__main__":
    main()
