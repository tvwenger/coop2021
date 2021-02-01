"""
Cleans up Bayesian MCMC data from pickle file
"""
from multiprocessing import Value
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

# Useful constants
_RAD_TO_DEG = 57.29577951308232  # 180/pi (Don't forget to % 360 after)
_KM_PER_KPC_S_TO_MAS_PER_YR = 0.21094952656969873  # (mas/yr) / (km/kpc/s)
_LN_SQRT_2PI = 0.918938533


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


def get_sigmas(plx, e_plx, e_mux, e_muy, e_vlsr):
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

    sigma_mux = np.sqrt(
        e_mux * e_mux
        + sigma_vir * sigma_vir
        / (dist * dist)
        * _KM_PER_KPC_S_TO_MAS_PER_YR * _KM_PER_KPC_S_TO_MAS_PER_YR
    )
    sigma_muy = np.sqrt(
        e_muy * e_muy
        + sigma_vir * sigma_vir
        / (dist * dist)
        * _KM_PER_KPC_S_TO_MAS_PER_YR * _KM_PER_KPC_S_TO_MAS_PER_YR
    )
    sigma_vlsr = np.sqrt(e_vlsr * e_vlsr + sigma_vir * sigma_vir)

    return sigma_mux, sigma_muy, sigma_vlsr


def ln_siviaskilling(x, mean, weight):
    """
    Returns the natural log of Sivia & Skilling's (2006) "Lorentzian-like" PDF.
    N.B. That the PDF is _not_ normalized.

    TODO: Finish docstring
    """

    residual = (x - mean) / weight
    return np.log((1 - np.exp(-0.5 * residual * residual)) / (residual * residual))


def ln_gauss_exp_part(x, mean, sigma):
    """
    Calculates the ln of the exponential part of a normal distribution
    i.e. returns -0.5 * (x-mean)^2/sigma^2
    """
    return -0.5 * (x - mean) * (x - mean) / sigma / sigma


def cleanup_data(data, trace, like_type, filter_method):
    """
    Cleans up data from pickle file
    (i.e. removes any sources with proper motion or vlsr > 3 sigma from predicted values)

    Returns:
      pandas DataFrame with cleaned up data
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

    # === Get data ===
    # Slice data into components (using np.asarray to prevent PyMC3 error with pandas)
    ra = data["ra"]  # deg
    dec = data["dec"]  # deg
    glon = data["glong"]  # deg
    glat = data["glat"]  # deg
    plx = data["plx"]  # mas
    e_plx = data["e_plx"]  # mas
    eqmux = data["mux"]  # mas/yr (equatorial frame)
    e_eqmux = data["e_mux"]  # mas/y (equatorial frame)
    eqmuy = data["muy"]  # mas/y (equatorial frame)
    e_eqmuy = data["e_muy"]  # mas/y (equatorial frame)
    vlsr = data["vlsr"]  # km/s
    e_vlsr = data["e_vlsr"]  # km/s

    # === Calculate predicted values from optimal parameters ===
    # Parallax to distance
    gdist = trans.parallax_to_dist(plx)
    # Galactic to barycentric Cartesian coordinates
    bary_x, bary_y, bary_z = trans.gal_to_bary(glon, glat, gdist)
    # Barycentric Cartesian to galactocentric Cartesian coodinates
    gcen_x, gcen_y, gcen_z = trans.bary_to_gcen(bary_x, bary_y, bary_z, R0=R0)
    # Galactocentric Cartesian frame to galactocentric cylindrical frame
    gcen_cyl_dist = np.sqrt(gcen_x * gcen_x + gcen_y * gcen_y)  # kpc
    azimuth = (np.arctan2(gcen_y, -gcen_x) * _RAD_TO_DEG) % 360  # deg in [0,360)
    v_circ_pred = urc(gcen_cyl_dist, a2=a2, a3=a3, R0=R0) + Vpec  # km/s
    v_rad = -Upec  # km/s
    v_vert = 0.0  # km/s, zero vertical velocity in URC
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
        use_theano=False,
    )

    # Calculating conditions for data cleaning
    # Reject all data w/ residuals larger than 3 sigma
    if filter_method == "sigma":
        print("Using Reid et al. (2014) weights to reject outliers")
        sigma_eqmux, sigma_eqmuy, sigma_vlsr = get_sigmas(
            plx, e_plx, e_eqmux, e_eqmuy, e_vlsr
        )

        # Throw away data with proper motion or vlsr residuals > 3 sigma
        bad_sigma = (
            (np.array(abs(eqmux_pred - eqmux) / sigma_eqmux) > 3)
            + (np.array(abs(eqmuy_pred - eqmuy) / sigma_eqmuy) > 3)
            + (np.array(abs(vlsr_pred - vlsr) / sigma_vlsr) > 3)
        )
    elif filter_method == "lnlike":
        print("Using log-likelihood to reject data")
        # Take median of all likelihood values per source (i.e. median of a whole "sheet")
        # Shape of distributions is (# MC iters, # samples, # sources)
        ln_eqmux_pred = np.median(trace["lnlike_eqmux_dist"], axis=(0, 1))
        ln_eqmuy_pred = np.median(trace["lnlike_eqmuy_dist"], axis=(0, 1))
        ln_vlsr_pred = np.median(trace["lnlike_vlsr_dist"], axis=(0, 1))
        print("min ln_mux:", min(ln_eqmux_pred))
        print("min ln_muy:", min(ln_eqmuy_pred))
        print("min ln_vlsr:", min(ln_vlsr_pred))

        if like_type == "gauss":  # Gaussian distribution of proper motions / vlsr
            ln_threshold = -4.5  # ln(exponential part) = -(3^2)/2
        elif like_type == "sivia":  # Lorentzian-like distribution of prop. motions / vlsr
            ln_threshold = -2.2 # ln((1-exp(-(3^2)/2)) / (3^2))
        else:
            raise ValueError(
                "Invalid like_type. Filtering only works with 'gauss' or 'sivia'."
            )

        bad_sigma = (
            (ln_eqmux_pred < ln_threshold)
            + (ln_eqmuy_pred < ln_threshold)
            + (ln_vlsr_pred < ln_threshold)
        )
    else:
        raise ValueError("Invalid filter_method. Please choose 'sigma' or 'lnlike'.")

    # Refilter data
    ra_good = ra[~bad_sigma]  # deg
    dec_good = dec[~bad_sigma]  # deg
    glon_good = glon[~bad_sigma]  # deg
    glat_good = glat[~bad_sigma]  # deg
    plx_good = plx[~bad_sigma]  # mas
    e_plx_good = e_plx[~bad_sigma]  # mas
    eqmux_good = eqmux[~bad_sigma]  # mas/yr (equatorial frame)
    e_eqmux_good = e_eqmux[~bad_sigma]  # mas/y (equatorial frame)
    eqmuy_good = eqmuy[~bad_sigma]  # mas/y (equatorial frame)
    e_eqmuy_good = e_eqmuy[~bad_sigma]  # mas/y (equatorial frame)
    vlsr_good = vlsr[~bad_sigma]  # km/s
    e_vlsr_good = e_vlsr[~bad_sigma]  # km/s

    # # === Compare sigma rejection vs ln(likelihood) rejection ===
    # # Retrieve data for histogram
    # lnlike_vlsr_dist = trace["lnlike_vlsr_dist"]
    # # Calculate ln(likelihood) given by best-fit parameters and database parallaxes
    # if like_type == "gauss":
    #     ln_vlsr_sigma = ln_gauss_exp_part(vlsr, vlsr_pred, sigma_vlsr)
    # else:  # like_type == "sivia"
    #     ln_vlsr_sigma = ln_siviaskilling(vlsr, vlsr_pred, sigma_vlsr)
    # # Find indices of rejected sources
    # bad_sigma_loc = np.asarray(np.where(bad_sigma))[0,:]
    # print("Total number of rejected sources:", len(bad_sigma_loc))
    # # Choose which rejected source to plot
    # reject_to_plot = 0  # any integer in [0, # rejected sources - 1]
    # print(data[bad_sigma])

    # reject_idx = bad_sigma_loc[reject_to_plot]  # index of rejected source in database
    # reject_data = lnlike_vlsr_dist[:,:,reject_idx].flatten()  # histogram values for reject
    # plt.hist(reject_data)
    # plt.axvline(np.median(reject_data), color="r", label="lnlike from median")
    # plt.axvline(np.asarray(ln_vlsr_sigma)[reject_idx], color="k", label="lnlike from sigma")
    # plt.legend()
    # plt.show()
    # # ===========================================================

    # Store filtered data in DataFrame
    data_cleaned = pd.DataFrame(
        {
            "ra": ra_good,
            "dec": dec_good,
            "glong": glon_good,
            "glat": glat_good,
            "plx": plx_good,
            "e_plx": e_plx_good,
            "mux": eqmux_good,
            "e_mux": e_eqmux_good,
            "muy": eqmuy_good,
            "e_muy": e_eqmuy_good,
            "vlsr": vlsr_good,
            "e_vlsr": e_vlsr_good,
        }
    )
    num_sources_cleaned = len(eqmux_good)
    print("num sources after filtering:", num_sources_cleaned)

    return data_cleaned, num_sources_cleaned


def main(prior_set, num_round, filter_method):
    # Binary file to read
    # infile = Path(__file__).parent / "MCMC_w_dist_uncer_outfile.pkl"
    infile = Path(
        f"/home/chengi/Documents/coop2021/bayesian_mcmc_rot_curve/mcmc_outfile_{prior_set}_{num_round}.pkl"
    )

    with open(infile, "rb") as f:
        file = dill.load(f)
        data = file["data"]
        model_obj = file["model_obj"]
        trace = file["trace"]
        prior_set = file["prior_set"]  # "A1", "A5", "B", "C", "D"
        like_type = file["like_type"]  # "gauss" or "cauchy" or "sivia"
        num_sources = file["num_sources"]
        num_samples = file["num_samples"]

    print(f"===\nExecuting outlier rejection after round {num_round}")
    print("prior_set:", prior_set)
    print("like_type:", like_type)
    print("num parallax samples:", num_samples)
    print("num sources before filtering:", num_sources)

    # print(data.to_markdown())
    # Clean data
    # _FILTER_METHOD = "lnlike"  # "sigma" or "lnlike"
    data_cleaned, num_sources_cleaned = cleanup_data(
        data, trace, like_type, filter_method
    )

    # Save results to same pickle file
    with open(infile, "wb") as f:
        dill.dump(
            {
                "data": data_cleaned,
                "model_obj": model_obj,
                "trace": trace,
                "prior_set": prior_set,
                "like_type": like_type,
                "num_sources": num_sources_cleaned,
                "num_samples": num_samples,
                "num_round": num_round,
            },
            f,
        )


if __name__ == "__main__":
    prior_set_file = input("prior_set of file (A1, A5, B, C, D): ")
    num_round_file = int(input("round number of file (int): "))
    filter_method = input("Outlier rejection method (sigma or lnlike): ")

    main(prior_set_file, num_round_file, filter_method)
