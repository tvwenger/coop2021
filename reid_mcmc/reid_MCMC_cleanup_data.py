"""
Cleans up Bayesian MCMC data from pickle file
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import dill

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


def cleanup_data(data, trace):
    """
    Cleans up data from pickle file
    (i.e. removes any sources with proper motion or vlsr > 3 sigma from predicted values)

    Returns:
      pandas DataFrame with cleaned up data
    """

    # === Get optimal parameters from MCMC trace ===
    # # Bad idea: via dynamic variable creation
    # opt_vals = {}
    # for varname in trace.varnames:
    #     if "interval" in varname:
    #         continue  # do not want to include non user-defined parameters
    #     opt_vals["{}".format(varname)] = np.median(trace[varname])
    # print(opt_vals)
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

    # Calculating uncertainties for data cleaning
    print("Using Reid et al. (2014) definitions of weights/uncertainties")
    # 1D Virial dispersion for stars in HMSFR w/ mass ~ 10^4 Msun w/in radius of ~ 1 pc
    sigma_vir = 5.0  # km/s
    sigma_eqmux = np.sqrt(
        e_eqmux * e_eqmux
        + sigma_vir * sigma_vir
        * plx * plx
        * _KM_PER_KPC_S_TO_MAS_PER_YR * _KM_PER_KPC_S_TO_MAS_PER_YR
    )
    sigma_eqmuy = np.sqrt(
        e_eqmuy * e_eqmuy
        + sigma_vir * sigma_vir
        * plx * plx
        * _KM_PER_KPC_S_TO_MAS_PER_YR * _KM_PER_KPC_S_TO_MAS_PER_YR
    )
    sigma_vlsr = np.sqrt(e_vlsr * e_vlsr + sigma_vir * sigma_vir)

    # Throw away data with proper motion or vlsr residuals > 3 sigma
    bad_sigma = (
        (np.array(abs(eqmux_pred - eqmux) / sigma_eqmux) > 3)
        + (np.array(abs(eqmuy_pred - eqmuy) / sigma_eqmuy) > 3)
        + (np.array(abs(vlsr_pred - vlsr) / sigma_vlsr) > 3)
    )

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


def main():
    # Binary file to read
    # infile = Path(__file__).parent / "reid_MCMC_outfile.pkl"
    infile = Path(
        "/home/chengi/Documents/coop2021/reid_mcmc/reid_MCMC_outfile.pkl"
    )

    with open(infile, "rb") as f:
        file = dill.load(f)
        data = file["data"]
        model_obj = file["model_obj"]
        trace = file["trace"]
        prior_set = file["prior_set"]  # "A1", "A5", "B", "C", "D"
        like_type = file["like_type"]  # "gauss" or "cauchy"
        num_sources = file["num_sources"]
        print("prior_set:", prior_set)
        print("like_type:", like_type)
        print("num sources before filtering:", num_sources)

    # print(data.to_markdown())
    # Clean data
    data_cleaned, num_sources_cleaned = cleanup_data(data, trace)

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
            },
            f,
        )


if __name__ == "__main__":
    main()
