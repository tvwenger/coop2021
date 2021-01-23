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
    # Upec = 6.1  # km/s, force Upec to this value since my current number of iterations does not do that
    Vpec = np.median(trace["Vpec"])  # km/s
    # Vpec = -2.1  # km/s, force Vpec to this value
    a2 = np.median(trace["a2"])  # dimensionless
    a3 = np.median(trace["a3"])  # dimensionless
    # R0 = 8.15  # kpc
    # Usun = 10.5  # km/s
    # Vsun = 10.7  # km/s
    # Wsun = 7.6  # km/s
    # Upec = 6.0  # km/s
    # Vpec = -4.3  # km/s
    # a2 = 0.96  # dimensionless
    # a3 = 1.62  # dimensionless

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
    v_circ_pred = urc(gcen_cyl_dist, a2=a2, a3=a3, R0=R0) + Upec  # km/s
    v_rad = Vpec  # km/s
    v_vert = 0.0  # km/s, zero vertical velocity in URC

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
        use_theano=False,
    )

    # Throw away data with proper motions or vlsr > 3 sigma
    bad_sigma = (
        (np.array(abs(eqmux_pred - eqmux) / e_eqmux) > 3)
        + (np.array(abs(eqmuy_pred - eqmuy) / e_eqmuy) > 3)
        + (np.array(abs(vlsr_pred - vlsr) / e_vlsr) > 3)
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
        "/home/chengi/Documents/coop2021/bayesian_mcmc_rot_curve/reid_MCMC_outfile.pkl"
    )

    with open(infile, "rb") as f:
        file = dill.load(f)
        data = file["data"]
        model_obj = file["model_obj"]
        trace = file["trace"]
        prior_set = file["prior_set"]  # "A1", "A5", "B", "C", "D"
        like_type = file["like_type"]  # "gaussian" or "cauchy"
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
