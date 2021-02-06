"""
test_mcmc.py

Simulates data for MCMC_w_dist_uncer.py to see
if MCMC_w_dist_uncer.py can recover parameters

Isaac Cheng - February 2021
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
import random
from pathlib import Path
from contextlib import closing
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
            "SELECT ra, dec, glong, glat, plx, e_plx, "
            "mux, muy, vlsr, e_mux, e_muy, e_vlsr FROM Parallax"
        )
        data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    return data


def filter_data(data, filter_e_plx, R0):
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
    all_radii = trans.get_gcen_cyl_radius(data["glong"], data["glat"], data["plx"], R0=R0)

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

def main():
    # .db file
    db = Path("/home/chengi/Documents/coop2021/data/hii_v2_20201203.db")
    data = get_data(db)

    # Specify parameters
    R0 = 8.15  # kpc
    Zsun = 5.5  # pc
    roll = 0.0  # deg
    Usun = 10.6  # km/s
    Vsun = 10.7  # km/s
    Wsun = 7.6  # km/s
    Upec = 6.1  # km/s
    Vpec = -4.3  # km/s
    Wpec = 0.0  # km/s
    a2 = 0.96  # dimensionless
    a3 = 1.62  # dimensionless

    # Filter data
    filter_e_plx = True
    data = filter_data(data, filter_e_plx, R0)

    # Slice data into components
    ra = data["ra"].values  # deg
    dec = data["dec"].values  # deg
    glon = data["glong"].values  # deg
    glat = data["glat"].values  # deg
    plx = data["plx"].values  # mas
    e_plx = data["e_plx"].values  # mas
    e_eqmux = data["e_mux"].values  # mas/y (equatorial frame)
    e_eqmuy = data["e_muy"].values  # mas/y (equatorial frame)
    e_vlsr = data["e_vlsr"].values  # km/s

    # Generate random parallaxes
    np.random.seed(158)
    plx_mc = np.random.normal(loc=plx, scale=e_plx)

    # Calculate number of sources used in fit
    num_sources = len(plx_mc)
    print("Number of data points used:", num_sources)

    # === Transform glon, glat, and plx_mc to proper motions ===
    dist = 1 / plx_mc
    # Galactic to galactocentric Cartesian coordinates
    bary_x, bary_y, bary_z = trans.gal_to_bary(glon, glat, dist)
    # Barycentric Cartesian to galactocentric Cartesian coodinates
    gcen_x, gcen_y, gcen_z = trans.bary_to_gcen(
        bary_x, bary_y, bary_z,R0=R0, Zsun=Zsun, roll=roll)
    # Galactocentric Cartesian frame to galactocentric cylindrical frame
    gcen_cyl_dist = np.sqrt(gcen_x * gcen_x + gcen_y * gcen_y)  # kpc
    azimuth = (np.arctan2(gcen_y, -gcen_x) * _RAD_TO_DEG) % 360  # deg in [0,360)
    # Predicted galactocentric cylindrical velocity components
    v_circ = urc(gcen_cyl_dist, a2=a2, a3=a3, R0=R0) + Vpec  # km/s
    v_rad = -1 * Upec  # km/s, negative bc toward GC
    Theta0 = urc(R0, a2=a2, a3=a3, R0=R0)  # km/s, circular rotation speed of Sun
    # Galactocentric cylindrical to equatorial proper motions & LSR velocity
    eqmux_pred, eqmuy_pred, vlsr_pred = trans.gcen_cyl_to_pm_and_vlsr(
        gcen_cyl_dist, azimuth, gcen_z, v_rad, v_circ, Wpec,
        R0=R0, Zsun=Zsun, roll=roll,
        Usun=Usun, Vsun=Vsun, Wsun=Wsun, Theta0=Theta0,
        use_theano=False)

    # Add noise to data
    # eqmux_pred += np.random.normal(loc=0, scale=e_eqmux, size=eqmux_pred.shape)
    # eqmuy_pred += np.random.normal(loc=0, scale=e_eqmuy, size=eqmux_pred.shape)
    # vlsr_pred += np.random.normal(loc=0, scale=e_vlsr, size=eqmux_pred.shape)
    eqmux_pred = np.random.normal(loc=eqmux_pred, scale=e_eqmux)
    eqmuy_pred = np.random.normal(loc=eqmuy_pred, scale=e_eqmuy)
    vlsr_pred = np.random.normal(loc=vlsr_pred, scale=e_vlsr)

    # Save data to new DataFrame
    data_new = pd.DataFrame(
        {
            "ra": ra,
            "dec": dec,
            "glong": glon,
            "glat": glat,
            "plx": plx_mc,
            "e_plx": e_plx,
            "mux": eqmux_pred,
            "e_mux": e_eqmux,
            "muy": eqmuy_pred,
            "e_muy": e_eqmuy,
            "vlsr": vlsr_pred,
            "e_vlsr": e_vlsr,
        }
    )

    # Place DataFrame in pickle file
    outfile = Path(
        f"/home/chengi/Documents/coop2021/bayesian_mcmc_rot_curve/mcmc_sim_data.pkl")
    with open(outfile, "wb") as f:
        dill.dump({"data": data_new}, f)

    # # View saved simulated data
    # with open(outfile, "rb") as f:
    #     file = dill.load(f)
    #     data = file["data"]
    # print(data.to_markdown())
    # print(np.shape(data))

    print("Finished generating data")


if __name__ == "__main__":
    main()
