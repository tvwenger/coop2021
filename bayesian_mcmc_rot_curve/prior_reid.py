"""
Bayesian MCMC using priors from Reid et al. (2019)
"""
import sys
from pathlib import Path
import sqlite3
from contextlib import closing
import numpy as np
import matplotlib.pyplot as plt
import corner

# from matplotlib.lines import Line2D
import pandas as pd
import theano.tensor as tt
import pymc3 as pm

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


def get_coords(db_file):
    """
    Retrieves ra, dec, glong, glat, plx, and e_plx from
    database connection specified by db_file

    Returns DataFrame with:
        ra (deg), dec (deg), glong (deg), glat (deg), plx (mas), and e_plx (mas)
    """

    with closing(sqlite3.connect(db_file).cursor()) as cur:  # context manager, auto-close
        cur.execute("SELECT ra, dec, glong, glat, plx, e_plx FROM Parallax")
        coords = pd.DataFrame(
            cur.fetchall(), columns=[desc[0] for desc in cur.description]
        )

    return coords


def get_vels(db_file):
    """
    Retrieves x & y proper motions (mux, muy) from J200 equatorial frame,
    and vlsr from database connection specified by db_file

    Returns DataFrame with: mux (mas/yr), muy (mas/yr), vlsr (km/s) + all uncertainties
    """

    with closing(sqlite3.connect(db_file).cursor()) as cur:  # context manager, auto-close
        cur.execute("SELECT mux, muy, vlsr, e_mux, e_muy, e_vlsr FROM Parallax")
        vels = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    return vels


def main():
    # # Specifying database file name & folder
    # filename = Path("data/hii_v2_20201203.db")
    # # Database folder in parent directory of this script (call .parent twice)
    # db = Path(__file__).parent.parent / filename

    # Specifying absolute file path instead
    # (allows file to be run in multiple locations as long as database location does not move)
    db = Path("/home/chengi/Documents/coop2021/data/hii_v2_20201203.db")

    # Get data + put into DataFrame
    coords = get_coords(db)  # coordinates
    vels = get_vels(db)  # velocities
    # print(coords.to_markdown())

    # Create condition to filter data
    all_radii = trans.get_gcen_cyl_radius(coords["glong"], coords["glat"], coords["plx"])
    # Bad data criteria (N.B. casting to array prevents "+" not supported warnings)
    bad = (np.array(all_radii) < 4.0) + (np.array(coords["e_plx"] / coords["plx"]) > 0.2)

    # Slice data into components
    ra = coords["ra"][~bad]  # deg
    dec = coords["dec"][~bad]  # deg
    glon = coords["glong"][~bad]  # deg
    glat = coords["glat"][~bad]  # deg
    plx = coords["plx"][~bad]  # mas
    e_plx = coords["e_plx"][~bad]  # mas
    eqmux = vels["mux"][~bad]  # mas/yr (equatorial frame)
    e_eqmux = vels["e_mux"][~bad]  # mas/y (equatorial frame)
    eqmuy = vels["muy"][~bad]  # mas/y (equatorial frame)
    e_eqmuy = vels["e_muy"][~bad]  # mas/y (equatorial frame)
    vlsr = vels["vlsr"][~bad]  # km/s
    e_vlsr = vels["e_vlsr"][~bad]  # km/s

    # Parallax to distance
    gdist = trans.parallax_to_dist(plx)
    # Galactic to galactocentric Cartesian coordinates
    bary_x, bary_y, bary_z = trans.gal_to_bary(glon, glat, gdist)
    # Zero vertical velocity in URC
    # ? Maybe make this Gaussian in future?
    v_vert = np.zeros(len(ra), float)

    # Make model
    rot_model = pm.Model()
    with rot_model:
        # 8 parameters from Reid et al. (2019): (see Section 4 & Table 3)
        #   R0, Usun, Vsun, Wsun, Upec, Vpec, a2, a3

        # === SET A priors ===
        # R0 = pm.Uniform("R0", lower=0, upper=np.inf)  # kpc
        R0 = pm.Uniform("R0", lower=4.0, upper=6.0)  # kpc (so my computer doesn't die)
        Usun = pm.Normal("Usun", mu=11.1, sigma=1.2)  # km/s
        Vsun = pm.Normal("Vsun", mu=15.0, sigma=10.0)  # km/s
        Wsun = pm.Normal("Wsun", my=7.2, sigma=1.1)  # km/s
        Upec = pm.Normal("Upec", mu=3.0, sigma=10.0)  # km/s
        Vpec = pm.Normal("Vpec", mu=-3.0, sigma=10.0)  # km/s
        a2 = pm.Uniform("a2", lower=0.5, upper=1.5)  # dimensionless (so comp. won't die)
        a3 = pm.Uniform("a3", lower=1.5, upper=1.7)  # dimensionless (so comp. won't die)

        # === Predicted values (using data) ===
        gcen_x, gcen_y, gcen_z = trans.bary_to_gcen(bary_x, bary_y, bary_z, R0=R0)
        # Galactocentric Cartesian coordinates to galactocentric cylindrical distance
        R = tt.sqrt(gcen_x * gcen_x + gcen_y * gcen_y)  # kpc
        azimuth = (tt.arctan2(gcen_y, -gcen_x) * _RAD_TO_DEG) % 360  # deg in [0,360)
        # height = gcen_z  # kpc
        v_circ = urc(R, a2=a2, a3=a3, R0=R0) + Upec  # km/s
        v_rad = Vpec  # km/s
        # v_vert = np.zeros(len(r_asc), float)  # moved outside model

        # Go in reverse!
        # Galactocentric cylindrical to galactocentric Cartesian
        (
            ra_rev,
            dec_rev,
            glon_rev,
            glat_rev,
            plx_rev,
            eqmux_rev,
            eqmuy_rev,
            vlsr_rev,
        ) = trans.gcen_cyl_to_eq(
            R,
            azimuth,
            gcen_z,
            v_rad,
            v_circ,
            v_vert,
            R0=R0,
            Usun=Usun,
            Vsun=Vsun,
            Wsun=Wsun,
        )

        # Likelihood function


if __name__ == "__main__":
    main()
