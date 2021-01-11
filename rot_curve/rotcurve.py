import sys
from pathlib import Path

# Want to add galaxymap.py as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
import astropy.coordinates as acoord
from galaxy_map import galaxymap as gm  # Remember to add coop2021 to $PATH
import transform_wenger as trans

# Define constants
_DEG_TO_RAD = 0.01745329251  # pi/180
_AU_PER_YR_TO_KM_PER_S = 4.743717361
_RSUN = 8.15  # kpc (Reid et al. 2019)
_ZSUN = 5.5  # pc (Reid et al. 2019)
_ROLL = 0  # deg (Anderson et al. 2019)
_LSR_X = 10.6  # km/s (Reid et al. 2019)
_LSR_Y = 10.7  # km/s (Reid et al. 2019)
_LSR_Z = 7.6  # km/s (Reid et al. 2019)


def create_connection(db_file):
    """
    Creates SQLite database connection specified by db_file
    """

    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        print(e)

    return conn


def get_coords(conn):
    """
    Retrieves glong, glat, and plx from database conn
    Returns DataFrame with glong, glat, and plx
    """

    cur = conn.cursor()
    cur.execute("SELECT glong, glat, plx FROM Parallax")
    coords = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    return coords


def get_vels(conn):
    """
    Retrieves ra, dec, x & y proper motions (mux, muy) from J200 equatorial frame,
    and vlsr from database conn
    
    Returns DataFrame with ra (deg), dec (deg), mux (mas/yr), muy (mas/yr), vlsr (km/s)
    """

    cur = conn.cursor()
    cur.execute("SELECT ra, dec, mux, muy, vlsr FROM Parallax")
    vels = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    return vels


def urc(R, a2, a3, R0=8.15):
    """
    Universal rotation curve from Persic et al. 1996
    
    TODO: finish docstring
    """

    lam = (a3 / 1.5) * (a3 / 1.5) * (a3 / 1.5) * (a3 / 1.5) * (a3 / 1.5)
    Ropt = a2 * R0
    rho = R / Ropt

    v1 = (200 * lam ** 0.41) / np.sqrt(
        0.8
        + 0.49 * np.log10(lam)
        + (0.75 * np.exp(-0.4 * lam) / (0.47 + 2.25 * lam ** 0.4))
    )

    v2 = (
        (0.72 + 0.44 * np.log10(lam))
        * (1.97 * (rho) ** 1.22)
        / (rho * rho + 0.61) ** 1.43
    )

    v3 = 1.6 * np.exp(-0.4 * lam) * rho * rho / (rho * rho + 2.25 * lam ** 0.4)

    return v1 * np.sqrt(v2 + v3)


def gal_to_bar_vel(glon, glat, dist, vlsr, gmul, gmub):
    """
    Convert Galactic velocities to a barycentric (heliocentric)
    Cartesian frame

    Inputs:
      glon : Array of scalars (deg)
        Galactic longitude
      glat : Array of scalars (deg)
        Galactic latitude
      dist : Array of scalars (kpc)
        Distance
      vlsr : radial velocity (km/s)
        Radial velocity relative to local standard of rest (i.e. vlsr)
      gmul : Array of scalars (mas/yr)
        Galactic longitudinal velocity
      gmub : Array of scalars (mas/yr)
        Galactic latitudinal velocity

    Returns: Vxb, Vyb, Vzb
      Ub, Vb, Wb : Arrays of scalars (km/s)
        Barycentric Cartesian velocities (i.e. vel_x, vel_y, vel_z)
    """

    # Calculate total velocity (km/s)
    v_transverse = np.sqrt(gmul * gmul + gmub * gmub) * dist * _AU_PER_YR_TO_KM_PER_S
    v_tot = np.sqrt(v_transverse * v_transverse + vlsr * vlsr)

    Ub = v_tot * np.cos(glat * _DEG_TO_RAD) * np.cos(glon * _DEG_TO_RAD)
    Vb = v_tot * np.cos(glat * _DEG_TO_RAD) * np.sin(glon * _DEG_TO_RAD)
    Wb = v_tot * np.sin(glat * _DEG_TO_RAD)

    return Ub, Vb, Wb


def bar_to_gcen_vel(Ub, Vb, Wb, R0=_RSUN, Zsun=_ZSUN, roll=_ROLL):
    """
    Convert barycentric Cartesian velocities to the Galactocentric
    Cartesian frame

    Inputs:
      Ub, Vb, Wb : Arrays of scalars (kpc)
        Barycentric Cartesian velocities (i.e. vel_x, vel_y, vel_z)
      R0 : scalar (kpc)
        Galactocentric radius of the Sun
      Zsun : scalar (pc)
        Height of the Sun above the Galactic midplane
      roll : scalar (deg)
        Angle between Galactic plane and b=0

    Returns: vel_xg, vel_yg, vel_zg
      vel_xg, vel_yg, vel_zg : Arrays of scalars (kpc)
        Galactocentric Cartesian velocities
    """

    # Tilt of b=0 relative to Galactic plane
    tilt = np.arcsin(0.001 * Zsun / R0)
    #
    # Roll CCW about the barycentric X-axis so that the Y-Z plane
    # is aligned with the Y-Z plane of the Galactocentric frame
    #
    roll_rad = roll * _DEG_TO_RAD
    Ub1 = np.copy(Ub)
    Vb1 = np.cos(roll_rad) * Vb - np.sin(roll_rad) * Wb
    Wb1 = np.sin(roll_rad) * Vb + np.cos(roll_rad) * Wb
    #
    # Translate to the Galactic Center
    #
    Ub1 -= R0
    #
    # Tilt to correct for Sun's height above midplane
    #
    # ?: Maybe difference due to diff convention? x, y --> y, -x
    vel_xg = np.cos(tilt) * Ub1 + np.sin(tilt) * Wb1 + _LSR_X
    vel_yg = Vb1 + _LSR_Y
    vel_zg = -np.sin(tilt) * Ub1 + np.cos(tilt) * Wb1 + _LSR_Z
    # vel_xg = np.cos(tilt) * Ub1 + np.sin(tilt) * Wb1
    # vel_yg = Vb1
    # vel_zg = -np.sin(tilt) * Ub1 + np.cos(tilt) * Wb1

    return vel_xg, vel_yg, vel_zg


def main():
    # Specifying database file name
    filename = Path("data/hii_v2_20201203.db")

    # Database in parent directory of this script (call .parent twice)
    db = Path(__file__).parent.parent / filename

    # Create database connection to db
    conn = create_connection(db)

    # Get data + put into DataFrame
    gcoords = get_coords(conn)  # galactic coordinates
    vels = get_vels(conn)
    # print(gcoords.to_markdown())

    # Slice data into components
    glon = gcoords["glong"]  # deg
    glat = gcoords["glat"]  # deg
    gdist = 1 / gcoords["plx"]  # kpc
    ra = vels["ra"]  # deg
    dec = vels["dec"]  # deg
    eqmux = vels["mux"]  # mas/yr (equatorial frame)
    eqmuy = vels["muy"]  # mas/y (equatorial frame)
    vlsr = vels["vlsr"]  # km/s

    # Transform from galactic to galactocentric Cartesian coordinates
    bary_x, bary_y, bary_z = gm.galactic_to_barycentric(glon, glat, gdist)
    gcen_x, gcen_y, gcen_z = gm.barycentric_to_galactocentric(bary_x, bary_y, bary_z)

    # # Change galactocentric coordinates to left-hand convention
    # gcen_x, gcen_y = gcen_y, -gcen_x

    # # Calculate distance from galactic centre
    Rcens = np.sqrt(gcen_x * gcen_x + gcen_y * gcen_y + gcen_z * gcen_z)  # kpc

    # Transform equatorial proper motions to galactic frame
    ############################## USING ASTROPY ##############################
    # eqmux2 = eqmux * (u.mas / u.yr)
    # eqmuy2 = eqmuy * (u.mas / u.yr)
    # vlsr2 = vlsr * (u.km / u.s)
    # _LSR_VEL = (10.6, 10.7, 7.6) * (u.km / u.s) # type =
    #                                             # 'astropy.units.quantity.Quantity'

    # # v_bary gives different results if you plug in _LSR_VEL vs. direct tuple
    # lsr = acoord.LSR(
    #     ra=ra * u.deg,
    #     dec=dec * u.deg,
    #     distance=gdist * u.kpc,
    #     pm_ra_cosdec=eqmux2,
    #     pm_dec=eqmuy2,
    #     radial_velocity=vlsr2,
    #     v_bary=(10.6, 10.7, 7.6),  # km/s
    # )

    # # _GAL_SUN_V = 247 * (u.km / u.s)
    # galactocentric = lsr.transform_to(
    #     acoord.Galactocentric(
    #         galcen_distance=8.15 * u.kpc,
    #         z_sun=5.5 * u.kpc,
    #         roll=0 * u.deg,
    #         # galcen_v_sun=_LSR_VEL,
    #     )
    # )

    # gcen_vx = galactocentric.v_x.value
    # gcen_vy = galactocentric.v_y.value
    # gcen_vz = galactocentric.v_z.value
    ##########################################################################

    ####################### ATTEMPT WITH OWN FUNCTIONS  ######################
    # NOTE: glon2, glat2 unnecessary
    # TODO: make own equatorial to galactic velocity function
    glon2, glat2, gmul, gmub = trans.eq_to_gal(ra, dec, eqmux, eqmuy)

    U, V, W = gal_to_bar_vel(glon, glat, gdist, vlsr, gmul, gmub)
    gcen_vx, gcen_vy, gcen_vz = bar_to_gcen_vel(U, V, W)
    # Result looks like astropy lsr.transform_to() with galcen_v_sun = (10.6, 10.7, 7.6)
    ##########################################################################

    # Calculate circular rotation speed
    # ?: Maybe no gcen_vz component?
    Vcens = np.sqrt(gcen_vx * gcen_vx + gcen_vy * gcen_vy + gcen_vz * gcen_vz)
    # Vcens = np.sqrt(gcen_vx * gcen_vx + gcen_vy * gcen_vy)

    # Create and plot dashed line for rotation curve
    fig, ax = plt.subplots()
    Rvals = np.linspace(0, 300, 1000)
    Vvals = urc(Rvals, 0.96, 1.62)
    ax.plot(Rvals, Vvals, "r-.", linewidth=0.5)

    # Plot data
    ax.plot(Rcens, Vcens, "o", markersize=2)
    # ax.axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    # ax.axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 300)

    # Set title and labels. Then save figure
    ax.set_title("Galactic Rotation Curve")
    ax.set_xlabel("R (kpc)")
    ax.set_ylabel("$\Theta$ (km $\mathrm{s}^{-1})$")
    fig.savefig(
        Path(__file__).parent / "rot_curve.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
