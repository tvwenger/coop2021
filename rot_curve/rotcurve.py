import sys
from pathlib import Path

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import astropy.units as u
import astropy.coordinates as acoord
from astropy.coordinates import CartesianDifferential as cd
import mytransforms as trans

# Want to add galaxymap.py as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

# User-defined constants
_RSUN = 8.15  # kpc (Reid et al. 2019)
_ZSUN = 5.5  # pc (Reid et al. 2019)
_ROLL = 0  # deg (Anderson et al. 2019)
_LSR_X = 10.6  # km/s (Reid et al. 2019)
_LSR_Y = 10.7  # km/s (Reid et al. 2019)
_LSR_Z = 7.6  # km/s (Reid et al. 2019)
_THETA_0 = 236  # km/s (Reid et al. 2019)

# Universal rotation curve parameters (Persic et al. 1996)
_A_TWO = 0.96  # (Reid et al. 2019)
_A_THREE = 1.62  # (Reid et al. 2019)

# IAU definition of the local standard of rest (km/s)
_USTD = 10.27
_VSTD = 15.32
_WSTD = 7.74

# Useful constants
_DEG_TO_RAD = 0.017453292519943295  # pi/180
_RAD_TO_DEG = 57.29577951308232  # 180/pi (Don't forget to %360 after)
_AU_PER_YR_TO_KM_PER_S = 4.740470463533348  # from astropy (uses tropical year)
_KPC_TO_KM = 3.085677581e16


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
    Retrieves ra, dec, glong, glat, and plx from database conn
    Returns DataFrame with ra (deg), dec (deg), glong (deg), glat (deg), and plx (mas)
    """

    cur = conn.cursor()
    cur.execute("SELECT ra, dec, glong, glat, plx FROM Parallax")
    coords = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    return coords


def get_vels(conn):
    """
    Retrieves x & y proper motions (mux, muy) from J200 equatorial frame,
    and vlsr from database conn

    Returns DataFrame with mux (mas/yr), muy (mas/yr), vlsr (km/s)
    """

    cur = conn.cursor()
    cur.execute("SELECT mux, muy, vlsr FROM Parallax")
    vels = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    return vels


def urc(R, a2=_A_TWO, a3=_A_THREE, R0=_RSUN):
    """
    Universal rotation curve from Persic et al. 1996

    Inputs:
      R : Array of scalars (kpc)
        Galactocentric radius of object
        (i.e. perpendicular distance from z-axis in cylindrical coordinates)
      a2 : Scalar (unitless)
        Defined as R_opt/R_0 (ratio of optical radius to Galactocentric radius of the Sun)
      a3 : Scalar (unitless)
        Defined as 1.5*(L/L*)^0.2
      R0 : Scalar (kpc)
        Galactocentric radius of the Sun perp. to z-axis (i.e. in cylindrical coordinates)

    Returns:
      Theta : Array of scalars (km/s)
        Circular rotation speed of objects at radius R
        (i.e. tangential velocity in cylindrical coordinates)
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

    return v1 * np.sqrt(v2 + v3)  # km/s; circular rotation speed at radius R


def gal_to_bar_vel(glon, glat, dist, vbary, gmul, gmub):
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
      vbary : Array of scalars (km/s)
        Radial velocity relative to barycentre (NOT vlsr)
      gmul : Array of scalars (mas/yr)
        Galactic longitudinal velocity
      gmub : Array of scalars (mas/yr)
        Galactic latitudinal velocity

    Returns: Vxb, Vyb, Vzb
      Ub, Vb, Wb : Array of scalars (km/s)
        Barycentric Cartesian velocities (i.e. vel_x, vel_y, vel_z)
    """

    # NOTE: (Ub, Vb, Wb) "omits" Sun's LSR velocity (implicitly included)
    # (This will be included in galactocentric transform)

    # Adopt method used by Jo Bovy (2019). Inverse of eqns (62) & (64)
    # https://github.com/jobovy/stellarkinematics/blob/master/stellarkinematics.pdf
    l = glon * _DEG_TO_RAD
    b = glat * _DEG_TO_RAD
    vl = dist * gmul * _AU_PER_YR_TO_KM_PER_S  # recall gmul has cos(b) correction
    vb = dist * gmub * _AU_PER_YR_TO_KM_PER_S

    Ub = vbary * np.cos(l) * np.cos(b) - vl * np.sin(l) - vb * np.sin(b) * np.cos(l)
    Vb = vbary * np.sin(l) * np.cos(b) + vl * np.cos(l) - vb * np.sin(b) * np.sin(l)
    Wb = vbary * np.sin(b) + vb * np.cos(b)
    # Ub = vlsr * np.cos(l) * np.cos(b) - vl * np.sin(l) - vb * np.sin(b) * np.cos(l) - _LSR_X
    # Vb = vlsr * np.sin(l) * np.cos(b) + vl * np.cos(l) - vb * np.sin(b) * np.sin(l) - _LSR_Y
    # Wb = vlsr * np.sin(b) + vb * np.cos(b) - _LSR_Z

    return Ub, Vb, Wb


def bar_to_gcen_vel(Ub, Vb, Wb, R0=_RSUN, Zsun=_ZSUN, roll=_ROLL):
    """
    Convert barycentric Cartesian velocities to the Galactocentric
    Cartesian frame

    Inputs:
      Ub, Vb, Wb : Arrays of scalars (km/s)
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
    # Tilt to correct for Sun's height above midplane
    #
    vel_xg = np.cos(tilt) * Ub1 + np.sin(tilt) * Wb1 + _LSR_X
    vel_yg = Vb1 + _LSR_Y + _THETA_0
    vel_zg = -np.sin(tilt) * Ub1 + np.cos(tilt) * Wb1 + _LSR_Z
    # vel_xg = np.cos(tilt) * Ub1 + np.sin(tilt) * Wb1
    # vel_yg = Vb1 + _THETA_0
    # vel_zg = -np.sin(tilt) * Ub1 + np.cos(tilt) * Wb1

    return vel_xg, vel_yg, vel_zg


def gcen_cart_to_gcen_cyl(x_kpc, y_kpc, z_kpc, vx, vy, vz):
    """
    Convert galactocentric Cartesian positions and velocities to
    galactocentric cylindrical positions and velocities

                +z +y
                 | /
                 |/
    Sun -x ------+------ +x
                /|
               / |

    Inputs:
      x_kpc, y_kpc, z_kpc : Array of scalars (kpc)
        Galactocentric Cartesian positions
      vx, vy, vz : Array of scalars (km/s)
        Galactocentric Cartesian velocities

    Returns: perp_distance, azimuth, height, v_radial, v_tangent, v_vertical;
             e_dist, e_azimuth, e_height, e_vrad, e_vtan, e_vvert (optional)
      perp_distance : Array of scalars (kpc)
        Radial distance perpendicular to z-axis
      azimuth : Array of scalars (deg)
        Azimuthal angle; positive CW from -x-axis (left-hand convention!)
      height : Array of scalars (kpc)
        Height above xy-plane (i.e. z_kpc)
      v_radial : Array of scalars (km/s)
        Radial velocity; positive away from z-axis
      v_tangent : Array of scalars (km/s)
        Tangential velocity; positive CW (left-hand convention!)
      v_vertical : Array of scalars (km/s)
        Velocity perp. to xy-plane; positive if pointing above xy-plane (i.e. vz)
    """

    y = y_kpc * _KPC_TO_KM  # km
    x = x_kpc * _KPC_TO_KM  # km

    perp_distance = np.sqrt(x_kpc * x_kpc + y_kpc * y_kpc)  # kpc
    perp_distance_km = perp_distance * _KPC_TO_KM  # km

    azimuth = (np.arctan2(y_kpc, -x_kpc) * _RAD_TO_DEG) % 360  # deg in [0,360)

    # #
    # # **Check if any object is on z-axis (i.e. object's x_kpc & y_kpc both zero)**
    # #
    # arr = np.array([x_kpc, y_kpc])  # array with x_kpc in 0th row, y_kpc in 1st row
    # if np.any(np.all(arr == 0, axis=0)):  # at least 1 object is on z_axis
    #     # Ensure vx & vy are arrays
    #     vx = np.atleast_1d(vx)
    #     vy = np.atleast_1d(vy)
    #     # Initialize arrays to store values
    #     v_radial = np.zeros(len(vx))
    #     v_tangent = np.zeros(len(vx))
    #     for i in range(len(vx)):
    #         if x[i] == 0 and y[i] == 0:  # this object is on z-axis
    #             # **all velocity in xy-plane is radial velocity**
    #             v_radial[i] = np.sqrt(vx[i] * vx[i] + vy[i] * vy[i])  # km/s
    #         else:  # this object is not on z-axis
    #             v_radial[i] = (x[i] * vx[i] + y[i] * vy[i]) / perp_distance_km[i]  # km/s
    #             v_tangent[i] = (x[i] * vy[i] - y[i] * vx[i]) / perp_distance_km[i]  # km/s
    # else:  # no object is on z-axis (no division by zero)
    #     v_radial = (x * vx + y * vy) / perp_distance_km  # km/s
    #     v_tangent = (y * vx - x * vy) / perp_distance_km  # km/s

    # Assuming no object is on z-axis
    v_radial = (x * vx + y * vy) / perp_distance_km  # km/s
    v_tangent = (y * vx - x * vy) / perp_distance_km  # km/s

    return perp_distance, azimuth, z_kpc, v_radial, v_tangent, vz


def get_gcen_cyl_radius_and_circ_velocity(x_kpc, y_kpc, vx, vy):
    """
    Convert galactocentric Cartesian velocities to
    galactocentric cylindrical tangential velocity

                +z +y
                 | /
                 |/
    Sun -x ------+------ +x
                /|
               / |

    Inputs:
      x_kpc, y_kpc : Arrays of scalars (kpc)
        Galactocentric x & y Cartesian positions
      vx, vy : Arrays of scalars (km/s)
        Galactocentric x & y Cartesian velocities

    Returns: perp_distance, v_tangent
      perp_distance : Array of scalars (kpc)
        Radial distance perpendicular to z-axis
      v_tangent : Array of scalars (km/s)
        Tangential velocity; positive CW (left-hand convention!)
    """

    y = y_kpc * _KPC_TO_KM  # km
    x = x_kpc * _KPC_TO_KM  # km

    perp_distance = np.sqrt(x_kpc * x_kpc + y_kpc * y_kpc)  # kpc

    # #
    # # **Check if any object is on z-axis (i.e. object's x_kpc & y_kpc both zero)**
    # #
    # arr = np.array([x_kpc, y_kpc])  # array with x_kpc in 0th row, y_kpc in 1st row
    # if np.any(np.all(arr == 0, axis=0)):  # at least 1 object is on z_axis
    #     # Ensure vx & vy are arrays
    #     vx = np.atleast_1d(vx)
    #     vy = np.atleast_1d(vy)
    #     # Initialize array (initially zero) to store tangential velocities
    #     v_tangent = np.zeros(len(vx))
    #     for i in range(len(vx)):
    #         if x[i] != 0 and y[i] != 0:  # this object is not on z-axis
    #             v_tangent[i] = (x[i] * vy[i] - y[i] * vx[i]) / perp_distance_km[i]  # km/s
    # else:  # no object is on z-axis (no division by zero)
    #     v_tangent = (y * vx - x * vy) / perp_distance_km  # km/s

    # Assuming no object is on z-axis
    v_tangent = (y * vx - x * vy) / perp_distance / _KPC_TO_KM  # km/s

    return perp_distance, v_tangent


def vlsr_to_vbary(vlsr, glon, glat):
    """
    Converts LSR (radial) velocity to radial velocity in barycentric frame

    Inputs:
      vlsr : Array of scalars (km/s)
        Radial velocity relative to local standard of rest
      glon, glat : Array of scalars (deg)
        Galactic longitude and latitude

    Returns: vbary
      vbary : Array of scalars (km/s)
        Radial velocity relative to barycentre of Solar System (NOT vlsr)
    """

    vbary = (
        vlsr
        - _USTD * np.cos(glon * _DEG_TO_RAD) * np.cos(glat * _DEG_TO_RAD)
        - _VSTD * np.sin(glon * _DEG_TO_RAD) * np.cos(glat * _DEG_TO_RAD)
        - _WSTD * np.sin(glat * _DEG_TO_RAD)
    )
    return vbary


def main():
    # Specifying database file name
    filename = Path("data/hii_v2_20201203.db")

    # Database in parent directory of this script (call .parent twice)
    db = Path(__file__).parent.parent / filename

    # Create database connection to db
    conn = create_connection(db)

    # Get data + put into DataFrame
    coords = get_coords(conn)  # galactic coordinates
    vels = get_vels(conn)
    # print(gcoords.to_markdown())

    # Slice data into components
    r_asc = coords["ra"]  # deg
    dec = coords["dec"]  # deg
    glon = coords["glong"]  # deg
    glat = coords["glat"]  # deg
    gdist = 1 / coords["plx"]  # kpc
    eqmux = vels["mux"]  # mas/yr (equatorial frame)
    eqmuy = vels["muy"]  # mas/y (equatorial frame)
    vlsr = vels["vlsr"]  # km/s
    vbary = vlsr_to_vbary(vlsr, glon, glat)  # km/s

    # Transform from galactic to galactocentric Cartesian coordinates
    bary_x, bary_y, bary_z = trans.gal_to_bar(glon, glat, gdist)
    gcen_x, gcen_y, gcen_z = trans.bar_to_gcen(bary_x, bary_y, bary_z)

    # Transform equatorial proper motions to galactic frame
    ############################## USING ASTROPY ##############################
    # icrs = acoord.ICRS(
    #     ra=ra * u.deg,
    #     dec=dec * u.deg,
    #     distance=gdist * u.kpc,
    #     pm_ra_cosdec=eqmux * (u.mas / u.yr),
    #     pm_dec=eqmuy * (u.mas / u.yr),
    #     radial_velocity=vbary * (u.km / u.s),
    # )

    # _GAL_V_SUN = cd(10.6, 246.7, 7.6, unit="km/s")
    # galactocentric = icrs.transform_to(
    #     acoord.Galactocentric(
    #         galcen_distance=8.15 * u.kpc,
    #         z_sun=5.5 * u.pc,
    #         roll=0 * u.deg,
    #         galcen_v_sun=_GAL_V_SUN,
    #     )
    # )

    # gcen_vx = galactocentric.v_x.value
    # gcen_vy = galactocentric.v_y.value
    # gcen_vz = galactocentric.v_z.value
    ##########################################################################

    ####################### ATTEMPT WITH OWN FUNCTIONS  ######################
    gmul, gmub = trans.eq_to_gal(r_asc, dec, eqmux, eqmuy, return_pos=False)

    U, V, W = gal_to_bar_vel(glon, glat, gdist, vbary, gmul, gmub)
    gcen_vx, gcen_vy, gcen_vz = bar_to_gcen_vel(U, V, W)
    ##########################################################################

    # Calculate circular rotation speed by converting to cylindrical frame
    perp_distance, v_circular = get_gcen_cyl_radius_and_circ_velocity(
        gcen_x, gcen_y, gcen_vx, gcen_vy
    )

    # Create and plot dashed line for rotation curve
    fig, ax = plt.subplots()
    Rvals = np.linspace(0, 17, 101)
    Vvals = urc(Rvals)
    ax.plot(Rvals, Vvals, "r-.", linewidth=0.5)

    # Plot data
    ax.plot(perp_distance, v_circular, "o", markersize=2)

    # Set title and labels. Then save figure
    ax.set_title("Galactic Rotation Curve with Reid et al. Parameters")
    ax.set_xlabel("R (kpc)")
    ax.set_ylabel("$\Theta$ (km $\mathrm{s}^{-1})$")
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 300)
    # Create legend to display current parameter values
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="k",
            markersize=0,
            label=f"a2 = {_A_TWO}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="k",
            markersize=0,
            label=f"a3 = {_A_THREE}",
        ),
    ]
    ax.legend(handles=legend_elements, handlelength=0, handletextpad=0)
    fig.savefig(
        Path(__file__).parent / "rot_curve.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
