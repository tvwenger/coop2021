from pathlib import Path
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
import astropy.coordinates as acoord
from galaxy_map import galaxymap as gm  # Remember to add coop2021 to $PYTHONPATH
# import transform_wenger as trans


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


# def gal_to_bar_vel(glong, glat, dist, gmul, gmub):
#     """
#     Convert Galactic velocities to a barycentric (heliocentric)
#     Cartesian frame

#     Inputs:
#       glon : Array of scalars (mas/yr)
#         Galactic longitudinal velocity
#       glat : Array of scalars (mas/yr)
#         Galactic latitudinal velocity
#       dist : Array of scalars (kpc)
#         Distance

#     Returns: Vxb, Vyb, Vzb
#       Xb, Yb, Zb : Arrays of scalars (km/s)
#         Barycentric Cartesian velocities
#     """

#     Xb = dist * np.cos(glat * _DEG_TO_RAD) * np.cos(glong * _DEG_TO_RAD)
#     Yb = dist * np.cos(glat * _DEG_TO_RAD) * np.sin(glong * _DEG_TO_RAD)
#     Zb = dist * np.sin(glat * _DEG_TO_RAD)

#     return Xb, Yb, Zb


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

    ############################# USING ASTROPY #############################
    # # Change data into SkyCoord object
    # galactic = acoord.SkyCoord(
    #     l=glon, b=glat, distance=gdist, frame="galactic", unit=["deg", "deg", "kpc"]
    # )
    # # print(galactic)

    # # Transform from galactic to galactocentric coordinates
    # # NOTE: astropy uses RHR instead of LHR for galactocentric coords
    # galactocentric = galactic.transform_to(
    #     acoord.Galactocentric(
    #         galcen_distance=8.15 * u.kpc, z_sun=5.5 * u.kpc, roll=0 * u.deg
    #     )
    # )
    # # print(galactocentric)
    # gcen_x = galactocentric.y  # left-hand convention
    # gcen_y = -galactocentric.x  # left-hand convention
    # # gcen_z = galactocentric.z   # not needed for this plot
    ##########################################################################

    # Transform from galactic to galactocentric Cartesian coordinates
    bary_x, bary_y, bary_z = gm.galactic_to_barycentric(glon, glat, gdist)
    gcen_x, gcen_y, gcen_z = gm.barycentric_to_galactocentric(bary_x, bary_y, bary_z)

    # # Change galactocentric coordinates to left-hand convention
    # gcen_x, gcen_y = gcen_y, -gcen_x

    # # Calculate distance from galactic centre
    Rcens = np.sqrt(gcen_x * gcen_x + gcen_y * gcen_y + gcen_z * gcen_z)  # kpc\

    # Transform equatorial proper motions to galactic frame
    # NOTE: glon2, glat2 unnecessary. TODO: make own function
    # glon2, glat2, gmul, gmub = trans.eq_to_gal(ra, dec, eqmux, eqmuy)

    # gvl = gdist * gmul
    # gvb = gdist * gmub
    # # c = acoord.SkyCoord(l = glon2*u.deg, b = glat2*u.deg, distance = gdist * u.kpc, pm_l_cosb=gmul *u.mas/u.yr, pm_b=gmub*u.mas/u.yr, frame="galactic")
    eqmux2 = eqmux * (u.mas / u.yr)
    eqmuy2 = eqmuy * (u.mas / u.yr)
    vlsr2 = vlsr * (u.km / u.s)
    lsr = acoord.LSR(
        ra=ra * u.deg,
        dec=dec * u.deg,
        distance=gdist * u.kpc,
        pm_ra_cosdec=eqmux2,
        pm_dec=eqmuy2,
        radial_velocity=vlsr2,
        v_bary=(10.6, 10.7, 7.6),  # in km/s
    )
    # _GAL_SUN_V = 247 * (u.km / u.s)
    galactocentric = lsr.transform_to(
        acoord.Galactocentric(
            galcen_distance=8.15 * u.kpc, z_sun=5.5 * u.kpc, roll=0 * u.deg
        )
    )
    # # c.transform_to(acoord.Galactocentric)
    gcen_vx = galactocentric.v_x.value
    gcen_vy = galactocentric.v_y.value
    gcen_vz = galactocentric.v_z.value

    Vcens = np.sqrt(gcen_vx * gcen_vx + gcen_vy * gcen_vy + gcen_vz * gcen_vz)

    # _MAS_TO_RAD = 4.84813681109536e-06  # pi/180/3600
    # _PER_YR_TO_PER_SEC = 3.1709791983764586e-08  # 1/365/24/3600
    # Vcens = (
    #     np.sqrt(gmul * gmul + gmub * gmub) * _MAS_TO_RAD * Rcens * _PER_YR_TO_PER_SEC
    #     + vlsr
    # )

    # # Calculate circular rotation speed
    # # Vcens = urc(Rcens, 0.96, 1.62)  # !INCORRECT. Must calculate Vcens from database

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
