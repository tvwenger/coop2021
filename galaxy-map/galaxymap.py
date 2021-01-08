from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import astropy.units as u
import astropy.coordinates as acoord

# Define constants
_DEG_TO_RAD = 0.01745329251  # pi/180
_RSUN = 8.15  # kpc (Reid et al. 2019)
_ZSUN = 5.5  # pc (Reid et al. 2019)
_ROLL = 0  # deg (Anderson et al. 2019)


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


def galactic_to_barycentric(glong, glat, dist):
    """
    Convert Galactic coordinates to a barycentric (heliocentric)
    Cartesian frame

    Inputs:
      glong : Array of scalars (deg)
        Galactic longitude
      glat : Array of scalars (deg)
        Galactic latitude
      dist : Array of scalars (kpc)
        Distance

    Returns: Xb, Yb, Zb
      Xb, Yb, Zb : Arrays of scalars (kpc)
        Barycentric Cartesian coordinates
    """

    Xb = dist * np.cos(glat * _DEG_TO_RAD) * np.cos(glong * _DEG_TO_RAD)
    Yb = dist * np.cos(glat * _DEG_TO_RAD) * np.sin(glong * _DEG_TO_RAD)
    Zb = dist * np.sin(glat * _DEG_TO_RAD)

    return Xb, Yb, Zb


def barycentric_to_galactocentric(Xb, Yb, Zb, R0=_RSUN, Zsun=_ZSUN, roll=_ROLL):
    """
    Convert barycentric Cartesian coordinates to the Galactocentric
    Cartesian frame

    Inputs:
      Xb, Yb, Zb : Arrays of scalars (kpc)
        Barycentric Cartesian coordinates
      R0 : scalar (kpc)
        Galactocentric radius of the Sun
      Zsun : scalar (pc)
        Height of the Sun above the Galactic midplane
      roll : scalar (deg)
        Angle between Galactic plane and b=0

    Returns: Xg, Yg, Zg
      Zg, Yg, Zg : Arrays of scalars (kpc)
        Galactocentric Cartesian coordinates
    """

    # Tilt of b=0 relative to Galactic plane
    tilt = np.arcsin(0.001 * Zsun / R0)
    #
    # Roll CCW about the barycentric X-axis so that the Y-Z plane
    # is aligned with the Y-Z plane of the Galactocentric frame
    #
    roll_rad = roll * _DEG_TO_RAD
    Xb1 = np.copy(Xb)  # OR: Xb1 = Xb
    Yb1 = np.cos(roll_rad) * Yb - np.sin(roll_rad) * Zb
    Zb1 = np.sin(roll_rad) * Yb + np.cos(roll_rad) * Zb
    #
    # Translate to the Galactic Center
    #
    Xb1 -= R0  # must use np.copy() above
    # OR: Xb1 = Xb1 - R0
    #
    # Tilt to correct for Sun's height above midplane
    #
    Xg = np.cos(tilt) * Xb1 + np.sin(tilt) * Zb1
    Yg = Yb1
    Zg = -np.sin(tilt) * Xb1 + np.cos(tilt) * Zb1

    return Xg, Yg, Zg


def main():
    # Specifying database file name
    filename = Path("hii_v2_20201203.db")

    # Database in parent directory of this script (call .parent twice)
    db = Path(__file__).parent.parent / filename

    # Create database connection to db
    conn = create_connection(db)

    # Get data + put into DataFrame
    coords = get_coords(conn)  # galactic coordinates
    # print(coords.to_markdown())

    # Slice data into components
    glon = coords["glong"]  # deg
    glat = coords["glat"]  # deg
    gdist = 1 / coords["plx"]  # kpc

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
    bary_x, bary_y, bary_z = galactic_to_barycentric(glon, glat, gdist)
    gcen_x, gcen_y, gcen_z = barycentric_to_galactocentric(bary_x, bary_y, bary_z)

    # Change galactocentric coordinates to left-hand convention
    gcen_x, gcen_y = gcen_y, -gcen_x

    # Plot data
    fig, ax = plt.subplots()
    ax.plot(gcen_x, gcen_y, "o", markersize=2)
    ax.axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax.axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax.set_xlim(-16, 16)
    ax.set_ylim(-16, 16)

    # Set title and labels. Then save figure
    ax.set_title("Face-on View of Masers")
    ax.set_xlabel("x (kpc)")
    ax.set_ylabel("y (kpc)")
    ax.set_aspect("equal")
    fig.savefig(
        Path(__file__).parent / "galaxy-map.jpg", format="jpg", dpi=300, bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
