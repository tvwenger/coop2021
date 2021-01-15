import sys
from pathlib import Path
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from scipy.optimize import curve_fit
import mytransforms as trans
from universal_rotcurve import urc

# Want to add galaxymap.py as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

# Universal rotation curve parameters (Persic et al. 1996)
_A_TWO = 0.96  # (Reid et al. 2019)
_A_THREE = 1.62  # (Reid et al. 2019)

# Sun's distance from galactic centre
_RSUN = 8.15  # kpc (Reid et al. 2019)


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
    Retrieves ra, dec, glong, glat, plx, and e_plx from database conn
    Returns DataFrame with:
        ra (deg), dec (deg), glong (deg), glat (deg), plx (mas), and e_plx (mas)
    """

    cur = conn.cursor()
    cur.execute("SELECT ra, dec, glong, glat, plx, e_plx FROM Parallax")
    coords = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    return coords


def get_vels(conn):
    """
    Retrieves x & y proper motions (mux, muy) from J200 equatorial frame,
    and vlsr from database conn

    Returns DataFrame with: mux (mas/yr), muy (mas/yr), vlsr (km/s) + all uncertainties
    """

    cur = conn.cursor()
    cur.execute("SELECT mux, muy, vlsr, e_mux, e_muy, e_vlsr FROM Parallax")
    vels = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    return vels


def main():
    # Specifying database file name & folder
    filename = Path("data/hii_v2_20201203.db")

    # Database folder in parent directory of this script (call .parent twice)
    db = Path(__file__).parent.parent / filename

    # Create database connection to db
    conn = create_connection(db)

    # Get data + put into DataFrame
    coords = get_coords(conn)  # coordinates
    vels = get_vels(conn)  # velocities
    # print(coords.to_markdown())

    # Slice data into components
    r_asc = coords["ra"]  # deg
    dec = coords["dec"]  # deg
    glon = coords["glong"]  # deg
    glat = coords["glat"]  # deg
    plx = coords["plx"]  # mas
    e_plx = coords["e_plx"]  # mas
    eqmux = vels["mux"]  # mas/yr (equatorial frame)
    e_eqmux = vels["e_mux"]  # mas/y (equatorial frame)
    eqmuy = vels["muy"]  # mas/y (equatorial frame)
    e_eqmuy = vels["e_muy"]  # mas/y (equatorial frame)
    vlsr = vels["vlsr"]  # km/s
    e_vlsr = vels["e_vlsr"]  # km/s

    # Parallax to distance
    gdist, e_gdist = trans.parallax_to_dist(plx, e_plx)  # kpc
    # LSR velocity to barycentric velocity
    vbary, e_vbary = trans.vlsr_to_vbary(vlsr, glon, glat, e_vlsr)  # km/s

    # Transform from galactic to galactocentric Cartesian coordinates
    bary_x, bary_y, bary_z, e_bary_x, e_bary_y, e_bary_z = trans.gal_to_bar(glon, glat, gdist, e_gdist)
    gcen_x, gcen_y, gcen_z, e_gcen_x, e_gcen_y, e_gcen_z = trans.bar_to_gcen(bary_x, bary_y, bary_z, e_bary_x, e_bary_y, e_bary_z)

    # Transform equatorial proper motions to galactic proper motions
    gmul, gmub, e_gmul, e_gmub = trans.eq_to_gal(r_asc, dec, eqmux, eqmuy, e_eqmux, e_eqmuy, return_pos=False)

    # Transform galactic proper motions to barycentric Cartesian velocities
    U, V, W, e_U, e_V, e_W = trans.gal_to_bar_vel(glon, glat, gdist, gmul, gmub, vbary, e_gdist, e_gmul, e_gmub, e_vbary)

    # Transform barycentric Cartesian velocities to galactocentric Cartesian velocities
    gcen_vx, gcen_vy, gcen_vz, e_gcen_vx, e_gcen_vy, e_gcen_vz = trans.bar_to_gcen_vel(U, V, W, e_U, e_V, e_W)

    # Calculate circular rotation speed by converting to cylindrical frame
    radius, v_circ, e_radius, e_v_circ = trans.get_gcen_cyl_radius_and_circ_velocity(
        gcen_x, gcen_y, gcen_vx, gcen_vy, e_gcen_x, e_gcen_y, e_gcen_vx, e_gcen_vy
    )

    ########################### IGNORING UNCERTAINTIES IN DATA ###########################
    # # Fit data to model (assuming no uncertainty in data)
    # optimal_params, cov = curve_fit(
    #     lambda r, a2, a3: urc(r, a2, a3, R0=_RSUN),
    #     radius,
    #     v_circ,
    #     p0=[_A_TWO, _A_THREE],  # inital guesses for a2, a3
    #     bounds=([0.9, 1.5], [1.1, 1.7]),  # bounds for a2, a3
    # )
    ######################################################################################

    ########################### INCLUDING UNCERTAINTIES IN DATA ##########################
    # Clean up data
    condition = e_v_circ < 20  # km/s

    # Fit data to model (with uncertainties in data)
    optimal_params, cov = curve_fit(
        lambda r, a2, a3: urc(r, a2, a3, R0=_RSUN),
        radius[condition],
        v_circ[condition],
        sigma = e_v_circ[condition],
        absolute_sigma=True,
        p0=[_A_TWO, _A_THREE],  # inital guesses for a2, a3
        bounds=([0.9, 1.5], [1.1, 1.7]),  # bounds for a2, a3
    )
    ######################################################################################
    a2_opt = optimal_params[0]
    a3_opt = optimal_params[1]
    e_a2_opt = np.sqrt(np.diag(cov))[0]
    e_a3_opt = np.sqrt(np.diag(cov))[1]
    print(f"a2: {a2_opt} +/- {e_a2_opt}")
    print(f"a3: {a3_opt} +/- {e_a3_opt}")

    # Create and plot dashed line for rotation curve using optimal parameters
    fig, ax = plt.subplots()
    Rvals = np.linspace(0, 17, 101)
    Vvals = urc(Rvals, a2=a2_opt, a3=a3_opt)
    ax.plot(Rvals, Vvals, "r-.", linewidth=0.5)

    # Plot data
    ax.errorbar(x=radius[condition], y=v_circ[condition], xerr=e_radius[condition], yerr=e_v_circ[condition], fmt="o", markersize=2, capsize=2)

    # Set title and labels. Then save figure
    ax.set_title("Galactic Rotation Curve with Fitted Parameters")
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
            label=f"a2 = {round(a2_opt, 2):.2f} $\pm$ {round(e_a2_opt, 2)}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="k",
            markersize=0,
            label=f"a3 = {round(a3_opt, 3):.3f} $\pm$ {round(e_a3_opt, 3)}",
        ),
    ]
    ax.legend(handles=legend_elements, handlelength=0, handletextpad=0)
    fig.savefig(
        Path(__file__).parent / "least_squares.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
