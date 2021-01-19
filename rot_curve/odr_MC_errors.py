"""
Orthogonal Distance Regression using Monte Carlo to estimate errors
i.e. 1 fit using ODR with 10000 MC samples

Isaac Cheng - January 2021
"""
import sys
from pathlib import Path
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import scipy.odr as odr

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans
from universal_rotcurve import urc_odr

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
    # # Specifying database file name & folder
    # filename = Path("data/hii_v2_20201203.db")
    # # Database folder in parent directory of this script (call .parent twice)
    # db = Path(__file__).parent.parent / filename

    # Specifying absolute file path instead
    # (allows file to be run in multiple locations as long as database location does not move)
    db = Path("/home/chengi/Documents/coop2021/data/hii_v2_20201203.db")

    # Create database connection to db
    conn = create_connection(db)

    # Get data + put into DataFrame
    coords = get_coords(conn)  # coordinates
    vels = get_vels(conn)  # velocities
    # print(coords.to_markdown())

    # Create condition to filter data
    all_radii = trans.get_gcen_cyl_radius(coords["glong"], coords["glat"], coords["plx"])
    # Bad data criteria (N.B. casting to array prevents "+" not supported warnings)
    bad = (np.array(all_radii) < 4.0) + (np.array(coords["e_plx"]/coords["plx"]) > 0.2)

    # Slice data into components
    r_asc = coords["ra"][~bad]  # deg
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

    _NUM_SAMPLES = 10000  # number of Monte Carlo samples of each parameter for each trial
    num_sources = len(r_asc)  # number of entries in each (and every) dataframe
    print("Number of data points included:", num_sources)

    # Sample measurements _NUM_SAMPLES times (e.g. 10000 times)
    #   Rows = Samples of 1 source, columns = distinct samples
    #   e.g. all samples from 1st parallax source: plx_mc[:,0]
    plx_mc = np.random.normal(loc=plx, scale=e_plx, size=(_NUM_SAMPLES, num_sources))
    eqmux_mc = np.random.normal(loc=eqmux, scale=e_eqmux, size=(_NUM_SAMPLES, num_sources))
    eqmuy_mc = np.random.normal(loc=eqmuy, scale=e_eqmuy, size=(_NUM_SAMPLES, num_sources))
    vlsr_mc = np.random.normal(loc=vlsr, scale=e_vlsr, size=(_NUM_SAMPLES, num_sources))
    r_asc_mc = np.array([r_asc, ] * _NUM_SAMPLES)  # _NUM_SAMPLES by num_sources
    dec_mc = np.array([dec, ] * _NUM_SAMPLES)  # _NUM_SAMPLES by num_sources
    glon_mc = np.array([glon, ] * _NUM_SAMPLES)  # _NUM_SAMPLES by num_sources
    glat_mc = np.array([glat, ] * _NUM_SAMPLES)  # _NUM_SAMPLES by num_sources

    # Transform raw data into galactocentric cylindrical distance & circular velocity
    radius_mc, v_circ_mc = trans.eq_and_gal_to_gcen_cyl(
        r_asc_mc, dec_mc, plx_mc, glon_mc, glat_mc, eqmux_mc, eqmuy_mc, vlsr_mc
    )

    # Store results
    radii_mc =  np.mean(radius_mc, axis=0)  # mean of each column (axis=0)
    e_radii_mc = np.std(radius_mc, axis=0)  # std of each column (axis=0)
    v_circs_mc = np.mean(v_circ_mc, axis=0)  # mean of each column (axis=0)
    e_v_circs_mc = np.std(v_circ_mc, axis=0)  # std of each column (axis=0)

    # Create model for fitting
    urc_model = odr.Model(urc_odr)
    # Create RealData object using galactocentric cylindrical radius & circular velocity
    model_data = odr.RealData(x=radii_mc, y=v_circs_mc, sx=e_radii_mc, sy=e_v_circs_mc)
    # Set up ODR with model and data
    my_odr = odr.ODR(model_data, urc_model, beta0=[_A_TWO, _A_THREE])
    # Run regression
    my_output = my_odr.run()
    # Print results
    my_output.pprint()

    # Get optimal parameters
    a2_opt = my_output.beta[0]
    a3_opt = my_output.beta[1]
    e_a2_opt = my_output.sd_beta[0]
    e_a3_opt = my_output.sd_beta[1]

    print(f"a2: {a2_opt} +/- {e_a2_opt}")
    print(f"a3: {a3_opt} +/- {e_a3_opt}")

    # Create and plot dashed line for rotation curve using optimal parameters
    fig1, ax1 = plt.subplots()
    Rvals = np.linspace(0, 17, 101)
    Vvals = urc_odr((a2_opt, a3_opt), Rvals)
    ax1.plot(Rvals, Vvals, "r-.", linewidth=0.5)

    # Plot data
    ax1.errorbar(x=radii_mc, y=v_circs_mc,
                xerr=e_radii_mc, yerr=e_v_circs_mc,
                fmt="o", linewidth=1, markersize=2, capsize=2)
    # ax1.plot(radii_mc, v_circs_mc, "o", markersize=2)

    # Set title and labels. Then save figure
    plt.suptitle("Galactic Rotation Curve using ODR and MC Errors", y=0.96)
    ax1.set_title(f"(errors derived using N={_NUM_SAMPLES} random samples)",
                fontsize=8)
    ax1.set_xlabel("R (kpc)")
    ax1.set_ylabel(r"$\Theta$ (km $\mathrm{s}^{-1})$")
    ax1.set_xlim(0, 17)
    ax1.set_ylim(0, 300)
    # Create legend to display current parameter values
    legend_elements = [
        Line2D(
            [0], [0],
            marker="o", color="w", markerfacecolor="k", markersize=0,
            label=fr"a2 = {round(a2_opt, 2)} $\pm$ {round(e_a2_opt, 2)}",
        ),
        Line2D(
            [0], [0],
            marker="o", color="w", markerfacecolor="k", markersize=0,
            label=fr"a3 = {round(a3_opt, 4)} $\pm$ {round(e_a3_opt, 4)}",
        ),
    ]
    ax1.legend(handles=legend_elements, handlelength=0, handletextpad=0)
    fig1.savefig(
        Path(__file__).parent / "odr_MC_errors.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()