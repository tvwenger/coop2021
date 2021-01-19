"""
Orthogonal Distance Regression using Monte Carlo to estimate errors
i.e. 1000 fits using ODR each with 1 MC sample

NOTE: THIS DOES NOT WORK! ODR does not allow parameter bounds, so the inputs to the
rotation curve yield a RuntimeWarning due to "invalid value encountered in power"

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
import corner

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
    print("Number of data points included:", len(r_asc))

    _NUM_TRIALS = 1000  # number of times (trials) to run curve_fit

    # Make arrays to store fit parameters
    a2_vals = np.zeros(_NUM_TRIALS, float)
    a3_vals = np.zeros(_NUM_TRIALS, float)

    # Create model for fitting
    urc_model = odr.Model(urc_odr)

    # Counter for number of times where ODR stopped due to reaching max iterations
    num_reach_maxiter = 0

    for trial in range(_NUM_TRIALS):
        # Sample measurements
        plx_mc = np.random.normal(loc=plx, scale=e_plx)
        eqmux_mc = np.random.normal(loc=eqmux, scale=e_eqmux)
        eqmuy_mc = np.random.normal(loc=eqmuy, scale=e_eqmuy)
        vlsr_mc = np.random.normal(loc=vlsr, scale=e_vlsr)

        # Transform raw data into galactocentric cylindrical distance & circular velocity
        radius_mc, v_circ_mc = trans.eq_and_gal_to_gcen_cyl(
            r_asc, dec, plx_mc, glon, glat, eqmux_mc, eqmuy_mc, vlsr_mc
        )

        # if np.any(radius_mc <= 0):
        #     print("<= negative distance")  # None, as expected
        # if np.any(v_circ_mc <= 0):
        #     print("<= circular velocity")  # None, as expected

        # Create RealData object w/ galactocentric cylindrical radius & circular velocity
        model_data = odr.RealData(x=radius_mc, y=v_circ_mc)
        # Set up ODR with model and data
        # my_odr = odr.ODR(model_data, urc_model, beta0=[_A_TWO, _A_THREE], ifixb=[1,1], maxit=100)
        my_odr = odr.ODR(model_data, urc_model, beta0=[_A_TWO, _A_THREE], maxit=100)
        # Run regression
        # ! RuntimeWarning: invalid value encountered in power (line 88 of universal_rotcurve.py)
        # ! "* (1.97 * (rho) ** 1.22)" --> due to invalid a2 or a3 parameters
        my_output = my_odr.run()
        # # Print results
        # my_output.pprint()

        # if trial in range(0,10):
        #     my_output.pprint()
        if my_output.stopreason == ["Iteration limit reached"]:
            num_reach_maxiter += 1

        # Store optimal parameters
        a2_vals[trial] = my_output.beta[0]
        a3_vals[trial] = my_output.beta[1]

        if np.isnan(my_output.beta[0]):
            print("a2 is nan in trial", trial) 
        elif np.isnan(my_output.beta[1]):
            print("a3 is nan in trial", trial)

    a2_opt = np.median(a2_vals)
    a3_opt = np.median(a3_vals)
    e_a2_opt = np.std(a2_vals)
    e_a3_opt = np.std(a3_vals)
    print(f"a2: {a2_opt} +/- {e_a2_opt}")
    print(f"a3: {a3_opt} +/- {e_a3_opt}")
    print("Number of times ODR stoppped due to reaching max iterations:", num_reach_maxiter, f" of {_NUM_TRIALS} trials")
    print("N.B. The reason for the nonsensical results is actually because of invalid a2/a3 parameters")

    # Create and plot dashed line for rotation curve using optimal parameters
    fig1, ax1 = plt.subplots()
    Rvals = np.linspace(0, 17, 101)
    Vvals = urc_odr((a2_opt, a3_opt), Rvals)
    ax1.plot(Rvals, Vvals, "r-.", linewidth=0.5)

    # Plot most recent data
    ax1.plot(radius_mc, v_circ_mc, "o", markersize=2)

    # Set title and labels. Then save figure
    ax1.set_title(f"Galactic Rotation Curve using N={_NUM_TRIALS} ODR fits")
    ax1.set_xlabel("R (kpc)")
    ax1.set_ylabel(r"$\Theta$ (km $\mathrm{s}^{-1})$")
    ax1.set_xlim(0, 17)
    ax1.set_ylim(0, 300)
    # Create legend to display current parameter values
    legend_elements = [
        Line2D(
            [0], [0],
            marker="o", color="w", markerfacecolor="k", markersize=0,
            label=fr"a2 = {round(a2_opt, 3):.3f} $\pm$ {round(e_a2_opt, 3)}",
        ),
        Line2D(
            [0], [0],
            marker="o", color="w", markerfacecolor="k", markersize=0,
            label=fr"a3 = {round(a3_opt, 2)} $\pm$ {round(e_a3_opt, 2)}",
        ),
    ]
    ax1.legend(handles=legend_elements, handlelength=0, handletextpad=0)
    fig1.savefig(
        Path(__file__).parent / "odr_MC_fit.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Plot histogram of a2 parameter values
    fig2, ax2 = plt.subplots()
    ax2.hist(a2_vals, bins=20)
    ax2.set_title(f"a2 parameter values for N={_NUM_TRIALS} trials")
    ax2.set_xlabel("a2 value")
    ax2.set_ylabel("Frequency")
    fig2.savefig(
        Path(__file__).parent / "a2_odr_MC_fit_histogram.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Plot histogram of a3 parameter values
    fig3, ax3 = plt.subplots()
    ax3.set_title(f"a3 parameter values for N={_NUM_TRIALS} trials")
    ax3.hist(a3_vals, bins=20)
    ax3.set_xlabel("a3 value")
    ax3.set_ylabel("Frequency")
    fig3.savefig(
        Path(__file__).parent / "a3_odr_MC_fit_histogram.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Make 2D histogram of parameters a2 & a3
    a2_a3 = np.vstack((a2_vals, a3_vals))  # a2_vals in first row, a3_vals in second row
    a2_a3 = a2_a3.T  # transpose (now array shape is _NUM_TRIALS rows by 2 columns)
    fig4 = corner.corner(a2_a3, labels=["a2", "a3"],
                        quantiles=[0.16,0.5,0.84], show_titles=True, title_fmt=".4f")
    # ax4.set_title(f"a2 & a3 distributions for N={_NUM_TRIALS} trials", fig="fig4")
    fig4.savefig(
        Path(__file__).parent / "a2_a3_odr_MC_fit_histogram.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()