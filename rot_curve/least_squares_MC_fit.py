import sys
from pathlib import Path
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from scipy.optimize import curve_fit
import corner

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

    ###################### MONTE CARLO METHOD (ENTIRE ARRAY, FILTERED) ###################
    _NUM_TRIALS = 1000  # number of times (trials) to run curve_fit

    # Make arrays to store fit parameters
    a2_vals = np.zeros(_NUM_TRIALS, float)
    a3_vals = np.zeros(_NUM_TRIALS, float)

    # # Make 3D arrays for all measurements (NO LONGER NECESSARY)
    # # (N.B. length of loc must equal size of final axis)
    # # 1st index is random samples, 2nd is each trial, 3rd is each unique source
    # # e.g. plx_mc_tot[:,:,0] will return all the samples from every trial
    # #      for the first parallax data point
    # #
    # # To get all samples from the ith variable in the jth trial, use _plx_mc_tot[:,j,i]
    # # e.g. Get 1st parallax measurement from the 2nd trial, use: plx_mc_tot[:,1,0]
    # plx_mc_tot = np.random.normal(loc=plx, scale=e_plx, size=(_NUM_SAMPLES, _NUM_TRIALS, num_sources))
    # eqmux_mc_tot = np.random.normal(loc=eqmux, scale=e_eqmux, size=(_NUM_SAMPLES, _NUM_TRIALS, num_sources))
    # eqmuy_mc_tot = np.random.normal(loc=eqmuy, scale=e_eqmuy, size=(_NUM_SAMPLES, _NUM_TRIALS, num_sources))
    # vlsr_mc_tot = np.random.normal(loc=vlsr, scale=e_vlsr, size=(_NUM_SAMPLES, _NUM_TRIALS, num_sources))

    for trial in range(_NUM_TRIALS):
        # Sample measurements
        plx_mc = np.random.normal(loc=plx, scale=e_plx)
        eqmux_mc = np.random.normal(loc=eqmux, scale=e_eqmux)
        eqmuy_mc = np.random.normal(loc=eqmuy, scale=e_eqmuy)
        vlsr_mc = np.random.normal(loc=vlsr, scale=e_vlsr)

        # Parallax to distance
        gdist_mc = trans.parallax_to_dist(plx_mc)

        # LSR velocity to barycentric velocity
        vbary_mc = trans.vlsr_to_vbary(vlsr_mc, glon, glat)

        # Transform from galactic to galactocentric Cartesian coordinates
        bary_x_mc, bary_y_mc, bary_z_mc = trans.gal_to_bary(glon, glat, gdist_mc)
        gcen_x_mc, gcen_y_mc, gcen_z_mc = trans.bary_to_gcen(bary_x_mc, bary_y_mc, bary_z_mc)

        # Transform equatorial proper motions to galactic proper motions
        gmul_mc, gmub_mc = trans.eq_to_gal(r_asc, dec, eqmux_mc, eqmuy_mc, return_pos=False)

        # Transform galactic proper motions to barycentric Cartesian velocities
        U_mc, V_mc, W_mc = trans.gal_to_bary_vel(glon, glat, gdist_mc, gmul_mc, gmub_mc, vbary_mc)

        # Transform barycentric Cartesian velocities to galactocentric Cartesian velocities
        gcen_vx_mc, gcen_vy_mc, gcen_vz_mc = trans.bary_to_gcen_vel(U_mc, V_mc, W_mc)

        # Calculate circular rotation speed by converting to cylindrical frame
        radius_mc, v_circ_mc = trans.get_gcen_cyl_radius_and_circ_velocity(gcen_x_mc, gcen_y_mc, gcen_vx_mc, gcen_vy_mc)

        # Fit data to model (WITHOUT uncertainties in data)
        optimal_params, cov = curve_fit(
            lambda r, a2, a3: urc(r, a2, a3, R0=_RSUN),
            radius_mc,
            v_circ_mc,
            p0=[_A_TWO, _A_THREE],  # inital guesses for a2, a3
            bounds=([0.5, 1.5], [1.95, 1.7]),  # [lower bound], [upper bound]
        )

        # Store parameter values
        a2_vals[trial] = optimal_params[0]
        a3_vals[trial] = optimal_params[1]
    ################### END MONTE CARLO METHOD (ENTIRE ARRAY, FILTERED) ##################

    a2_opt = np.median(a2_vals)
    a3_opt = np.median(a3_vals)
    e_a2_opt = np.std(a2_vals)
    e_a3_opt = np.std(a3_vals)
    print(f"a2: {a2_opt} +/- {e_a2_opt}")
    print(f"a3: {a3_opt} +/- {e_a3_opt}")

    # Create and plot dashed line for rotation curve using optimal parameters
    fig1, ax1 = plt.subplots()
    Rvals = np.linspace(0, 17, 101)
    Vvals = urc(Rvals, a2=a2_opt, a3=a3_opt)
    ax1.plot(Rvals, Vvals, "r-.", linewidth=0.5)

    # Plot most recent trial
    ax1.plot(radius_mc, v_circ_mc, "o", markersize=2)

    # Set title and labels. Then save figure
    plt.suptitle("Galactic Rotation Curve with MC Least Squares Fit", y=0.96)
    ax1.set_title(f"(Fit N={_NUM_TRIALS} times, no errors in each fit)",
                fontsize=8)
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
            label=fr"a3 = {round(a3_opt, 4):.4f} $\pm$ {round(e_a3_opt, 4)}",
        ),
    ]
    ax1.legend(handles=legend_elements, handlelength=0, handletextpad=0)
    fig1.savefig(
        Path(__file__).parent / "least_squares_MC_fit.jpg",
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
        Path(__file__).parent / "a2_MC_fit_histogram.jpg",
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
        Path(__file__).parent / "a3_MC_fit_histogram.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Make 2D histogram of parameters a2 & a3
    # fig4, ax4 = plt.subplots()
    # cmap = plt.cm.get_cmap('viridis', _NUM_TRIALS)
    # colors = [cmap(i) for i in range(_NUM_TRIALS)]
    a2_a3 = np.vstack((a2_vals, a3_vals))  # a2_vals in first row, a3_vals in second row
    a2_a3 = a2_a3.T  # transpose (now array shape is _NUM_TRIALS rows by 2 columns)
    fig4 = corner.corner(a2_a3, labels=["a2", "a3"],
                        quantiles=[0.16,0.5,0.84], show_titles=True, title_fmt=".4f")
    # ax4.set_title(f"a2 & a3 distributions for N={_NUM_TRIALS} trials", fig="fig4")
    fig4.savefig(
        Path(__file__).parent / "a2_a3_MC_fit_histogram.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

if __name__ == "__main__":
    main()
