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

    # ########################### MONTE CARLO METHOD (ROW BY ROW) ##########################
    # # Make arrays to store fit parameters
    # _NUM_TRIALS = 50  # number of times to run curve_fit
    # a2_vals = np.zeros(_NUM_TRIALS, float)
    # a3_vals = np.zeros(_NUM_TRIALS, float)
    # e_a2_vals = np.zeros(_NUM_TRIALS, float)
    # e_a3_vals = np.zeros(_NUM_TRIALS, float)

    # # Arrays to store results (will be overwritten with each iteration)
    # radii_mc = np.zeros(len(plx), float)
    # v_circs_mc = np.zeros(len(plx), float)
    # e_radii_mc = np.zeros(len(plx), float)
    # e_v_circs_mc = np.zeros(len(plx), float)

    # for trial in range(_NUM_TRIALS):
    #     # Iterate through each row
    #     for row in range(len(plx)):
    #         # Sample observed parameters 10000 times
    #         plx_mc = np.random.normal(loc=plx[row], scale=e_plx[row], size=10000)
    #         eqmux_mc = np.random.normal(loc=eqmux[row], scale=e_eqmux[row], size=10000)
    #         eqmuy_mc = np.random.normal(loc=eqmuy[row], scale=e_eqmuy[row], size=10000)
    #         vlsr_mc = np.random.normal(loc=vlsr[row], scale=e_vlsr[row], size=10000)

    #         # Make quantities (those with no uncertainty) same size as MC array
    #         r_asc_mc = np.full(10000, r_asc[row])
    #         dec_mc = np.full(10000, dec[row])
    #         glon_mc = np.full(10000, glon[row])
    #         glat_mc = np.full(10000, glat[row])

    #         # Parallax to distance
    #         gdist_mc = trans.parallax_to_dist(plx_mc)

    #         # LSR velocity to barycentric velocity
    #         vbary_mc = trans.vlsr_to_vbary(vlsr_mc, glon_mc, glat_mc)

    #         # Transform from galactic to galactocentric Cartesian coordinates
    #         bary_x_mc, bary_y_mc, bary_z_mc = trans.gal_to_bar(glon_mc, glat_mc, gdist_mc)
    #         gcen_x_mc, gcen_y_mc, gcen_z_mc = trans.bar_to_gcen(bary_x_mc, bary_y_mc, bary_z_mc)

    #         # Transform equatorial proper motions to galactic proper motions
    #         gmul_mc, gmub_mc = trans.eq_to_gal(r_asc_mc, dec_mc, eqmux_mc, eqmuy_mc, return_pos=False)

    #         # Transform galactic proper motions to barycentric Cartesian velocities
    #         U_mc, V_mc, W_mc = trans.gal_to_bar_vel(glon_mc, glat_mc, gdist_mc, gmul_mc, gmub_mc, vbary_mc)

    #         # Transform barycentric Cartesian velocities to galactocentric Cartesian velocities
    #         gcen_vx_mc, gcen_vy_mc, gcen_vz_mc = trans.bar_to_gcen_vel(U_mc, V_mc, W_mc)

    #         # Calculate circular rotation speed by converting to cylindrical frame
    #         radius_mc, v_circ_mc = trans.get_gcen_cyl_radius_and_circ_velocity(gcen_x_mc, gcen_y_mc, gcen_vx_mc, gcen_vy_mc)

    #         # Store results
    #         radii_mc[row] =  np.mean(radius_mc)
    #         e_radii_mc[row] = np.std(radius_mc)
    #         v_circs_mc[row] = np.mean(v_circ_mc)
    #         e_v_circs_mc[row] = np.std(v_circ_mc)

    #     # Condition to help clean up data
    #     condition = e_v_circs_mc < 20  # km/s
    #     # Fit data to model (with uncertainties in data)
    #     optimal_params, cov = curve_fit(
    #         lambda r, a2, a3: urc(r, a2, a3, R0=_RSUN),
    #         radii_mc[condition],
    #         v_circs_mc[condition],
    #         sigma = e_v_circs_mc[condition],
    #         absolute_sigma=True,
    #         p0=[_A_TWO, _A_THREE],  # inital guesses for a2, a3
    #         bounds=([0.8, 1.5], [1.1, 1.7]),  # bounds for a2, a3
    #     )

    #     # Store parameter values
    #     a2_vals[trial] = optimal_params[0]
    #     a3_vals[trial] = optimal_params[1]
    #     e_a2_vals[trial] = np.sqrt(np.diag(cov))[0]
    #     e_a3_vals[trial] = np.sqrt(np.diag(cov))[1]
    # ######################### END MONTE CARLO METHOD (ROW BY ROW) ########################

    ########################### MONTE CARLO METHOD (ENTIRE ARRAY) ########################
    _NUM_TRIALS = 50  # number of times to run curve_fit

    # Make arrays to store fit parameters
    a2_vals = np.zeros(_NUM_TRIALS, float)
    a3_vals = np.zeros(_NUM_TRIALS, float)
    e_a2_vals = np.zeros(_NUM_TRIALS, float)
    e_a3_vals = np.zeros(_NUM_TRIALS, float)

    # Make arrays to store Monte Carlo data (which will be processed all at once)
    # Arrays will be overwritten with each new trial
    plx_mc_tot = np.zeros([len(plx), 10000], float)
    eqmux_mc_tot = np.zeros([len(plx), 10000], float)
    eqmuy_mc_tot = np.zeros([len(plx), 10000], float)
    vlsr_mc_tot =  np.zeros([len(plx), 10000], float)
    r_asc_mc_tot = np.zeros([len(plx), 10000], float)
    dec_mc_tot = np.zeros([len(plx), 10000], float)
    glon_mc_tot = np.zeros([len(plx), 10000], float)
    glat_mc_tot = np.zeros([len(plx), 10000], float)

    for trial in range(_NUM_TRIALS):
        # Iterate through each row to populate Monte Carlo data arrays
        for row in range(len(plx)):
            # Sample observed parameters 10000 times
            plx_mc = np.random.normal(loc=plx[row], scale=e_plx[row], size=10000)
            eqmux_mc = np.random.normal(loc=eqmux[row], scale=e_eqmux[row], size=10000)
            eqmuy_mc = np.random.normal(loc=eqmuy[row], scale=e_eqmuy[row], size=10000)
            vlsr_mc = np.random.normal(loc=vlsr[row], scale=e_vlsr[row], size=10000)
            # Populate arrays
            plx_mc_tot[row] = plx_mc
            eqmux_mc_tot[row] = eqmux_mc
            eqmuy_mc_tot[row] = eqmuy_mc
            vlsr_mc_tot[row] = vlsr_mc
            r_asc_mc_tot[row] = np.full(10000, r_asc[row])
            dec_mc_tot[row] = np.full(10000, dec[row])
            glon_mc_tot[row] = np.full(10000, glon[row])
            glat_mc_tot[row] = np.full(10000, glat[row])

        # Pass whole Monte Carlo arrays into functions:
        # Parallax to distance
        gdist_mc = trans.parallax_to_dist(plx_mc_tot)

        # LSR velocity to barycentric velocity
        vbary_mc = trans.vlsr_to_vbary(vlsr_mc_tot, glon_mc_tot, glat_mc_tot)

        # Transform from galactic to galactocentric Cartesian coordinates
        bary_x_mc, bary_y_mc, bary_z_mc = trans.gal_to_bar(glon_mc_tot, glat_mc_tot, gdist_mc)
        gcen_x_mc, gcen_y_mc, gcen_z_mc = trans.bar_to_gcen(bary_x_mc, bary_y_mc, bary_z_mc)

        # Transform equatorial proper motions to galactic proper motions
        gmul_mc, gmub_mc = trans.eq_to_gal(r_asc_mc_tot, dec_mc_tot, eqmux_mc_tot, eqmuy_mc_tot, return_pos=False)

        # Transform galactic proper motions to barycentric Cartesian velocities
        U_mc, V_mc, W_mc = trans.gal_to_bar_vel(glon_mc_tot, glat_mc_tot, gdist_mc, gmul_mc, gmub_mc, vbary_mc)

        # Transform barycentric Cartesian velocities to galactocentric Cartesian velocities
        gcen_vx_mc, gcen_vy_mc, gcen_vz_mc = trans.bar_to_gcen_vel(U_mc, V_mc, W_mc)

        # Calculate circular rotation speed by converting to cylindrical frame
        radius_mc, v_circ_mc = trans.get_gcen_cyl_radius_and_circ_velocity(gcen_x_mc, gcen_y_mc, gcen_vx_mc, gcen_vy_mc)

        # Store results
        radii_mc =  np.mean(radius_mc, axis=1)  # mean of each row (axis=1)
        e_radii_mc = np.std(radius_mc, axis=1)  # std of each row (axis=1)
        v_circs_mc = np.mean(v_circ_mc, axis=1)  # mean of each row (axis=1)
        e_v_circs_mc = np.std(v_circ_mc, axis=1)  # std of each row (axis=1)

        # Condition to help clean up data
        condition = e_v_circs_mc < 20  # km/s
        # Fit data to model (with uncertainties in data)
        optimal_params, cov = curve_fit(
            lambda r, a2, a3: urc(r, a2, a3, R0=_RSUN),
            radii_mc[condition],
            v_circs_mc[condition],
            sigma=e_v_circs_mc[condition],
            absolute_sigma=True,
            p0=[_A_TWO, _A_THREE],  # inital guesses for a2, a3
            bounds=([0.8, 1.5], [1.1, 1.7]),  # bounds for a2, a3
        )

        # Store parameter values
        a2_vals[trial] = optimal_params[0]
        a3_vals[trial] = optimal_params[1]
        e_a2_vals[trial] = np.sqrt(np.diag(cov))[0]
        e_a3_vals[trial] = np.sqrt(np.diag(cov))[1]
    ######################## END MONTE CARLO METHOD (ENTIRE ARRAY) #######################

    a2_opt = np.mean(a2_vals)
    a3_opt = np.mean(a3_vals)
    e_a2_opt = np.mean(e_a2_vals)
    e_a3_opt = np.mean(e_a3_vals)
    print(f"a2: {a2_opt} +/- {e_a2_opt}")
    print(f"a3: {a3_opt} +/- {e_a3_opt}")

    # Create and plot dashed line for rotation curve using optimal parameters
    fig1, ax1 = plt.subplots()
    Rvals = np.linspace(0, 17, 101)
    Vvals = urc(Rvals, a2=a2_opt, a3=a3_opt)
    ax1.plot(Rvals, Vvals, "r-.", linewidth=0.5)

    # Condition to help clean up data
    condition = e_v_circs_mc < 20  # km/s
    # Plot most recent data
    ax1.errorbar(x=radii_mc[condition], y=v_circs_mc[condition],
                xerr=e_radii_mc[condition], yerr=e_v_circs_mc[condition],
                fmt="o", markersize=2, capsize=2)

    # Set title and labels. Then save figure
    ax1.set_title("Galactic Rotation Curve (MC Least Squares)")
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
            label=fr"a3 = {round(a3_opt, 4):.4f} $\pm$ {round(e_a3_opt, 4)}",
        ),
    ]
    ax1.legend(handles=legend_elements, handlelength=0, handletextpad=0)
    fig1.savefig(
        Path(__file__).parent / "least_squares_MC.jpg",
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
        Path(__file__).parent / "a2_MC_histogram.jpg",
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
        Path(__file__).parent / "a3_MC_histogram.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Make 2D histogram of parameters a2 & a3
    a2_a3 = np.vstack((a2_vals, a3_vals))  # a2_vals in first row, a3_vals in second row
    a2_a3 = a2_a3.T  # transpose (now array shape is _NUM_TRIALS rows by 2 columns)
    fig4 = corner.corner(a2_a3, labels=["a2", "a3"],
                        quantiles=[0.16,0.5,0.84], show_titles=True)
    fig4.savefig(
        Path(__file__).parent / "a2_a3_MC_histogram.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

if __name__ == "__main__":
    main()
