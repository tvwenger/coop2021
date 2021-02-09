"""
plot_vrad_vtan.py

Plots the maser sources colour-coded by ratio of radial to azimuthal velocity

Isaac Cheng - February 2021
"""
import sys
from pathlib import Path
import numpy as np
import dill
import matplotlib as mpl
import matplotlib.pyplot as plt

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans
from universal_rotcurve import urc

# Roll angle between galactic midplane and galactocentric frame
_ROLL = 0.0  # deg (Anderson et al. 2019)
# Sun's height above galactic midplane (Reid et al. 2019)
_ZSUN = 5.5  # pc
# Useful constants
_RAD_TO_DEG = 57.29577951308232  # 180/pi (Don't forget to % 360 after)

def data_to_gcen_cyl_residuals(
  data, trace, free_Zsun=False, free_roll=False, free_Wpec=False
):
    """
    Converts database data to galactocentric cylindrical coordinates and
    calculates the residual motions of the sources.

    Returns galactocentric cylindrical coordinates & their
    residual motions in the galactocentric cylindrical frame

    Inputs:
      data :: pandas DataFrame
        Contains maser galactic longitudes, latitudes, right ascensions, declinations,
        parallaxes, equatorial proper motions, and LSR velocities
        with all associated uncertainties
      trace :: PyMC3 MultiTrace object
        Contains each iteration of the Bayesian MCMC algorithm for every parameter
      free_Zsun, free_roll, free_Wpec :: booleans (default: False)
        True iff Zsun, roll, or Wpec are free parameters in the model

    Returns: eqmux_pred, eqmuy_pred, vlsr_pred
      eqmux_pred :: Array of scalars (mas/yr)
        RA proper motion with cos(Declination) correction
      eqmuy_pred :: Array of scalars (mas/yr)
        Declination proper motion
      vlsr_pred :: Array of scalars (km/s)
        LSR velocity

    TODO: fix docstring
    """
    # === Get optimal parameters from MCMC trace ===
    R0 = np.median(trace["R0"])  # kpc
    Vsun = np.median(trace["Vsun"])  # km/s
    Usun = np.median(trace["Usun"])  # km/s
    Wsun = np.median(trace["Wsun"])  # km/s
    Upec = np.median(trace["Upec"])  # km/s
    Vpec = np.median(trace["Vpec"])  # km/s
    a2 = np.median(trace["a2"])  # dimensionless
    a3 = np.median(trace["a3"])  # dimensionless
    Zsun = np.median(trace["Zsun"]) if free_Zsun else _ZSUN  # pc
    roll = np.median(trace["roll"]) if free_roll else _ROLL  # deg
    Wpec = np.median(trace["Wpec"]) if free_Wpec else 0.0  # km/s
    # # ? Set Upec = Vpec = 0 as per Reid et al. Fig. 6
    # Upec = 0.0  # km/s
    # Vpec = 0.0  # km/s

    # === Get data ===
    # Slice data into components
    ra = data["ra"]  # deg
    dec = data["dec"]  # deg
    glon = data["glong"]  # deg
    glat = data["glat"]  # deg
    plx = data["plx"]  # mas
    e_plx = data["e_plx"]  # mas
    eqmux = data["mux"]  # mas/yr (equatorial frame)
    e_eqmux = data["e_mux"]  # mas/y (equatorial frame)
    eqmuy = data["muy"]  # mas/y (equatorial frame)
    e_eqmuy = data["e_muy"]  # mas/y (equatorial frame)
    vlsr = data["vlsr"]  # km/s
    e_vlsr = data["e_vlsr"]  # km/s

    # === Calculate predicted values from optimal parameters ===
    Theta0 = urc(R0, a2=a2, a3=a3, R0=R0)  # km/s, LSR circular rotation speed
    radius, azimuth, height, v_radial, v_circ, v_vert = trans.eq_and_gal_to_gcen_cyl(
      ra, dec, plx, glon, glat, eqmux, eqmuy, vlsr,
      R0=R0, Zsun=Zsun, roll=roll,
      Usun=Usun, Vsun=Vsun, Wsun=Wsun, Theta0=Theta0,
      use_theano=False, return_only_r_and_theta=False
    )

    # v_rad_pred = -Upec  # km/s
    # v_circ_pred = urc(radius, a2=a2, a3=a3, R0=R0) + Vpec  # km/s
    # v_vert_pred = Wpec  # km/s
    # v_rad_residual = v_radial - v_rad_pred
    # v_circ_residual = v_circ - v_circ_pred
    # v_vert_residual = v_vert - v_vert_pred

    # # Parallax to distance
    # dist = trans.parallax_to_dist(plx)
    # # Galactic to barycentric Cartesian coordinates
    # bary_x, bary_y, bary_z = trans.gal_to_bary(glon, glat, dist)
    # # Barycentric Cartesian to galactocentric Cartesian coodinates
    # gcen_x, gcen_y, gcen_z = trans.bary_to_gcen(
    #     bary_x, bary_y, bary_z, R0=R0, Zsun=Zsun, roll=roll)
    # # Galactocentric Cartesian frame to galactocentric cylindrical frame
    # gcen_cyl_dist = np.sqrt(gcen_x * gcen_x + gcen_y * gcen_y)  # kpc
    # azimuth = (np.arctan2(gcen_y, -gcen_x) * _RAD_TO_DEG) % 360  # deg in [0,360)
    # Theta0 = urc(R0, a2=a2, a3=a3, R0=R0)  # km/s, LSR circular rotation speed

    # # Go in reverse!
    # # Galactocentric cylindrical to equatorial proper motions & LSR velocity
    # eqmux_pred, eqmuy_pred, vlsr_pred = trans.gcen_cyl_to_pm_and_vlsr(
    #     gcen_cyl_dist, azimuth, gcen_z, v_rad, v_circ_pred, Wpec,
    #     R0=R0, Zsun=Zsun, roll=roll,
    #     Usun=Usun, Vsun=Vsun, Wsun=Wsun, Theta0=Theta0,
    #     use_theano=False)

    return radius, azimuth, height, v_radial, v_circ, v_vert


def main(prior_set, num_samples, num_rounds):
    # Binary file to read
    # infile = Path(__file__).parent / "reid_MCMC_outfile.pkl"
    infile = Path(
        "/home/chengi/Documents/coop2021/bayesian_mcmc_rot_curve/"
        f"mcmc_outfile_{prior_set}_{num_samples}dist_{num_rounds}.pkl"
    )

    with open(infile, "rb") as f:
        file = dill.load(f)
        data = file["data"]
        trace = file["trace"]
        like_type = file["like_type"]  # "gauss", "cauchy", or "sivia"
        num_sources = file["num_sources"]
        # reject_method = file["reject_method"] if num_rounds != 1 else None
        free_Zsun = file["free_Zsun"]
        free_roll = file["free_roll"]
        free_Wpec = file["free_Wpec"]

    print("=== Plotting v_rad/v_tan plot for "
          f"({prior_set} priors & {num_rounds} MCMC rounds) ===")
    print("Number of sources:", num_sources)
    print("Likelihood function:", like_type)

    # Convert database data to galactocentric cylindrical positions & residual velocities
    (
      radius,
      azimuth,
      height,
      v_rad,
      v_circ,
      v_vert,
    ) = data_to_gcen_cyl_residuals(
        data, trace, free_Zsun=free_Zsun, free_roll=free_roll, free_Wpec=free_Wpec)

    # Convert galactocentric cylindrical to galactocentric Cartesian
    # x, y, z, vx, vy, vz = trans.gcen_cyl_to_gcen_cart(
    #   radius, azimuth, height,
    #   v_radial=v_rad, v_tangent=v_circ, v_vertical=v_vert)
    x, y, z = trans.gcen_cyl_to_gcen_cart(radius, azimuth, height)

    # Change galactocentric coordinates to Reid's convention
    # (our convention is detailed in the docstring of trans.gcen_cyl_to_gcen_cart)
    x, y = y, -x

    # Ratio of radial velocity to circular rotation speed
    vrad_vcirc = v_rad / v_circ
    vrad_vcirc_min = np.min(vrad_vcirc)
    vrad_vcirc_max = np.max(vrad_vcirc)
    print("Min v_rad/v_circ:", vrad_vcirc_min)
    print("Max v_rad/v_circ:", vrad_vcirc_max)
    print("Mean v_rad/v_circ:", np.mean(vrad_vcirc))

    # Plot data
    cmap = "inferno"  # "coolwarm" is another option
    cmap_min = np.floor(100 * vrad_vcirc_min) / 100
    cmap_max = np.ceil(100 * vrad_vcirc_max) / 100
    norm = mpl.colors.Normalize(vmin=cmap_min, vmax=cmap_max)

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=vrad_vcirc, cmap=cmap, s=2)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(r'$v_{rad}/v_{circ}$', rotation=270)
    ax.axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax.axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax.set_xlim(-8, 12)
    ax.set_xticks([-5, 0, 5, 10])
    ax.set_ylim(-5, 15)
    ax.set_yticks([-5, 0, 5, 10, 15])

    # Set title and labels. Then save figure
    ax.set_title(f"Face-on View of {num_sources} Masers & "
                 r"their Ratio of $v_{rad}$ to $v_{circ}$"
                 f"\nUsed best-fit parameters from {prior_set} priors")
    ax.set_xlabel("x (kpc)")
    ax.set_ylabel("y (kpc)")
    ax.set_aspect("equal")
    fig.savefig(
        Path(__file__).parent / "vrad_vtan.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    # # === Extract parallax, proper motions & LSR velocity ===
    # plx = data["plx"].values  # mas
    # eqmux = data["mux"].values  # mas/yr (equatorial frame)
    # e_eqmux = data["e_mux"].values  # mas/y (equatorial frame)
    # eqmuy = data["muy"].values  # mas/y (equatorial frame)
    # e_eqmuy = data["e_muy"].values  # mas/y (equatorial frame)
    # vlsr = data["vlsr"].values  # km/s
    # e_vlsr = data["e_vlsr"].values  # km/s
    # eqmux_pred, eqmuy_pred, vlsr_pred = data_to_gcen_cyl_residuals(
    #     data, trace, free_Zsun=free_Zsun, free_roll=free_roll, free_Wpec=free_Wpec)

    # # === Peculiar motions ===
    # eqmux_pec = eqmux - eqmux_pred
    # eqmuy_pec = eqmuy - eqmuy_pred
    # vlsr_pec = vlsr - vlsr_pred


if __name__ == "__main__":
    prior_set_file = input("prior_set of file (A1, A5, B, C, D): ")
    num_samples_file = int(input("Number of distance samples per source in file (int): "))
    num_rounds_file = int(input("round number of file for best-fit parameters (int): "))

    main(prior_set_file, num_samples_file, num_rounds_file)