"""
plot_vrad_vtan.py

Plots the peculiar (non-circular) motions of the sources,
colour-coded by ratio of radial to azimuthal velocity

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

def data_to_gcen_cyl(data, trace, free_Zsun=False, free_roll=False):
    """
    Converts database data to
    galactocentric cylindrical coordinates and velocities

    Returns galactocentric cylindrical coordinates & their
    residual motions in the galactocentric cylindrical frame

    Inputs:
      data :: pandas DataFrame
        Contains maser galactic longitudes, latitudes, right ascensions, declinations,
        parallaxes, equatorial proper motions, and LSR velocities
        with all associated uncertainties
      trace :: PyMC3 MultiTrace object
        Contains each iteration of the Bayesian MCMC algorithm for every parameter
      free_Zsun, free_roll :: booleans (default: False)
        True iff Zsun or roll are free parameters in the model

    Returns: radius, azimuth, height, v_radial, v_circ, v_vert
      radius : Array of scalars (kpc)
        Radial distance perpendicular to z-axis
      azimuth : Array of scalars (deg)
        Azimuthal angle; positive CW from -x-axis (left-hand convention!)
      height : Array of scalars (kpc)
        Height above xy-plane (i.e. z_kpc)
      v_radial : Array of scalars (km/s)
        Radial velocity; positive away from z-axis
      v_circ : Array of scalars (km/s)
        Tangential velocity; positive CW (left-hand convention!)
      v_vert : Array of scalars (km/s)
        Velocity perp. to xy-plane; positive if pointing toward NGP (i.e. vz)
    """
    # === Get optimal parameters from MCMC trace ===
    R0 = np.median(trace["R0"])  # kpc
    Vsun = np.median(trace["Vsun"])  # km/s
    Usun = np.median(trace["Usun"])  # km/s
    Wsun = np.median(trace["Wsun"])  # km/s
    a2 = np.median(trace["a2"])  # dimensionless
    a3 = np.median(trace["a3"])  # dimensionless
    Zsun = np.median(trace["Zsun"]) if free_Zsun else _ZSUN  # pc
    roll = np.median(trace["roll"]) if free_roll else _ROLL  # deg

    # R0 = 8.15
    # Usun = 10.6
    # Vsun = 10.7
    # Wsun = 7.6
    # a2 = 0.96
    # a3 = 1.62

    # === Get data ===
    # Slice data into components
    ra = data["ra"]  # deg
    dec = data["dec"]  # deg
    glon = data["glong"]  # deg
    glat = data["glat"]  # deg
    plx = data["plx"]  # mas
    eqmux = data["mux"]  # mas/yr (equatorial frame)
    eqmuy = data["muy"]  # mas/y (equatorial frame)
    vlsr = data["vlsr"]  # km/s

    # === Calculate predicted values from optimal parameters ===
    Theta0 = urc(R0, a2=a2, a3=a3, R0=R0)  # km/s, LSR circular rotation speed
    radius, azimuth, height, v_radial, v_circ, v_vert = trans.eq_and_gal_to_gcen_cyl(
      ra, dec, plx, glon, glat, eqmux, eqmuy, vlsr,
      R0=R0, Zsun=Zsun, roll=roll,
      Usun=Usun, Vsun=Vsun, Wsun=Wsun, Theta0=Theta0,
      use_theano=False, return_only_r_and_theta=False
    )

    return radius, azimuth, height, v_radial, v_circ, v_vert


def data_to_vcirc_pred(data, trace, free_Zsun=False, free_roll=False):
    """
    TODO: add description here

    Inputs:
      data :: pandas DataFrame
        Contains maser galactic longitudes, latitudes, right ascensions, declinations,
        parallaxes, equatorial proper motions, and LSR velocities
        with all associated uncertainties
      trace :: PyMC3 MultiTrace object
        Contains each iteration of the Bayesian MCMC algorithm for every parameter
      free_Zsun, free_roll :: booleans (default: False)
        True iff Zsun or roll are free parameters in the model

    Returns: v_circ_pred
      v_circ_pred :: Array of scalars (km/s)
        Circular rotation speed of source around galactic centre predicted by
        Persic et al.'s 1996 universal rotation curve
    TODO: fix docstring (data & returns descriptions)
    eqmux_res, eqmuy_res, vlsr_res
      eqmux_res :: Array of scalars (mas/yr)
        Residual RA proper motion with cos(Declination) correction
      eqmuy_res :: Array of scalars (mas/yr)
        Residual declination proper motion
      vlsr_res :: Array of scalars (km/s)
        Residual LSR velocity
    """
    # === Get optimal parameters from MCMC trace ===
    R0 = np.median(trace["R0"])  # kpc
    Vsun = np.median(trace["Vsun"])  # km/s
    Usun = np.median(trace["Usun"])  # km/s
    Wsun = np.median(trace["Wsun"])  # km/s
    # Upec = np.median(trace["Upec"])  # km/s
    # Vpec = np.median(trace["Vpec"])  # km/s
    a2 = np.median(trace["a2"])  # dimensionless
    a3 = np.median(trace["a3"])  # dimensionless
    Zsun = np.median(trace["Zsun"]) if free_Zsun else _ZSUN  # pc
    roll = np.median(trace["roll"]) if free_roll else _ROLL  # deg
    # Set Upec = Vpec = 0 as per Reid et al. Fig. 6
    Upec = 0.0  # km/s
    Vpec = 0.0  # km/s

    # R0 = 8.15
    # Usun = 10.6
    # Vsun = 10.7
    # Wsun = 7.6
    # a2 = 0.96
    # a3 = 1.62

    # === Get data ===
    # Slice data into components
    glon = data["glong"].values  # deg
    glat = data["glat"].values  # deg
    plx = data["plx"].values  # mas

    # === Calculate predicted values from optimal parameters ===
    # Parallax to distance
    dist = trans.parallax_to_dist(plx)
    # Galactic to barycentric Cartesian coordinates
    bary_x, bary_y, bary_z = trans.gal_to_bary(glon, glat, dist)
    # Barycentric Cartesian to galactocentric Cartesian coodinates
    gcen_x, gcen_y, gcen_z = trans.bary_to_gcen(
        bary_x, bary_y, bary_z, R0=R0, Zsun=Zsun, roll=roll)
    # Galactocentric Cartesian frame to galactocentric cylindrical frame
    gcen_cyl_dist = np.sqrt(gcen_x * gcen_x + gcen_y * gcen_y)  # kpc
    # azimuth = (np.arctan2(gcen_y, -gcen_x) * _RAD_TO_DEG) % 360  # deg in [0,360)
    v_circ_pred = urc(gcen_cyl_dist, a2=a2, a3=a3, R0=R0) + Vpec  # km/s
    # v_rad = -Upec  # km/s
    # Theta0 = urc(R0, a2=a2, a3=a3, R0=R0)  # km/s, LSR circular rotation speed

    return v_circ_pred


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

    print("=== Plotting v_rad/v_tan plot for "
          f"({prior_set} priors & {num_rounds} MCMC rounds) ===")
    print("Number of sources:", num_sources)
    print("Likelihood function:", like_type)

    # ===
    # Convert database data to galactocentric cylindrical positions & velocities
    (
      radius,
      azimuth,
      height,
      v_rad,
      v_circ,
      v_vert,
    ) = data_to_gcen_cyl(data, trace, free_Zsun=free_Zsun, free_roll=free_roll)

    # # Convert galactocentric cylindrical to galactocentric Cartesian
    # # x, y, z, vx, vy, vz = trans.gcen_cyl_to_gcen_cart(
    # #   radius, azimuth, height,
    # #   v_radial=v_rad, v_tangent=v_circ, v_vertical=v_vert)
    # x, y, z = trans.gcen_cyl_to_gcen_cart(radius, azimuth, height)

    # # Change galactocentric coordinates to Reid's convention
    # # (our convention is detailed in the docstring of trans.gcen_cyl_to_gcen_cart)
    # x, y = y, -x
    # ===

    # ===
    # Find residual motions
    v_circ_pred = data_to_vcirc_pred(
      data, trace, free_Zsun=free_Zsun, free_roll=free_roll)
    v_circ_res = v_circ - v_circ_pred

    # Transform galactocentric cylindrical residual velocities
    # to galactocentric Cartesian residuals
    x, y, z, vx_res, vy_res, vz_res = trans.gcen_cyl_to_gcen_cart(
      radius, azimuth, height,
      v_radial=v_rad, v_tangent=v_circ_res, v_vertical=v_vert)

    # Change galactocentric coordinates to Reid's convention
    # (our convention is detailed in the docstring of trans.gcen_cyl_to_gcen_cart)
    x, y = y, -x
    vx_res, vy_res = vy_res, -vx_res

    # Ratio of radial velocity to circular rotation speed
    vrad_vcirc = v_rad / v_circ
    vrad_vcirc_min = np.min(vrad_vcirc)
    vrad_vcirc_max = np.max(vrad_vcirc)
    print()
    print("Mean v_rad/v_circ:", np.mean(vrad_vcirc))
    print("Min & Max v_rad/v_circ:", vrad_vcirc_min, vrad_vcirc_max)
    print("# v_rad/v_circ > 0:", np.sum(vrad_vcirc > 0))
    print("# v_rad/v_circ < 0:", np.sum(vrad_vcirc < 0))
    print("# v_rad < 0:", np.sum(v_rad < 0))
    print("# v_circ < 0:", np.sum(v_circ < 0))
    print()
    print("Mean residual x-velocity:", np.mean(vx_res))
    print("Mean residual y-velocity:", np.mean(vy_res))
    print("Min & Max residual x-velocity:", np.min(vx_res), np.max(vx_res))
    print("Min & Max residual y-velocity:", np.min(vy_res), np.max(vy_res))
    print()
    v_tot = np.sqrt(vx_res * vx_res + vy_res * vy_res)
    print("Mean magnitude of peculiar velocity:", np.mean(v_tot))
    print("Min & Max magnitudes of peculiar velocity:", np.min(v_tot), np.max(v_tot))
    print("="*6)

    # Define plotting parameters
    fig, ax = plt.subplots()
    cmap = "viridis"  # "coolwarm" is another option
    cmap_min = np.floor(100 * vrad_vcirc_min) / 100
    cmap_max = np.ceil(100 * vrad_vcirc_max) / 100
    norm = mpl.colors.Normalize(vmin=cmap_min, vmax=cmap_max)
    ticks = np.linspace(cmap_min, cmap_max, 8)

    # Plot v_rad / v_circ
    ax.scatter(x, y, c=vrad_vcirc, cmap=cmap, s=2)
    cbar = fig.colorbar(
      mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=ticks, format="%.2f")
    cbar.ax.set_ylabel(r'$v_{rad}/v_{circ}$', rotation=270)
    cbar.ax.get_yaxis().labelpad = 20

    # # Normalize x & y components & scale
    # v_length = np.sqrt(vx_res * vx_res + vy_res * vy_res)
    # scale = 50  # km/s
    # vx_res_norm = vx_res / v_length * scale
    # vy_res_norm = vy_res / v_length * scale
    # # Plot residual velocity vectors
    # ax.quiver(x, y, vx_res_norm, vy_res_norm, vrad_vcirc, cmap=cmap,
    #           # width=0.005,
    #           # headlength=2, headwidth=2,
    #           minlength=0.01 * scale,
    #           linewidth=0.5)
    # Plot residual velocity vectors
    vectors = ax.quiver(x, y, vx_res, vy_res, vrad_vcirc, cmap=cmap,
              minlength=3, width=0.002)
    ax.quiverkey(vectors, X=0.25, Y=0.1, U=-50,
                 label="50 km/s", labelpos="N", fontproperties={"size": 10})

    ax.axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax.axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax.set_xlim(-8, 12)
    ax.set_xticks([-5, 0, 5, 10])
    ax.set_ylim(-5, 15)
    ax.set_yticks([-5, 0, 5, 10, 15])
    # # Using our coordinate convention
    # ax.set_xlim(-15, 5)
    # ax.set_xticks([-15, -10, -5, 0, 5])
    # ax.set_ylim(-8, 12)
    # ax.set_yticks([-5, 0, 5, 10])

    # Set title and labels. Then save figure
    fig.suptitle(f"Face-on View of {num_sources} Masers \& Their Peculiar Motions",
                 x=0.55, y=0.94)
    ax.set_title(r"Colour-coded by their ratio of $v_{rad}$ to $v_{circ}$"
                 f"\nUsed best-fit parameters from {prior_set} priors",
                 fontsize=12)
    ax.set_xlabel("x (kpc)")
    ax.set_ylabel("y (kpc)")
    ax.set_aspect("equal")
    ax.grid(False)
    fig.tight_layout()
    filename = f"vrad_vtan_{prior_set}_{num_samples}dist_{num_rounds}.jpg"
    fig.savefig(
        Path(__file__).parent / filename,
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
    # eqmux_pred, eqmuy_pred, vlsr_pred = data_to_gcen_cyl(
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