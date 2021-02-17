"""
fit_pec_motions_krige.py

Fits the vector peculiar motions of sources using kriging.

Isaac Cheng - February 2021
"""

import sys
from pathlib import Path
import numpy as np
import dill
import matplotlib as mpl
import matplotlib.pyplot as plt
from pykrige.uk import UniversalKriging
from kriging import kriging

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans
from plot_vrad_vtan import (
    get_pos_and_residuals_and_vrad_vtan,
    get_cart_pos_and_cyl_residuals,
)

# Roll angle between galactic midplane and galactocentric frame
_ROLL = 0.0  # deg (Anderson et al. 2019)
# Sun's height above galactic midplane (Reid et al. 2019)
_ZSUN = 5.5  # pc
# Useful constants
_RAD_TO_DEG = 57.29577951308232  # 180/pi (Don't forget to % 360 after)


def plot_cartesian_residuals(
    prior_set,
    num_samples,
    num_rounds,
    data,
    trace,
    num_sources,
    free_Zsun=False,
    free_roll=False,
):
    """
    Plots peculiar motion components in Cartesian coordinates
    (i.e. residual velocities in x- & y-components)
    """

    # Get residual motions & ratio of radial to circular velocity
    x, y, z, vx_res, vy_res, vz_res, vrad_vcirc = get_pos_and_residuals_and_vrad_vtan(
        data, trace, free_Zsun=free_Zsun, free_roll=free_roll
    )
    # Remove very far sources (not enough data to accurately interpolate)
    vx_res = vx_res[y > -5]
    vy_res = vy_res[y > -5]
    x = x[y > -5]
    y = y[y > -5]

    variogram_model = "spherical"  # use "gaussian" or "spherical"
    print("Variogram Model:", variogram_model)
    vx_res_fit = UniversalKriging(
        x,
        y,
        vx_res,
        variogram_model=variogram_model,
        exact_values=False,
        # verbose=False,
        # enable_plotting=True,
    )
    vy_res_fit = UniversalKriging(
        x,
        y,
        vy_res,
        variogram_model=variogram_model,
        exact_values=False,
        # verbose=False,
        # enable_plotting=True,
    )

    # gridx, gridy = np.mgrid[-8:12:500j, -5:15:500j]
    gridx = np.linspace(-8, 12, 500)
    gridy = np.linspace(-5, 15, 500)

    # Interpolate
    vx_res_interp, vx_res_interp_sigmasq = vx_res_fit.execute("grid", gridx, gridy)
    vy_res_interp, vy_res_interp_sigmasq = vy_res_fit.execute("grid", gridx, gridy)

    # Standard deviations of interpolated values
    vx_res_interp_sigma = np.sqrt(vx_res_interp_sigmasq)
    vy_res_interp_sigma = np.sqrt(vy_res_interp_sigmasq)

    print("mean vx_res_interp:", np.mean(vx_res_interp))
    print("min & max vx_res_interp:", np.min(vx_res_interp), np.max(vx_res_interp))
    print("vx_res_interp mean sd:", np.mean(vx_res_interp_sigma))
    print(
        "vx_res_interp min & max sd:",
        np.min(vx_res_interp_sigma),
        np.max(vx_res_interp_sigma),
    )
    print()
    print("mean vy_res_interp:", np.mean(vy_res_interp))
    print("min & max vy_res_interp:", np.min(vy_res_interp), np.max(vy_res_interp))
    print("vy_res_interp mean sd:", np.mean(vy_res_interp_sigma))
    print(
        "vy_res_interp min & max sd:",
        np.min(vy_res_interp_sigma),
        np.max(vy_res_interp_sigma),
    )

    print("vx_res_interp nan detected!") if np.sum(np.isnan(vx_res_interp)) else None
    print("vy_res_interp nan detected!") if np.sum(np.isnan(vy_res_interp)) else None
    print("vx_res_interp_sigmasq nan detected!") if np.sum(
        np.isnan(vx_res_interp_sigmasq)
    ) else None
    print("vy_res_interp_sigmasq nan detected!") if np.sum(
        np.isnan(vy_res_interp_sigmasq)
    ) else None
    print("=" * 6)

    # === Peculiar motion plot parameters ===
    fig, ax = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    cmap = "viridis"

    # Plot residual x-component
    norm_x = mpl.colors.Normalize(
        vmin=np.min([np.min(vx_res_interp), np.min(vx_res)]),
        vmax=np.max([np.max(vx_res_interp), np.max(vx_res)]),
    )
    ax[0].imshow(vx_res_interp.T, origin="lower", extent=(-8, 12, -5, 15), norm=norm_x)
    cbar_x = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_x, cmap=cmap), ax=ax[0], format="%.0f"
    )
    ax[0].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[0].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[0].set_xlabel("x (kpc)")
    ax[0].set_ylabel("y (kpc)")
    ax[0].set_title("x-component")
    cbar_x.ax.set_ylabel("Residual x-velocity", rotation=270)
    cbar_x.ax.get_yaxis().labelpad = 15
    ax[0].set_aspect("equal")
    ax[0].grid(False)

    # Plot residual y-components
    norm_y = mpl.colors.Normalize(
        vmin=np.min([np.min(vy_res_interp), np.min(vy_res)]),
        vmax=np.max([np.max(vy_res_interp), np.max(vy_res)]),
    )
    ax[1].imshow(vy_res_interp.T, origin="lower", extent=(-8, 12, -5, 15), norm=norm_y)
    cbar_y = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_y, cmap=cmap), ax=ax[1], format="%.0f"
    )
    ax[1].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[1].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[1].set_xlabel("x (kpc)")
    ax[1].set_ylabel("y (kpc)")
    ax[1].set_title("y-component")
    cbar_y.ax.set_ylabel("Residual y-velocity", rotation=270)
    cbar_y.ax.get_yaxis().labelpad = 15
    ax[1].set_aspect("equal")
    ax[1].grid(False)

    # Plot actual residual motion data
    ax[0].scatter(
        x, y, c=vx_res, norm=norm_x, cmap=cmap, s=10, edgecolors="k", label="Masers"
    )
    ax[0].set_xlim(-8, 12)
    ax[0].set_xticks([-5, 0, 5, 10])
    ax[0].set_ylim(-5, 15)
    ax[0].set_yticks([-5, 0, 5, 10, 15])
    ax[0].legend(loc="lower left", fontsize=9)
    ax[1].scatter(
        x, y, c=vy_res, norm=norm_y, cmap=cmap, s=10, edgecolors="k", label="Masers"
    )
    ax[1].set_xlim(-8, 12)
    ax[1].set_xticks([-5, 0, 5, 10])
    ax[1].set_ylim(-5, 15)
    ax[1].set_yticks([-5, 0, 5, 10, 15])
    ax[1].legend(loc="lower left", fontsize=9)

    fig.suptitle(
        f"Interpolated Peculiar Motions of {num_sources} Masers\n"
        fr"(Universal Kriging, \texttt{{variogram\_model={variogram_model}}})"
    )
    # fig.suptitle("Interpolated Peculiar Motions of", len(x), "Masers")
    fig.tight_layout()
    filename = f"pec_mot_krige_{prior_set}_{num_samples}dist_{num_rounds}_cart.jpg"
    fig.savefig(
        Path(__file__).parent / filename, format="jpg", dpi=300, bbox_inches="tight",
    )
    plt.show()

    # === Standard deviation of pec motion plot parameters ===
    fig2, ax2 = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    cmap2 = "viridis"

    # Plot residual x-component standard deviations
    norm_x_sd = mpl.colors.Normalize(
        vmin=np.min(vx_res_interp_sigma), vmax=np.max(vx_res_interp_sigma),
    )
    ax2[0].imshow(
        vx_res_interp_sigma.T, origin="lower", extent=(-8, 12, -5, 15), norm=norm_x_sd
    )
    cbar2_x = fig2.colorbar(
        mpl.cm.ScalarMappable(norm=norm_x_sd, cmap=cmap2), ax=ax2[0], format="%.0f"
    )
    ax2[0].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax2[0].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax2[0].set_xlabel("x (kpc)")
    ax2[0].set_ylabel("y (kpc)")
    ax2[0].set_title("x-component")
    cbar2_x.ax.set_ylabel("Standard Deviation", rotation=270)
    cbar2_x.ax.get_yaxis().labelpad = 15
    ax2[0].set_aspect("equal")
    ax2[0].grid(False)

    # Plot residual y-component standard deviations
    norm_y_sd = mpl.colors.Normalize(
        vmin=np.min(vy_res_interp_sigma), vmax=np.max(vy_res_interp_sigma),
    )
    ax2[1].imshow(
        vy_res_interp_sigma.T, origin="lower", extent=(-8, 12, -5, 15), norm=norm_y_sd
    )
    cbar2_y = fig2.colorbar(
        mpl.cm.ScalarMappable(norm=norm_y_sd, cmap=cmap2), ax=ax2[1], format="%.1f"
    )
    ax2[1].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax2[1].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax2[1].set_xlabel("x (kpc)")
    ax2[1].set_ylabel("y (kpc)")
    ax2[1].set_title("y-component")
    cbar2_y.ax.set_ylabel("Standard Deviation", rotation=270)
    cbar2_y.ax.get_yaxis().labelpad = 15
    ax2[1].set_aspect("equal")
    ax2[1].grid(False)

    fig2.suptitle(
        f"Standard Deviations of Interpolated Peculiar Motions ({num_sources} Masers)\n"
        fr"(Universal Kriging, \texttt{{variogram\_model={variogram_model}}})"
    )
    # fig.suptitle("Interpolated Peculiar Motions of", len(x), "Masers")
    fig2.tight_layout()
    filename2 = f"pec_mot_krige_{prior_set}_{num_samples}dist_{num_rounds}_cart_sd.jpg"
    fig2.savefig(
        Path(__file__).parent / filename2, format="jpg", dpi=300, bbox_inches="tight",
    )
    plt.show()


def plot_cylindrical_residuals_old(
    prior_set,
    num_samples,
    num_rounds,
    data,
    trace,
    num_sources,
    free_Zsun=False,
    free_roll=False,
):
    """
    Plots radial & tangential peculiar motion components
    """

    # Get residual motions & ratio of radial to circular velocity
    # x, y, z, vx_res, _circ, vz_res, vrad_vcirc = get_pos_and_residuals_and_vrad_vtan(
    #     data, trace, free_Zsun=free_Zsun, free_roll=free_roll
    # )
    x, y, z, v_rad_res, v_circ_res, v_vert_res = get_cart_pos_and_cyl_residuals(
        data, trace, free_Zsun=free_Zsun, free_roll=free_roll
    )
    # Remove very far sources (not enough data to accurately interpolate)
    print(len(y))
    v_rad_res = v_rad_res[y > -5]
    v_circ_res = v_circ_res[y > -5]
    x = x[y > -5]
    y = y[y > -5]
    print(len(y))

    variogram_model = "power"  # use "gaussian" or "spherical"
    print("Variogram Model:", variogram_model)
    v_rad_res_fit = UniversalKriging(
        x,
        y,
        v_rad_res,
        variogram_model=variogram_model,
        exact_values=False,
        # verbose=False,
        enable_plotting=True,
    )
    v_circ_res_fit = UniversalKriging(
        x,
        y,
        v_circ_res,
        variogram_model=variogram_model,
        exact_values=False,
        # verbose=False,
        enable_plotting=True,
    )

    # gridx, gridy = np.mgrid[-8:12:500j, -5:15:500j]
    gridx = np.linspace(-8, 12, 500)
    gridy = np.linspace(-5, 15, 500)

    # Interpolate
    vrad_res_interp, vrad_res_interp_sigmasq = v_rad_res_fit.execute("grid", gridx, gridy)
    vcirc_res_interp, vcirc_res_interp_sigmasq = v_circ_res_fit.execute(
        "grid", gridx, gridy
    )

    # Standard deviations of interpolated values
    vrad_res_interp_sigma = np.sqrt(vrad_res_interp_sigmasq)
    vcirc_res_interp_sigma = np.sqrt(vcirc_res_interp_sigmasq)

    print("mean vrad_res_interp:", np.mean(vrad_res_interp))
    print("min & max vrad_res_interp:", np.min(vrad_res_interp), np.max(vrad_res_interp))
    print("vrad_res_interp mean sd:", np.mean(vrad_res_interp_sigma))
    print(
        "vrad_res_interp min & max sd:",
        np.min(vrad_res_interp_sigma),
        np.max(vrad_res_interp_sigma),
    )
    print()
    print("mean vcirc_res_interp:", np.mean(vcirc_res_interp))
    print(
        "min & max vcirc_res_interp:", np.min(vcirc_res_interp), np.max(vcirc_res_interp)
    )
    print("vcirc_res_interp mean sd:", np.mean(vcirc_res_interp_sigma))
    print(
        "vcirc_res_interp min & max sd:",
        np.min(vcirc_res_interp_sigma),
        np.max(vcirc_res_interp_sigma),
    )

    print("vrad_res_interp nan detected!") if np.sum(np.isnan(vrad_res_interp)) else None
    print("vcirc_res_interp nan detected!") if np.sum(
        np.isnan(vcirc_res_interp)
    ) else None
    print("vrad_res_interp_sigmasq nan detected!") if np.sum(
        np.isnan(vrad_res_interp_sigmasq)
    ) else None
    print("vcirc_res_interp_sigmasq nan detected!") if np.sum(
        np.isnan(vcirc_res_interp_sigmasq)
    ) else None
    print("=" * 6)

    # === Peculiar motion plot parameters ===
    fig, ax = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    cmap = "viridis"

    # Plot residual radial components
    norm_rad = mpl.colors.Normalize(
        vmin=np.min([np.min(vrad_res_interp), np.min(v_rad_res)]),
        vmax=np.max([np.max(vrad_res_interp), np.max(v_rad_res)]),
    )
    ax[0].imshow(
        vrad_res_interp.T, origin="lower", extent=(-8, 12, -5, 15), norm=norm_rad
    )
    cbar_rad = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_rad, cmap=cmap), ax=ax[0], format="%.0f"
    )
    ax[0].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[0].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[0].set_xlabel("x (kpc)")
    ax[0].set_ylabel("y (kpc)")
    ax[0].set_title("Radial Component")
    cbar_rad.ax.set_ylabel("Residual Radial Velocity", rotation=270)
    cbar_rad.ax.get_yaxis().labelpad = 15
    ax[0].set_aspect("equal")
    ax[0].grid(False)

    # Plot residual tangential components
    norm_circ = mpl.colors.Normalize(
        vmin=np.min([np.min(vcirc_res_interp), np.min(v_circ_res)]),
        vmax=np.max([np.max(vcirc_res_interp), np.max(v_circ_res)]),
    )
    ax[1].imshow(
        vcirc_res_interp.T, origin="lower", extent=(-8, 12, -5, 15), norm=norm_circ
    )
    cbar_circ = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_circ, cmap=cmap), ax=ax[1], format="%.0f"
    )
    ax[1].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[1].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[1].set_xlabel("x (kpc)")
    ax[1].set_ylabel("y (kpc)")
    ax[1].set_title("Tangential Component")
    cbar_circ.ax.set_ylabel("Residual Tangential Velocity", rotation=270)
    cbar_circ.ax.get_yaxis().labelpad = 15
    ax[1].set_aspect("equal")
    ax[1].grid(False)

    # Plot actual residual motion data
    ax[0].scatter(
        x, y, c=v_rad_res, norm=norm_rad, cmap=cmap, s=10, edgecolors="k", label="Masers"
    )
    ax[0].set_xlim(-8, 12)
    ax[0].set_xticks([-5, 0, 5, 10])
    ax[0].set_ylim(-5, 15)
    ax[0].set_yticks([-5, 0, 5, 10, 15])
    ax[0].legend(loc="lower left", fontsize=9)
    ax[1].scatter(
        x,
        y,
        c=v_circ_res,
        norm=norm_circ,
        cmap=cmap,
        s=10,
        edgecolors="k",
        label="Masers",
    )
    ax[1].set_xlim(-8, 12)
    ax[1].set_xticks([-5, 0, 5, 10])
    ax[1].set_ylim(-5, 15)
    ax[1].set_yticks([-5, 0, 5, 10, 15])
    ax[1].legend(loc="lower left", fontsize=9)

    fig.suptitle(
        f"Interpolated Peculiar Motions of {num_sources} Masers\n"
        fr"(Universal Kriging, \texttt{{variogram\_model={variogram_model}}})"
    )
    # fig.suptitle("Interpolated Peculiar Motions of", len(x), "Masers")
    fig.tight_layout()
    filename = f"pec_mot_krige_{prior_set}_{num_samples}dist_{num_rounds}_cyl.jpg"
    fig.savefig(
        Path(__file__).parent / filename, format="jpg", dpi=300, bbox_inches="tight",
    )
    plt.show()

    # === Standard deviation of pec motion plot parameters ===
    fig2, ax2 = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    cmap2 = "viridis"

    # Plot residual radial component standard deviations
    norm_rad_sd = mpl.colors.Normalize(
        vmin=np.min(vrad_res_interp_sigma), vmax=np.max(vrad_res_interp_sigma),
    )
    ax2[0].imshow(
        vrad_res_interp_sigma.T, origin="lower", extent=(-8, 12, -5, 15), norm=norm_rad_sd
    )
    cbar2_rad = fig2.colorbar(
        mpl.cm.ScalarMappable(norm=norm_rad_sd, cmap=cmap2), ax=ax2[0], format="%.0f"
    )
    ax2[0].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax2[0].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax2[0].set_xlabel("x (kpc)")
    ax2[0].set_ylabel("y (kpc)")
    ax2[0].set_title("Radial Component")
    cbar2_rad.ax.set_ylabel("Standard Deviation", rotation=270)
    cbar2_rad.ax.get_yaxis().labelpad = 15
    ax2[0].set_aspect("equal")
    ax2[0].grid(False)

    # Plot residual tangential component standard deviations
    norm_circ_sd = mpl.colors.Normalize(
        vmin=np.min(vcirc_res_interp_sigma), vmax=np.max(vcirc_res_interp_sigma),
    )
    ax2[1].imshow(
        vcirc_res_interp_sigma.T,
        origin="lower",
        extent=(-8, 12, -5, 15),
        norm=norm_circ_sd,
    )
    cbar2_circ = fig2.colorbar(
        mpl.cm.ScalarMappable(norm=norm_circ_sd, cmap=cmap2), ax=ax2[1], format="%.1f"
    )
    ax2[1].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax2[1].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax2[1].set_xlabel("x (kpc)")
    ax2[1].set_ylabel("y (kpc)")
    ax2[1].set_title("Tangential Component")
    cbar2_circ.ax.set_ylabel("Standard Deviation", rotation=270)
    cbar2_circ.ax.get_yaxis().labelpad = 15
    ax2[1].set_aspect("equal")
    ax2[1].grid(False)

    fig2.suptitle(
        f"Standard Deviations of Interpolated Peculiar Motions ({num_sources} Masers)\n"
        fr"(Universal Kriging, \texttt{{variogram\_model={variogram_model}}})"
    )
    fig2.tight_layout()
    filename2 = f"pec_mot_krige_{prior_set}_{num_samples}dist_{num_rounds}_cyl_sd.jpg"
    fig2.savefig(
        Path(__file__).parent / filename2, format="jpg", dpi=300, bbox_inches="tight",
    )
    plt.show()


def plot_cylindrical_residuals(
    prior_set,
    num_samples,
    num_rounds,
    data,
    trace,
    num_sources,
    free_Zsun=False,
    free_roll=False,
):
    """
    Plots radial & tangential peculiar motion components
    """

    # Get residual motions & ratio of radial to circular velocity
    x, y, z, v_rad_res, v_circ_res, v_vert_res = get_cart_pos_and_cyl_residuals(
        data, trace, free_Zsun=free_Zsun, free_roll=free_roll
    )

    # Remove very far sources (not enough data to accurately interpolate)
    # print(len(y))
    v_rad_res = v_rad_res[y > -5]
    v_circ_res = v_circ_res[y > -5]
    x = x[y > -5]
    y = y[y > -5]
    # print(len(y))

    # Data coordinates (size=(num_data, 2))
    coord_obs = np.vstack((x, y)).T

    # Interpolation grid
    # num_xpoints = 500j
    # num_ypoints = 500j
    gridx, gridy = np.mgrid[-8:12:500j, -5:15:500j]
    coord_interp = np.vstack((gridx.flatten(), gridy.flatten())).T

    # Universal kriging with linear drift term
    variogram_model = "gaussian"  # "gaussian", "spherical", or "exponential"
    print("Variogram Model:", variogram_model)

    vrad_res_interp, vrad_res_interp_sigmasq = kriging.kriging(
        coord_obs,
        v_rad_res,
        coord_interp,
        model=variogram_model,
        deg=1,
        nbins=10,
        bin_number=True,
        plot=Path(__file__).parent
        / f"semivariogram_vrad_{prior_set}_{num_samples}dist_{num_rounds}_{variogram_model}.jpg",
    )
    vcirc_res_interp, vcirc_res_interp_sigmasq = kriging.kriging(
        coord_obs,
        v_circ_res,
        coord_interp,
        model=variogram_model,
        deg=1,
        nbins=10,
        bin_number=True,
        plot=Path(__file__).parent
        / f"semivariogram_vcirc_{prior_set}_{num_samples}dist_{num_rounds}_{variogram_model}.jpg",
    )
    vrad_res_interp = vrad_res_interp.reshape(500, 500)
    vrad_res_interp_sigmasq = vrad_res_interp_sigmasq.reshape(500, 500)
    vcirc_res_interp = vcirc_res_interp.reshape(500, 500)
    vcirc_res_interp_sigmasq = vcirc_res_interp_sigmasq.reshape(500, 500)
    print("min vcirc variance", np.min(vcirc_res_interp_sigmasq))

    # Standard deviations of interpolated values
    vrad_res_interp_sigma = np.sqrt(vrad_res_interp_sigmasq)
    vcirc_res_interp_sigma = np.sqrt(vcirc_res_interp_sigmasq)

    print("mean interpolated radial peculiar motion:", np.mean(vrad_res_interp))
    print(
        "min & max interpolated radial peculiar motion:",
        np.min(vrad_res_interp), np.max(vrad_res_interp),
    )
    print(
        "mean standard deviation of interpolated radial peculiar motion:",
        np.nanmean(vrad_res_interp_sigma),
    )
    print(
        "min & max standard deviation of interpolated radial peculiar motion:",
        np.nanmin(vrad_res_interp_sigma), np.nanmax(vrad_res_interp_sigma),
    )
    print()
    print("mean interpolated tangential peculiar motion:", np.mean(vcirc_res_interp))
    print(
        "min & max interpolated tangential peculiar motion:",
        np.min(vcirc_res_interp), np.max(vcirc_res_interp),
    )
    print(
        "mean standard deviation of interpolated tangential peculiar motion:",
        np.nanmean(vcirc_res_interp_sigma),
    )
    print(
        "min & max standard deviation of interpolated tangential peculiar motion:",
        np.nanmin(vcirc_res_interp_sigma), np.nanmax(vcirc_res_interp_sigma),
    )

    # Check for nans
    print("nan in vrad variance detected!") if np.sum(
        np.isnan(vrad_res_interp_sigmasq)
    ) else None
    print("nan in vcirc variance detected!") if np.sum(
        np.isnan(vcirc_res_interp_sigmasq)
    ) else None

    num_nans_vrad = np.sum(np.isnan(vrad_res_interp))
    num_nans_vrad_sigma = np.sum(np.isnan(vrad_res_interp_sigma))
    num_nans_vcirc = np.sum(np.isnan(vcirc_res_interp))
    num_nans_vcirc_sigma = np.sum(np.isnan(vcirc_res_interp_sigma))
    print(
        "# nans in interpolated radial peculiar motion:", num_nans_vrad
    ) if num_nans_vrad else None
    print(
        "# nans in standard deviation of interpolated radial peculiar motion:",
        num_nans_vrad_sigma,
    ) if num_nans_vrad_sigma else None
    print(
        "# nans in interpolated tangential peculiar motion:", num_nans_vcirc
    ) if num_nans_vcirc else None
    print(
        "# nans in standard deviation of interpolated tangential peculiar motion:",
        num_nans_vcirc_sigma,
    ) if num_nans_vcirc_sigma else None
    print("=" * 6)

    # === Peculiar motion plot parameters ===
    fig, ax = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    cmap = "viridis"

    # Plot residual radial components
    norm_rad = mpl.colors.Normalize(
        vmin=np.nanmin([np.nanmin(vrad_res_interp), np.nanmin(v_rad_res)]),
        vmax=np.nanmax([np.nanmax(vrad_res_interp), np.nanmax(v_rad_res)]),
    )
    ax[0].imshow(
        vrad_res_interp.T, origin="lower", extent=(-8, 12, -5, 15), norm=norm_rad
    )
    cbar_rad = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_rad, cmap=cmap), ax=ax[0], format="%.0f"
    )
    ax[0].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[0].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[0].set_xlabel("x (kpc)")
    ax[0].set_ylabel("y (kpc)")
    ax[0].set_title("Radial Component")
    cbar_rad.ax.set_ylabel("Residual Radial Velocity", rotation=270)
    cbar_rad.ax.get_yaxis().labelpad = 15
    ax[0].set_aspect("equal")
    ax[0].grid(False)

    # Plot residual tangential components
    norm_circ = mpl.colors.Normalize(
        vmin=np.nanmin([np.nanmin(vcirc_res_interp), np.nanmin(v_circ_res)]),
        vmax=np.nanmax([np.nanmax(vcirc_res_interp), np.nanmax(v_circ_res)]),
    )
    ax[1].imshow(
        vcirc_res_interp.T, origin="lower", extent=(-8, 12, -5, 15), norm=norm_circ
    )
    cbar_circ = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_circ, cmap=cmap), ax=ax[1], format="%.0f"
    )
    ax[1].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[1].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[1].set_xlabel("x (kpc)")
    ax[1].set_ylabel("y (kpc)")
    ax[1].set_title("Tangential Component")
    cbar_circ.ax.set_ylabel("Residual Tangential Velocity", rotation=270)
    cbar_circ.ax.get_yaxis().labelpad = 15
    ax[1].set_aspect("equal")
    ax[1].grid(False)

    # Plot actual residual motion data
    ax[0].scatter(
        x, y, c=v_rad_res, norm=norm_rad, cmap=cmap, s=10, edgecolors="k", label="Masers"
    )
    ax[0].set_xlim(-8, 12)
    ax[0].set_xticks([-5, 0, 5, 10])
    ax[0].set_ylim(-5, 15)
    ax[0].set_yticks([-5, 0, 5, 10, 15])
    ax[0].legend(loc="lower left", fontsize=9)
    ax[1].scatter(
        x,
        y,
        c=v_circ_res,
        norm=norm_circ,
        cmap=cmap,
        s=10,
        edgecolors="k",
        label="Masers",
    )
    ax[1].set_xlim(-8, 12)
    ax[1].set_xticks([-5, 0, 5, 10])
    ax[1].set_ylim(-5, 15)
    ax[1].set_yticks([-5, 0, 5, 10, 15])
    ax[1].legend(loc="lower left", fontsize=9)

    fig.suptitle(
        f"Interpolated Peculiar Motions of {num_sources} Masers\n"
        fr"(Universal Kriging, \texttt{{variogram\_model={variogram_model}}})"
    )
    # fig.suptitle("Interpolated Peculiar Motions of", len(x), "Masers")
    fig.tight_layout()
    filename = f"pec_mot_krige_{prior_set}_{num_samples}dist_{num_rounds}_cyl_{variogram_model}.jpg"
    fig.savefig(
        Path(__file__).parent / filename, format="jpg", dpi=300, bbox_inches="tight",
    )
    plt.show()

    # === Standard deviation of pec motion plot parameters ===
    fig2, ax2 = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    cmap2 = "viridis"

    # Plot residual radial component standard deviations
    norm_rad_sd = mpl.colors.Normalize(
        vmin=np.nanmin(vrad_res_interp_sigma), vmax=np.nanmax(vrad_res_interp_sigma),
    )
    ax2[0].imshow(
        vrad_res_interp_sigma.T, origin="lower", extent=(-8, 12, -5, 15), norm=norm_rad_sd
    )
    cbar2_rad = fig2.colorbar(
        mpl.cm.ScalarMappable(norm=norm_rad_sd, cmap=cmap2), ax=ax2[0], format="%.0f"
    )
    ax2[0].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax2[0].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax2[0].set_xlabel("x (kpc)")
    ax2[0].set_ylabel("y (kpc)")
    ax2[0].set_title("Radial Component")
    cbar2_rad.ax.set_ylabel("Standard Deviation", rotation=270)
    cbar2_rad.ax.get_yaxis().labelpad = 15
    ax2[0].set_aspect("equal")
    ax2[0].grid(False)

    # Plot residual tangential component standard deviations
    norm_circ_sd = mpl.colors.Normalize(
        vmin=np.nanmin(vcirc_res_interp_sigma), vmax=np.nanmax(vcirc_res_interp_sigma),
    )
    ax2[1].imshow(
        vcirc_res_interp_sigma.T,
        origin="lower",
        extent=(-8, 12, -5, 15),
        norm=norm_circ_sd,
    )
    cbar2_circ = fig2.colorbar(
        mpl.cm.ScalarMappable(norm=norm_circ_sd, cmap=cmap2), ax=ax2[1], format="%.1f"
    )
    ax2[1].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax2[1].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax2[1].set_xlabel("x (kpc)")
    ax2[1].set_ylabel("y (kpc)")
    ax2[1].set_title("Tangential Component")
    cbar2_circ.ax.set_ylabel("Standard Deviation", rotation=270)
    cbar2_circ.ax.get_yaxis().labelpad = 15
    ax2[1].set_aspect("equal")
    ax2[1].grid(False)

    fig2.suptitle(
        f"Standard Deviations of Interpolated Peculiar Motions ({num_sources} Masers)\n"
        fr"(Universal Kriging, \texttt{{variogram\_model={variogram_model}}})"
    )
    fig2.tight_layout()
    filename2 = f"pec_mot_krige_{prior_set}_{num_samples}dist_{num_rounds}_cyl_sd_{variogram_model}.jpg"
    fig2.savefig(
        Path(__file__).parent / filename2, format="jpg", dpi=300, bbox_inches="tight",
    )
    plt.show()


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

    print(
        "=== Fitting peculiar motions (KRIGING) for "
        f"({prior_set} priors & {num_rounds} MCMC rounds) ==="
    )
    print("Number of sources:", num_sources)
    print("Likelihood function:", like_type)
    plot_cylindrical_residuals(
        prior_set,
        num_samples,
        num_rounds,
        data,
        trace,
        num_sources,
        free_Zsun=free_Zsun,
        free_roll=free_roll,
    )


if __name__ == "__main__":
    # prior_set_file = input("prior_set of file (A1, A5, B, C, D): ")
    # num_samples_file = int(input("Number of distance samples per source in file (int): "))
    # num_rounds_file = int(input("round number of file for best-fit parameters (int): "))
    prior_set_file = "A5"
    num_samples_file = 100
    num_rounds_file = 5
    main(prior_set_file, num_samples_file, num_rounds_file)
