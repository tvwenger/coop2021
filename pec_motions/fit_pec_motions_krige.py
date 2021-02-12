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
import pykrige.kriging_tools as kt
from pykrige.uk import UniversalKriging

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans
from plot_vrad_vtan import get_pos_and_residuals_and_vrad_vtan

# Roll angle between galactic midplane and galactocentric frame
_ROLL = 0.0  # deg (Anderson et al. 2019)
# Sun's height above galactic midplane (Reid et al. 2019)
_ZSUN = 5.5  # pc
# Useful constants
_RAD_TO_DEG = 57.29577951308232  # 180/pi (Don't forget to % 360 after)


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

    # Get residual motions & ratio of radial to circular velocity
    x, y, z, vx_res, vy_res, vz_res, vrad_vcirc = get_pos_and_residuals_and_vrad_vtan(
        data, trace, free_Zsun=free_Zsun, free_roll=free_roll
    )
    # Remove very far sources (not enough data to accurately interpolate)
    vx_res = vx_res[y > -5]
    vy_res = vy_res[y > -5]
    x = x[y > -5]
    y = y[y > -5]

    variogram_model = "gaussian"  # use "gaussian" or "spherical"
    vx_res_fit = UniversalKriging(
        x,
        y,
        vx_res,
        variogram_model=variogram_model,
        exact_values=False,
        # verbose=False,
        enable_plotting=True,
    )
    vy_res_fit = UniversalKriging(
        x,
        y,
        vy_res,
        variogram_model=variogram_model,
        exact_values=False,
        # verbose=False,
        enable_plotting=True,
    )
    print("Variogram Model:", variogram_model)

    # gridx, gridy = np.mgrid[-8:12:500j, -5:15:500j]
    gridx = np.linspace(-8, 12, 500)
    gridy = np.linspace(-5, 15, 500)

    # Interpolate
    vx_res_interp, vx_res_interp_sigmasq = vx_res_fit.execute("grid", gridx, gridy)
    vy_res_interp, vy_res_interp_sigmasq = vy_res_fit.execute("grid", gridx, gridy)
    print("mean vx_res_interp:", np.mean(vx_res_interp))
    print("min & max vx_res_interp:", np.min(vx_res_interp), np.max(vx_res_interp))
    print("vx_res_interp mean sd:", np.mean(np.sqrt(vx_res_interp_sigmasq)))
    print(
        "vx_res_interp min & max sd:",
        np.min(np.sqrt(vx_res_interp_sigmasq)),
        np.max(np.sqrt(vx_res_interp_sigmasq)),
    )
    print()
    print("mean vy_res_interp:", np.mean(vy_res_interp))
    print("min & max vy_res_interp:", np.min(vy_res_interp), np.max(vy_res_interp))
    print("vy_res_interp mean sd:", np.mean(np.sqrt(vy_res_interp_sigmasq)))
    print(
        "vy_res_interp min & max sd:",
        np.min(np.sqrt(vy_res_interp_sigmasq)),
        np.max(np.sqrt(vy_res_interp_sigmasq)),
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

    # Plot parameters
    fig, ax = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    # cmap = copy.copy(mpl.cm.get_cmap("viridis"))
    # cmap.set_bad(color="black")
    cmap = "viridis"

    # Plot residual x-component
    norm_x = mpl.colors.Normalize(
        vmin=np.min([np.min(vx_res_interp), np.min(vx_res)]),
        vmax=np.max([np.max(vx_res_interp), np.max(vx_res)]),
    )
    # norm_x = mpl.colors.Normalize(vmin=-150, vmax=150)
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
    # norm_y = mpl.colors.Normalize(vmin=-150, vmax=150)
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
        x, y, c=vx_res, norm=norm_x, cmap="viridis", s=10, edgecolors="k", label="Masers"
    )
    ax[0].set_xlim(-8, 12)
    ax[0].set_xticks([-5, 0, 5, 10])
    ax[0].set_ylim(-5, 15)
    ax[0].set_yticks([-5, 0, 5, 10, 15])
    ax[0].legend(loc="lower left", fontsize=9)
    ax[1].scatter(
        x, y, c=vy_res, norm=norm_y, cmap="viridis", s=10, edgecolors="k", label="Masers"
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
    filename = f"pec_mot_interp_{prior_set}_{num_samples}dist_{num_rounds}_krige.jpg"
    fig.savefig(
        Path(__file__).parent / filename, format="jpg", dpi=300, bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    # prior_set_file = input("prior_set of file (A1, A5, B, C, D): ")
    # num_samples_file = int(input("Number of distance samples per source in file (int): "))
    # num_rounds_file = int(input("round number of file for best-fit parameters (int): "))
    prior_set_file = "A5"
    num_samples_file = 100
    num_rounds_file = 5
    main(prior_set_file, num_samples_file, num_rounds_file)
