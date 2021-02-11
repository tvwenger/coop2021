"""
fit_pec_motions.py

Fits the vector peculiar motions of sources using a cubic spline interpolation.

Isaac Cheng - February 2021
"""
import sys
from pathlib import Path
import numpy as np
import dill
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

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

    print("=== Fitting peculiar motions for "
          f"({prior_set} priors & {num_rounds} MCMC rounds) ===")
    print("Number of sources:", num_sources)
    print("Likelihood function:", like_type)

    # Get residual motions & ratio of radial to circular velocity
    x, y, z, vx_res, vy_res, vz_res, vrad_vcirc = get_pos_and_residuals_and_vrad_vtan(
      data, trace, free_Zsun=free_Zsun, free_roll=free_roll)

    # 500 by 500 grid
    grid_x, grid_y = np.mgrid[np.min(x):np.max(x):500j, np.min(y):np.max(y):500j]
    points = np.vstack((x, y)).T  # shape: (num_sources, 2) --> 2D data
    # v_res = np.sqrt(vx_res * vx_res + vy_res * vy_res)
    # print(x.shape)
    # print(y.shape)
    # print(vx_res.shape)
    # print(vy_res.shape)
    # print(v_res.shape)
    vx_res_fit = griddata(points, vx_res, (grid_x, grid_y), method="cubic")
    norm = mpl.colors.Normalize(vmin=np.min(vx_res_fit), vmax=np.max(vx_res_fit))
    plt.imshow(vx_res_fit.T, origin="lower", extent=(-8,12,-5,15))
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), format="%.4f")
    plt.show()


if __name__ == "__main__":
    prior_set_file = input("prior_set of file (A1, A5, B, C, D): ")
    num_samples_file = int(input("Number of distance samples per source in file (int): "))
    num_rounds_file = int(input("round number of file for best-fit parameters (int): "))

    main(prior_set_file, num_samples_file, num_rounds_file)