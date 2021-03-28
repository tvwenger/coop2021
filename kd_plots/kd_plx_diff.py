"""
kd_plx_diff.py

Computes + saves + plots the differences between
kinematic distances and peak of parallax distances

Isaac Cheng - March 2021
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf
from re import findall  # for regex

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans

# CW21 A6 rotation model parameters
_R0 = 8.1746
_USUN = 10.879
_VSUN = 10.697
_WSUN = 8.088
_ROLL = -0.011
_ZSUN = 5.399


def gaussian_cdf(x, mu=0, sigma=1):
    """
    Computes the cumulative distribution function (CDF)
    of the normal distribution at x.
    The standard normal distribution has mu=0, sigma=1.
    """
    cdf = 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))
    return cdf


def assign_kd_distances(database_data, kd_results, vlsr_tol=20):
    """
    Returns the closest kinematic distance to parallax distance.
    If vlsr (km/s) is within vlsr_tol (km/s) of tangent point vlsr,
    use the tangent point vlsr

    Inputs:
      database_data :: pandas DataFrame
      kd_results :: pandas DataFrame
      vlsr_tol :: scalar
    """
    glong = database_data["glong"].values
    vlsr = database_data["vlsr_med"].values
    peak_dist = database_data["dist_mode"].values
    # Assign tangent kd to any source with vlsr w/in vlsr_tol of tangent vlsr
    use_tangent = abs(kd_results["vlsr_tangent"] - vlsr) < vlsr_tol
    # Otherwise, select kd that is closest to distance from parallax
    # peak_dist = plx_to_peak_dist(plx, e_plx)
    near_err = abs(kd_results["near"] - peak_dist)
    far_err = abs(kd_results["far"] - peak_dist)
    tangent_err = abs(kd_results["tangent"] - peak_dist)
    min_err = np.fmin.reduce([near_err, far_err, tangent_err])  # ignores NaNs
    # Select distance corresponding to smallest error
    tol = 1e-9  # tolerance for float equality
    is_near = (abs(near_err - min_err) < tol) & (~use_tangent)
    is_far = (abs(far_err- min_err) < tol) & (~use_tangent)
    is_tangent = (abs(tangent_err - min_err) < tol) | (use_tangent)
    conditions = [is_near, is_far, is_tangent]
    choices = [kd_results["near"], kd_results["far"], kd_results["tangent"]]
    dists = np.select(conditions, choices, default=np.nan)
    # Exclude any sources w/in 15 deg of GC or 20 deg of GAC
    glong[glong > 180] -= 360  # force -180 < glong <= 180
    is_unreliable = (abs(glong) < 15.) | (abs(glong) > 160.)
    #
    print("=" * 6)
    is_nan = (~is_near) & (~is_far) & (~is_tangent)
    # num_sources = np.sum(np.isfinite(dists)) + np.sum((is_unreliable) & (~is_nan)) - \
    #               np.sum((np.isfinite(dists)) & ((is_unreliable) & (~is_nan)))
    num_sources = np.sum(np.isfinite(dists))
    print("Total number of (non NaN) sources:", num_sources)
    print(f"Num near: {np.sum((is_near) & (~is_unreliable))}"
          + f"\tNum far: {np.sum((is_far) & (~is_unreliable))}"
          + f"\tNum tangent: {np.sum((is_tangent) & (~is_unreliable))}"
          + f"\tNum unreliable: {np.sum((is_unreliable) & (~is_nan))}")
    print("Number of NaN sources (i.e. all dists are NaNs):", np.sum(is_nan))
    print("Num NaNs in near, far, tangent:",
          np.sum(np.isnan(near_err)), np.sum(np.isnan(far_err)),
          np.sum(np.isnan(tangent_err)))
    # Print following if two distances are selected:
    num_near_far = np.sum((is_near) & (is_far))
    num_near_tan = np.sum((is_near) & (is_tangent))
    num_far_tan = np.sum((is_far) & (is_tangent))
    if any([num_near_far, num_near_tan, num_far_tan]):
        print("Both near and far (should be 0):", num_near_far)
        print("Both near and tan (should be 0):", num_near_tan)
        print("Both far and tan (should be 0):", num_far_tan)
    #
    e_near = 0.5 * (kd_results["near_err_pos"] + kd_results["near_err_neg"])
    e_far = 0.5 * (kd_results["far_err_pos"] + kd_results["far_err_neg"])
    e_tan = 0.5 * (kd_results["distance_err_pos"] + kd_results["distance_err_neg"])
    e_conditions = [is_near, is_far, is_tangent]
    e_choices = [e_near, e_far, e_tan]
    e_dists = np.select(e_conditions, e_choices, default=np.nan)
    print("Num of NaN errors (i.e. all errors are NaNs):", np.sum(np.isnan(e_dists)))
    # conditions = [(is_near) & (is_unreliable), (is_far) & (is_unreliable), (is_tangent) & (is_unreliable)]
    # choices = [e_near, e_far, e_tan]
    # e_unreliable = np.select(conditions, choices, default=np.nan)

    return dists, e_dists, is_near, is_far, is_tangent, is_unreliable


def main(kdfile, vlsr_tol=20):
    #
    # Load plx data
    #
    plxfile = Path("pec_motions/csvfiles/alldata_HPDmode_NEW.csv")
    plxdata = pd.read_csv(Path(__file__).parent.parent / plxfile)
    dist_plx = plxdata["dist_mode"].values
    e_dist_plx = plxdata["dist_halfhpd"].values
    #
    # RegEx stuff
    #
    # Find num_samples used in kd
    num_samples = findall(r"\d+", kdfile)
    if len(num_samples) != 1:
        print("regex num_samples:", num_samples)
        raise ValueError("Invalid number of samples parsed")
    num_samples = int(num_samples[0])
    # Find if kd used kriging
    use_kriging = findall("True", kdfile)
    if len(use_kriging) > 1:
        print("regex use_kriging:", use_kriging)
        raise ValueError("Invalid use_kriging parsed")
    use_kriging = bool(use_kriging)
    #
    # Load kd data
    #
    kddata = pd.read_csv(Path(__file__).parent / kdfile)
    dist_kd, e_dist_kd, is_near, is_far, is_tangent, is_unreliable = assign_kd_distances(
        plxdata, kddata, vlsr_tol=vlsr_tol)
    #
    # Check that the distances and errors are the same as those
    # in face_on_view_plx.py
    #
    # Convert to galactocentric frame
    glong = plxdata["glong"].values
    glat = plxdata["glat"].values
    Xb, Yb, Zb = trans.gal_to_bary(glong, glat, dist_kd)
    Xg, Yg, Zg = trans.bary_to_gcen(Xb, Yb, Zb, R0=_R0, Zsun=_ZSUN, roll=_ROLL)
    # Rotate 90 deg CW (so Sun is on +y-axis)
    Xg, Yg = Yg, -Xg
    #
    fig, ax = plt.subplots()
    size_scale = 4  # scaling factor for size
    #
    ax.scatter(Xg[(is_near) & (~is_unreliable)], Yg[(is_near) & (~is_unreliable)],
               c="tab:cyan", s=e_dist_kd[(is_near) & (~is_unreliable)] * size_scale,
               label="Near")
    #
    ax.scatter(Xg[(is_far) & (~is_unreliable)], Yg[(is_far) & (~is_unreliable)],
               c="tab:purple", s=e_dist_kd[(is_far) & (~is_unreliable)] * size_scale,
               label="Far")
    #
    ax.scatter(Xg[(is_tangent) & (~is_unreliable)], Yg[(is_tangent) & (~is_unreliable)],
               c="tab:green", s=e_dist_kd[(is_tangent) & (~is_unreliable)] * size_scale,
               label="Tangent")
    #
    ax.scatter(Xg[is_unreliable], Yg[is_unreliable],
               c="tab:red", s=e_dist_kd[is_unreliable] * size_scale,
               label="Unreliable")
    ax.legend(fontsize=9)
    ax.set_xlabel("$x$ (kpc)")
    ax.set_ylabel("$y$ (kpc)")
    ax.axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax.axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax.set_xlim(-8, 12)
    ax.set_xticks([-5, 0, 5, 10])
    ax.set_ylim(-5, 15)
    ax.set_yticks([-5, 0, 5, 10, 15])
    ax.grid(False)
    ax.set_aspect("equal")
    fig.savefig(Path(__file__).parent / "test_HII_errs.pdf", bbox_inches="tight")
    plt.show()
    return None
    #
    # Calculate difference between kd and plx distances
    #
    dist_diff = dist_kd - dist_plx
    e_dist_diff = np.sqrt(e_dist_plx**2 + e_dist_kd**2)
    #
    # Plot
    #
    # Histogram of differences
    fig, ax = plt.subplots()
    ax.hist(dist_diff, bins=20, histtype="step", color="k")
    ax.axvline(np.median(dist_diff), color="k")
    plt.show()
    # CDF of differences over total error
    fig, ax = plt.subplots()
    xlims = (-3, 3)
    cdf_x = np.linspace(*xlims)
    ax.plot(cdf_x, gaussian_cdf(cdf_x), "k--")
    cdf_data = np.sort(dist_diff/e_dist_diff)
    ax.plot(cdf_data, np.arange(cdf_data.size)/cdf_data.size, "k-")
    ax.set_xlim(*xlims)
    plt.show()


if __name__ == "__main__":
    # kdfile_input = input("Name of kd .csv file in this folder: ")
    kdfile_input = "kd_plx_results_10000x_krigeTrue.csv"
    main(kdfile_input)
