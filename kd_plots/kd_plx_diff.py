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
from scipy.stats import gaussian_kde
from re import search  # for regex

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
    #
    # Assign tangent kd to any source with vlsr w/in vlsr_tol of tangent vlsr
    #
    # 1st quadrant
    # is_q1 = (glong >= 0) & (glong < 180)
    is_q1 = (glong >= 0) & (glong <= 90)
    use_tan_q1 = (vlsr) > (kd_results["vlsr_tangent"].values - vlsr_tol)
    # 4th quadrant
    # is_q4 = (glong >= 180) & (glong < 360)
    is_q4 = (glong >= 270) & (glong < 360)
    use_tan_q4 = (vlsr) < (kd_results["vlsr_tangent"].values + vlsr_tol)
    use_tangent = ((is_q1) & (use_tan_q1)) | ((is_q4) & (use_tan_q4))
    # use_tangent = abs(kd_results["vlsr_tangent"] - vlsr) < vlsr_tol
    #
    # Otherwise, select kd that is closest to distance from parallax
    #
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
    # Print some stats
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
    print("Number of sources with NaN in both 'near' and 'far':",
          sum(np.isnan(kd_results["near"]) & np.isnan(kd_results["far"])))
    print("Number of sources with NaN in both 'near' and 'tangent':",
          sum(np.isnan(kd_results["near"]) & np.isnan(kd_results["tangent"])))
    print("Number of sources with NaN in both 'far' and 'tangent':",
          sum(np.isnan(kd_results["far"]) & np.isnan(kd_results["tangent"])))
    # Print following if two distances are selected:
    num_near_far = np.sum((is_near) & (is_far))
    num_near_tan = np.sum((is_near) & (is_tangent))
    num_far_tan = np.sum((is_far) & (is_tangent))
    if any([num_near_far, num_near_tan, num_far_tan]):
        print("Both near and far (should be 0):", num_near_far)
        print("Both near and tan (should be 0):", num_near_tan)
        print("Both far and tan (should be 0):", num_far_tan)
    #
    # Get corresponding kd errors
    #
    e_near = 0.5 * (kd_results["near_err_pos"] + kd_results["near_err_neg"])
    e_far = 0.5 * (kd_results["far_err_pos"] + kd_results["far_err_neg"])
    e_tan = 0.5 * (kd_results["distance_err_pos"] + kd_results["distance_err_neg"])
    e_conditions = [is_near, is_far, is_tangent]
    e_choices = [e_near, e_far, e_tan]
    e_dists = np.select(e_conditions, e_choices, default=np.nan)
    print("Num of NaN errors (i.e. all errors are NaNs):", np.sum(np.isnan(e_dists)))

    return dists, e_dists, is_near, is_far, is_tangent, is_unreliable


def main(kdfile, vlsr_tol=20):
    _PLOT_FIGS = True
    _SAVE_FIGS = False
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
    # Find num_samples in kd
    num_samples = int(search('(\d+)x', kdfile).group(1))
    # Find if kd used kriging
    use_kriging = search("krige(.+?)", kdfile).group(1)
    use_kriging = use_kriging.lower() == "t"
    # Find if kd used peculiar motions
    use_peculiar = search("pec(.+?)", kdfile).group(1)
    use_peculiar = use_peculiar.lower() == "t"
    #
    # Print stats
    #
    print("=" * 6)
    print("Number of MC kd samples:", num_samples)
    print("Including peculiar motions in kd:", use_peculiar)
    print("Using kriging:", use_kriging)
    print("vlsr tolerance (km/s):", vlsr_tol)
    print("=" * 6)
    #
    # Load kd data
    #
    kddata = pd.read_csv(Path(__file__).parent / kdfile)
    dist_kd, e_dist_kd, is_near, is_far, is_tangent, is_unreliable = assign_kd_distances(
        plxdata, kddata, vlsr_tol=vlsr_tol)
    if _PLOT_FIGS:
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
        figname = f"HII_faceonplx_{num_samples}x_pec{use_peculiar}_krige{use_kriging}_vlsrTolerance{vlsr_tol}.pdf"
        figname = "reid19_" + figname if "reid19" in kdfile else figname
        fig.savefig(Path(__file__).parent / figname, bbox_inches="tight") if _SAVE_FIGS else None
        plt.show()
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
        ax.hist(dist_diff[~is_unreliable], bins=15, histtype="step", color="k", alpha=0.5)
        ax.axvline(np.median(dist_diff[~is_unreliable]), color="k")
        # KDE
        kde = gaussian_kde(dist_diff[~is_unreliable])
        xlabels = [-5, 0, 5]
        diffs = np.linspace(xlabels[0], xlabels[-1], 200)
        kde = kde(diffs)
        ax.plot(diffs, kde * np.sum(kde) * 2, color="k")
        ax.set_xlabel(r"$d_{\rm kd} - d_\pi$ (kpc)")
        ax.set_ylabel("Frequency")
        ax.set_xticks(xlabels)
        figname = f"kd_plx_diff_hist_{num_samples}x_pec{use_peculiar}_krige{use_kriging}_vlsrTolerance{vlsr_tol}.pdf"
        figname = "reid19_" + figname if "reid19" in kdfile else figname
        fig.savefig(Path(__file__).parent / figname, bbox_inches="tight") if _SAVE_FIGS else None
        plt.show()
        # CDF of differences over total error
        fig, ax = plt.subplots()
        xlims = (-3, 3)
        cdf_x = np.linspace(*xlims)
        ax.plot(cdf_x, gaussian_cdf(cdf_x), "k--")
        cdf_data = np.sort(dist_diff[~is_unreliable]/e_dist_diff[~is_unreliable])
        ax.plot(cdf_data, np.arange(cdf_data.size)/cdf_data.size, "k-")
        # ax.set_xlim(*xlims)
        ax.set_xlabel(r"$(d_{\rm kd} - d_\pi) / \sqrt(\sigma^2_{\rm kd} + \sigma^2_\pi)$")
        ax.set_ylabel("CDF")
        figname = f"kd_plx_diff_CDF_{num_samples}x_pec{use_peculiar}_krige{use_kriging}_vlsrTolerance{vlsr_tol}.pdf"
        figname = "reid19_" + figname if "reid19" in kdfile else figname
        fig.savefig(Path(__file__).parent / figname, bbox_inches="tight") if _SAVE_FIGS else None
        plt.show()
    #
    # Print stats
    #
    kdtypes = [(is_near) & (~is_unreliable),
               (is_far) & (~is_unreliable),
               (is_tangent) & (~is_unreliable),
               is_unreliable
    ]
    kdnames = ["Near", "Far", "Tangent", "Unreliable"]
    for kdtype, kdname in zip(kdtypes, kdnames):
        print("="*3, kdname + " sources", "="*3)
        df_plx = plxdata[["gname", "glong", "glat", "vlsr_med", "dist_mode"]][kdtype]
        df_kd = kddata[["near", "far", "tangent", "vlsr_tangent"]][kdtype]
        df_tot = pd.concat([df_plx, df_kd], axis=1)
        print(df_tot.to_string())
        print(f"Number of {kdname} sources:", len(df_tot))
        print()


if __name__ == "__main__":
    # kdfile_input = input("Name of kd .csv file in this folder: ")
    kdfile_input = "cw21_kd_plx_results_10000x_pecTrue_krigeTrue.csv"
    # kdfile_input = "reid19_kd_plx_results_10000x_pecTrue_krigeFalse.csv"
    main(kdfile_input, vlsr_tol=20)
