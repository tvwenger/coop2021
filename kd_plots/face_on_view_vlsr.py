"""
face_on_view_vlsr.py

Plots a face-on (galactocentric) view of the Milky Way
with the Sun on +y-axis using kinematic distances colour-coded by
their tangent LSR velocity.

Isaac Cheng - March 2021
"""
import sys
import sqlite3
from contextlib import closing
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm  # for centred colourbar
from re import search  # for regex
from kd import cw21_rotcurve


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


def str2bool(string, empty_condition=None):
    yes_conditions = ["yes", "y", "true", "t", "1"]
    no_conditions = ["no", "n", "false", "f", "0"]
    if empty_condition is not None:
        yes_conditions.append("") if empty_condition else no_conditions.append("")
    if string.lower() in yes_conditions:
        return True
    elif string.lower() in no_conditions:
        return False
    else:
        raise ValueError("Cannot convert input to boolean.")


# def assign_kd_distances(database_data, kd_results, vlsr_tol=20):
#     """
#     Returns the closest kinematic distance to parallax distance.
#     If vlsr (km/s) is within vlsr_tol (km/s) of tangent point vlsr,
#     use the tangent point vlsr

#     Inputs:
#       database_data :: pandas DataFrame
#       kd_results :: pandas DataFrame
#       vlsr_tol :: scalar
#     """
#     glong = database_data["glong"].values
#     vlsr = database_data["vlsr_med"].values
#     peak_dist = database_data["dist_mode"].values
#     #
#     # Assign tangent kd to any source with vlsr w/in vlsr_tol of tangent vlsr
#     #
#     # 1st quadrant
#     # is_q1 = (glong >= 0) & (glong < 180)
#     is_q1 = (glong >= 0) & (glong <= 90)
#     use_tan_q1 = (vlsr) > (kd_results["vlsr_tangent"].values - vlsr_tol)
#     # 4th quadrant
#     # is_q4 = (glong >= 180) & (glong < 360)
#     is_q4 = (glong >= 270) & (glong < 360)
#     use_tan_q4 = (vlsr) < (kd_results["vlsr_tangent"].values + vlsr_tol)
#     use_tangent = ((is_q1) & (use_tan_q1)) | ((is_q4) & (use_tan_q4))
#     print("Number of sources automatically assigned to tangent based on vlsr_tol:",
#           np.sum(use_tangent))
#     #
#     # Otherwise, select kd that is closest to distance from parallax
#     #
#     # peak_dist = plx_to_peak_dist(plx, e_plx)
#     near_err = abs(kd_results["near"] - peak_dist)
#     far_err = abs(kd_results["far"] - peak_dist)
#     tangent_err = abs(kd_results["tangent"] - peak_dist)
#     min_err = np.fmin.reduce([near_err, far_err, tangent_err])  # ignores NaNs
#     # Select distance corresponding to smallest error
#     tol = 1e-9  # tolerance for float equality
#     is_near = (abs(near_err - min_err) < tol) & (~use_tangent)
#     is_far = (abs(far_err- min_err) < tol) & (~use_tangent)
#     is_tangent = (abs(tangent_err - min_err) < tol) | (use_tangent)
#     conditions = [is_near, is_far, is_tangent]
#     choices = [kd_results["near"], kd_results["far"], kd_results["tangent"]]
#     dists = np.select(conditions, choices, default=np.nan)
#     # Exclude any sources w/in 15 deg of GC or 20 deg of GAC
#     glong[glong > 180] -= 360  # force -180 < glong <= 180
#     is_unreliable = (abs(glong) < 15.) | (abs(glong) > 160.)
#     #
#     # Print some stats
#     #
#     print("=" * 6)
#     is_nan = (~is_near) & (~is_far) & (~is_tangent)
#     num_sources = np.sum(np.isfinite(dists))
#     print("Total number of (non NaN) sources:", num_sources)
#     print(f"Num near: {np.sum((is_near) & (~is_unreliable))}"
#           + f"\tNum far: {np.sum((is_far) & (~is_unreliable))}"
#           + f"\tNum tangent: {np.sum((is_tangent) & (~is_unreliable))}"
#           + f"\tNum unreliable: {np.sum((is_unreliable) & (~is_nan))}")
#     print("Number of NaN sources (i.e. all dists are NaNs):", np.sum(is_nan))
#     print("Num NaNs in near, far, tangent:",
#           np.sum(np.isnan(near_err)), np.sum(np.isnan(far_err)),
#           np.sum(np.isnan(tangent_err)))
#     print("Number of sources with NaN in both 'near' and 'far':",
#           sum(np.isnan(kd_results["near"]) & np.isnan(kd_results["far"])))
#     print("Number of sources with NaN in both 'near' and 'tangent':",
#           sum(np.isnan(kd_results["near"]) & np.isnan(kd_results["tangent"])))
#     print("Number of sources with NaN in both 'far' and 'tangent':",
#           sum(np.isnan(kd_results["far"]) & np.isnan(kd_results["tangent"])))
#     # Print following if two distances are selected:
#     num_near_far = np.sum((is_near) & (is_far))
#     num_near_tan = np.sum((is_near) & (is_tangent))
#     num_far_tan = np.sum((is_far) & (is_tangent))
#     if any([num_near_far, num_near_tan, num_far_tan]):
#         print("Both near and far (should be 0):", num_near_far)
#         print("Both near and tan (should be 0):", num_near_tan)
#         print("Both far and tan (should be 0):", num_far_tan)
#     #
#     # Get corresponding kd errors
#     #
#     e_near = 0.5 * (kd_results["near_err_pos"] + kd_results["near_err_neg"])
#     e_far = 0.5 * (kd_results["far_err_pos"] + kd_results["far_err_neg"])
#     e_tan = 0.5 * (kd_results["distance_err_pos"] + kd_results["distance_err_neg"])
#     e_conditions = [is_near, is_far, is_tangent]
#     e_choices = [e_near, e_far, e_tan]
#     e_dists = np.select(e_conditions, e_choices, default=np.nan)
#     print("Num of NaN errors (i.e. all errors are NaNs):", np.sum(np.isnan(e_dists)))

#     return dists, e_dists, is_near, is_far, is_tangent, is_unreliable


# def main(kdfile, vlsr_tol=20, save_figs=True):
#     #
#     # Load plx data
#     #
#     plxfile = Path("pec_motions/csvfiles/alldata_HPDmode_NEW2.csv")
#     plxdata = pd.read_csv(Path(__file__).parent.parent / plxfile)
#     dist_plx = plxdata["dist_mode"].values
#     e_dist_plx = plxdata["dist_halfhpd"].values
#     #
#     # RegEx stuff
#     #
#     # Find num_samples in kd
#     num_samples = int(search('(\d+)x', kdfile).group(1))
#     # Find if kd used kriging
#     use_kriging = search("krige(.+?)", kdfile).group(1)
#     use_kriging = use_kriging.lower() == "t"
#     # Find if kd used peculiar motions
#     use_peculiar = search("pec(.+?)", kdfile).group(1)
#     use_peculiar = use_peculiar.lower() == "t"
#     #
#     # Print stats
#     #
#     print("=" * 6)
#     print("Number of MC kd samples:", num_samples)
#     print("Including peculiar motions in kd:", use_peculiar)
#     print("Using kriging:", use_kriging)
#     print("vlsr tolerance (km/s):", vlsr_tol)
#     print("=" * 6)
#     #
#     # Load kd data
#     #
#     kddata = pd.read_csv(Path(__file__).parent / kdfile)
#     dist_kd, e_dist_kd, is_near, is_far, is_tangent, is_unreliable = assign_kd_distances(
#         plxdata, kddata, vlsr_tol=vlsr_tol)
#     #
#     # Transform kd to galactocentric Cartesian frame
#     #
#     # Convert to galactocentric frame
#     glong = plxdata["glong"].values
#     glat = plxdata["glat"].values
#     Xb, Yb, Zb = trans.gal_to_bary(glong, glat, dist_kd)
#     Xg, Yg, Zg = trans.bary_to_gcen(Xb, Yb, Zb, R0=_R0, Zsun=_ZSUN, roll=_ROLL)
#     # Rotate 90 deg CW (so Sun is on +y-axis)
#     Xg, Yg = Yg, -Xg
#     #
#     # Plot
#     #
#     vlsr_tan = kddata["vlsr_tangent"].values
#     e_vlsr_tan = 0.5 * (kddata["vlsr_tangent_err_neg"].values
#                         + kddata["vlsr_tangent_err_pos"].values)
#     # # Centred colourbar
#     # norm = CenteredNorm(vcenter=0, halfrange=np.nanmax(abs(vlsr_tan[~is_unreliable])))
#     norm = mpl.colors.Normalize(vmin=np.nanmin(vlsr_tan[~is_unreliable]),
#                                 vmax=np.nanmax(vlsr_tan[~is_unreliable]))
#     cmap = "coolwarm"
#     #
#     fig, ax = plt.subplots()
#     size_scale = 2  # scaling factor for size
#     #
#     ax.scatter(Xg[(vlsr_tan > 0) & (~is_unreliable)], Yg[(vlsr_tan > 0) & (~is_unreliable)],
#             s=e_vlsr_tan[(vlsr_tan > 0) & (~is_unreliable)] * size_scale,
#             c=vlsr_tan[(vlsr_tan > 0) & (~is_unreliable)],
#             cmap=cmap, norm=norm, label="Positive")
#     #
#     ax.scatter(Xg[(vlsr_tan < 0) & (~is_unreliable)], Yg[(vlsr_tan < 0) & (~is_unreliable)],
#             s=e_vlsr_tan[(vlsr_tan < 0) & (~is_unreliable)] * size_scale,
#             c=vlsr_tan[(vlsr_tan < 0) & (~is_unreliable)],
#             cmap=cmap, norm=norm, label="Negative")
#     #
#     ax.scatter(Xg[is_unreliable], Yg[is_unreliable],
#             s=e_vlsr_tan[is_unreliable] * size_scale,
#             c="grey", label="Unreliable")
#     fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
#     ax.legend(fontsize=9)
#     ax.set_xlabel("$x$ (kpc)")
#     ax.set_ylabel("$y$ (kpc)")
#     ax.axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
#     ax.axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
#     ax.set_xlim(-8, 12)
#     ax.set_xticks([-5, 0, 5, 10])
#     ax.set_ylim(-5, 15)
#     ax.set_yticks([-5, 0, 5, 10, 15])
#     ax.grid(False)
#     ax.set_aspect("equal")
#     figname = f"HII_faceonVlsr_{num_samples}x_pec{use_peculiar}_krige{use_kriging}_vlsrTolerance{vlsr_tol}.pdf"
#     figname = "reid19_" + figname if "reid19" in kdfile else figname
#     fig.savefig(Path(__file__).parent / figname, bbox_inches="tight") if save_figs else None
#     plt.show()


# if __name__ == "__main__":
#     # kdfile_input = input("Filename of kd .csv file in this folder: ")
#     vlsr_tol_input = input("(int) Assign tangent distance if vlsr is within ___ " +
#                             "km/s of tangent velocity? (default 20): ")
#     vlsr_tol_input = 20 if vlsr_tol_input == "" else int(vlsr_tol_input)
#     # save_figs_input = str2bool(input("(y/n) Save figures (default y): "),
#     #                           empty_condition=True)
#     kdfile_input = "cw21_kd_plx_results_10000x_pecTrue_krigeTrue.csv"
#     save_figs_input = True
#     main(kdfile_input, vlsr_tol=vlsr_tol_input, save_figs=save_figs_input)

def main(use_kriging=False):
    # Galactocentric Cartesian positions
    xlow, xhigh = -8, 12
    ylow, yhigh = -5, 15
    # ylow, yhigh = ylow - _R0, yhigh - _R0
    gridx, gridy = np.mgrid[xlow:xhigh:500j, ylow:yhigh:500j]
    # print(gridx)
    # print()
    # Rotate 90 deg CCW
    gridx, gridy = -gridy, gridx
    gridz = np.zeros_like(gridx)
    # Convert to galactic coordinates
    xb, yb, zb = trans.gcen_to_bary(gridx, gridy, gridz, R0=_R0, Zsun=_ZSUN, roll=_ROLL)
    # print(gridx)
    # print()
    # print(xb)
    # return None
    glong, glat, dist = trans.bary_to_gal(xb, yb, zb)
    # Calculate LSR velocity at positions
    nom_params = cw21_rotcurve.nominal_params(glong, glat, dist,
                                              use_kriging=use_kriging, resample=False)
    vlsr = cw21_rotcurve.calc_vlsr(glong, glat, dist, peculiar=True, **nom_params)
    #
    # Plot
    #
    fig, ax = plt.subplots()
    cmap = "viridis"
    # Change limits for barycentric
    # xlow, xhigh = -8, 12
    # ylow, yhigh = ylow - _R0, yhigh - _R0
    extent = (xlow, xhigh, ylow, yhigh)
    norm = mpl.colors.Normalize(vmin=np.min(vlsr), vmax=np.max(vlsr))
    ax.imshow(vlsr.T, origin="lower", extent=extent, norm=norm)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    ax.axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax.axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax.set_xlabel("$x$ (kpc)")
    ax.set_ylabel("$y$ (kpc)")
    cbar.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar.ax.get_yaxis().labelpad = 15
    ax.set_aspect("equal")
    ax.grid(False)
    figname = f"cw21_faceonVlsr_krige{use_kriging}.pdf"
    fig.savefig(Path(__file__).parent / figname, bbox_inches="tight")
    plt.show()


def plot_diff():
    # Galactocentric Cartesian positions
    xlow, xhigh = -8, 12
    ylow, yhigh = -5, 15
    gridx, gridy = np.mgrid[xlow:xhigh:500j, ylow:yhigh:500j]
    # Rotate 90 deg CCW
    gridx, gridy = -gridy, gridx
    gridz = np.zeros_like(gridx)
    # Convert to galactic coordinates
    xb, yb, zb = trans.gcen_to_bary(gridx, gridy, gridz, R0=_R0, Zsun=_ZSUN, roll=_ROLL)
    glong, glat, dist = trans.bary_to_gal(xb, yb, zb)
    # Calculate LSR velocity at positions
    nom_params_nokrige = cw21_rotcurve.nominal_params(glong, glat, dist,
                                              use_kriging=False, resample=False)
    vlsr_nokrige = cw21_rotcurve.calc_vlsr(glong, glat, dist, peculiar=True,
                                           **nom_params_nokrige)
    nom_params_krige = cw21_rotcurve.nominal_params(glong, glat, dist,
                                              use_kriging=True, resample=False)
    vlsr_krige = cw21_rotcurve.calc_vlsr(glong, glat, dist, peculiar=True,
                                           **nom_params_krige)
    vlsr_diff = vlsr_nokrige - vlsr_krige
    #
    # Plot
    #
    fig, ax = plt.subplots()
    cmap = "viridis"
    extent = (xlow, xhigh, ylow, yhigh)
    norm = mpl.colors.Normalize(vmin=np.min(vlsr_diff), vmax=np.max(vlsr_diff))
    ax.imshow(vlsr_diff.T, origin="lower", extent=extent, norm=norm)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    ax.axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax.axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax.set_xlabel("$x$ (kpc)")
    ax.set_ylabel("$y$ (kpc)")
    cbar.ax.set_ylabel(r"$v_{\scriptscriptstyle LSR\text{, noKriging}} - v_{\scriptscriptstyle LSR\text{, Kriging}}$ (km s$^{-1}$)", rotation=270)
    cbar.ax.get_yaxis().labelpad = 15
    ax.set_aspect("equal")
    ax.grid(False)
    figname = "cw21_faceonVlsr_differences.pdf"
    fig.savefig(Path(__file__).parent / figname, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # main(use_kriging=True)
    plot_diff()