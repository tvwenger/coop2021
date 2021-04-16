"""
face_on_view_vlsr.py

Plots a face-on (galactocentric) view of the Milky Way LSR velocity field
using kriging and without using kriging, as well as their differences.

Isaac Cheng - March 2021
"""
import sys
import dill
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm  # for centred colourbar
from re import search  # for regex
from kd import cw21_rotcurve, cw21_rotcurve_w_mc


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

def main(use_kriging=False, normalization=20):
    """
    Plots the LSR velocity with kriging and without kriging
    (N.B. This does not plot their differences! See plot_diff() for that)
    """

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
                                              use_kriging=use_kriging, resample=False,
                                              norm=normalization)
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


def plot_diff(normalization=20, as_png=False):
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
                                              use_kriging=False, resample=False,
                                              norm=normalization)
    vlsr_nokrige = cw21_rotcurve.calc_vlsr(glong, glat, dist, peculiar=True,
                                           **nom_params_nokrige)
    nom_params_krige = cw21_rotcurve.nominal_params(glong, glat, dist,
                                              use_kriging=True, resample=False,
                                              norm=normalization)
    vlsr_krige = cw21_rotcurve.calc_vlsr(glong, glat, dist, peculiar=True,
                                           **nom_params_krige)
    vlsr_diff = vlsr_nokrige - vlsr_krige
    #
    # Plot
    #
    linecolor = "k"  # colour of dashed line at x=0 and y=0
    if as_png:
        white_params = {
            # "ytick.color" : "#17becf",
            # "xtick.color" : "#17becf",
            # "axes.labelcolor" : "#17becf",
            # "axes.edgecolor" : "#17becf",
            "ytick.color" : "w",
            "xtick.color" : "w",
            "axes.labelcolor" : "w",
            "axes.edgecolor" : "w",
        }
        plt.rcParams.update(white_params)
        linecolor = "w"
    fig, ax = plt.subplots()
    cmap = "viridis"
    extent = (xlow, xhigh, ylow, yhigh)
    norm = mpl.colors.Normalize(vmin=np.min(vlsr_diff), vmax=np.max(vlsr_diff))
    ax.imshow(vlsr_diff.T, origin="lower", extent=extent, norm=norm, cmap=cmap)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    ax.axhline(y=0, linewidth=0.5, linestyle="--", color=linecolor)  # horizontal line
    ax.axvline(x=0, linewidth=0.5, linestyle="--", color=linecolor)  # vertical line
    ax.set_xlabel("$x$ (kpc)")
    ax.set_ylabel("$y$ (kpc)")
    # cbar.ax.set_ylabel(r"$v_{\scriptscriptstyle LSR\text{, noKriging}} - v_{\scriptscriptstyle LSR\text{, Kriging}}$ (km s$^{-1}$)", rotation=270)
    cbar.ax.set_ylabel("LSR Velocity Difference (km s$^{-1}$)", rotation=270)
    cbar.ax.get_yaxis().labelpad = 15
    ax.set_aspect("equal")
    ax.grid(False)
    figname = "cw21_faceonVlsr_differences"
    if as_png:
        figname += ".png"
        fig.savefig(Path(__file__).parent / figname, bbox_inches="tight",
                    dpi=300, transparent=True)
    else:
        figname += ".pdf"
        fig.savefig(Path(__file__).parent / figname, bbox_inches="tight")
    plt.show()


# def plot_diff_sd_old(normalization=20, samples=500):
#     # Galactocentric Cartesian positions
#     xlow, xhigh = -8, 12
#     ylow, yhigh = -5, 15
#     gridx, gridy = np.mgrid[xlow:xhigh:500j, ylow:yhigh:500j]
#     # Rotate 90 deg CCW
#     gridx, gridy = -gridy, gridx
#     gridz = np.zeros_like(gridx)
#     # Convert to galactic coordinates
#     xb, yb, zb = trans.gcen_to_bary(gridx, gridy, gridz, R0=_R0, Zsun=_ZSUN, roll=_ROLL)
#     glong, glat, dist = trans.bary_to_gal(xb, yb, zb)  # all shapes: (500, 500)
#     #
#     # Get non-kriging values
#     #
#     nom_params_nokrige = cw21_rotcurve.nominal_params(glong, glat, dist,
#                                               use_kriging=False, resample=True,
#                                               norm=normalization)
#     # Change to array to store each MC sample:
#     # nom_params_nokrige["Upec"] = np.array([nom_params_nokrige["Upec"],] * samples)
#     # nom_params_nokrige["Vpec"] = np.array([nom_params_nokrige["Vpec"],] * samples)
#     #
#     # Get kriging values
#     #
#     nom_params_krige = cw21_rotcurve.nominal_params(glong, glat, dist,
#                                               use_kriging=True, resample=True,
#                                               norm=normalization)
#     # Change to 3D array to store each MC sample:
#     # print(nom_params_krige["Upec"].shape)
#     # print(type(nom_params_krige["Upec"]))
#     nom_params_krige["Upec"] = np.dstack([nom_params_krige["Upec"]] * samples)  # (500, 500, samples)
#     nom_params_krige["Vpec"] = np.dstack([nom_params_krige["Vpec"]] * samples)  # (500, 500, samples)
#     # print(nom_params_krige["Upec"].shape)
#     # print(type(nom_params_krige["Upec"]))
#     # print(nom_params_krige["Upec"][0:5,0:5,0])
#     # print(nom_params_krige["Upec"][0:5,0:5,1])
#     # print(np.all(nom_params_krige["Upec"][:,:,0] == nom_params_krige["Upec"][:,:,1]))
#     # return None
#     #
#     # MC resample
#     #
#     kde_file = Path("/mnt/c/Users/ichen/OneDrive/Documents/Jobs/WaterlooWorks/2A Job Search/ACCEPTED__NRC_EXT-10708-JuniorResearcher/Work Documents/kd/kd/cw21_kde_krige.pkl")
#     with open(kde_file, "rb") as f:
#         kde = dill.load(f)["full"]
#     mc_params = kde.resample(samples)
#     Upec, Vpec = mc_params[5], mc_params[6]
#     # print(Upec)
#     # print(Upec.shape)
#     # return None
#     # print(np.shape(nom_params_nokrige["Upec"]))
#     # print(np.shape(nom_params_krige["Upec"]))
#     # print(nom_params_krige["Upec"][:, 0])
#     # return None
#     nom_params_nokrige["Upec"] += Upec
#     nom_params_nokrige["Vpec"] += Vpec
#     nom_params_krige["Upec"] += Upec  # Add MC sample: one resample to entire sheet
#     nom_params_krige["Vpec"] += Vpec  # Add MC sample: one resample to entire sheet
#     print(nom_params_krige["Upec"].shape)
#     #
#     # Calculate LSR velocity at positions
#     #
#     # Arrays to store vlsr results
#     vlsr_nokrige = np.zeros_like(nom_params_krige["Upec"])  # (500, 500, samples)
#     vlsr_krige = np.zeros_like(nom_params_krige["Upec"])  # (500, 500, samples)
#     print("In for loop")
#     for i in range(samples):
#         vlsr_nokrige[:,:,i] = cw21_rotcurve.calc_vlsr(glong, glat, dist, peculiar=True,
#                                             R0=nom_params_nokrige["R0"],
#                                             Usun=nom_params_nokrige["Usun"],
#                                             Vsun=nom_params_nokrige["Vsun"],
#                                             Wsun=nom_params_nokrige["Wsun"],
#                                             Upec=nom_params_nokrige["Upec"][i],
#                                             Vpec=nom_params_nokrige["Vpec"][i],
#                                             a2=nom_params_nokrige["a2"],
#                                             a3=nom_params_nokrige["a3"],
#                                             Zsun=nom_params_nokrige["Zsun"],
#                                             roll=nom_params_nokrige["roll"])
#         vlsr_krige[:,:,i] = cw21_rotcurve.calc_vlsr(glong, glat, dist, peculiar=True,
#                                             R0=nom_params_krige["R0"],
#                                             Usun=nom_params_krige["Usun"],
#                                             Vsun=nom_params_krige["Vsun"],
#                                             Wsun=nom_params_krige["Wsun"],
#                                             Upec=nom_params_krige["Upec"][:,:,i],
#                                             Vpec=nom_params_krige["Vpec"][:,:,i],
#                                             a2=nom_params_krige["a2"],
#                                             a3=nom_params_krige["a3"],
#                                             Zsun=nom_params_krige["Zsun"],
#                                             roll=nom_params_krige["roll"])
#     print(vlsr_nokrige.shape)
#     print(vlsr_nokrige[0:10,0:10,0])
#     print(vlsr_krige.shape)
#     print(vlsr_krige[0:10,0:10,0])
#     vlsr_diff = vlsr_nokrige - vlsr_krige
#     vlsr_diff_sd = np.std(vlsr_diff, axis=2)
#     print(vlsr_diff.shape)
#     print(vlsr_diff_sd.shape)
#     # Plot
#     #
#     fig, ax = plt.subplots()
#     cmap = "viridis"
#     extent = (xlow, xhigh, ylow, yhigh)
#     norm = mpl.colors.Normalize(vmin=np.min(vlsr_diff_sd), vmax=np.max(vlsr_diff_sd))
#     ax.imshow(vlsr_diff_sd.T, origin="lower", extent=extent, norm=norm)
#     cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
#     ax.axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
#     ax.axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
#     ax.set_xlabel("$x$ (kpc)")
#     ax.set_ylabel("$y$ (kpc)")
#     cbar.ax.set_ylabel("Standard Deviation (km s$^{-1}$)", rotation=270)
#     cbar.ax.get_yaxis().labelpad = 15
#     ax.set_aspect("equal")
#     ax.grid(False)
#     figname = "cw21_faceonVlsr_differences_sd.pdf"
#     fig.savefig(Path(__file__).parent / figname, bbox_inches="tight")
#     plt.show()


def plot_diff_sd_mean(normalization=20, samples=10, load_pkl=False, as_png=False):
    # Galactocentric Cartesian positions
    xlow, xhigh = -8, 12
    ylow, yhigh = -5, 15
    pkl = Path(__file__).parent / f"vlsr_sd_mean_{samples}samples.pkl"
    if not load_pkl:
        gridx, gridy = np.mgrid[xlow:xhigh:500j, ylow:yhigh:500j]
        # Rotate 90 deg CCW
        gridx, gridy = -gridy, gridx
        gridz = np.zeros_like(gridx)
        # Convert to galactic coordinates
        xb, yb, zb = trans.gcen_to_bary(gridx, gridy, gridz, R0=_R0, Zsun=_ZSUN, roll=_ROLL)
        glong, glat, dist = trans.bary_to_gal(xb, yb, zb)
        # Arrays to store results
        vlsr_nokrige = np.zeros((500, 500, samples))  # (500, 500, samples)
        vlsr_krige = np.zeros((500, 500, samples))  # (500, 500, samples)
        # Calculate LSR velocity at positions
        for i in range(samples):
            print("Computing sample", i, end="\r")
            nom_params_nokrige = cw21_rotcurve_w_mc.nominal_params(glong, glat, dist,
                                                    use_kriging=False, resample=False,
                                                    norm=normalization, krige_resample=True)
            vlsr_nokrige[:,:,i] = cw21_rotcurve_w_mc.calc_vlsr(glong, glat, dist, peculiar=True,
                                                **nom_params_nokrige)
            nom_params_krige = cw21_rotcurve_w_mc.nominal_params(glong, glat, dist,
                                                    use_kriging=True, resample=False,
                                                    norm=normalization, krige_resample=True)
            vlsr_krige[:,:,i] = cw21_rotcurve_w_mc.calc_vlsr(
                glong, glat, dist, peculiar=True, **nom_params_krige)
            # * N.B. No need to resample_params() since will just add then subtract same constant after
        vlsr_diff = vlsr_nokrige - vlsr_krige
        vlsr_diff_sd = np.std(vlsr_diff, axis=2)
        # Save to pickle file
        with open(pkl, "wb") as f:
            dill.dump(
                {
                    "vlsr_nokrige": vlsr_nokrige,
                    "vlsr_krige": vlsr_krige,
                    "vlsr_diff": vlsr_diff,
                    "vlsr_diff_sd": vlsr_diff_sd,
                }, f
            )
        print("Saved pickle file!")
    #
    # Plot
    #
    else:
        # Load pickle file
        with open(pkl, "rb") as f:
            vlsr_diff_sd = dill.load(f)["vlsr_diff_sd"]
    linecolor = "k"  # colour of dashed line at x=0 and y=0
    if as_png:
        white_params = {
            "ytick.color" : "w",
            "xtick.color" : "w",
            "axes.labelcolor" : "w",
            "axes.edgecolor" : "w",
        }
        plt.rcParams.update(white_params)
        linecolor = "white"
    fig, ax = plt.subplots()
    cmap = "viridis"
    extent = (xlow, xhigh, ylow, yhigh)
    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(vlsr_diff_sd))
    ax.imshow(vlsr_diff_sd.T, origin="lower", extent=extent, norm=norm, cmap=cmap)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    ax.axhline(y=0, linewidth=0.5, linestyle="--", color=linecolor)  # horizontal line
    ax.axvline(x=0, linewidth=0.5, linestyle="--", color=linecolor)  # vertical line
    ax.set_xlabel("$x$ (kpc)")
    ax.set_ylabel("$y$ (kpc)")
    cbar.ax.set_ylabel(r"Standard Deviation of Differences (km s$^{-1}$)", rotation=270)
    cbar.ax.get_yaxis().labelpad = 15
    ax.set_aspect("equal")
    ax.grid(False)
    figname = "cw21_faceonVlsr_differences_sd_mean"
    if as_png:
        figname += ".png"
        fig.savefig(Path(__file__).parent / figname, bbox_inches="tight",
                    format="png", dpi=300, transparent=True)
    else:
        figname += ".pdf"
        fig.savefig(Path(__file__).parent / figname, bbox_inches="tight",
                    format="pdf")
    plt.show()


def plot_diff_sd_tot(normalization=20, samples=10, load_pkl=False, as_png=False):
    # Galactocentric Cartesian positions
    xlow, xhigh = -8, 12
    ylow, yhigh = -5, 15
    pkl = Path(__file__).parent / f"vlsr_sd_tot_{samples}samples.pkl"
    if not load_pkl:
        gridx, gridy = np.mgrid[xlow:xhigh:500j, ylow:yhigh:500j]
        # Rotate 90 deg CCW
        gridx, gridy = -gridy, gridx
        gridz = np.zeros_like(gridx)
        # Convert to barycentric Cartesian coordinates
        xb, yb, zb = trans.gcen_to_bary(gridx, gridy, gridz, R0=_R0, Zsun=_ZSUN, roll=_ROLL)
        glong, glat, dist = trans.bary_to_gal(xb, yb, zb)
        # Arrays to store results
        # vlsr_nokrige = np.zeros((500, 500, samples))  # (500, 500, samples)
        # vlsr_krige = np.zeros((500, 500, samples))  # (500, 500, samples)
        vlsr_mc = np.zeros((500, 500, samples))  # (500, 500, samples)
        kde_file = Path("/mnt/c/Users/ichen/OneDrive/Documents/Jobs/WaterlooWorks/2A Job Search/ACCEPTED__NRC_EXT-10708-JuniorResearcher/Work Documents/kd/kd/cw21_kde_krige.pkl")
        with open(kde_file, "rb") as f:
            file = dill.load(f)
            kde = file["full"]
            Upec_krige = file["Upec_krige"]
            Vpec_krige = file["Vpec_krige"]
            var_threshold = file["var_threshold"]  # sum of Upec/Vpec variances
        # mc_params = kde.resample(samples)
        # Upec, Vpec = mc_params[5], mc_params[6]
        # Get average Upec and average Vpec --> scalars
        nom_params_nokrige = cw21_rotcurve.nominal_params(
            glong, glat, dist, use_kriging=False, resample=False, norm=normalization)
        Upec_avg_var = 1.1 ** 2
        Vpec_avg_var = 6.1 ** 2
        #
        # Calculate expected Upec and Vpec differences at source location(s)
        interp_pos = np.vstack((xb.flatten(), yb.flatten())).T
        Upec_diff, Upec_diff_var = Upec_krige.interp(interp_pos, resample=False)
        Vpec_diff, Vpec_diff_var = Vpec_krige.interp(interp_pos, resample=False)
        # Gaussian-like weighting function
        var_tot = Upec_diff_var + Vpec_diff_var  # total variance (not in quadrature)
        pec_weights = np.exp(normalization * (var_threshold / var_tot - 1))
        zero_weights = np.ones_like(pec_weights)
        weights = np.vstack((pec_weights, zero_weights))
        zero_diff = np.zeros_like(Upec_diff)
        Upec_diff = np.average([Upec_diff, zero_diff], weights=weights, axis=0)
        Vpec_diff = np.average([Vpec_diff, zero_diff], weights=weights, axis=0)
        Upec_diff_var = np.average([Upec_diff_var, np.full_like(zero_diff, Upec_avg_var)],
                                   weights=weights, axis=0)
        Vpec_diff_var = np.average([Vpec_diff_var, np.full_like(zero_diff, Vpec_avg_var)],
                                   weights=weights, axis=0)
        # Reshape
        Upec = Upec_diff.reshape(np.shape(xb)) + nom_params_nokrige["Upec"]
        Vpec = Vpec_diff.reshape(np.shape(xb)) + nom_params_nokrige["Vpec"]
        Upec_sd = np.sqrt(Upec_diff_var).reshape(np.shape(xb))
        Vpec_sd = np.sqrt(Vpec_diff_var).reshape(np.shape(xb))
        # MC sample Upec & Vpec at each point
        # Method 1 (?): (This method uses too much memory)
        # Upec = np.dstack([Upec] * samples)
        # Vpec = np.dstack([Vpec] * samples)
        # Upec_sd = np.dstack([Upec_sd] * samples)
        # Vpec_sd = np.dstack([Vpec_sd] * samples)
        # print("Upec.shape", Upec.shape)
        # Upec_mc = np.random.normal(loc=Upec, scale=Upec_sd)
        # Vpec_mc = np.random.normal(loc=Vpec, scale=Vpec_sd)
        # vlsr_mc = cw21_rotcurve.calc_vlsr(
        #     glong, glat, dist, peculiar=True,
        #     Upec=Upec_mc[np.newaxis].T, Vpec=Vpec_mc[np.newaxis].T)
        # Method 2:
        for i in range(samples):
            # print("Computing sample", i, end="\r")
            Upec_mc = np.random.normal(loc=Upec, scale=Upec_sd)
            Vpec_mc = np.random.normal(loc=Vpec, scale=Vpec_sd)
            # # Use nominal params for all other values in calc_vlsr()
            # vlsr_mc[:, :, i] = cw21_rotcurve.calc_vlsr(
            #     glong, glat, dist, Upec=Upec_mc, Vpec=Vpec_mc, peculiar=True)
            mc_params = cw21_rotcurve.resample_params(kde)
            mc_params["Upec"] = Upec_mc
            mc_params["Vpec"] = Vpec_mc
            vlsr_mc[:, :, i] = cw21_rotcurve.calc_vlsr(
                glong, glat, dist, peculiar=True, **mc_params)
        vlsr_mc_sd = np.std(vlsr_mc, axis=2)
        # Save to pickle file
        with open(pkl, "wb") as f:
            dill.dump(
                {
                    "vlsr_mc": vlsr_mc,
                    "vlsr_mc_sd": vlsr_mc_sd,
                }, f
            )
        print("Saved pickle file!")
    else:
        # Load pickle file
        with open(pkl, "rb") as f:
            vlsr_mc_sd = dill.load(f)["vlsr_mc_sd"]
    # print("Shape of vlsr_mc_sd:", np.shape(vlsr_mc_sd))
    #
    # Plot
    #
    linecolor = "k"  # colour of dashed line at x=0 and y=0
    if as_png:
        white_params = {
            "ytick.color" : "w",
            "xtick.color" : "w",
            "axes.labelcolor" : "w",
            "axes.edgecolor" : "w",
        }
        plt.rcParams.update(white_params)
        linecolor = "white"
    fig, ax = plt.subplots()
    cmap = "viridis"
    extent = (xlow, xhigh, ylow, yhigh)
    norm = mpl.colors.Normalize(vmin=np.min(vlsr_mc_sd), vmax=np.max(vlsr_mc_sd))
    # norm = mpl.colors.Normalize(vmin=0, vmax=np.max(vlsr_mc_sd))
    ax.imshow(vlsr_mc_sd.T, origin="lower", extent=extent, norm=norm, cmap=cmap)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    ax.axhline(y=0, linewidth=0.5, linestyle="--", color=linecolor)  # horizontal line
    ax.axvline(x=0, linewidth=0.5, linestyle="--", color=linecolor)  # vertical line
    ax.set_xlabel("$x$ (kpc)")
    ax.set_ylabel("$y$ (kpc)")
    cbar.ax.set_ylabel(r"Full Standard Deviation of Differences (km s$^{-1}$)", rotation=270)
    cbar.ax.get_yaxis().labelpad = 15
    ax.set_aspect("equal")
    ax.grid(False)
    figname = "cw21_faceonVlsr_differences_sd_tot"
    if as_png:
        figname += ".png"
        fig.savefig(Path(__file__).parent / figname, bbox_inches="tight",
                    format="png", dpi=300, transparent=True)
    else:
        figname += ".pdf"
        fig.savefig(Path(__file__).parent / figname, bbox_inches="tight",
                    format="pdf")
    plt.show()


if __name__ == "__main__":
    normalization_factor = 20
    # main(use_kriging=False, normalization=normalization_factor)
    # plot_diff(normalization=normalization_factor, as_png=True)
    # plot_diff_sd_mean(normalization=normalization_factor, samples=10, load_pkl=True,
    #              as_png=True)
    plot_diff_sd_tot(normalization=normalization_factor, samples=1000,
                     load_pkl=False, as_png=False)
