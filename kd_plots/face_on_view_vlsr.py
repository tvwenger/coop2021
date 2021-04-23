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
from kd import wc21_rotcurve, wc21_rotcurve_w_mc


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
    nom_params = wc21_rotcurve.nominal_params(glong, glat, dist,
                                              use_kriging=use_kriging, resample=False,
                                              norm=normalization)
    vlsr = wc21_rotcurve.calc_vlsr(glong, glat, dist, peculiar=True, **nom_params)
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
    nom_params_nokrige = wc21_rotcurve.nominal_params(glong, glat, dist,
                                              use_kriging=False, resample=False,
                                              norm=normalization)
    vlsr_nokrige = wc21_rotcurve.calc_vlsr(glong, glat, dist, peculiar=True,
                                           **nom_params_nokrige)
    nom_params_krige = wc21_rotcurve.nominal_params(glong, glat, dist,
                                              use_kriging=True, resample=False,
                                              norm=normalization)
    vlsr_krige = wc21_rotcurve.calc_vlsr(glong, glat, dist, peculiar=True,
                                           **nom_params_krige)
    vlsr_diff = vlsr_krige - vlsr_nokrige
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
    figname = "wc21_faceonVlsr_differences"
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
#     nom_params_nokrige = wc21_rotcurve.nominal_params(glong, glat, dist,
#                                               use_kriging=False, resample=True,
#                                               norm=normalization)
#     # Change to array to store each MC sample:
#     # nom_params_nokrige["Upec"] = np.array([nom_params_nokrige["Upec"],] * samples)
#     # nom_params_nokrige["Vpec"] = np.array([nom_params_nokrige["Vpec"],] * samples)
#     #
#     # Get kriging values
#     #
#     nom_params_krige = wc21_rotcurve.nominal_params(glong, glat, dist,
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
#         vlsr_nokrige[:,:,i] = wc21_rotcurve.calc_vlsr(glong, glat, dist, peculiar=True,
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
#         vlsr_krige[:,:,i] = wc21_rotcurve.calc_vlsr(glong, glat, dist, peculiar=True,
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
    pkl = Path(__file__).parent / f"vlsr_sd_{samples}samples.pkl"
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
            nom_params_nokrige = wc21_rotcurve_w_mc.nominal_params(glong, glat, dist,
                                                    use_kriging=False, resample=False,
                                                    norm=normalization, krige_resample=True)
            vlsr_nokrige[:,:,i] = wc21_rotcurve_w_mc.calc_vlsr(glong, glat, dist, peculiar=True,
                                                **nom_params_nokrige)
            nom_params_krige = wc21_rotcurve_w_mc.nominal_params(glong, glat, dist,
                                                    use_kriging=True, resample=False,
                                                    norm=normalization, krige_resample=True)
            vlsr_krige[:,:,i] = wc21_rotcurve_w_mc.calc_vlsr(
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
    figname = "wc21_faceonVlsr_differences_sd_mean"
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
        nom_params_nokrige = wc21_rotcurve.nominal_params(
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
        # vlsr_mc = wc21_rotcurve.calc_vlsr(
        #     glong, glat, dist, peculiar=True,
        #     Upec=Upec_mc[np.newaxis].T, Vpec=Vpec_mc[np.newaxis].T)
        # Method 2:
        for i in range(samples):
            # print("Computing sample", i, end="\r")
            Upec_mc = np.random.normal(loc=Upec, scale=Upec_sd)
            Vpec_mc = np.random.normal(loc=Vpec, scale=Vpec_sd)
            # # Use nominal params for all other values in calc_vlsr()
            # vlsr_mc[:, :, i] = wc21_rotcurve.calc_vlsr(
            #     glong, glat, dist, Upec=Upec_mc, Vpec=Vpec_mc, peculiar=True)
            mc_params = wc21_rotcurve.resample_params(kde)
            mc_params["Upec"] = Upec_mc
            mc_params["Vpec"] = Vpec_mc
            vlsr_mc[:, :, i] = wc21_rotcurve.calc_vlsr(
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
    figname = "wc21_faceonVlsr_differences_sd_tot"
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
    plot_diff(normalization=normalization_factor, as_png=False)
    # plot_diff_sd_mean(normalization=normalization_factor, samples=1000,
    #                   load_pkl=True, as_png=False)
    # plot_diff_sd_tot(normalization=normalization_factor, samples=1000,
    #                  load_pkl=False, as_png=False)
