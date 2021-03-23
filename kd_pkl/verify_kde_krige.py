"""
verify_kde_krige.py

Checks that the kernel density estimator (KDE) produces
the same results as the MCMC algorithm.

Isaac Cheng - March 2021
"""

from pathlib import Path
import dill
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# from kriging import kriging

#
# CW21 A6 rotation model parameters
#
__R0 = 8.1746
__Usun = 10.879
__Vsun = 10.697
__Wsun = 8.088
__Upec = 4.907
__Vpec = -4.522
__a2 = 0.977
__a3 = 1.626
__Zsun = 5.399
__roll = -0.011

#
# IAU defined LSR
#
__Ustd = 10.27
__Vstd = 15.32
__Wstd = 7.74


def resample_params(size=None):
    """
    Resample the  rotation curve parameters within their
    uncertainties using the Wenger+2020 kernel density estimator
    to include parameter covariances.

    Parameters:
      size :: integer
        The number of random samples to generate. If None, generate
        only one sample and return a scalar.

    Returns: params
      params :: dictionary
        params['a1'], etc. : scalar or array of scalars
                             The re-sampled parameters
    """
    kdefile = Path(__file__).parent / "cw21_kdefull_krige.pkl"
    with open(kdefile, "rb") as f:
        kde = dill.load(f)["kde"]
    if size is None:
        samples = kde.resample(1)
        params = {
            "R0": samples[0][0],
            "Zsun": samples[1][0],
            "Usun": samples[2][0],
            "Vsun": samples[3][0],
            "Wsun": samples[4][0],
            "Upec": samples[5][0],
            "Vpec": samples[6][0],
            "roll": samples[7][0],
            "a2": samples[8][0],
            "a3": samples[9][0],
        }
    else:
        samples = kde.resample(size)
        params = {
            "R0": samples[0],
            "Zsun": samples[1],
            "Usun": samples[2],
            "Vsun": samples[3],
            "Wsun": samples[4],
            "Upec": samples[5],
            "Vpec": samples[6],
            "roll": samples[7],
            "a2": samples[8],
            "a3": samples[9],
        }
    return params


def nominal_param(x):
    """
    Get nominal value
    """
    return {
        "R0": __R0,
        "Zsun": __Zsun,
        "Usun": __Usun,
        "Vsun": __Vsun,
        "Wsun": __Wsun,
        "Upec": __Upec,
        "Vpec": __Vpec,
        "roll": __roll,
        "a2": __a2,
        "a3": __a3,
    }.get(x, np.nan)


def plot_kde():
    params = resample_params(size=10000)

    param_lst = ["R0", "Zsun", "Usun", "Vsun", "Wsun", "Upec", "Vpec", "roll", "a2", "a3"]
    fig, axes = plt.subplots(np.shape(param_lst)[0], figsize=plt.figaspect(10))
    for ax, param in zip(axes, param_lst):
        var = params[param]
        nominal_mean = nominal_param(param)
        mean = np.mean(var)
        ax.hist(var, bins=20)
        ax.axvline(mean, color="r", linewidth=1, label=f"KDE mean {round(mean, 2)}")
        ax.axvline(
            nominal_mean,
            color="k",
            linewidth=1,
            label=f"MCMC mean {round(nominal_mean, 2)}",
        )
        ax.legend(fontsize=6)
        ax.set_title(param)
    fig.tight_layout()
    fig.savefig(
        Path(__file__).parent / "kde_vs_mcmc.pdf", format="pdf", bbox_inches="tight",
    )
    plt.show()


def plot_kriging():
    krigefile = Path(__file__).parent / "cw21_kde_krige.pkl"
    with open(krigefile, "rb") as f:
        file = dill.load(f)
        Upec_krige = file["Upec_krige"]
        Vpec_krige = file["Vpec_krige"]
    xlow, xhigh = -8, 12
    ylow, yhigh = -5, 15
    gridx, gridy = np.mgrid[xlow:xhigh:500j, ylow:yhigh:500j]
    coord_interp = np.vstack((gridx.flatten(), gridy.flatten())).T

    Upec, Upec_var = Upec_krige.interp(coord_interp)
    Vpec, Vpec_var = Vpec_krige.interp(coord_interp)
    Upec_sd = np.sqrt(Upec_var)
    Vpec_sd = np.sqrt(Vpec_var)
    # Reshape
    Upec = Upec.reshape(gridx.shape)
    Upec_sd = Upec_sd.reshape(gridx.shape)
    Vpec = Vpec.reshape(gridx.shape)
    Vpec_sd = Vpec_sd.reshape(gridx.shape)
    # ----
    # Plot
    # ----
    fig, ax = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    cmap = "viridis"
    extent = (xlow, xhigh, ylow, yhigh)
    # Plot interpolated Upec
    norm_Upec = mpl.colors.Normalize(vmin=np.min(Upec), vmax=np.max(Upec))
    ax[0].imshow(Upec.T, origin="lower", extent=extent, norm=norm_Upec)
    cbar_Upec = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_Upec, cmap=cmap), ax=ax[0], format="%.0f"
    )
    ax[0].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[0].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[0].set_xlabel("$x$ (kpc)")
    ax[0].set_ylabel("$y$ (kpc)")
    ax[0].set_title("$\overline{U_s}$")
    cbar_Upec.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar_Upec.ax.get_yaxis().labelpad = 15
    ax[0].set_aspect("equal")
    ax[0].grid(False)
    # Plot interpolated Vpec
    norm_Vpec = mpl.colors.Normalize(vmin=np.min(Vpec), vmax=np.max(Vpec))
    ax[1].imshow(Vpec.T, origin="lower", extent=extent, norm=norm_Vpec)
    cbar_Vpec = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_Vpec, cmap=cmap), ax=ax[1], format="%.0f"
    )
    ax[1].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[1].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[1].set_xlabel("$x$ (kpc)")
    ax[1].set_ylabel("$y$ (kpc)")
    ax[1].set_title("$\overline{V_s}$")
    cbar_Vpec.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar_Vpec.ax.get_yaxis().labelpad = 15
    ax[1].set_aspect("equal")
    ax[1].grid(False)
    ax[0].set_xlim(xlow, xhigh)
    ax[0].set_ylim(ylow, yhigh)
    ax[1].set_xlim(xlow, xhigh)
    ax[1].set_ylim(ylow, yhigh)
    plt.show()
    #
    # Plot standard deviation
    #
    fig, ax = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    cmap = "viridis"
    extent = (xlow, xhigh, ylow, yhigh)
    # Plot interpolated Upec sd
    norm_Upec_sd = mpl.colors.Normalize(vmin=np.min(Upec_sd), vmax=np.max(Upec_sd))
    ax[0].imshow(Upec_sd.T, origin="lower", extent=extent, norm=norm_Upec_sd)
    cbar_Upec = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_Upec_sd, cmap=cmap), ax=ax[0], format="%.0f"
    )
    ax[0].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[0].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[0].set_xlabel("$x$ (kpc)")
    ax[0].set_ylabel("$y$ (kpc)")
    ax[0].set_title("$\overline{U_s}$")
    cbar_Upec.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar_Upec.ax.get_yaxis().labelpad = 15
    ax[0].set_aspect("equal")
    ax[0].grid(False)
    # Plot interpolated Vpec sd
    norm_Vpec_sd = mpl.colors.Normalize(vmin=np.min(Vpec_sd), vmax=np.max(Vpec_sd))
    ax[1].imshow(Vpec_sd.T, origin="lower", extent=extent, norm=norm_Vpec_sd)
    cbar_Vpec = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_Vpec_sd, cmap=cmap), ax=ax[1], format="%.0f"
    )
    ax[1].axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax[1].axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax[1].set_xlabel("$x$ (kpc)")
    ax[1].set_ylabel("$y$ (kpc)")
    ax[1].set_title("$\overline{V_s}$")
    cbar_Vpec.ax.set_ylabel("km s$^{-1}$", rotation=270)
    cbar_Vpec.ax.get_yaxis().labelpad = 15
    ax[1].set_aspect("equal")
    ax[1].grid(False)
    ax[0].set_xlim(xlow, xhigh)
    ax[0].set_ylim(ylow, yhigh)
    ax[1].set_xlim(xlow, xhigh)
    ax[1].set_ylim(ylow, yhigh)
    plt.show()


def main():
    # plot_kde()
    plot_kriging()


if __name__ == "__main__":
    main()
