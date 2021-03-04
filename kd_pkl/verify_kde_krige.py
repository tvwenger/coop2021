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
from kriging import kriging

#
# CW21 A5 rotation model parameters
#
__R0 = 8.181
__Usun = 10.407
__Vsun = 10.213
__Wsun = 8.078
__Upec = 4.430
__Vpec = -4.823
__a2 = 0.971
__a3 = 1.625
__Zsun = 5.583
__roll = 0.010

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


# def plot_kriging():
#     kdefile = Path(__file__).parent / "cw21_kdefull_krige.pkl"
#     with open(kdefile, "rb") as f:
#         krige = dill.load(f)["krige"]
#     xlow, xhigh = -8, 12
#     ylow, yhigh = -5, 15
#     gridx, gridy = np.mgrid[xlow:xhigh:500j, ylow:yhigh:500j]
#     coord_interp = np.vstack((gridx.flatten(), gridy.flatten())).T

#     # Can't do this... krige only takes one x- & one y-value
#     Upec_interp, Upec_interp_var, Vpec_interp, Vpec_interp_var = krige(coord_interp)


def main():
    plot_kde()


if __name__ == "__main__":
    main()
