"""
compare_wc_reid.py

Compare the KDE from Wenger & Cheng (2021) to Reid (2019)

Isaac Cheng - March 2021
"""

from pathlib import Path
import dill
import numpy as np
import matplotlib.pyplot as plt
from kriging import kriging

#
# Values from WC21 A6
#
__R0 = 8.174602364395952
__Zsun = 5.398550615892994
__Usun = 10.878914326160878
__Vsun = 10.696801784160257
__Wsun = 8.087892505141708
__Upec = 4.9071771802606285
__Vpec = -4.521832904300172
__roll = -0.010742182667190958
__a2 = 0.9768982857793898
__a3 = 1.626400628724733

#
# IAU defined LSR
#
__Ustd = 10.27
__Vstd = 15.32
__Wstd = 7.74


def resample_cw_params(size=None):
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


def nominal_cw_params(x):
    """
    Cheng & Wenger (2021) A5 rotation model parameters
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


def resample_reid_params(size=None):
    """
    Resample the Reid+2019 rotation curve parameters within their
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
    kdefile = Path(__file__).parent / "reid19_params.pkl"
    with open(kdefile, "rb") as f:
        kde = dill.load(f)
    if size is None:
        samples = kde["full"].resample(1)
        params = {
            "R0": samples[0][0],
            "Usun": samples[1][0],
            "Vsun": samples[2][0],
            "Wsun": samples[3][0],
            "Upec": samples[4][0],
            "Vpec": samples[5][0],
            "a2": samples[6][0],
            "a3": samples[7][0],
        }
    else:
        samples = kde["full"].resample(size)
        params = {
            "R0": samples[0],
            "Usun": samples[1],
            "Vsun": samples[2],
            "Wsun": samples[3],
            "Upec": samples[4],
            "Vpec": samples[5],
            "a2": samples[6],
            "a3": samples[7],
        }
    return params


def nominal_reid_params(x):
    """
    Reid (2019) A5 rotation model parameters
    """
    R0 = 8.166
    Usun = 10.449
    Vsun = 12.092
    Wsun = 7.729
    Upec = 5.796
    Vpec = -3.489
    a2 = 0.977
    a3 = 1.623
    Zsun = 5.5
    roll = 0.0
    return {
        "R0": R0,
        "Zsun": Zsun,
        "Usun": Usun,
        "Vsun": Vsun,
        "Wsun": Wsun,
        "Upec": Upec,
        "Vpec": Vpec,
        "roll": roll,
        "a2": a2,
        "a3": a3,
    }.get(x, np.nan)


def plot_cw_kde(num_samples=10000):
    params = resample_cw_params(size=num_samples)

    param_lst = ["R0", "Zsun", "Usun", "Vsun", "Wsun", "Upec", "Vpec", "roll", "a2", "a3"]
    fig, axes = plt.subplots(np.shape(param_lst)[0], figsize=plt.figaspect(10))
    for ax, param in zip(axes, param_lst):
        var = params[param]
        nominal_mean = nominal_cw_params(param)
        mean = np.mean(var)
        ax.hist(var, bins=20)
        ax.axvline(mean, color="r", linewidth=1, label=f"CW KDE mean {round(mean, 3)}")
        ax.axvline(
            nominal_mean,
            color="k",
            linewidth=1,
            label=f"CW A5 mean {round(nominal_mean, 3)}",
        )
        ax.legend(fontsize=6)
        ax.set_title(param)
    fig.tight_layout()
    fig.savefig(
        Path(__file__).parent / "cw_kde_vs_A5.pdf", format="pdf", bbox_inches="tight",
    )
    plt.show()


def plot_reid_kde(num_samples=10000):
    params = resample_reid_params(size=num_samples)

    param_lst = ["R0", "Usun", "Vsun", "Wsun", "Upec", "Vpec", "a2", "a3"]
    fig, axes = plt.subplots(np.shape(param_lst)[0], figsize=plt.figaspect(10))
    for ax, param in zip(axes, param_lst):
        var = params[param]
        nominal_mean = nominal_reid_params(param)
        mean = np.mean(var)
        ax.hist(var, bins=20)
        ax.axvline(mean, color="r", linewidth=1, label=f"Reid KDE mean {round(mean, 3)}")
        ax.axvline(
            nominal_mean,
            color="k",
            linewidth=1,
            label=f"Reid A5 mean {round(nominal_mean, 3)}",
        )
        ax.legend(fontsize=6)
        ax.set_title(param)
    fig.tight_layout()
    fig.savefig(
        Path(__file__).parent / "reid_kde_vs_A5.pdf", format="pdf", bbox_inches="tight",
    )
    plt.show()


def plot_cw_reid_kde_hist(num_samples=10000):
    params_cw = resample_cw_params(size=num_samples)
    params_reid = resample_reid_params(size=num_samples)
    param_lst = ["R0", "Usun", "Vsun", "Wsun", "Upec", "Vpec", "a2", "a3"]

    # Plot side-by-side
    fig, axes = plt.subplots(
        np.shape(param_lst)[0], 2, sharex="row", sharey="row", figsize=plt.figaspect(10)
    )
    # Plot CW KDE
    for ax, param in zip(axes[:, 0], param_lst):
        var = params_cw[param]
        nominal_mean = nominal_cw_params(param)
        mean = np.mean(var)
        ax.hist(var, bins=20)
        ax.axvline(mean, color="r", linewidth=1, label=f"CW KDE mean {round(mean, 3)}")
        ax.axvline(
            nominal_mean,
            color="k",
            linewidth=1,
            label=f"CW A5 mean {round(nominal_mean, 3)}",
        )
        ax.legend(fontsize=6)
        ax.set_title(param)
    # Plot Reid KDE
    for ax, param in zip(axes[:, 1], param_lst):
        var = params_reid[param]
        nominal_mean = nominal_reid_params(param)
        mean = np.mean(var)
        ax.hist(var, bins=20)
        ax.axvline(mean, color="r", linewidth=1, label=f"Reid KDE mean {round(mean, 3)}")
        ax.axvline(
            nominal_mean,
            color="k",
            linewidth=1,
            label=f"Reid A5 mean {round(nominal_mean, 3)}",
        )
        ax.legend(fontsize=6)
        ax.set_title(param)
    # Save
    fig.tight_layout()
    fig.savefig(
        Path(__file__).parent / "cw_vs_reid_sidebyside.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()

    # Plot histograms on same plot
    fig, axes = plt.subplots(np.shape(param_lst)[0], figsize=plt.figaspect(10))
    for ax, param in zip(axes, param_lst):
        var_cw = params_cw[param]
        mean_cw = np.mean(var_cw)
        ax.hist(var_cw, color="c", alpha=0.5, bins=20, density=True)
        ax.axvline(
            mean_cw, color="b", linewidth=1, label=f"CW KDE mean {round(mean_cw, 3)}"
        )
        var_reid = params_reid[param]
        mean_reid = np.mean(var_reid)
        ax.hist(var_reid, color="m", alpha=0.5, bins=20, density=True)
        ax.axvline(
            mean_reid,
            color="r",
            linewidth=1,
            label=f"Reid KDE mean {round(mean_reid, 3)}",
        )
        ax.legend(fontsize=6)
        ax.set_title(param)
    # Save
    fig.tight_layout()
    fig.savefig(
        Path(__file__).parent / "cw_vs_reid_overlay.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_cw_reid_kdefull(num_samples=10000):
    """
    Plots the KDEs from Cheng & Wenger (2021) and Reid et al. (2019).
    This uses the CW pickle file where only the full KDE is stored (i.e. no components)
    """

    # Get KDEs
    cwfile = Path(__file__).parent / "cw21_kdefull_krige.pkl"
    with open(cwfile, "rb") as f:
        kde_cw = dill.load(f)["kde"]
    reidfile = Path(__file__).parent / "reid19_params.pkl"
    with open(reidfile, "rb") as f:
        kde_reid = dill.load(f)

    # Plot each KDE
    param_lst = ["R0", "Usun", "Vsun", "Wsun", "Upec", "Vpec", "a2", "a3"]
    fig, axes = plt.subplots(np.shape(param_lst)[0], figsize=plt.figaspect(10))
    for ax, param in zip(axes, param_lst):
        print("Plotting", param)
        R0 = [__R0,] * num_samples
        Zsun = [__Zsun,] * num_samples
        Usun = [__Usun,] * num_samples
        Vsun = [__Vsun,] * num_samples
        Wsun = [__Wsun,] * num_samples
        Upec = [__Upec,] * num_samples
        Vpec = [__Vpec,] * num_samples
        roll = [__roll,] * num_samples
        a2 = [__a2,] * num_samples
        a3 = [__a3,] * num_samples
        vals = {
            "R0": np.linspace(8.0, 8.30, num_samples),
            "Usun": np.linspace(6.0, 14.0, num_samples),
            "Vsun": np.linspace(-2, 24.0, num_samples),
            "Wsun": np.linspace(6.0, 10.0, num_samples),
            "Upec": np.linspace(0.0, 10.0, num_samples),
            "Vpec": np.linspace(-16.0, 10.0, num_samples),
            "a2": np.linspace(0.88, 1.08, num_samples),
            "a3": np.linspace(1.57, 1.67, num_samples),
        }.get(param, np.nan)

        all_vals = {
            "R0": np.vstack((vals, Zsun, Usun, Vsun, Wsun, Upec, Vpec, roll, a2, a3)),
            "Usun": np.vstack((R0, Zsun, vals, Vsun, Wsun, Upec, Vpec, roll, a2, a3)),
            "Vsun": np.vstack((R0, Zsun, Usun, vals, Wsun, Upec, Vpec, roll, a2, a3)),
            "Wsun": np.vstack((R0, Zsun, Usun, Vsun, vals, Upec, Vpec, roll, a2, a3)),
            "Upec": np.vstack((R0, Zsun, Usun, Vsun, Wsun, vals, Vpec, roll, a2, a3)),
            "Vpec": np.vstack((R0, Zsun, Usun, Vsun, Wsun, Upec, vals, roll, a2, a3)),
            "a2": np.vstack((R0, Zsun, Usun, Vsun, Wsun, Upec, Vpec, roll, vals, a3)),
            "a3": np.vstack((R0, Zsun, Usun, Vsun, Wsun, Upec, Vpec, roll, a2, vals)),
        }.get(param, np.nan)

        pdf_cw = kde_cw(all_vals)
        pdf_reid = kde_reid[param](vals)
        ax.plot(vals, pdf_cw / np.sum(pdf_cw), "c", label=f"CW {param}")
        ax.plot(vals, pdf_reid / np.sum(pdf_reid), "m", label=f"Reid {param}")
        ax.legend(fontsize=6)
        ax.set_title(param)

    # Save
    fig.tight_layout()
    fig.savefig(
        Path(__file__).parent / "cw_vs_reid_kdefull.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()
    # * NOTE: The saved plot will look weird (cf. plot_cw_reid_kde -> cw_vs_reid_kde.pdf)
    #   This behaviour is because Vsun and Vpec are very very tightly correlated, and
    #   since I am holding all the other parameters fixed, the KDE is really only allowing
    #   Vsun and Vpec to be 1 value. For example, this does not happen to Vsun or Vpec if
    #   I use kde.sample(size=10000) (e.g. in the histogram functions above); I am
    #   allowing all the parameters to be free and thus Vsun and Vpec are a lot broader


def plot_cw_reid_kde(num_samples=10000):
    """
    Plots the KDEs from Cheng & Wenger (2021) and Reid et al. (2019).
    This uses the CW pickle file where the full KDE + components are stored
    (cf. plot_cw_reid_kdefull)
    """

    # Get KDEs
    cwfile = Path(__file__).parent / "cw21_kde_krige.pkl"
    with open(cwfile, "rb") as f:
        kde_cw = dill.load(f)
    reidfile = Path(__file__).parent / "reid19_params.pkl"
    with open(reidfile, "rb") as f:
        kde_reid = dill.load(f)

    # Plot each KDE
    param_lst = ["R0", "Usun", "Vsun", "Wsun", "Upec", "Vpec", "a2", "a3"]
    fig, axes = plt.subplots(np.shape(param_lst)[0], figsize=plt.figaspect(10))
    for ax, param in zip(axes, param_lst):
        print("Plotting", param)
        vals = {
            "R0": np.linspace(8.0, 8.30, num_samples),
            "Usun": np.linspace(6.0, 14.0, num_samples),
            "Vsun": np.linspace(-2, 24.0, num_samples),
            "Wsun": np.linspace(6.0, 10.0, num_samples),
            "Upec": np.linspace(0.0, 10.0, num_samples),
            "Vpec": np.linspace(-16.0, 10.0, num_samples),
            "a2": np.linspace(0.88, 1.08, num_samples),
            "a3": np.linspace(1.57, 1.67, num_samples),
        }.get(param, np.nan)

        pdf_cw = kde_cw[param](vals)
        pdf_reid = kde_reid[param](vals)
        ax.plot(vals, pdf_cw / np.sum(pdf_cw), "c", label=f"CW {param}")
        ax.plot(vals, pdf_reid / np.sum(pdf_reid), "m", label=f"Reid {param}")
        ax.legend(fontsize=6)
        ax.set_title(param)

    # Save
    fig.tight_layout()
    fig.savefig(
        Path(__file__).parent / "cw_vs_reid_kde.pdf", format="pdf", bbox_inches="tight",
    )
    plt.show()


def main():
    plot_cw_reid_kde()

if __name__ == "__main__":
    main()
