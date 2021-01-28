"""
Plots Bayesian MCMC data from pickle file
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import corner
import dill


def plot_MCMC(trace, prior_set, like_type, num_sources, num_samples, filter_type):
    """
    Plots walkers and corner plot from trace. Returns None
    """
    if like_type != "gauss" and like_type != "cauchy" and like_type != 'sivia':
        raise ValueError("Invalid like_type. Allowed: 'gauss', 'cauchy', or 'sivia'.")

    sample_lst = []
    varnames = []

    # Get names of variables & data associated with each variable
    for varname in trace.varnames:
        if "interval" in varname or "lnlike" in varname:
            continue  # do not want to include non user-defined parameters
        varnames.append(varname)
        sample_lst.append(trace[varname])
    samples = np.array(sample_lst)

    num_iters = len(trace)
    num_chains = len(trace.chains)
    # print("varnames:", varnames)
    # print("samples shape", np.shape(samples))

    # === Plot MCMC chains for each parameter ===
    # Reshape to (# params, # chains, # iter per chain)
    param_lst = [param.reshape((num_chains, num_iters)) for param in samples]
    # print("param_lst shape", np.shape(param_lst))

    # Make # subplots same as # params & make figure twice as tall as is wide
    fig1, axes1 = plt.subplots(np.shape(param_lst)[0], figsize=plt.figaspect(2))

    for ax, parameter, varname in zip(axes1, param_lst, varnames):
        for chain in parameter:
            ax.plot(chain, "k-", alpha=0.5, linewidth=0.5)  # plot chains of parameter
        ax.set_title(varname, fontsize=8)  # add parameter name as title
        # Make x & y ticks smaller
        ax.tick_params(axis="both", which="major", labelsize=5)
        ax.tick_params(axis="both", which="minor", labelsize=3)

    if like_type == "gauss":
        fig1.suptitle(
            f"MCMC walkers: {num_chains} chains with {num_iters} iters each. Each parallax sampled {num_samples}×.\nGaussian (+ SS 2006) PDF with {prior_set} priors\n{num_sources} sources used in fit. Used {filter_type} to reject outliers",
            fontsize=9,
        )
    elif like_type == "cauchy":
        fig1.suptitle(
            f"MCMC walkers: {num_chains} chains with {num_iters} iters each. Each parallax sampled {num_samples}×.\n(Cauchy PDF with {prior_set} priors. {num_sources} sources used in fit)",
            fontsize=9,
        )
    else:  # like_type == "sivia"
        fig1.suptitle(
            f"MCMC walkers: {num_chains} chains with {num_iters} iters each. Each parallax sampled {num_samples}×.\nSivia & Skilling (2006) PDF with {prior_set} priors\n{num_sources} sources used in fit. Used {filter_type} to reject outliers",
            fontsize=9,
        )
    fig1.tight_layout()  # Need this below suptitle()
    fig1.savefig(
        Path(__file__).parent / f"MCMC_chains_{like_type}_{prior_set}_{num_samples}_samples.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Plot histogram of parameters
    fig2 = corner.corner(
        samples.T,
        labels=varnames,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".2f",
    )
    fig2.savefig(
        Path(__file__).parent / f"MCMC_hist_{like_type}_{prior_set}_{num_samples}_samples.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def main():
    # Binary file to read
    # infile = Path(__file__).parent / "reid_MCMC_outfile.pkl"
    infile = Path(
        "/home/chengi/Documents/coop2021/bayesian_mcmc_rot_curve/MCMC_w_dist_uncer_outfile.pkl"
    )

    with open(infile, "rb") as f:
        file = dill.load(f)
        trace = file["trace"]
        prior_set = file["prior_set"]  # "A1", "A5", "B", "C", "D"
        like_type = file["like_type"]  # "gauss", "cauchy", or "sivia"
        num_sources = file["num_sources"]
        num_samples = file["num_samples"]
        print("prior_set:", prior_set)
        print("like_type:", like_type)
        print("num_sources:", num_sources)
        print("num_samples:", num_samples)
        _FILTER_TYPE = "ln(likelihood)"  # "3 sigma" or "ln(likelihood)"

        plot_MCMC(trace, prior_set, like_type, num_sources, num_samples, _FILTER_TYPE)


if __name__ == "__main__":
    main()
