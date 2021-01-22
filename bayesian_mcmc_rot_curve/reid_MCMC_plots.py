"""
Plots Bayesian MCMC data from pickle file
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import corner
import dill


def plot_MCMC(trace, like_type):
    """
    Plots walkers and corner plot from trace. Returns None
    """
    if like_type != "gaussian" and like_type != "cauchy":
        raise ValueError("Invalid like_type. Please select 'gaussian' or 'cauchy'.")

    sample_lst = []
    varnames = []

    # Get names of variables & data associated with each variable
    for varname in trace.varnames:
        if "interval" in varname:
            continue  # do not want to include non user-defined parameters
        varnames.append(varname)
        sample_lst.append(trace[varname])
    samples = np.array(sample_lst)

    num_iters = len(trace)
    num_chains = len(trace.chains)
    print("varnames:", varnames)
    print("samples shape", np.shape(samples))

    # === Plot MCMC chains for each parameter ===
    # Reshape to (# params, # chains, # iter per chain)
    param_lst = [param.reshape((num_chains, num_iters)) for param in samples]
    print("param_lst shape", np.shape(param_lst))

    # Make # subplots same as # params & make figure twice as tall as is wide
    fig1, axes1 = plt.subplots(np.shape(param_lst)[0], figsize=plt.figaspect(2))

    for ax, parameter, varname in zip(axes1, param_lst, varnames):
        for chain in parameter:
            ax.plot(chain, "k-", alpha=0.5, linewidth=0.5)  # plot chains of parameter
        ax.set_title(varname, fontsize=8)  # add parameter name as title
        # Make x & y ticks smaller
        ax.tick_params(axis="both", which="major", labelsize=5)
        ax.tick_params(axis="both", which="minor", labelsize=3)

    if like_type == "gaussian":
        fig1.suptitle(
            f"MCMC walkers: {num_chains} chains each with {num_iters} iterations\n(Gaussian PDF)",
            fontsize=10,
        )
        fig1.tight_layout()  # Need this below suptitle()
        fig1.savefig(
            Path(__file__).parent / "reid_MCMC_chains_gauss.jpg",
            format="jpg",
            dpi=300,
            bbox_inches="tight",
        )
    else:  # like_type == "cauchy"
        fig1.suptitle(
            f"MCMC walkers: {num_chains} chains each with {num_iters} iterations\n(Lorentzian-Like PDF)",
            fontsize=10,
        )
        fig1.tight_layout()  # Need this below suptitle()
        fig1.savefig(
            Path(__file__).parent / "reid_MCMC_chains_lorentz.jpg",
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
    if like_type == "gaussian":
        fig2.savefig(
            Path(__file__).parent / "reid_MCMC_histogram_gauss.jpg",
            format="jpg",
            dpi=300,
            bbox_inches="tight",
        )
    else:  # like_type == "cauchy"
        fig2.savefig(
            Path(__file__).parent / "reid_MCMC_histogram_lorentz.jpg",
            format="jpg",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def main():
    # Binary file to read
    infile = Path(__file__).parent / "reid_MCMC_outfile.pkl"

    with open(infile, "rb") as f:
        file = dill.load(f)
        trace = file["trace"]
        like_type = file["like_type"]  # "gaussian" or "cauchy"
        print("like_type:", like_type)

        plot_MCMC(trace, like_type)


if __name__ == "__main__":
    main()
