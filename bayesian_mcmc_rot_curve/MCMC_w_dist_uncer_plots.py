"""
Plots Bayesian MCMC data from pickle file
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import corner
import dill


def plot_MCMC(trace, prior_set, like_type, num_sources, num_samples, reject_method,
              num_rounds, free_Zsun=False, free_roll=False, free_Wpec=False):
    """
    Plots walkers and corner plot from trace. Returns None
    """
    if like_type != "gauss" and like_type != "cauchy" and like_type != 'sivia':
        raise ValueError("Invalid like_type. Allowed: 'gauss', 'cauchy', or 'sivia'.")

    # Varnames order: [R0, Zsun, Usun, Vsun, Wsun, Upec, Vpec, Wpec, roll, a2, a3]
    varnames = ['R0', 'Usun', 'Vsun', 'Wsun', 'Upec', 'Vpec', 'a2', 'a3']
    samples = [trace[varname] for varname in varnames]
    if free_roll:
        varnames.insert(6, "roll")
        samples.insert(6, trace["roll"])
    if free_Wpec:
        varnames.insert(6, "Wpec")
        samples.insert(6, trace["Wpec"])
    if free_Zsun:
        varnames.insert(1, "Zsun")
        samples.insert(1, trace["Zsun"])
    samples = np.array(samples)

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
            ax.plot(chain, "k-", alpha=0.1, linewidth=0.5)  # plot chains of parameter
        ax.set_title(varname, fontsize=8)  # add parameter name as title
        # Make x & y ticks smaller
        ax.tick_params(axis="both", which="major", labelsize=5)
        ax.tick_params(axis="both", which="minor", labelsize=3)

    if like_type == "gauss":
        fig1.suptitle(
            f"MCMC walkers: {num_chains} chains with {num_iters} iters each. "
            f"Each distance sampled {num_samples}×."
            f"\nGaussian (+ Cauchy) PDF with {prior_set} priors. {num_rounds} MCMC fits."
            f"\n{num_sources} sources used in fit. Outlier rejection method: {reject_method}",
            fontsize=9,
        )
    elif like_type == "cauchy":
        fig1.suptitle(
            f"MCMC walkers: {num_chains} chains with {num_iters} iters each. "
            f"Each distance sampled {num_samples}×."
            f"\nCauchy PDF with {prior_set} priors. {num_rounds} MCMC fits."
            f"\n{num_sources} sources used in fit. Outlier rejection method: {reject_method}",
            fontsize=9,
        )
    else:  # like_type == "sivia"
        fig1.suptitle(
            f"MCMC walkers: {num_chains} chains with {num_iters} iters each. "
            f"Each distance sampled {num_samples}×."
            f"\nSivia & Skilling (2006) PDF with {prior_set} priors. {num_rounds} MCMC fits."
            f"\n{num_sources} sources used in fit. Outlier rejection method: {reject_method}",
            fontsize=9,
        )
    fig1.tight_layout()  # Need this below suptitle()
    fig1.savefig(
        Path(__file__).parent / f"MCMC_chains_{prior_set}_{num_samples}dist_{num_rounds}.jpg",
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
        # truths order (choose all variables that apply):
        # [R0, Zsun, Usun, Vsun, Wsun, Upec, Vpec, Wpec, roll, a2, a3]
        # truths=[8.15, 5.5, 10.6, 10.7, 7.6, 6.1, -4.3, 0., 0., 0.96, 1.62],
    )
    fig2.savefig(
        Path(__file__).parent / f"MCMC_hist_{prior_set}_{num_samples}dist_{num_rounds}.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def main(prior_set, num_samples, num_rounds):
    # Binary file to read
    # infile = Path(__file__).parent / "reid_MCMC_outfile.pkl"
    infile = Path(
        "/home/chengi/Documents/coop2021/bayesian_mcmc_rot_curve/"
        f"mcmc_outfile_{prior_set}_{num_samples}dist_{num_rounds}.pkl"
    )

    with open(infile, "rb") as f:
        file = dill.load(f)
        trace = file["trace"]
        # prior_set = file["prior_set"]  # "A1", "A5", "B", "C", "D"
        like_type = file["like_type"]  # "gauss", "cauchy", or "sivia"
        num_sources = file["num_sources"]
        # num_samples = file["num_samples"]
        reject_method = file["reject_method"] if num_rounds != 1 else None
        free_Zsun = file["free_Zsun"]
        free_roll = file["free_roll"]
        free_Wpec = file["free_Wpec"]

    print("prior_set:", prior_set)
    print("like_type:", like_type)
    print("num_sources:", num_sources)
    print("num_samples:", num_samples)
    print("MCMC iteration to plot:", num_rounds)
    print("Outlier rejection method:", reject_method) if num_rounds != 1 else None

    if reject_method is None:
        reject_string = "n/a"
    elif reject_method == "sigma":
        reject_string = "3 sigma"
    elif reject_method == "lnlike":
        reject_string = "ln(likelihood)"
    else:
        raise ValueError("Invalid reject_method. Please choose sigma or lnlike.")

    plot_MCMC(trace, prior_set, like_type, num_sources, num_samples, reject_string,
              num_rounds, free_Zsun=free_Zsun, free_roll=free_roll, free_Wpec=free_Wpec)


if __name__ == "__main__":
    prior_set_file = input("prior_set of file (A1, A5, B, C, D): ")
    num_samples_file = int(input("Number of distance samples per source in file (int): "))
    num_rounds_file = int(input("Number of times MCMC has run. i.e., this_round of file (int): "))
    # reject_method_file = input("Outlier rejection method used (sigma or lnlike): ")

    main(prior_set_file, num_samples_file, num_rounds_file)
