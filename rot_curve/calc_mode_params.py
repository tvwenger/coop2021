"""
calc_mode_params.py

Calculates the mode and upper_hpd-mode and mode-lower_hpd
of each of the MCMC parameters from trace file

Isaac Cheng - March 2021
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import dill

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

from calc_hpd import calc_hpd


def main():
    prior_set = "D"
    num_rounds = 4
    tracefile = Path(__file__).parent.parent / Path(
        f"bayesian_mcmc_rot_curve/mcmc_outfile_{prior_set}_102dist_{num_rounds}.pkl"
    )
    with open(tracefile, "rb") as f:
        trace = dill.load(f)["trace"]

    # print(np.shape(trace["R0"]))  # shape: (10000,)
    print("Prior set", prior_set, "& run", num_rounds)

    var_lst = ["R0", "Zsun", "Usun", "Vsun", "Wsun", "Upec", "Vpec", "roll", "a2", "a3"]
    results = {}
    for var in var_lst:
        print(var, "HPD:")
        _, var_mode, var_low, var_high = calc_hpd(trace[var], "scipy")
        up_err = var_high - var_mode
        low_err = var_mode - var_low
        print("Mode, high-mode, low-mode:", var_mode, up_err, low_err)
        name_mode = var + "_mode"
        name_hpdlow = var + "_hpdlow"
        name_hpdhigh = var + "_hpdhigh"
        name_up_err = var + "_up_err"
        name_low_err = var + "_low_err"
        tmp_results = {
            name_mode: var_mode,
            name_hpdlow: var_low,
            name_hpdhigh: var_high,
            name_up_err: up_err,
            name_low_err: low_err,
        }
        results.update(tmp_results)
    df = pd.DataFrame.from_dict([results])
    df.to_csv(
        path_or_buf=Path(__file__).parent / Path(f"{prior_set}_params_HPDmode.csv"),
        sep=",",
        index=False,
        header=True,
    )
    print("Saved .csv file!")


if __name__ == "__main__":
    main()
