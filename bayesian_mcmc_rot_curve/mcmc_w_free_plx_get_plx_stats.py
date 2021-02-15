"""
mcmc_w_free_plx_get_plx_stats.py

Gets key results for the MCMC parallaxes from a pickle file

Isaac Cheng - February 2021
"""

from pathlib import Path
import pymc3 as pm
import dill
import numpy as np

infile = Path(
    "/home/chengi/Documents/coop2021/bayesian_mcmc_rot_curve/"
    f"mcmc_outfile_A1_1dist_1.pkl"
)

with open(infile, "rb") as f:
    file = dill.load(f)
    trace = file["trace"]
    data = file["data"]

print(pm.summary(trace, var_names="plx").to_string())
print(data.to_markdown())

# plx_mcmc = pm.summary(trace, var_names="plx")["mean"].values
# e_plx_mcmc = pm.summary(trace, var_names="plx")["sd"].values
plx_mcmc = np.mean(trace["plx"], axis=0)
e_plx_mcmc = np.std(trace["plx"], axis=0)
# rhat = pm.summary(trace, var_names="plx")["r_hat"].values
plx_data = data["plx"].values
e_plx_data = data["e_plx"].values

plx_diff = (plx_mcmc - plx_data) / e_plx_data
# plx_diff = (plx_mcmc[rhat < 1.4] - plx_data[rhat < 1.4]) / e_plx_data [rhat < 1.4]
e_plx_diff = (e_plx_mcmc - e_plx_data) / e_plx_data

print("min & max abs plx diff (in termns of e_plx)", np.min(abs(plx_diff)), np.max(abs(plx_diff)))
print("mean abs plx diff (in termns of e_plx)", np.mean(abs(plx_diff)))
print("median abs plx diff (in termns of e_plx)", np.median(abs(plx_diff)))
print("min & max e_plx diff (in termns of e_plx)", np.min(e_plx_diff), np.max(e_plx_diff))
print("mean e_plx diff (in termns of e_plx)", np.mean(e_plx_diff))
print("median e_plx diff (in termns of e_plx)", np.median(e_plx_diff))
print("num MCMC e_plx > e_plx", np.sum(e_plx_diff > 0))