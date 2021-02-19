"""
mcmc_w_free_plx_get_plx_stats.py

Gets key results for the MCMC parallaxes from a pickle file

Isaac Cheng - February 2021
"""

# from pathlib import Path
# import pymc3 as pm
# import dill
# import numpy as np

# infile = Path(
#     "/home/chengi/Documents/coop2021/bayesian_mcmc_rot_curve/"
#     f"mcmc_outfile_A1_1dist_3.pkl"
# )

# with open(infile, "rb") as f:
#     file = dill.load(f)
#     trace = file["trace"]
#     data = file["data"]

# # print(pm.summary(trace, var_names="plx").to_string())
# # print(data.to_markdown())

# # plx_mcmc = pm.summary(trace, var_names="plx")["mean"].values
# # e_plx_mcmc = pm.summary(trace, var_names="plx")["sd"].values
# plx_mcmc = np.mean(trace["plx"], axis=0)
# e_plx_mcmc = np.std(trace["plx"], axis=0)
# # rhat = pm.summary(trace, var_names="plx")["r_hat"].values
# plx_data = data["plx"].values
# e_plx_data = data["e_plx"].values
# Upec_mcmc = np.mean(trace["Upec"], axis=0)
# e_Upec_mcmc = np.std(trace["Upec"], axis=0)
# Vpec_mcmc = np.mean(trace["Vpec"], axis=0)
# e_Vpec_mcmc = np.std(trace["Vpec"], axis=0)
# Upec_A5 = 4.853
# e_Upec_A5 = 1.296
# Vpec_A5 = -4.55
# e_Vpec_A5 = 5.853

# plx_diff = np.round((plx_mcmc - plx_data) / e_plx_data, decimals=4)
# # plx_diff = (plx_mcmc[rhat < 1.4] - plx_data[rhat < 1.4]) / e_plx_data [rhat < 1.4]
# e_plx_diff = np.round((e_plx_mcmc - e_plx_data) / e_plx_data, decimals=4)
# Upec_diff = np.round(Upec_mcmc - Upec_A5, decimals=4)
# Vpec_diff = np.round(Vpec_mcmc - Vpec_A5, decimals=4)

# print(
#     "min & max abs plx diff (in terms of database standard deviations)",
#     np.min(abs(plx_diff)),
#     np.max(abs(plx_diff)),
# )
# print(
#     "mean abs plx diff (in terms of database standard deviations)", np.mean(abs(plx_diff))
# )
# print(
#     "median abs plx diff (in terms of database standard deviations)",
#     np.median(abs(plx_diff)),
# )
# print("# MC parallax > 1 sigma from the database value:", np.sum(abs(plx_diff) > 1))
# print(
#     "\n=== Comparing MC parallax uncertainties to database uncertainties "
#     "to see if model constrains parallaxes further ==="
# )
# print(
#     "min & max e_plx diff (in terms of database e_plx)",
#     np.min(e_plx_diff),
#     np.max(e_plx_diff),
# )
# print("mean e_plx diff (in terms of database e_plx)", np.mean(e_plx_diff))
# print(
#     "median e_plx diff (in terms of database e_plx)", np.median(e_plx_diff)
# )
# print(
#     "# of MCMC parallaxes with uncertainties larger than corresponding database parallax:",
#     np.sum(e_plx_diff > 0),
# )
# print(
#     "\n=== Comparing individual Upec & Vpec to average Upec & Vpec "
#     "from fit that used 100 MC distances ==="
# )
# print("Average Upec from 100 MC distance fit (km/s):", Upec_A5, "+/-", e_Upec_A5)
# print("Average Vpec from 100 MC distance fit (km/s):", Vpec_A5, "+/-", e_Vpec_A5)
# print(
#     "min, max, mean, & median abs Upec diff (km/s)",
#     np.min(abs(Upec_diff)),
#     np.max(abs(Upec_diff)),
#     np.mean(abs(Upec_diff)),
#     np.median(abs(Upec_diff))
# )
# print(
#     "min, max, mean, & median abs Vpec diff (km/s)",
#     np.min(abs(Vpec_diff)),
#     np.max(abs(Vpec_diff)),
#     np.mean(abs(Vpec_diff)),
#     np.median(abs(Vpec_diff))
# )
# print(
#     "min, max, mean, & median abs Upec diff (in terms of A5's Upec standard deviation)",
#     np.round(np.min(abs(Upec_diff) / e_Upec_A5), decimals=2),
#     np.round(np.max(abs(Upec_diff) / e_Upec_A5), decimals=2),
#     np.round(np.mean(abs(Upec_diff) / e_Upec_A5), decimals=2),
#     np.round(np.median(abs(Upec_diff) / e_Upec_A5), decimals=2),
# )
# print(
#     "min, max, mean, & median abs Vpec diff (in terms of A5's Vpec standard deviation)",
#     np.round(np.min(abs(Vpec_diff) / e_Vpec_A5), decimals=2),
#     np.round(np.max(abs(Vpec_diff) / e_Vpec_A5), decimals=2),
#     np.round(np.mean(abs(Vpec_diff) / e_Vpec_A5), decimals=2),
#     np.round(np.median(abs(Vpec_diff) / e_Vpec_A5), decimals=2),
# )
# print("# Upec > 1 sigma from the A5 value:", np.sum(abs(Upec_diff / e_Upec_A5) > 1))
# print("# Upec > 2 sigma from the A5 value:", np.sum(abs(Upec_diff / e_Upec_A5) > 2))
# print("# Vpec > 1 sigma from the A5 value:", np.sum(abs(Vpec_diff / e_Vpec_A5) > 1))
# print("# Vpec > 2 sigma from the A5 value:", np.sum(abs(Vpec_diff / e_Vpec_A5) > 2))

from pathlib import Path
# import pymc3 as pm
import dill
import numpy as np
import pandas as pd
import arviz as az

infile = Path(
    "/home/chengi/Documents/coop2021/bayesian_mcmc_rot_curve/"
    f"mcmc_outfile_A1_1dist_2.pkl"
)

with open(infile, "rb") as f:
    file = dill.load(f)
    trace = file["trace"]
    data = file["data"]

# num_iters = len(trace)
# num_chains = len(trace.chains)
num_sources = np.shape(trace["plx"])[1]
plx_hdi_minus1sigma = np.array([az.hdi(trace["plx"][:, idx], hdi_prob=.6827)[0] for idx in range(num_sources)])
plx_hdi_plus1sigma = np.array([az.hdi(trace["plx"][:, idx], hdi_prob=.6827)[1] for idx in range(num_sources)])
Upec_hdi_minus1sigma = np.array([az.hdi(trace["Upec"][:, idx], hdi_prob=.6827)[0] for idx in range(num_sources)])
Upec_hdi_plus1sigma = np.array([az.hdi(trace["Upec"][:, idx], hdi_prob=.6827)[1] for idx in range(num_sources)])
Vpec_hdi_minus1sigma = np.array([az.hdi(trace["Vpec"][:, idx], hdi_prob=.6827)[0] for idx in range(num_sources)])
Vpec_hdi_plus1sigma = np.array([az.hdi(trace["Vpec"][:, idx], hdi_prob=.6827)[1] for idx in range(num_sources)])
df = pd.DataFrame({
    "glong": data["glong"],
    "glat": data["glat"],
    "plx_mean": np.mean(trace["plx"], axis=0),
    "plx_med": np.median(trace["plx"], axis=0),
    "plx_sd": np.std(trace["plx"], axis=0),
    "plx_hdi_minus1sigma": plx_hdi_minus1sigma,
    "plx_hdi_plus1sigma": plx_hdi_plus1sigma,
    "Upec_mean": np.mean(trace["Upec"], axis=0),
    "Upec_med": np.median(trace["Upec"], axis=0),
    "Upec_sd": np.std(trace["Upec"], axis=0),
    "Upec_hdi_minus1sigma": Upec_hdi_minus1sigma,
    "Upec_hdi_plus1sigma": Upec_hdi_plus1sigma,
    "Vpec_mean": np.mean(trace["Vpec"], axis=0),
    "Vpec_med": np.median(trace["Vpec"], axis=0),
    "Vpec_sd": np.std(trace["Vpec"], axis=0),
    "Vpec_hdi_minus1sigma": Vpec_hdi_minus1sigma,
    "Vpec_hdi_plus1sigma": Vpec_hdi_plus1sigma,
})

np.savetxt(r"/home/chengi/Documents/coop2021/pec_motions/freeplx_freeUpecVpec.txt", df)
df.to_csv(
    path_or_buf=r"/home/chengi/Documents/coop2021/pec_motions/freeplx_freeUpecVpec.csv",
    sep=",",
    index=False,
    header=True,
)