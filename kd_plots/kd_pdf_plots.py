"""
kd_pdf_plots.py

Plots 1 source's  probability density function (PDF) for
various kinematic distance parameters.

Isaac Cheng - March 2021
"""
# %%
import sys
from pathlib import Path
import sqlite3
from contextlib import closing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kd import pdf_kd

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


# %%
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


# %%
def get_data(db_file):
    """
    Puts all relevant data from db_file's Parallax table into pandas DataFrame.

    Inputs:
      db_file :: pathlib Path object
        Path object to SQL database containing HII region data in Parallax table.

    Returns: data
      data :: pandas DataFrame
        Contains db_file data in pandas DataFrame. Specifically, it includes:
        gname, ra (deg), dec (deg), glong (deg), glat (deg),
        plx (mas), e_plx (mas), vlsr (km/s), e_vlsr (km/s)
    """

    with closing(sqlite3.connect(db_file).cursor()) as cur:  # context manager, auto-close
        cur.execute(
            "SELECT gname, ra, dec, glong, glat, plx, e_plx, vlsr, e_vlsr FROM Parallax"
        )
        data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    return data


# %%
def run_kd(source_to_plot, rotcurve="cw21_rotcurve", num_samples=100,
           use_peculiar=True, use_kriging=False):
    # if load_csv:
    #     # Find (all) numbers in csv_filename --> num_samples
    #     num_samples = findall(r"\d+", csv_filename)
    #     if len(num_samples) != 1:
    #         print("regex num_samples:", num_samples)
    #         raise ValueError("Invalid number of samples parsed")
    #     num_samples = int(num_samples[0])
    #     # Find if kd used kriging
    #     use_kriging = findall("True", csv_filename)
    #     if len(use_kriging) > 1:
    #         print("regex use_kriging:", use_kriging)
    #         raise ValueError("Invalid use_kriging parsed")
    #     use_kriging = bool(use_kriging)
    #     # Load csv results
    #     csvfile = Path(__file__).parent / csv_filename
    #     kd_results = pd.read_csv(csvfile)
    #     glong = kd_results["glong"].values
    #     glat = kd_results["glat"].values
    #     plx = kd_results["plx"].values
    #     e_plx = kd_results["e_plx"].values
    #     vlsr = kd_results["vlsr"].values
    #     e_vlsr = kd_results["e_vlsr"].values
    # else:
    # Get HII region data
    dbfile = Path("/home/chengi/Documents/coop2021/data/hii_v2_20201203.db")
    # dbfile = Path(__file__).parent.parent / Path("data/hii_v2_20201203.db")
    data = get_data(dbfile)
    data = data.iloc[source_to_plot-2]  # only select data of source to plot
    print("=" * 6)
    print("Source to plot:", data["gname"], f"(row #{source_to_plot} in .csv)")
    print("Rotation model:", rotcurve)
    print("Number of MC kd samples:", num_samples)
    print("Including peculiar motions in kd:", use_peculiar)
    print("Using kriging:", use_kriging)
    print("=" * 6)

    glong = data["glong"].values % 360.
    glat = data["glat"].values
    # plx = data["plx"].values
    # e_plx = data["e_plx"].values
    vlsr = data["vlsr"].values
    e_vlsr = data["e_vlsr"].values

    # MC kinematic distances
    print("kd in progress...")
    kd_results = pdf_kd.pdf_kd(
        glong,
        glat,
        vlsr,
        velo_err=e_vlsr,
        rotcurve=rotcurve,
        num_samples=num_samples,
        peculiar=use_peculiar,
        use_kriging=use_kriging,
        plot_pdf=True
    )
    print("Done kd")

    # Save results
    kd_df = pd.DataFrame(kd_results)
    print("Results shape:", np.shape(kd_df))
    # Add kd results to data (.reset_index() ensures rows have
    #                         same number & can concat properly)
    results = pd.concat([data.reset_index(drop=True),
                            kd_df.reset_index(drop=True)], axis=1)
    csv_filename = f"kd_source{source_to_plot}_{data['gname']}.csv"
    results.to_csv(
        path_or_buf=Path(__file__).parent / csv_filename,
        sep=",",
        index=False,
        header=True,
    )
    print("Saved to .csv")

# %%
source_to_plot_input = int(input(
    "(int) 'Index' of the source to plot (i.e. row num from .csv file): "))
# rotcurve_input = input("rotcurve file (default cw21_rotcurve): ")
# rotcurve_input = "cw21_rotcurve" if rotcurve_input == "" else rotcurve_input
num_samples_input = int(input("(int) Number of MC kd samples: "))
use_pec_input = str2bool(
    input("(y/n) Include peculiar motions in kd (default y): "),
    empty_condition=True)
use_kriging_input = str2bool(
    input("(y/n) Use kriging in kd (default n): "),
    empty_condition=False)
rotcurve_input = "cw21_rotcurve"
run_kd(
    source_to_plot_input,
    rotcurve=rotcurve_input,
    num_samples=num_samples_input,
    use_peculiar=use_pec_input,
    use_kriging=use_kriging_input,
)

# # %%
# def assign_kd_distances(database_data, kd_results, vlsr_tol=20):
#     """
#     Returns the closest kinematic distance to parallax distance.
#     If vlsr (km/s) is within vlsr_tol (km/s) of tangent point vlsr,
#     use the tangent point vlsr

#     Inputs:
#       database_data :: pandas DataFrame
#       kd_results :: pandas DataFrame
#       vlsr_tol :: scalar
#     """
#     glong = database_data["glong"].values
#     vlsr = database_data["vlsr_med"].values
#     peak_dist = database_data["dist_mode"].values
#     # Assign tangent kd to any source with vlsr w/in vlsr_tol of tangent vlsr
#     # TODO: fix use_tangent
#     use_tangent = abs(kd_results["vlsr_tangent"] - vlsr) < vlsr_tol
#     # Otherwise, select kd that is closest to distance from parallax
#     # peak_dist = plx_to_peak_dist(plx, e_plx)
#     near_err = abs(kd_results["near"] - peak_dist)
#     far_err = abs(kd_results["far"] - peak_dist)
#     tangent_err = abs(kd_results["tangent"] - peak_dist)
#     min_err = np.fmin.reduce([near_err, far_err, tangent_err])  # ignores NaNs
#     # Select distance corresponding to smallest error
#     tol = 1e-9  # tolerance for float equality
#     is_near = (abs(near_err - min_err) < tol) & (~use_tangent)
#     is_far = (abs(far_err - min_err) < tol) & (~use_tangent)
#     is_tangent = (abs(tangent_err - min_err) < tol) | (use_tangent)
#     conditions = [is_near, is_far, is_tangent]
#     choices = [kd_results["near"], kd_results["far"], kd_results["tangent"]]
#     dists = np.select(conditions, choices, default=np.nan)
#     # Exclude any sources w/in 15 deg of GC or 20 deg of GAC
#     glong[glong > 180] -= 360  # force -180 < glong <= 180
#     is_unreliable = (abs(glong) < 15.0) | (abs(glong) > 160.0)
#     #
#     print("=" * 6)
#     is_nan = (~is_near) & (~is_far) & (~is_tangent)
#     # num_sources = np.sum(np.isfinite(dists)) + np.sum((is_unreliable) & (~is_nan)) - \
#     #               np.sum((np.isfinite(dists)) & ((is_unreliable) & (~is_nan)))
#     num_sources = np.sum(np.isfinite(dists))
#     print("Total number of (non NaN) sources:", num_sources)
#     print(
#         f"Num near: {np.sum((is_near) & (~is_unreliable))}"
#         + f"\tNum far: {np.sum((is_far) & (~is_unreliable))}"
#         + f"\tNum tangent: {np.sum((is_tangent) & (~is_unreliable))}"
#         + f"\tNum unreliable: {np.sum((is_unreliable) & (~is_nan))}"
#     )
#     print("Number of NaN sources (i.e. all dists are NaNs):", np.sum(is_nan))
#     print(
#         "Num NaNs in near, far, tangent:",
#         np.sum(np.isnan(near_err)),
#         np.sum(np.isnan(far_err)),
#         np.sum(np.isnan(tangent_err)),
#     )
#     # Print following if two distances are selected:
#     num_near_far = np.sum((is_near) & (is_far))
#     num_near_tan = np.sum((is_near) & (is_tangent))
#     num_far_tan = np.sum((is_far) & (is_tangent))
#     if any([num_near_far, num_near_tan, num_far_tan]):
#         print("Both near and far (should be 0):", num_near_far)
#         print("Both near and tan (should be 0):", num_near_tan)
#         print("Both far and tan (should be 0):", num_far_tan)
#     #
#     e_near = 0.5 * (kd_results["near_err_pos"] + kd_results["near_err_neg"])
#     e_far = 0.5 * (kd_results["far_err_pos"] + kd_results["far_err_neg"])
#     e_tan = 0.5 * (kd_results["distance_err_pos"] + kd_results["distance_err_neg"])
#     e_conditions = [is_near, is_far, is_tangent]
#     e_choices = [e_near, e_far, e_tan]
#     e_dists = np.select(e_conditions, e_choices, default=np.nan)
#     print("Num of NaN errors (i.e. all errors are NaNs):", np.sum(np.isnan(e_dists)))

#     return dists, e_dists, is_near, is_far, is_tangent, is_unreliable


# # %%
# vlsr_tol = 20
# #
# # Load plx data
# #
# plxfile = Path("pec_motions/csvfiles/alldata_HPDmode_NEW.csv")
# plxdata = pd.read_csv(Path(__file__).parent.parent / plxfile)
# dist_plx = plxdata["dist_mode"].values
# e_dist_plx = plxdata["dist_halfhpd"].values
# #
# # RegEx stuff
# #
# # Find num_samples used in kd
# # num_samples = findall(r"\d+", kdfile)
# # if len(num_samples) != 1:
# #     print("regex num_samples:", num_samples)
# #     raise ValueError("Invalid number of samples parsed")
# # num_samples = int(num_samples[0])
# # # Find if kd used kriging
# # use_kriging = findall("True", kdfile)
# # if len(use_kriging) > 1:
# #     print("regex use_kriging:", use_kriging)
# #     raise ValueError("Invalid use_kriging parsed")
# # use_kriging = bool(use_kriging)
# #
# # Load kd data
# #
# kdfile = "kd_plx_results_10000x_krigeTrue.csv"
# kddata = pd.read_csv(Path(__file__).parent / kdfile)
# dist_kd, e_dist_kd, is_near, is_far, is_tangent, is_unreliable = assign_kd_distances(
#     plxdata, kddata, vlsr_tol=vlsr_tol
# )
