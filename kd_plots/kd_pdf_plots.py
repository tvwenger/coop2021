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
           use_peculiar=True, use_kriging=False, norm=20):
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
    # dbfile = Path("/home/chengi/Documents/coop2021/data/hii_v2_20201203.db")
    dbfile = Path(__file__).parent.parent / Path("data/hii_v2_20201203.db")
    data = get_data(dbfile)
    data = data.iloc[source_to_plot-2]  # only select data of source to plot
    print("=" * 6)
    print("Source to plot:", data["gname"], f"(row #{source_to_plot} in .csv)")
    print("Rotation model:", rotcurve)
    print("Number of MC kd samples:", num_samples)
    print("Including peculiar motions in kd:", use_peculiar)
    print("Using kriging:", use_kriging)
    print("=" * 6)

    glong = data["glong"] % 360.
    glat = data["glat"]
    # plx = data["plx"].values
    # e_plx = data["e_plx"].values
    vlsr = data["vlsr"]
    e_vlsr = data["e_vlsr"]

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
        plot_pdf=True,
        norm=norm,
    )
    print("Done kd")

    # # Save results
    # kd_df = pd.DataFrame(kd_results)
    # print("Results shape:", np.shape(kd_df))
    # # Add kd results to data (.reset_index() ensures rows have
    # #                         same number & can concat properly)
    # results = pd.concat([data.reset_index(drop=True),
    #                         kd_df.reset_index(drop=True)], axis=1)
    # csv_filename = f"kd_source{source_to_plot}_{data['gname']}.csv"
    # results.to_csv(
    #     path_or_buf=Path(__file__).parent / csv_filename,
    #     sep=",",
    #     index=False,
    #     header=True,
    # )
    # print("Saved to .csv")

# %%
# source_to_plot_input = int(input(
#     "(int) 'Index' of the source to plot (i.e. row num from .csv file): "))
# rotcurve_input = input("rotcurve file (default cw21_rotcurve): ")
# rotcurve_input = "cw21_rotcurve" if rotcurve_input == "" else rotcurve_input
# num_samples_input = int(input("(int) Number of MC kd samples: "))
# use_pec_input = str2bool(
#     input("(y/n) Include peculiar motions in kd (default y): "),
#     empty_condition=True)
# use_kriging_input = str2bool(
#     input("(y/n) Use kriging in kd (default n): "),
#     empty_condition=False)
# if use_kriging_input:
#     norm_input = float(input("(float) normalization factor for kriging: "))
source_to_plot_input = 147
rotcurve_input = "wc21_rotcurve"
num_samples_input = 10000
use_pec_input = True
use_kriging_input = False
norm_input = 20
run_kd(
    source_to_plot_input,
    rotcurve=rotcurve_input,
    num_samples=num_samples_input,
    use_peculiar=use_pec_input,
    use_kriging=use_kriging_input,
    norm=norm_input,
)