"""
vlsr_dist_plot.py

Plots the line-of-sight velocity as a function of
galactic distance along a given longitude and latitude.

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
from kd import rotcurve_kd_vlsrDistPlot
from re import search  # for regex

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)
_SCRIPT_DIR = str(Path.cwd().parent)
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
def plot_vlsr_dist(source_to_plot, rotcurve="cw21_rotcurve", num_cores=None,
                   use_peculiar=True, use_kriging=False, resample=False, size=1):
    # Get HII region data
    # dbfile = Path("/home/chengi/Documents/coop2021/data/hii_v2_20201203.db")
    dbfile = Path(__file__).parent.parent / Path("data/hii_v2_20201203.db")
    data = get_data(dbfile)
    data = data.iloc[source_to_plot-2]  # only select data of source to plot
    print("=" * 6)
    print("Source to plot:", data["gname"], f"(row #{source_to_plot} in .csv)")
    print("Rotation model:", rotcurve)
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
    kd_results = rotcurve_kd_vlsrDistPlot.rotcurve_kd_vlsrDistPlot(
        glong,
        glat,
        vlsr,
        velo_err=e_vlsr,
        rotcurve=rotcurve,
        peculiar=use_peculiar,
        use_kriging=use_kriging,
        processes=num_cores,
        resample=resample,
        size=size,
    )
    print("Done kd")

# %%
# source_to_plot_input = int(input(
#     "(int) 'Index' of the source to plot (i.e. row num from .csv file): "))
# # rotcurve_input = input("rotcurve file (default cw21_rotcurve): ")
# # rotcurve_input = "cw21_rotcurve" if rotcurve_input == "" else rotcurve_input
# num_cores_input = input("(int) Number of CPU threads to use (default None): ")
# num_cores_input = None if num_cores_input == "" or num_cores_input.lower() == "none" else int(num_cores_input)
# use_pec_input = str2bool(
#     input("(y/n) Include peculiar motions in kd (default y): "),
#     empty_condition=True)
# use_kriging_input = str2bool(
#     input("(y/n) Use kriging in kd (default n): "),
#     empty_condition=False)
rotcurve_input = "cw21_rotcurve"
source_to_plot_input = 32
num_cores_input = 4
use_pec_input = True
use_kriging_input = False
plot_vlsr_dist(
    source_to_plot_input,
    rotcurve=rotcurve_input,
    num_cores=num_cores_input,
    use_peculiar=use_pec_input,
    use_kriging=use_kriging_input,
    resample=True,
    size=100,
)
