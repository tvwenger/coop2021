"""
galaxymap_hpd.py

Plots a face-on galactocentric Cartesian map of
Parallax HII regions using the peak of the distributions

Isaac Cheng - March 2021
"""
import sys
import sqlite3
from contextlib import closing
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans


def get_data(db_file):
    """
    Puts all relevant data from db_file's Parallax table into pandas DataFrame.

    Inputs:
      db_file :: pathlib Path object
        Path object to SQL database containing HII region data in Parallax table.

    Returns: data
      data :: pandas DataFrame
        Contains glong (deg), glat (deg), plx (mas), e_plx (mas)
    """

    with closing(sqlite3.connect(db_file).cursor()) as cur:  # context manager, auto-close
        cur.execute("SELECT gname, glong, glat, plx, e_plx FROM Parallax")
        data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    return data


def main():
    # db = Path(__file__).parent.parent / "data/hii_v2_20201203.db"
    # data = get_data(db)
    # gname = data["gname"].values
    # glong = data["glong"].values
    # glat = data["glat"].values
    # plx = data["plx"].values
    # e_plx = data["e_plx"].values

    # # Difference between analytic and numerical distances
    # dist = trans.parallax_to_dist(plx, e_parallax=e_plx)
    # data2 = Path(__file__).parent.parent / "pec_motions/alldata_HPDmode_NEW.csv"
    # dist_hpd = pd.read_csv(data2)["dist_mode"].values
    # dist_diff = dist - dist_hpd
    # diffs = {
    #     "gname": gname,
    #     "peak_pdf": dist,
    #     "peak_hpd": dist_hpd,
    #     "error": dist_diff
    # }
    # pd.DataFrame(diffs).to_csv(
    #     path_or_buf=Path(__file__).parent/"diff_mode_analytical_plx.csv",
    #     sep=",",
    #     index=False,
    #     header=True,
    # )
    # print(np.max(dist_diff), np.min(dist_diff))
    # print(np.max(abs(dist_diff)), np.min(abs(dist_diff)))
    # print(np.mean(dist_diff), np.median(dist_diff))

    csv_filepath = Path("pec_motions/csvfiles/alldata_HPDmode_NEW2.csv")
    data = pd.read_csv(Path(__file__).parent.parent / csv_filepath)
    x = data["x_mode"].values
    y = data["y_mode"].values
    # x_err = data["x_halfhpd"].values
    # y_err = data["y_halfhpd"].values
    # err = np.sqrt(x_err * x_err + y_err * y_err)
    err = data["dist_halfhpd"].values

    # Plot
    fig, ax = plt.subplots()
    size_scale = 4  # scaling factor for size

    ax.scatter(x, y, s=err*size_scale, c="k")
    ax.set_xlabel("$x$ (kpc)")
    ax.set_ylabel("$y$ (kpc)")
    ax.axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    ax.axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    ax.set_xlim(-8, 12)
    ax.set_xticks([-5, 0, 5, 10])
    ax.set_ylim(-5, 15)
    ax.set_yticks([-5, 0, 5, 10, 15])
    ax.grid(False)
    ax.set_aspect("equal")
    fig.savefig(Path(__file__).parent / "galaxymap_hpd_distErr.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
