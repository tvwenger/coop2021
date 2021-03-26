"""
face_on_view_plx.py

Plots a face-on (galactocentric) view of the Milky Way
with the Sun on +y-axis using kinematic distances.
This file uses data from the Parallax table in the database.

Isaac Cheng - March 2021
"""
import sys
import sqlite3
from contextlib import closing
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kd import pdf_kd, rotcurve_kd


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
        # id, name, ra (deg), dec (deg), glong (deg), glat (deg),
        # vlsr (km/s), e_vlsr (km/s)
    """

    with closing(sqlite3.connect(db_file).cursor()) as cur:  # context manager, auto-close
        # cur.execute("SELECT id, name, ra, dec, glong, glat, vlsr, e_vlsr FROM Detections")
        cur.execute("SELECT gname, ra, dec, glong, glat, plx, e_plx, vlsr, e_vlsr FROM Parallax")
        data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    return data


def plx_to_peak_dist (plx, e_plx):
    """
    Computes the peak of the parallax to distance distribution
    """

    plx_dist = 1 / plx
    sigma_sq = e_plx * e_plx
    return (np.sqrt(8 * sigma_sq * plx_dist * plx_dist + 1) - 1) \
           / (4 * sigma_sq * plx_dist)


def main(load_csv=False, rotcurve="cw21_rotcurve", num_samples=100,
         use_peculiar=True, use_kriging=False):
    if load_csv:
        csvfile = Path(__file__).parent / f"kd_plx_results_{num_samples}x_krige{use_kriging}.csv"
        kd_results = pd.read_csv(csvfile)
        glong = kd_results["glong"].values
        glat = kd_results["glat"].values
        plx = kd_results["plx"].values
        e_plx = kd_results["e_plx"].values
        vlsr = kd_results["vlsr"].values
        e_vlsr = kd_results["e_vlsr"].values
    else:
        print("=" * 6)
        print("Rotation model:", rotcurve)
        print("Number of MC kd samples:", num_samples)
        print("Including peculiar motions in kd:", use_peculiar)
        print("=" * 6)
        # Get HII region data
        dbfile = Path("/home/chengi/Documents/coop2021/data/hii_v2_20201203.db")
        data = get_data(dbfile)
        glong = data["glong"].values % 360.
        glat = data["glat"].values
        plx = data["plx"].values
        e_plx = data["e_plx"].values
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
        )
        # kd_results = rotcurve_kd.rotcurve_kd(
        #     glong, glat, vlsr, velo_tol=0.1, rotcurve=rotcurve, peculiar=use_peculiar
        # )
        print("Done kd")

        # Save results
        kd_df = pd.DataFrame(kd_results)
        print("Results shape:", np.shape(kd_df))
        # Add kd results to data (.reset_index() ensures rows have
        #                         same number & can concat properly)
        results = pd.concat([data.reset_index(drop=True),
                             kd_df.reset_index(drop=True)], axis=1)
        results.to_csv(
            path_or_buf=Path(__file__).parent / f"kd_plx_results_{num_samples}x_krige{use_kriging}.csv",
            sep=",",
            index=False,
            header=True,
        )
        print("Saved to .csv")
        kd_df = None  # free memory

    # Assign tangent kd to any source with vlsr w/in 20 km/s of tangent vlsr
    use_tangent = abs(kd_results["vlsr_tangent"] - vlsr) < 20
    # Otherwise, select kd that is closest to distance from parallax
    peak_dist = plx_to_peak_dist(plx, e_plx)
    near_err = abs(kd_results["near"] - peak_dist)
    far_err = abs(kd_results["far"] - peak_dist)
    tangent_err = abs(kd_results["tangent"] - peak_dist)
    min_err = np.fmin.reduce([near_err, far_err, tangent_err])  # ignores NaNs
    # Select distance corresponding to smallest error
    tol = 0.001  # tolerance for float equality
    is_near = (abs(near_err - min_err) < tol) & (~use_tangent)
    is_far = (abs(far_err- min_err) < tol) & (~use_tangent)
    is_tangent = (abs(tangent_err - min_err) < tol) | (use_tangent)
    conditions = [is_near, is_far, is_tangent]
    choices = [kd_results["near"], kd_results["far"], kd_results["tangent"]]
    dists = np.select(conditions, choices, default=np.nan)
    # Exclude any sources w/in 15 deg of GC or 20 deg of GAC
    glong_red = np.copy(glong)
    glong_red[glong > 180] -= 360  # force -180 < glong_red <= 180
    is_unreliable = (abs(glong_red) < 15.) | (abs(glong_red) > 160.)

    print("=" * 6)
    is_nan = (~is_near) & (~is_far) & (~is_tangent)
    # num_sources = np.sum(np.isfinite(dists)) + np.sum((is_unreliable) & (~is_nan)) - \
    #               np.sum((np.isfinite(dists)) & ((is_unreliable) & (~is_nan)))
    num_sources = np.sum(np.isfinite(dists))
    print("Total number of (non NaN) sources:", num_sources)
    print(f"Num near: {np.sum((is_near) & (~is_unreliable))}"
          + f"\tNum far: {np.sum((is_far) & (~is_unreliable))}"
          + f"\tNum tangent: {np.sum((is_tangent) & (~is_unreliable))}"
          + f"\tNum unreliable: {np.sum((is_unreliable) & (~is_nan))}")
    print("Number of NaN sources (i.e. all dists are NaNs):", np.sum(is_nan))
    print("Num NaNs in near, far, tangent:",
          np.sum(np.isnan(near_err)), np.sum(np.isnan(far_err)),
          np.sum(np.isnan(tangent_err)))
    # Print following if two distances are selected:
    num_near_far = np.sum((is_near) & (is_far))
    num_near_tan = np.sum((is_near) & (is_tangent))
    num_far_tan = np.sum((is_far) & (is_tangent))
    if any([num_near_far, num_near_tan, num_far_tan]):
        print("Both near and far (should be 0):", num_near_far)
        print("Both near and tan (should be 0):", num_near_tan)
        print("Both far and tan (should be 0):", num_far_tan)

    # Convert to galactocentric frame
    Xb, Yb, Zb = trans.gal_to_bary(glong, glat, dists)
    Xg, Yg, Zg = trans.bary_to_gcen(Xb, Yb, Zb, R0=_R0, Zsun=_ZSUN, roll=_ROLL)
    # Rotate 90 deg CW (so Sun is on +y-axis)
    Xg, Yg = Yg, -Xg

    # Plot
    fig, ax = plt.subplots()
    size = 2
    ax.scatter(Xg[(is_near) & (~is_unreliable)], Yg[(is_near) & (~is_unreliable)],
               c="tab:cyan", s=size, label="Near")
    ax.scatter(Xg[(is_far) & (~is_unreliable)], Yg[(is_far) & (~is_unreliable)],
               c="tab:purple", s=size, label="Far")
    ax.scatter(Xg[(is_tangent) & (~is_unreliable)], Yg[(is_tangent) & (~is_unreliable)],
               c="tab:green", s=size, label="Tangent")
    ax.scatter(Xg[is_unreliable], Yg[is_unreliable],
               c="tab:red", s=size, label="Unreliable")
    ax.legend(fontsize=9)
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
    fig.savefig(Path(__file__).parent / f"HII_faceonplx_{num_samples}x_krige{use_kriging}.pdf",
                bbox_inches="tight")
    plt.show()

    # dists = kd_results["distance"]
    # print("Total number of sources:", sum(~np.isnan(dists)))
    # # Sources with all nans
    # # by index+1 (aka line number in csv file): 191, 193, 194, 196, 198

    # # Convert to galactocentric frame
    # Xb, Yb, Zb = trans.gal_to_bary(glong, glat, dists)
    # Xg, Yg, Zg = trans.bary_to_gcen(Xb, Yb, Zb, R0=_R0, Zsun=_ZSUN, roll=_ROLL)
    # # Rotate 90 deg CW (so Sun is on +y-axis)
    # Xg, Yg = Yg, -Xg

    # # Plot
    # fig, ax = plt.subplots()
    # size = 2
    # ax.scatter(Xg, Yg, c="tab:cyan", s=size, label="Distance")
    # ax.legend(fontsize=9)
    # ax.set_xlabel("$x$ (kpc)")
    # ax.set_ylabel("$y$ (kpc)")
    # ax.axhline(y=0, linewidth=0.5, linestyle="--", color="k")  # horizontal line
    # ax.axvline(x=0, linewidth=0.5, linestyle="--", color="k")  # vertical line
    # ax.set_xlim(-8, 12)
    # ax.set_xticks([-5, 0, 5, 10])
    # ax.set_ylim(-5, 15)
    # ax.set_yticks([-5, 0, 5, 10, 15])
    # ax.grid(False)
    # fig.savefig(Path(__file__).parent / "HII_faceonplx_distance.pdf", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    load_csv_input = str2bool(input("(y/n) Plot results from csv? (default n): "),
                              empty_condition=False)
    if load_csv_input:
        main(load_csv=load_csv_input)
    else:
        rotcurve_input = input("rotcurve file (default cw21_rotcurve): ")
        rotcurve_input = "cw21_rotcurve" if rotcurve_input == "" else rotcurve_input
        num_samples_input = int(input("(int) Number of MC kd samples: "))
        use_pec_input = str2bool(
            input("(y/n) Include peculiar motions in kd (default y): "),
            empty_condition=True)
        use_kriging_input = str2bool(
            input("(y/n) Use kriging in kd (default n): "),
            empty_condition=False)
        main(
            load_csv=load_csv_input,
            rotcurve=rotcurve_input,
            num_samples=num_samples_input,
            use_peculiar=use_pec_input,
            use_kriging=use_kriging_input,
        )