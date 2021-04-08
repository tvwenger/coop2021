"""
face_on_view_plx.py

! Use kd_plx_diff.py instead !

Plots a face-on (galactocentric) view of the Milky Way
with the Sun on +y-axis using kinematic distances.
This file uses data from the Parallax table in the database.

Isaac Cheng - March 2021
"""
import sys
import sqlite3
from contextlib import closing
from pathlib import Path
import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from re import findall  # for regex
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


def correct_vlsr(glong, glat, vlsr, e_vlsr,
                 Ustd=10.27, Vstd=15.32, Wstd=7.74,
                 Usun=10.5, e_Usun=1.7, Vsun=14.4, e_Vsun=6.8,
                 Wsun=8.9, e_Wsun=0.9):
    """
    NOTE: This function is verbatim from Dr. Trey Wenger's kd_utils (commit 5ea4996)

    Return the "corrected" LSR velocity by updating the IAU-defined
    solar motion components (Ustd,Vstd,Wstd) to newly-measured
    values (Usun,Vsun,Wsun).
    Also computes the new LSR velocity uncertainty including the
    uncertainties in the newly-measured values (e_Usun,e_Vsun,e_Wsun).
    Parameters:
      glong : scalar or 1-D array
              Galactic longitude (deg). If it is an array, it must
              have the same size as glat, vlsr, and e_vlsr.
      glat : scalar or 1-D array
             Galactic latitude (deg). If it is an array, it
             must have the same size as glong, vlsr, and e_vlsr.
      vlsr : scalar or 1-D array
             Measured LSR velocity (km/s). If it is an array, it
             must have the same size as glong, glat, and e_vlsr.
      e_vlsr : scalar or 1-D array
               Uncertainty on measured LSR velocity (km/s). If it is
               an array, it must have the same size as glong, glat,
               and vlsr.
      Ustd,Vstd,Wstd : scalar (optional)
                       IAU-defined solar motion parameters (km/s).
      Usun,Vsun,Wsun : scalar (optional)
                       Newly measured solar motion parameters (km/s).
                       Defaults are from Reid et al. (2014)
      e_Usun,e_Vsun,e_Wsun : scalar (optional)
                       Newly measured solar motion parameter
                       uncertainties (km/s).
                       Defaults are from Reid et al. (2014)
    Returns: (new_vlsr,e_new_vlsr)
      new_vlsr : scalar or 1-D array
                 Re-computed LSR velocity. Same shape as vlsr.
      e_vlsr : scalar or 1-D array
               Re-computed LSR velocity uncertainty. Same shape as
               e_vlsr.
    Raises:
      ValueError : if glong, glat, vlsr, and e_vlsr are not 1-D; or
                   if glong, glat, vlsr, and e_vlsr are arrays and
                   not the same size
    """
    #
    # check inputs
    #
    # convert scalar to array if necessary
    glong_inp, glat_inp, vlsr_inp, e_vlsr_inp = \
      np.atleast_1d(glong, glat, vlsr, e_vlsr)
    # check shape of inputs
    if (glong_inp.ndim != 1 or glat_inp.ndim != 1 or
        vlsr_inp.ndim != 1 or e_vlsr_inp.ndim != 1):
        raise ValueError("glong, glat, vlsr, and e_vlsr must be 1-D")
    if glong_inp.size != 1 and (glong_inp.size != glat_inp.size or
                                glong_inp.size != vlsr_inp.size or
                                glong_inp.size != e_vlsr_inp.size):
        raise ValueError("glong, glat, vlsr, and e_vlsr must have same size")
    #
    # Useful values
    #
    cos_glong = np.cos(np.deg2rad(glong_inp))
    sin_glong = np.sin(np.deg2rad(glong_inp))
    cos_glat = np.cos(np.deg2rad(glat_inp))
    sin_glat = np.sin(np.deg2rad(glat_inp))
    #
    # Compute heliocentric velocity by subtracting IAU defined solar
    # motion components
    #
    U_part = Ustd*cos_glong
    V_part = Vstd*sin_glong
    W_part = Wstd*sin_glat
    UV_part = (U_part+V_part)*cos_glat
    v_helio = vlsr_inp - UV_part - W_part
    #
    # Compute corrected VLSR
    #
    U_part = Usun*cos_glong
    V_part = Vsun*sin_glong
    W_part = Wsun*sin_glat
    UV_part = (U_part+V_part)*cos_glat
    new_vlsr = v_helio + UV_part + W_part
    #
    # Compute corrected LSR velocity uncertainty
    #
    U_part = (e_Usun*cos_glong*cos_glat)**2.
    V_part = (e_Vsun*sin_glong*cos_glat)**2.
    W_part = (e_Wsun*sin_glat)**2.
    e_new_vlsr = np.sqrt(e_vlsr_inp**2.+U_part+V_part+W_part)
    #
    # Convert back to scalar if necessary
    #
    if glong_inp.size == 1:
        return new_vlsr[0],e_new_vlsr[0]
    else:
        return new_vlsr,e_new_vlsr


def main(load_csv=False, csv_filename=None, rotcurve="cw21_rotcurve", num_samples=100,
         use_peculiar=True, use_kriging=False, vlsr_tol=20, norm=20):
    if load_csv:
        # Find (all) numbers in csv_filename --> num_samples
        num_samples = findall(r"\d+", csv_filename)
        if len(num_samples) != 1:
            print("regex num_samples:", num_samples)
            raise ValueError("Invalid number of samples parsed")
        num_samples = int(num_samples[0])
        # Find if kd used kriging
        use_kriging = findall("True", csv_filename)
        if len(use_kriging) > 1:
            print("regex use_kriging:", use_kriging)
            raise ValueError("Invalid use_kriging parsed")
        use_kriging = bool(use_kriging)
        # Load csv results
        csvfile = Path(__file__).parent / csv_filename
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
        print("Using kriging:", use_kriging)
        print("vlsr tolerance (km/s):", vlsr_tol)
        print("Normalization factor:", norm)
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
        if rotcurve == "reid14_rotcurve":
            print("Correcting vlsr")
            vlsr_corr, e_vlsr_corr = correct_vlsr(glong, glat, vlsr, e_vlsr)
            vlsr, e_vlsr = vlsr_corr, e_vlsr_corr

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
            norm=norm,
        )
        # kd_results = rotcurve_kd.rotcurve_kd(
        #     glong, glat, vlsr, velo_tol=0.1, rotcurve=rotcurve, peculiar=use_peculiar
        # )
        print("Done kd")

        # Save to pickle file
        pkl_filename = f"kd_plx_results_{num_samples}x_krige{use_kriging}"
        if use_kriging:
            pkl_filename += f"_norm{norm}"
        pkl_outfile = Path(__file__).parent / f"{pkl_filename}.pkl"
        with open(pkl_outfile, "wb") as f:
            dill.dump(
                {
                    "data": data,
                    "kd": kd_results,
                },
                f
            )

        # Save to csv file
        kd_df = pd.DataFrame(kd_results)
        print("Results shape:", np.shape(kd_df))
        # Add kd results to data (.reset_index() ensures rows have
        #                         same number & can concat properly)
        results = pd.concat([data.reset_index(drop=True),
                             kd_df.reset_index(drop=True)], axis=1)
        csv_filename = f"kd_plx_results_{num_samples}x_krige{use_kriging}"
        if use_kriging:
            csv_filename += f"_norm{norm}"
        results.to_csv(
            path_or_buf=Path(__file__).parent / f"{csv_filename}.csv",
            sep=",",
            index=False,
            header=True,
        )
        print("Saved to .csv")
        kd_df = None  # free memory

    # Assign tangent kd to any source with vlsr w/in vlsr_tol of tangent vlsr
    use_tangent = abs(kd_results["vlsr_tangent"] - vlsr) < vlsr_tol
    # Otherwise, select kd that is closest to distance from parallax
    peak_dist = plx_to_peak_dist(plx, e_plx)
    near_err = abs(kd_results["near"] - peak_dist)
    far_err = abs(kd_results["far"] - peak_dist)
    tangent_err = abs(kd_results["tangent"] - peak_dist)
    min_err = np.fmin.reduce([near_err, far_err, tangent_err])  # ignores NaNs
    # Select distance corresponding to smallest error
    tol = 1e-9  # tolerance for float equality
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
    size_scale = 4  # scaling factor for size
    #
    size_near = 0.5 * (kd_results["near_err_pos"] + kd_results["near_err_neg"])
    ax.scatter(Xg[(is_near) & (~is_unreliable)], Yg[(is_near) & (~is_unreliable)],
               c="tab:cyan", s=size_near[(is_near) & (~is_unreliable)] * size_scale,
               label="Near")
    #
    size_far = 0.5 * (kd_results["far_err_pos"] + kd_results["far_err_neg"])
    ax.scatter(Xg[(is_far) & (~is_unreliable)], Yg[(is_far) & (~is_unreliable)],
               c="tab:purple", s=size_far[(is_far) & (~is_unreliable)] * size_scale,
               label="Far")
    #
    size_tan = 0.5 * (kd_results["distance_err_pos"] + kd_results["distance_err_neg"])
    ax.scatter(Xg[(is_tangent) & (~is_unreliable)], Yg[(is_tangent) & (~is_unreliable)],
               c="tab:green", s=size_tan[(is_tangent) & (~is_unreliable)] * size_scale,
               label="Tangent")
    #
    conditions = [(is_near) & (is_unreliable), (is_far) & (is_unreliable), (is_tangent) & (is_unreliable)]
    choices = [size_near, size_far, size_tan]
    size_unreliable = np.select(conditions, choices, default=np.nan)
    if np.sum(~np.isnan(size_unreliable)) != np.sum((is_unreliable) & (~is_nan)):
        print("Number of NaNs in size_unreliable (should be equal to 'Num unreliable'):",
              np.sum(np.isnan(size_unreliable)))
    ax.scatter(Xg[is_unreliable], Yg[is_unreliable],
               c="tab:red", s=size_unreliable[is_unreliable] * size_scale,
               label="Unreliable")
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
    fig_filename = f"HII_faceonplx_{num_samples}x_krige{use_kriging}_vlsrTolerance{vlsr_tol}.pdf"
    fig.savefig(Path(__file__).parent / fig_filename, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    load_csv_input = str2bool(input("(y/n) Plot results from csv? (default n): "),
                              empty_condition=False)
    if load_csv_input:
        csv_filename_input = input(".csv filename in this folder: ")
        rotcurve_input = num_samples_input = use_pec_input = use_kriging_input = None
        # main(load_csv=load_csv_input, csv_filename=csv_filename_input)
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
        csv_filename_input = None
        if use_kriging_input:
            norm_input = float(input("(float) normalization factor for kriging: "))
        else:
            norm_input = None
    vlsr_tol_input = input("(int) Assign tangent distance if vlsr is within ___ " +
                            "km/s of tangent velocity? (default 20): ")
    vlsr_tol_input = 20 if vlsr_tol_input == "" else int(vlsr_tol_input)
    main(
        load_csv=load_csv_input,
        csv_filename=csv_filename_input,
        rotcurve=rotcurve_input,
        num_samples=num_samples_input,
        use_peculiar=use_pec_input,
        use_kriging=use_kriging_input,
        vlsr_tol=vlsr_tol_input,
        norm=norm_input,
    )
