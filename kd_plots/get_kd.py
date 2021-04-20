"""
get_kd.py

Finds kinematic distances of Parallax data in database.

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

# WC21 A6 rotation model parameters
_R0 = 8.1746
_USUN = 10.879
_VSUN = 10.697
_WSUN = 8.088
_E_USUN = 1.1
_E_VSUN = 6.1
_E_WSUN = 0.6
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


def main(rotcurve="wc21_rotcurve", num_samples=100,
         use_peculiar=True, use_kriging=False, norm=20, use_revised_lsr=False):
    print("=" * 6)
    print("Rotation model:", rotcurve)
    print("Number of MC kd samples:", num_samples)
    print("Including peculiar motions in kd:", use_peculiar)
    print("Using kriging:", use_kriging)
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
        use_revised_lsr = True
        print("Correcting vlsr for Reid 2014")
        vlsr_corr, e_vlsr_corr = correct_vlsr(glong, glat, vlsr, e_vlsr)
        vlsr, e_vlsr = vlsr_corr, e_vlsr_corr
    elif use_revised_lsr:
        print("Correcting vlsr for non-reid2014_rotcurve")
        vlsr_corr, e_vlsr_corr = correct_vlsr(
            glong, glat, vlsr, e_vlsr,
            Usun=_USUN, Vsun=_VSUN, Wsun=_WSUN,
            e_Usun=_E_USUN, e_Vsun=_E_VSUN, e_Wsun=_E_WSUN)
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
    rotcurve_truncated = rotcurve[:-9]
    filename = f"{rotcurve_truncated}_kd_{num_samples}x_krige{use_kriging}"
    if use_kriging:
        filename += f"_norm{norm}"
    filename += f"_revLsr{use_revised_lsr}"
    #
    pkl_outfile = Path(__file__).parent / f"{filename}.pkl"
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
    results = pd.concat(
        [data.reset_index(drop=True), kd_df.reset_index(drop=True)], axis=1)
    results.to_csv(
        path_or_buf=Path(__file__).parent / f"{filename}.csv",
        sep=",",
        index=False,
        header=True,
    )
    print("Saved to .csv")
    kd_df = None  # free memory


if __name__ == "__main__":
    rotcurve_input = input("rotcurve file (default wc21_rotcurve): ")
    rotcurve_input = "wc21_rotcurve" if rotcurve_input == "" else rotcurve_input
    num_samples_input = int(input("(int) Number of MC kd samples: "))
    use_pec_input = str2bool(
        input("(y/n) Include peculiar motions in kd (default y): "),
        empty_condition=True)
    use_kriging_input = str2bool(
        input("(y/n) Use kriging in kd (default n): "),
        empty_condition=False)
    if use_kriging_input:
        norm_input = float(input("(float) normalization factor for kriging: "))
    else:
        norm_input = None
    use_revised_lsr_input = str2bool(
        input("Use revised LSR velocity (default n): "), empty_condition=False)
    main(
        rotcurve=rotcurve_input,
        num_samples=num_samples_input,
        use_peculiar=use_pec_input,
        use_kriging=use_kriging_input,
        norm=norm_input,
        use_revised_lsr=use_revised_lsr_input,
    )
