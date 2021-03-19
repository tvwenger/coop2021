"""
face_on_view.py

Plots a face-on (galactocentric) view of the Milky Way
with the Sun on +y-axis using kinematic distances.

Isaac Cheng - March 2021
"""
import sys
import sqlite3
from contextlib import closing
from pathlib import Path
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


def get_data(db_file):
    """
    Puts all relevant data from db_file's Detections table into pandas DataFrame.

    Inputs:
      db_file :: pathlib Path object
        Path object to SQL database containing HII region data in Detections table.

    Returns: data
      data :: pandas DataFrame
        Contains db_file data in pandas DataFrame. Specifically, it includes:
        id, name, ra (deg), dec (deg), glong (deg), glat (deg),
        vlsr (km/s), e_vlsr (km/s)
    """

    with closing(sqlite3.connect(db_file).cursor()) as cur:  # context manager, auto-close
        # cur.execute("SELECT id, name, ra, dec, glong, glat, vlsr, e_vlsr FROM Detections")
        cur.execute("SELECT gname, ra, dec, glong, glat, vlsr, e_vlsr FROM Parallax")
        data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    return data


def main():
    # Get HII region data
    dbfile = Path("/home/chengi/Documents/coop2021/data/hii_v2_20201203.db")
    data = get_data(dbfile)
    glong = data["glong"].values
    glat = data["glat"].values
    vlsr = data["vlsr"].values
    e_vlsr = data["e_vlsr"].values

    # MC kinematic distances
    rotcurve = "cw21_rotcurve"  # the name of the script containing the rotation curve
    num_samples = 2  # number of MC re-samples
    peculiar = True  # include peculiar motion (either average or kriging)
    use_kriging = True  # use kriging to estimate peculiar motions

    print("kd in progress...")
    kd_dict = pdf_kd.pdf_kd(
        glong,
        glat,
        vlsr,
        velo_err=e_vlsr,
        rotcurve=rotcurve,
        num_samples=num_samples,
        peculiar=peculiar,
        use_kriging=use_kriging,
    )
    print("Done kd")

    # Save results
    kd_df = pd.DataFrame.from_dict(kd_dict)
    results = data.append(kd_df, sort=False)  # add kd results to data
    results.to_csv(
        path_or_buf=Path(__file__).parent / Path("kd_results.csv"),
        sep=",",
        index=False,
        header=True,
    )
    print("Saved to .csv")

    # Convert to galactocentric frame
    # ? How to get azimuth... Which distance to use?
    # For now, use distance with smallest error
    near_err = 0.5 * (kd_dict["near_err_pos"] + kd_dict["near_err_neg"])
    far_err = 0.5 * (kd_dict["far_err_pos"] + kd_dict["far_err_neg"])
    tangent_err = 0.5 * (kd_dict["tangent_err_pos"] + kd_dict["tangent_err_neg"])
    # Ignore NaNs
    min_err = np.fmin.reduce([near_err, far_err, tangent_err])
    # Select distance corresponding to smallest error
    is_near = near_err == min_err
    is_far = far_err == min_err
    is_tangent = tangent_err == min_err
    conditions = [is_near, is_far, is_tangent]
    choices = [kd_dict["near"], kd_dict["far"], kd_dict["tangent"]]
    dists = np.select(conditions, choices, default=np.nan)
    # near_dists = kd_dict["near"][near_err == min_err]
    # far_dists = kd_dict["far"][far_err == min_err]
    # tangent_dists = kd_dict["tangent"][tangent_err == min_err]
    print("Total number of sources:", sum(~np.isnan(dists)))

    # Convert to galactocentric frame
    Xb, Yb, Zb = trans.gal_to_bary(glong, glat, dists)
    Xg, Yg, Zg = trans.bary_to_gcen(Xb, Yb, Zb, R0=_R0, Zsun=_ZSUN, roll=_ROLL)
    # Rotate 90 deg CW (so Sun is on +y-axis)
    Xg, Yg = Yg, -Xg

    # Plot
    fig, ax = plt.subplots()
    size = 2
    ax.scatter(Xg[is_near], Yg[is_near], c="tab:cyan", s=size, label="Near")
    ax.scatter(Xg[is_far], Yg[is_far], c="tab:purple", s=size, label="Far")
    ax.scatter(Xg[is_tangent], Yg[is_tangent], c="tab:red", s=size, label="Tangent")
    ax.legend(fontsize=9)
    ax.grid(False)
    fig.savefig("HII_faceon.pdf")
    plt.show()


if __name__ == "__main__":
    main()
