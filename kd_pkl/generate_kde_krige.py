"""
generate_kde_krige.py

Generates kernel density estimator and kriging function from pickled MCMC data.

Isaac Cheng - March 2021
"""
import sys
from pathlib import Path
import numpy as np
import dill
import pandas as pd
from kriging import kriging
from scipy.stats import gaussian_kde
from scipy.spatial import Delaunay

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans
from universal_rotcurve import urc

# Values from WC21 A6
_R0_MODE = 8.174602364395952
_ZSUN_MODE = 5.398550615892994
_USUN_MODE = 10.878914326160878
_VSUN_MODE = 10.696801784160257
_WSUN_MODE = 8.087892505141708
_UPEC_MODE = 4.9071771802606285
_VPEC_MODE = -4.521832904300172
_ROLL_MODE = -0.010742182667190958
_A2_MODE = 0.9768982857793898
_A3_MODE = 1.626400628724733


def get_kde(pkl_file):
    # with open(pkl_file, "rb") as f1:
    #     file1 = dill.load(f1)
    #     print("loaded file1")
    #     trace = file1["trace"]
    #     print("loaded trace")
    #     free_Zsun = file1["free_Zsun"]
    #     free_roll = file1["free_roll"]
    #     free_Wpec = file1["free_Wpec"]
    #     individual_Upec = file1["individual_Upec"]
    #     individual_Vpec = file1["individual_Vpec"]

    # # Varnames order: [R0, Zsun, Usun, Vsun, Wsun, Upec, Vpec, Wpec, roll, a2, a3]
    # varnames = ["R0", "Usun", "Vsun", "Wsun", "a2", "a3"]
    # samples = [trace[varname] for varname in varnames]

    # if free_roll:
    #     varnames.insert(4, "roll")
    #     samples.insert(4, trace["roll"])
    # if free_Wpec:
    #     varnames.insert(4, "Wpec")
    #     samples.insert(4, trace["Wpec"])
    # varnames.insert(4, "Vpec")
    # if individual_Vpec:
    #     # Take median Vpec for all sources
    #     samples.insert(4, np.median(trace["Vpec"], axis=1))
    # else:
    #     samples.insert(4, trace["Vpec"])
    # varnames.insert(4, "Upec")
    # if individual_Upec:
    #     # Take median Upec for all sources
    #     samples.insert(4, np.median(trace["Upec"], axis=1))
    # else:
    #     samples.insert(4, trace["Upec"])
    # if free_Zsun:
    #     varnames.insert(1, "Zsun")
    #     samples.insert(1, trace["Zsun"])
    # samples = np.array(samples)  # shape: (# params, # total iterations)
    # print("variables in kde:", varnames)

    # # Create KDEs
    # kde_full = gaussian_kde(samples)
    # kde_R0 = gaussian_kde(trace["R0"])
    # kde_Zsun = gaussian_kde(trace["Zsun"])
    # kde_Usun = gaussian_kde(trace["Usun"])
    # kde_Vsun = gaussian_kde(trace["Vsun"])
    # kde_Wsun = gaussian_kde(trace["Wsun"])
    # kde_Upec = gaussian_kde(trace["Upec"])
    # kde_Vpec = gaussian_kde(trace["Vpec"])
    # kde_roll = gaussian_kde(trace["roll"])
    # kde_a2 = gaussian_kde(trace["a2"])
    # kde_a3 = gaussian_kde(trace["a3"])

    with open(pkl_file, "rb") as f1:
        file = dill.load(f1)
        kde_full = file["full"]
        kde_R0 = file["R0"]
        kde_Zsun = file["Zsun"]
        kde_Usun = file["Usun"]
        kde_Vsun = file["Vsun"]
        kde_Wsun = file["Wsun"]
        kde_Upec = file["Upec"]
        kde_Vpec = file["Vpec"]
        kde_roll = file["roll"]
        kde_a2 = file["a2"]
        kde_a3 = file["a3"]


    return (
        kde_full,
        kde_R0,
        kde_Zsun,
        kde_Usun,
        kde_Vsun,
        kde_Wsun,
        kde_Upec,
        kde_Vpec,
        kde_roll,
        kde_a2,
        kde_a3,
    )


if __name__ == "__main__":
    # ---- KDE
    infile = Path("/mnt/c/Users/ichen/OneDrive/Documents/Jobs/WaterlooWorks/2A Job Search/ACCEPTED__NRC_EXT-10708-JuniorResearcher/Work Documents/kd/kd/cw21_kde_krige.pkl")
    kdes = get_kde(infile)

    # ---- Kriging
    datafile = Path("/mnt/c/Users/ichen/OneDrive/Documents/Jobs/WaterlooWorks/2A Job Search/ACCEPTED__NRC_EXT-10708-JuniorResearcher/Work Documents/coop2021/pec_motions/csvfiles/alldata_HPDmode_NEW2.csv")
    pearsonrfile = Path("/mnt/c/Users/ichen/OneDrive/Documents/Jobs/WaterlooWorks/2A Job Search/ACCEPTED__NRC_EXT-10708-JuniorResearcher/Work Documents/coop2021/pec_motions/pearsonr_cov.pkl")
    data = pd.read_csv(datafile)
    with open(pearsonrfile, "rb") as f:
        file = dill.load(f)
        cov_Upec = file["cov_Upec"]
        cov_Vpec = file["cov_Vpec"]
    # Only choose sources that have R > 4 kpc
    # & are not outliers & have ~ Gaussian uncertainties
    Upec_halfhpd = data["Upec_halfhpd"].values
    Vpec_halfhpd = data["Vpec_halfhpd"].values
    Upec_err_high = data["Upec_hpdhigh"].values -  data["Upec_mode"].values
    Upec_err_low = data["Upec_mode"].values - data["Upec_hpdlow"].values
    Vpec_err_high = data["Vpec_hpdhigh"].values -  data["Vpec_mode"].values
    Vpec_err_low = data["Vpec_mode"].values - data["Vpec_hpdlow"].values
    is_good = (
        (data["is_tooclose"].values == 0)
        & (data["is_outlier"].values == 0)
        & (abs(Upec_err_high - Upec_err_low) < 0.33 * Upec_halfhpd)
        & (abs(Vpec_err_high - Vpec_err_low) < 0.33 * Vpec_halfhpd)
    )
    data = data[is_good]
    print("Num good:", len(data))
    cov_Upec = cov_Upec[:, is_good][is_good]
    cov_Vpec = cov_Vpec[:, is_good][is_good]
    # Get data
    Upec = data["Upec_mode"].values
    Vpec = data["Vpec_mode"].values
    Wpec = data["Wpec_mode"].values
    x = data["x_mode"].values
    y = data["y_mode"].values
    z = data["z_mode"].values
    #
    # Convert data to barycentric Cartesian coordinates
    #
    # Calculate circular rotation speed
    theta0 = urc(_R0_MODE, a2=_A2_MODE, a3=_A3_MODE, R0=_R0_MODE)
    v_circ_noVpec = urc(np.sqrt(x * x + y * y), a2=_A2_MODE, a3=_A3_MODE, R0=_R0_MODE)
    v_circ = v_circ_noVpec + Vpec
    # Rotate 90 deg CCW
    x, y = -y, x
    # Convert peculiar motions to galactocentric Cartesian
    az = np.arctan2(y, -x)
    cos_az = np.cos(az)
    sin_az = np.sin(az)
    vx_g = Upec * cos_az + v_circ * sin_az
    vy_g = -Upec * sin_az + v_circ * cos_az
    vz_g = Wpec
    # Galactocentric Cartesian to Barycentric Cartesian
    xb, yb, zb, vxb, vyb, vzb = trans.gcen_to_bary(
        x, y, z, Vxg=vx_g, Vyg=vy_g, Vzg=vz_g,
        R0=_R0_MODE, Zsun=_ZSUN_MODE, roll=_ROLL_MODE,
        Usun=_USUN_MODE, Vsun=_VSUN_MODE, Wsun=_WSUN_MODE, Theta0=theta0)
    # # Rotate 90 deg CW (Sun is on +y-axis)
    # xb, yb = yb, -xb
    # Rename Barycentric Cartesian coordinates
    x, y, z = xb, yb, zb  # NOTE: Sun is on -x-axis
    #
    coord_obs = np.vstack((x, y)).T
    # Initialize kriging object
    Upec_krige = kriging.Kriging(coord_obs, Upec - _UPEC_MODE, obs_data_cov=cov_Upec)
    Vpec_krige = kriging.Kriging(coord_obs, Vpec - _VPEC_MODE, obs_data_cov=cov_Vpec)
    # Fit semivariogram
    variogram_model = "gaussian"
    nbins = 10
    bin_number = False
    lag_cutoff = 0.5
    Upec_semivar = Upec_krige.fit(
        model=variogram_model,
        deg=1,
        nbins=nbins,
        bin_number=bin_number,
        lag_cutoff=lag_cutoff,
    )
    Vpec_semivar = Vpec_krige.fit(
        model=variogram_model,
        deg=1,
        nbins=nbins,
        bin_number=bin_number,
        lag_cutoff=lag_cutoff,
    )
    # Threshold values where Upec and Vpec are no longer reliable
    # (Based on standard deviation)
    # Upec_var_threshold = 225.0  # km^2/s^2, (15)^2
    # Vpec_var_threshold = 225.0  # km^2/s^2, (15)^2
    var_threshold = 250.0  # (km/s)^2 (sum of Upec & Vpec variances, not in quadrature)
    # Compute convex hull; x is first column, y is 2nd column, shape=(num_sources, 2)
    hull = Delaunay(coord_obs)
    # Save KDE & kriging function to pickle file
    filename = "cw21_kde_krige.pkl"
    outfile = Path(__file__).parent / filename
    with open(outfile, "wb") as f:
        dill.dump(
            {
                "full": kdes[0],
                "R0": kdes[1],
                "Zsun": kdes[2],
                "Usun": kdes[3],
                "Vsun": kdes[4],
                "Wsun": kdes[5],
                "Upec": kdes[6],
                "Vpec": kdes[7],
                "roll": kdes[8],
                "a2": kdes[9],
                "a3": kdes[10],
                "Upec_krige": Upec_krige,
                "Vpec_krige": Vpec_krige,
                # "Upec_var_threshold": Upec_var_threshold,
                # "Vpec_var_threshold": Vpec_var_threshold,
                "var_threshold": var_threshold,
                "hull": hull,
            },
            f,
        )
    print("Saved!")
