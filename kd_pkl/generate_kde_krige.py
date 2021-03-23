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

# Want to add my own programs as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)

import mytransforms as trans


def krige_old(x, y):
    """
    Calculates kriging result of Upec & Vpec and their variances at 1 position.
    Requires kriging module from https://github.com/tvwenger/kriging and numpy as np
    N.B. Uses tvw's kriging v1.2. OUTDATED. DO NOT USE THIS!

    Inputs:
      x, y :: scalars (kpc)
        The x- and y- Cartesian positions of the kriging point

    Returns: Upec_interp, Upec_interp_var, Vpec_interp, Vpec_interp_var
      Upec_interp, Vpec_inter :: scalars (km/s)
        The radial (positive toward GC) and tangential (positive in direction
        of galactic rotation) peculiar motions of the source
      Upec_interp_var, Vpec_interp_var :: scalars (km/s)
        The variances of Upec_interp and Vpec_interp, respectively
    """
    import numpy as np
    from kriging import kriging

    coord_obs = np.array(
        [[ 0.52848393,  4.92364912], [ 0.52241911,  5.31311558],
        [ 0.68051412,  5.93654582], [ 0.55079817,  6.38225185],
        [ 0.94199302,  5.50117487], [ 1.64661405,  4.17774697],
        [ 2.39585762,  4.89761027], [ 3.89882719,  4.85023093],
        [ 2.62628549,  7.45643312], [ 3.10629653,  9.38844023],
        [ 2.46047618,  9.15263738], [ 1.45495123,  9.733154  ],
        [-0.29705464, 14.5337487 ], [-0.30817625, 10.17344442],
        [ 0.0160902 ,  5.25745539], [ 0.30693676,  5.20320964],
        [ 0.41084152,  4.73357774], [ 0.22671741,  6.6356576 ],
        [ 0.5293325 ,  5.48559642], [ 0.24907593,  6.95687882],
        [ 0.69527461,  4.88714766], [ 0.64658902,  5.33853923],
        [ 0.55210512,  5.76855315], [ 0.54721684,  5.79227906],
        [ 0.56412736,  5.7199621 ], [ 0.89793454,  4.50146031],
        [ 0.94402077,  4.35924606], [ 0.27718161,  7.09646939],
        [ 0.46277915,  6.40939919], [ 0.51980708,  6.24610297],
        [ 1.0230826 ,  4.74625309], [ 0.60674836,  6.26296169],
        [ 0.45292341,  6.75686832], [ 0.62904476,  6.2838269 ],
        [ 1.08916837,  4.9953893 ], [ 1.90672517,  3.69155832],
        [ 1.64869273,  4.33589552], [ 1.90850874,  3.76872863],
        [ 1.28196526,  5.25496555], [ 2.14547242,  4.19758714],
        [ 2.05877415,  4.37325063], [ 2.41121554,  3.8012442 ],
        [ 3.33713699,  2.561575  ], [ 1.75042024,  5.26499785],
        [ 6.82457239, -3.06827716], [ 2.8219139 ,  3.53661733],
        [ 2.86175861,  3.52616435], [ 2.67970209,  3.90038201],
        [ 1.49718082,  6.02656214], [ 1.33469372,  6.27695947],
        [ 1.26391131,  6.38948243], [ 1.39848677,  6.19896856],
        [ 5.17568375,  1.00320041], [ 1.14204618,  6.68918105],
        [ 2.55079468,  4.93044709], [ 2.17691588,  5.6127934 ],
        [ 5.26486239,  2.15805349], [ 5.83210908,  1.52553642],
        [ 5.24971815,  2.55926353], [ 7.60132788,  0.077252  ],
        [ 5.48823365,  2.70672916], [ 4.9524976 ,  3.31331178],
        [ 5.23319371,  3.09306856], [ 5.96860892,  2.46905381],
        [ 8.06647113,  1.07134481], [ 4.23924638,  4.494969  ],
        [ 4.60408848,  4.18524017], [ 3.8419812 ,  4.86392177],
        [ 6.70553623,  2.40655441], [ 3.14802139,  5.47831203],
        [ 4.10950693,  4.67010971], [ 3.49323205,  5.20833818],
        [ 4.78155059,  4.45900627], [ 3.50710839,  5.64356457],
        [ 3.1286739 ,  6.02066453], [ 3.19005247,  6.03730563],
        [ 2.85980007,  6.44765535], [ 1.61014778,  7.23207939],
        [ 1.86635629,  7.09436452], [ 3.57232044,  6.10492479],
        [ 7.19834393,  4.12126603], [ 2.3073509 ,  7.32055278],
        [ 6.02775692,  6.00884827], [ 4.44685253,  6.67723161],
        [ 3.42391737,  7.03721485], [ 1.52782331,  7.74428685],
        [ 3.40091165,  7.31831655], [ 3.71398458,  7.24032288],
        [ 1.33493391,  7.93960588], [ 1.58745336,  7.89790838],
        [ 1.59126226,  7.92362533], [ 1.43709463,  7.95017876],
        [ 1.48589009,  7.9659684 ], [ 1.28220032,  7.99820917],
        [ 0.67374485,  8.18381771], [ 1.62719369,  8.25721696],
        [ 4.83301031,  8.62945368], [ 7.44237189,  9.16493589],
        [ 3.39700223,  8.80361763], [ 0.84118967,  8.41320994],
        [ 0.7376802 ,  8.41104958], [ 0.86596795,  8.46576135],
        [ 4.18448831,  9.55773611], [ 2.37154381,  8.97141556],
        [ 3.06583519,  9.20557383], [ 0.76532   ,  8.45794934],
        [ 3.32817966,  9.47602694], [ 2.5446013 ,  9.37107672],
        [ 0.79333427,  8.66366742], [ 1.8292477 ,  9.32527305],
        [ 1.9785933 ,  9.46965303], [ 1.40595973,  9.53656408],
        [ 1.72217148,  9.88092749], [ 4.20855322, 12.43071024],
        [ 1.38975653, 12.03011602], [ 1.10598902, 13.41271682],
        [ 0.3050625 , 10.03612893], [ 0.19092567,  9.85240746],
        [ 0.21636571, 10.31173027], [ 0.05855199,  9.14296378],
        [-0.1029212 ,  9.76466306], [-0.32665947, 10.25637491],
        [-0.31845395,  9.65929693], [-0.36297425,  9.80518203],
        [-1.16990831, 12.14275346], [-0.18891791,  8.52221495],
        [-2.19160768, 11.74468722], [-0.47170796,  8.88870541],
        [-3.75932768, 13.02821721], [-3.49178123, 11.15613611],
        [-1.33307819,  9.19979887], [-2.56570634,  9.85921506],
        [-1.00227515,  8.77530763], [-4.64580523, 10.82960529],
        [-3.2685687 ,  5.87562547], [-0.71632416,  6.22561998],
        [-0.66164665,  4.86902446], [-0.19780827,  6.86644794],
        [-0.01793087,  5.51478809]]
    )

    Upec_good = np.array(
        [ 20.18355779,  -6.67225625,  -6.57607902,   4.2000377 ,  -2.93899109,
        25.89094442,  -4.52511784,   0.73370822,  -2.83111961,  32.68975895,
        17.89369676,   7.53793282,  11.32956242,  14.20989142,  17.41991785,
        -2.74776503,   3.02948033,  -2.26508969,   8.75538996,  -0.75857301,
        2.45156726,   1.89877773,  14.4757572 ,  16.20324128,  16.4691059 ,
        4.76215892,   7.60173252,  11.13272519,   2.24063214,   0.40765283,
        21.17304858,  20.40637146,   7.46848207,   5.27474329,  -7.22304519,
        15.40465985,  20.39350614,   9.12262145,  33.20050096,  12.16744682,
        -2.7392967 ,  11.83684225, -13.22733147,  24.61450473, -16.45726849,
        6.39225781,  -3.33890512,  14.4827658 ,  -6.69843273,  11.71074256,
        -4.16997489,   2.84992572,   3.11944887,   7.45434299,  25.72733066,
        6.24296021,  -6.27521382, -10.35638682,  22.53793003,  -4.34153262,
        9.48549083,  -8.67253258,  -1.00814941,   7.2475087 ,  -4.95556097,
        0.40724713,  27.70908822,   8.38032262,  14.86594333,  11.75454672,
        -8.94346259,  15.32852912,   5.71525921,  10.48830992,  25.1882342 ,
        17.69715857,  15.41196898,  14.60313684,  -0.51162659,  16.30296115,
        17.54757534,   6.5120495 ,  11.02541716,   6.32408232,  -4.5062177 ,
        7.62634163,  -0.05192267,  -4.12803192,   1.23924613,   8.49947151,
        -3.0180938 ,   6.85434609,  -0.62909728,  -1.11252452,  -4.79165286,
        -17.00901466,  -7.55166034,   9.67825334,  12.81364589,   7.75288574,
        4.32034847,  -2.07193877, -11.16170318,  12.29317425,  20.14426843,
        2.81189181,   1.06238629,  16.15820636,   9.56820964,  29.96751715,
        7.29434256,  20.53210071,   3.00680804,  14.63624601, -11.64345667,
        5.23695758,  10.95683554,   8.46563152,  -1.7432229 ,  15.00882178,
        -0.93246002,  -0.25977681,   2.72111659,  -0.06096162,   4.14341847,
        -0.82552185,  -5.00208561,  -6.33870864,  13.3222905 ,  13.4160818 ,
        1.62920323,  -7.9231827 ,  -0.61744261,  -7.63792626,  -4.91597903,
        -6.50673082,  13.98534604,   0.1208676 ,  21.87509731]
    )

    Vpec_good = np.array(
        [ -3.13660859, -23.58136981, -10.50298905,  -8.2866297 , -16.93772422,
        -13.12478866,   6.14447393,  -5.41153916, -14.76577414,   4.31874766,
        -23.84873124, -18.57726424,   0.75620205, -19.80152708,  -6.71101356,
        -13.7183526 ,  -2.67840572,  -9.44814569,  -7.27826375,   9.50340312,
        8.42952789,  13.0167603 ,  -4.60677107,  -6.15953849,  -6.25533856,
        -5.74655299,  -8.35824925,   3.07001128,  -3.2063974 ,   3.76732313,
        -8.20752294,   5.03050758,   3.25086721,  -1.93323971,  -1.53700485,
        -17.53128829,  -1.91349383, -16.29736502,   7.9972358 ,  -0.05440143,
        0.25297802,  -8.72016508, -15.3815636 ,   1.69090597,  -7.31086433,
        13.76046046,   2.18336992,   3.52057395,   3.87794904,   4.09677884,
        -8.16982944,  -2.59542262, -16.66841755,   3.32580867,  -6.81553129,
        14.10019688, -13.97270252,  -5.7622628 ,  -5.98895292, -12.72993593,
        1.39644101, -10.94167698,   0.73426267, -21.0304591 ,  13.0737745 ,
        4.40827141, -19.83149011,   4.70182425, -27.78389309,   8.17972464,
        -3.54478259,   1.49863691,  -9.33601155, -10.48748775, -12.78014775,
        -20.78541662,  -6.9809948 ,  -5.4451151 ,  -6.14142246,  -3.66502659,
        13.32184063,  -8.45119181, -15.46964431,   1.21442124,  -5.31701925,
        -9.11244992, -17.76813201,  -6.48112162, -12.25955831, -13.88645591,
        -11.47969038, -11.20728373, -10.68022389,  -0.70943807,  -8.06969269,
        -7.52816223,  -5.10814247,   6.91005019,  -2.91849337,  -2.62299488,
        -8.84381144,  -8.13796992, -13.34643462, -25.21988201, -12.51105694,
        -2.40726834,  -1.04537112,   7.21767773, -11.30673054,  -6.06192128,
        4.38573089, -11.8808445 , -13.07423799,   8.41106442, -16.64527498,
        -6.11731257, -11.91333153,  -5.67946895,  10.86633946, -15.75837206,
        1.94346346,  -4.53497842,  -4.67289639,   6.37175119,  -5.72857724,
        -3.79742577,  -4.27322082,   4.86927642,  -9.92200509,  -9.02854072,
        -3.30190403,  -0.09246264, -10.08428383,   0.70542043,  -5.06797262,
        1.77465458, -25.15735207,   1.38799339,   1.6087242 ]
    )

    if np.isscalar(x) and np.isscalar(y):
        coord_interp = np.array([[x, y]], float)
    else:
        coord_interp = np.vstack((x.flatten(), y.flatten())).T

    Upec_interp, Upec_interp_var = kriging.kriging(
        coord_obs,
        Upec_good,
        coord_interp,
        model="gaussian",
        deg=1,
        nbins=10,
        bin_number=True,
    )
    Vpec_interp, Vpec_interp_var = kriging.kriging(
        coord_obs,
        Vpec_good,
        coord_interp,
        model="gaussian",
        deg=1,
        nbins=10,
        bin_number=True,
    )

    if np.size(x) == 1 and np.size(y) == 1:
        Upec_interp = Upec_interp[0]
        Upec_interp_var = Upec_interp_var[0]
        Vpec_interp = Vpec_interp[0]
        Vpec_interp_var = Vpec_interp_var[0]

    return Upec_interp, Upec_interp_var, Vpec_interp, Vpec_interp_var

def get_kde(pkl_file):
    with open(pkl_file, "rb") as f1:
        file1 = dill.load(f1)
        trace = file1["trace"]
        free_Zsun = file1["free_Zsun"]
        free_roll = file1["free_roll"]
        free_Wpec = file1["free_Wpec"]
        individual_Upec = file1["individual_Upec"]
        individual_Vpec = file1["individual_Vpec"]

    # Varnames order: [R0, Zsun, Usun, Vsun, Wsun, Upec, Vpec, Wpec, roll, a2, a3]
    varnames = ["R0", "Usun", "Vsun", "Wsun", "a2", "a3"]
    samples = [trace[varname] for varname in varnames]

    if free_roll:
        varnames.insert(4, "roll")
        samples.insert(4, trace["roll"])
    if free_Wpec:
        varnames.insert(4, "Wpec")
        samples.insert(4, trace["Wpec"])
    varnames.insert(4, "Vpec")
    if individual_Vpec:
        # Take median Vpec for all sources
        samples.insert(4, np.median(trace["Vpec"], axis=1))
    else:
        samples.insert(4, trace["Vpec"])
    varnames.insert(4, "Upec")
    if individual_Upec:
        # Take median Upec for all sources
        samples.insert(4, np.median(trace["Upec"], axis=1))
    else:
        samples.insert(4, trace["Upec"])
    if free_Zsun:
        varnames.insert(1, "Zsun")
        samples.insert(1, trace["Zsun"])
    samples = np.array(samples)  # shape: (# params, # total iterations)
    print("variables in kde:", varnames)

    # Create KDEs
    kde_full = gaussian_kde(samples)
    kde_R0 = gaussian_kde(trace["R0"])
    kde_Zsun = gaussian_kde(trace["Zsun"])
    kde_Usun = gaussian_kde(trace["Usun"])
    kde_Vsun = gaussian_kde(trace["Vsun"])
    kde_Wsun = gaussian_kde(trace["Wsun"])
    kde_Upec = gaussian_kde(trace["Upec"])
    kde_Vpec = gaussian_kde(trace["Vpec"])
    kde_roll = gaussian_kde(trace["roll"])
    kde_a2 = gaussian_kde(trace["a2"])
    kde_a3 = gaussian_kde(trace["a3"])

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
    # prior_set = input("prior_set of file (A1, A5, B, C, D): ")
    # num_samples = int(input("Number of distance samples per source in file (int): "))
    # num_rounds = int(
    #     input("Number of times MCMC has run. i.e., this_round of file (int): ")
    # )

    # # Binary file to read
    # infile = (
    #     Path(__file__).parent.parent
    #     / Path("bayesian_mcmc_rot_curve")
    #     / f"mcmc_outfile_{prior_set}_{num_samples}dist_{num_rounds}.pkl"
    # )
    infile = (
        Path(__file__).parent.parent
        / Path("bayesian_mcmc_rot_curve")
        / "mcmc_outfile_A5_102dist_6.pkl"
    )

    kdes = get_kde(infile)

    # ---- Kriging
    datafile = Path("/home/chengi/Documents/coop2021/pec_motions/csvfiles/alldata_HPDmode.csv")
    pearsonrfile = Path("/home/chengi/Documents/coop2021/pec_motions/pearsonr_cov.pkl")
    data = pd.read_csv(datafile)
    with open(pearsonrfile, "rb") as f:
        file = dill.load(f)
        cov_Upec = file["cov_Upec"]
        cov_Vpec = file["cov_Vpec"]
    # Only choose sources that have R > 4 kpc & are not outliers
    is_good = (data["is_tooclose"].values == 0) & (data["is_outlier"].values == 0)
    data = data[is_good]
    cov_Upec = cov_Upec[:, is_good][is_good]
    cov_Vpec = cov_Vpec[:, is_good][is_good]
    # Get data
    Upec = data["Upec_mode"].values
    Vpec = data["Vpec_mode"].values
    x = data["x_mode"].values
    y = data["y_mode"].values
    coord_obs = np.vstack((x, y)).T
    # Initialize kriging object
    Upec_krige = kriging.Kriging(coord_obs, Upec, obs_data_cov=cov_Upec)
    Vpec_krige = kriging.Kriging(coord_obs, Vpec, obs_data_cov=cov_Vpec)
    # Fit semivariogram
    variogram_model = "gaussian"
    nbins = 10
    bin_number = False
    lag_cutoff = 0.55
    Upec_semivar, Upec_corner = Upec_krige.fit(
        model=variogram_model,
        deg=1,
        nbins=nbins,
        bin_number=bin_number,
        lag_cutoff=lag_cutoff,
        nsims=1000,
    )
    Vpec_semivar, Vpec_corner = Vpec_krige.fit(
        model=variogram_model,
        deg=1,
        nbins=nbins,
        bin_number=bin_number,
        lag_cutoff=lag_cutoff,
        nsims=1000,
    )
    # Threshold values where Upec and Vpec are no longer reliable
    # (Based on standard deviation)
    Upec_var_threshold = 225.0  # km^2/s^2, (15)^2
    Vpec_var_threshold = 225.0  # km^2/s^2, (15)^2

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
                "Upec_var_threshold": Upec_var_threshold,
                "Vpec_var_threshold": Vpec_var_threshold,
            },
            f,
        )
    print("Saved!")
