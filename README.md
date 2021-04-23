# coop2021

Isaac Cheng's 2A work term (winter 2021) with Dr. Trey V. Wenger.

If you have any questions, feel free to email me or message me on Slack!

For generating any plots, I add the following line to my `matplotlibrc` file:

```markdown
text.latex.preamble : \usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{mathtools}\usepackage{newtxtext, newtxmath}
```

This (i.e., the last `\usepackage` command) changes the font to a Times New Roman
look-alike, which better matches the final AASTeX font (I think they use Spivak's _MathTime
Professional 2_, but that font package costs money...).

---

## General Utilities

- `calc_hpd.py` calculates the highest posterior density region given some data. Straight
  from TVW's `kd_utils.py`.
- `mytransforms.py` contains many many functions to transform between various frames. Also
  contains some functions that improve ease-of-use like `gcen_cyl_to_pm_and_vlsr()`, which
  is used in the Bayesian MCMC program to convert from the Galactocentric cylindrical
  frame to the equatorial frame's proper motions and LSR velocity.
- `universal_rotcurve.py` is just a re-production of Reid et al.'s (2019) Fortran code.
  The only important function in this is `urc()`.

---

## Important Folders

### `bayesian_mcmc_rot_curve`

- `MCMC_w_dist_uncer.py` is the main Bayesian MCMC program.
- `mcmc_cleanup.py` is the outlier rejection program (called by the main MCMC program)
- `mcmc_bic.py` calculates the Bayesian Information Criterion (BIC) of each MCMC run. This
  is also called by the main MCMC program.
- `MCMC_w_dist_uncer_plots.py` is the plotting program for the Bayesian MCMC run. The
  input questions will automatically determine the pickle file to plot. Thus, you must
  ensure that whatever file you want to plot is in the same folder as this Python script.
- `mcmc_results` contains every run's MCMC results (i.e., the info printed in the terminal
  after a run is complete) for all the prior sets. Only the results at the top of the file
  are correct (i.e., before the "OLD DATA THAT DID NOT USE PEAK OF DISTANCE PDF FOR
  INITIAL R < 4 KPC rejection" break).
- `run_mcmc.sh` is the bash script used to run `MCMC_w_dist_uncer.py`.
- `plot_lorentz_vs_lorentzianlike.py` is just a small program to visualize the difference
  between the Cauchy (aka Lorentzian) distribution and the Sivia & Skilling 2006
  distribution. It also plots the parallax-to-distance PDF, which is included in the
  Appendix of the paper.
- (not very important now) `test_mcmc.py` generates some noisy data from a set of known values
  to test the `MCMC_w_dist_uncer.py` program.

The rest of the files are irrelevant.

### `galaxy_map`

(Not particularly important, but may be useful.)

- `galaxymap_hpd.py` plots the coordinates of the maser sources in a face-on
  Galactocentric Cartesian frame. The area of the points is proportional to the
  uncertainty in the parallax-derived distance. This uses the peak of the distance PDF.
  - Requires `alldata_HPDmode_NEW2.csv` (see `pec_motions` folder)
- `galaxymap.py` simply plots the distances of the maser sources in a face-on
  Galactocentric Cartesian frame. This was one of my warm-up exercises and did not use the
  peak of the distance PDF.

### `kd_pkl`

- `generate_kde_krige.py` generates the KDE and the kriging object for the kinematic
  distance program. Also generates convex hull just in case we ever need it (probably not).
  - Requires either an MCMC pickle file (i.e., the A6 results) or an existing pickle file
    that contains a KDE (i.e., in the BearBearCodes fork of
    tvwenger/kd—`wc21_kde_krige.pkl`)
- `verify_kde_krige.py` just checks that the KDE and kriging objects in the pickle file
  produced by `generate_kde_krige.py` is correct. Note that the plots are in
  Galactocentric Cartesian coordinates, not barycentric Cartesian. Depending on the
  coordinate system, you may have to rotate your head a bit. I didn't spend the time
  transforming the coordinates only because this is just a quick check to make sure the
  file is saved properly.
- `compare_wc_reid.py` compares the KDEs used in Wenger & Cheng (our work-in-progress
  paper) to the KDE containing the Reid et al. (2019) parameters. This was just another
  sanity check.

### `kd_plots`

<span style="color:Tomato"> VERY IMPORTANT: also read the `pec_motions` section in this
README. </span>

The scripts in this folder all require TVW's [kinematic distance
program](https://github.com/tvwenger/kd). The BearBearCodes fork is located
[here](https://github.com/BearBearCodes/kd), which includes the non-axisymmetric Galactic
rotation model (implemented in `wc21_rotcurve.py`). The BearBearCodes fork has 2 main
branches: (1) [`master`](https://github.com/BearBearCodes/kd/tree/master), which uses
straight kriging results to produce the non-axisymmetric GRM, and (2)
[`multiply-kriging`](https://github.com/BearBearCodes/kd/tree/multiply-kriging), which
currently multiplies the kriging magnitudes by a factor of 3.

- `get_kd.py` is the program that runs the kinematic distance program. You will need to
  modify the database path to wherever your database is located. This will also save the
  KD results in a pickle file + a csv file (useful to quickly check sources).
- `kd_pdf_plots.py` simply plots the probability distributions of various MC kinematic
  distance quantities in a PDF file.
- `vlsr_dist_plot.py` plots the LSR velocity vs. distance graph for 1 source. Here, you
  can specify if you want to use MC sampling, how many MC samples to use (I typically use
  100 samples for the plots), and what rotation curve to use (and if you want to use the
  kriging estimates in the `wc21_rotcurve`).
- `kd_plx_diff.py` is an OLD file. Do not use! See `kd_plots.ipynb` instead.
- `face_on_view_vlsr.py` plots the LSR velocity prediction differences between an
  axisymmetric GRM and a non-aisymmetric GRM. It also has a method to plot the standard
  deviation (i.e., the error in the mean) map. Finally, there is a method to plot the
  total error in kriging estimates (i.e., by resampling Upec/Vpec at every point).
- `kd_plots.ipynb` is a Jupyter notebook that conatins a lot of functions to assign and
  compare kinematic distances. There are usage instructions in the notebook, but the most
  important sections of `kd_plots.ipynb` are probably the first 2 cells, the `Using
  vlsr_tol` section (which will generate a pickle file), the `general plotting function
  using pickled data` section, and finally the `Plots after removing outliers` section.
  - This notebook automatically generates 5 different plots from a pickle file: 2 face-on
    maps, 2 histograms, and 1 CDF. See the notebook for more details.

### `pec_motions`

<span style="color:Tomato"> VERY IMPORTANT: Should only use
`./csvfiles/alldata_HPDmode_NEW2.csv` as data (e.g., for kriging). Other .csv files are
incomplete. </span>

- `./csvfiles/100dist_meanUpecVpec_cauchyOutlierRejection_peakEverything.csv` contains the database
  data for all 202 maser parallax sources from Reid et al. (2019) in addition to if they
  were initially rejected from MCMC fitting (i.e., R < 4 kpc, denoted by the `is_tooclose`
  header) and if the source is an MCMC fit A6 outlier (denoted by the `is_outlier`
  header).
- `./csvfiles/alldata_HPDmode_NEW2.csv` contains the database data PLUS a whole bunch of
  associated quantities derived using 10000 MC samples and a KDE fit to the MCMC A6
  results. I Monte Carlo sample the proper motions and LSR velocities from a normal
  distribution centred on their database value and with a standard deviation equal to
  their uncertainties. I also generate 10000 distance samples using our derived
  parallax-to-distance PDF. Any values with the suffix `_mode`, `_hpdlow`, `hpdhigh`, or
  `_halfhpd` denote quantities that were calculated using the `calc_hpd()` function (taken
  from TVW's `kd_utils.py`). The mode is the peak of distribution (aka the most likely
  value), the hpdlow and hpdhigh values are the lower and upper HPD limits respectively
  (e.g., the upper error would be hpdhigh - mode), and halfhpd is simply half of the HPD
  range (i.e., (hpdhigh-hpdlow)/2) It is important to note that the `x`, `y`, `z`, `vx`,
  `vy`, `vz`, `vx_res`, `vy_res`, and `vz_res` values are Galactocentric Cartesian
  coordinates/velocities/residual velocities (from the rotation curve) _with the Sun on
  the +y-axis_ (i.e., they have already been rotated 90 degrees CW)!
  - This file was generated using `generate_mc_pec_motions.py`
- `generate_mc_pec_motions.py` generates a .csv file with many useful quantities like the
  `./csvfiles/alldata_HPDmode_NEW2.csv` file. It also has a method to print various
  statistics about the quantities like the mean, median, standard deviation, as well as
  HPD modes and limits of the peculiar motions in the .csv file. I've saved a copy of the
  results in the Overleaf project under `tex/pec_motion_stats.txt`.
- `calc_cov.py` calculates the Pearson product-moment correlation coefficient and
  covariance values for various quantities and stores them in a pickle file titled
  `pearsonr_cov.pkl`.
- `krige.py` is the main kriging function using v2.3 of TVW's
  [`kriging`](https://github.com/tvwenger/kriging) package. This uses data from the
  `_NEW2.csv` file and the covariances from the `pearsonr_cov.pkl` file to perform
  universal kriging of Upec and Vpec. Uses Galactocentric Cartesian coordinates.
- `krige_mc.py` generates MC kriging maps to plot a map of the kriging standard deviation
  in the mean. The arrays are saved in a pickle file. Uses Galactocentric Cartesian
  coordinates.
- `krige_diff.py` kriges the _difference_ between the individual peculiar motions and the
  MCMC A6 average peculiar motion values in Barycentric Cartesian coordinates. This is
  used in the kinematic distances package (i.e., if the user chooses to use kriging with
  `wc21_rotcurve`). Also see `coop2021/kd_pkl/generate_kde_krige.py`.
- `krige_vlsr.py` kriges the LSR velocity difference in Barycentric Cartesian coordinates.
  Not used in the kinematic distances program.
- `plot_pec_mot_csv.py` plots the peculiar motions of the masers in a face-on view of the
  Milky Way with the Sun on the +y-axis. Requires `_NEW2.csv`.
- (meh importance) `pec_motions_hist.py` contain various functions to plot histograms of the peculiar
  motions (formerly used to identify outliers for kriging before realizing Gaussian
  uncertainties were important to kriging).
- (meh importance) `plot_vrad_vtan.py` has nothing to do with plotting (it used to...
  Sorry!). This program generates the
  `./csvfiles/100dist_meanUpecVpec_cauchyOutlierRejection_peakEverything.csv`file, though
  you can ignore all the Upec, Vpec, and Wpec info saved. The main thing this does is
  identify which sources were never used in the MCMC fits (i.e., those with R < 4 kpc) and
  which sources were MCMC outliers by comparing the database data to the MCMC pickle file
  data. Hence this program requires you to modify the path so it knows where the database
  is and also requires you to have an MCMC pickle file (e.g., the MCMC A6 pickle file).

#### Unimportant files in `pec_motions`

- all the files with a `fit_pec_motions_` prefix. These used PyKrige or a radial basis
  function instead of TVW's kriging package.
- `plot_pec_mot_csv_mcmcoutliers.py` is just an old version of `plot_pec_mot_csv.py`.
- `plot_pec_mot_no_colour.py` is `plot_pec_mot_csv.py` except the colours are themed to
  match Isaac's HAA presentation because I'm extra.
- `krige_xy.py` is an attempt to krige the Cartesian velocities vx and vy. Never used.


## `rot_curve`

- `calc_mode_params.py` calculates the mode and HPD values of the MCMC parameters from
  their pickled trace file. This then saves the HPD results in a .csv file just in case.
  These data should be in the paper already.
- `my_rotcurve.py` plots our new rotation curve with data and errorbars calculated using
  `calc_hpd()`.
- (not very important but cool) `param_effects.py` is a small script to visualize the
  effects of changing a2 and a3 in the Persic et al. (1996) Universal Rotation Curve.

#### Unimportant files in `rot_curve`

- Everything else not mentioned above. The rest of the files were warm-up exercises for me.

---

## Unimportant Folders

### `glong_vlsr_plot`

One of my warm-up exercises. Plots a longitude vs. LSR velocity diagram.

### `reid_mcmc`

Contains code to reproduce Reid et al. (2019) results. These were also warm-up exercises.
