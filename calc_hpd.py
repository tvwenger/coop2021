"""
calc_hpd.py

Calculates the HPD of a value using a KDE.
Returns KDE, mode, lower HPD bound, upper HPD bound.
This is Dr. Trey V. Wenger's function!

Isaac Cheng - March 2021
"""
import numpy as np
from scipy.stats.kde import gaussian_kde
from pyqt_fit import kde as pyqt_kde
from pyqt_fit import kde_methods


def calc_hpd(samples, kdetype, alpha=0.683, pdf_bins=1000):
    """
    From Dr. Trey V. Wenger's kd_utils.py

    Fit a kernel density estimator (KDE) to the posterior given
    by a collection of samples. Return the mode (posterior peak)
    and the highest posterior density (HPD) determined by the minimum
    width Bayesian credible interval (BCI) containing a fraction of
    the posterior samples. The posterior should be well described by a
    single-modal distribution.

    Parameters:
      samples :: 1-D array of scalars
        The samples being fit with a KDE

      kdetype :: string
        Which KDE method to use
          'pyqt' uses pyqt_fit with boundary at 0
          'scipy' uses gaussian_kde with no boundary

      alpha :: scalar (optional)
        The fraction of samples included in the BCI.

      pdf_bins :: integer (optional)
        Number of bins used in calculating the PDF

    Returns: kde, mode, lower, upper
      kde :: scipy.gaussian_kde or pyqt_fit.1DKDE object
        The KDE calculated for this kinematic distance

      mode :: scalar
        The mode of the posterior

      lower :: scalar
        The lower bound of the BCI

      upper :: scalar
        The upper bound of the BCI
    """
    # check inputs
    if (alpha <= 0.0) or (alpha >= 1.0):
        raise ValueError("alpha should be between 0 and 1.")
    #
    # Fit KDE
    #
    nans = np.isnan(samples)
    if np.sum(~nans) < 2:
        # skip if fewer than two non-nans
        return (None, np.nan, np.nan, np.nan)
    try:
        if kdetype == "scipy":
            kde = gaussian_kde(samples[~nans])
        elif kdetype == "pyqt":
            kde = pyqt_kde.KDE1D(
                samples[~nans], lower=0, method=kde_methods.linear_combination
            )
        else:
            raise ValueError("Invalid KDE method: {0}".format(kdetype))
    except np.linalg.LinAlgError:
        # catch singular matricies (i.e. all values are the same)
        return (None, np.nan, np.nan, np.nan)
    #
    # Compute PDF
    #
    xdata = np.linspace(np.nanmin(samples), np.nanmax(samples), pdf_bins)
    pdf = kde(xdata)
    #
    # Get the location of the mode
    #
    mode = xdata[np.argmax(pdf)]
    if np.isnan(mode):
        return (None, np.nan, np.nan, np.nan)
    #
    # Reverse sort the PDF and xdata and find the BCI
    #
    sort_pdf = sorted(zip(xdata, pdf / np.sum(pdf)), key=lambda x: x[1], reverse=True)
    cum_prob = 0.0
    bci_xdata = np.empty(len(xdata), dtype=float) * np.nan
    for i, dat in enumerate(sort_pdf):
        cum_prob += dat[1]
        bci_xdata[i] = dat[0]
        if cum_prob >= alpha:
            break
    lower = np.nanmin(bci_xdata)
    upper = np.nanmax(bci_xdata)
    return kde, mode, lower, upper
