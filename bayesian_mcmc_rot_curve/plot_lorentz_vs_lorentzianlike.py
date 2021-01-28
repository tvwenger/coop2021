import numpy as np
import matplotlib.pyplot as plt

# def lorentzian(x, mean, sigma):
#     residual = (x-mean)/sigma
#     return np.log((1-np.exp(-0.5 * residual * residual))/ (residual * residual))

# def cauchy(x, mean ,hwhm):
#     return np.log(1/(np.pi * hwhm) * (hwhm * hwhm / ((x - mean)**2 + hwhm * hwhm)))

# x = np.linspace(0,17,101)
# mean = 8
# sigma = 1
# hwhm = np.sqrt(2 * np.log(2)) * sigma

# plt.plot(x, lorentzian(x, mean, sigma), label="lorentzian-like")
# plt.plot(x,cauchy(x,mean,hwhm), label="cauchy")
# plt.legend()
# plt.show()


def plx_to_peak_dist(plx, e_plx):
    """
    Computes peak of distance distribution given the
    parallax & the uncertainty in the parallax (assuming the parallax is Gaussian)

    TODO: finish docstring
    """

    mean_dist = 1 / plx
    sigma_sq = e_plx * e_plx
    return (np.sqrt(8 * sigma_sq * mean_dist * mean_dist + 1) - 1) / (
        4 * sigma_sq * mean_dist
    )


def dist_PDF(dist, mean_dist, sigma_plx):
    # plx = 1 / dist
    term1 = 1 / (dist**2 * sigma_plx * np.sqrt(2*np.pi))
    term2 = np.exp(-0.5 * (dist - mean_dist)**2 / (dist**2 * mean_dist**2 * sigma_plx**2))
    return term1 * term2


plx = np.random.normal(loc=1, scale=0.25, size=1000)
dist = 1 / plx
peak = plx_to_peak_dist(1, 0.25)
plt.hist(dist[dist<2], bins=50, density=True)
x = np.linspace(0.5,2,101)
y = dist_PDF(x, 1, 0.25)
plt.plot(x, y, "r-")
plt.axvline(peak, color="k")
plt.show()
