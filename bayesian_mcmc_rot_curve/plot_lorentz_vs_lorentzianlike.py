import numpy as np
import matplotlib.pyplot as plt

def lorentzian(x, mean, sigma):
    residual = (x - mean)/sigma
    return np.log(1 - np.exp(-0.5 * residual * residual)) - 2 * np.log(abs(residual))
    # return np.log((1-np.exp(-0.5 * residual * residual))/ (residual * residual))

def cauchy(x, mean ,hwhm):
    return np.log(1 / (np.pi * hwhm) * (hwhm * hwhm / ((x - mean)**2 + hwhm * hwhm)))

x = np.linspace(0,17,101)
mean = 8
sigma = 1
hwhm = np.sqrt(2 * np.log(2)) * sigma

plt.plot(x, lorentzian(x, mean, sigma), label="lorentzian-like")
plt.plot(x,cauchy(x,mean,hwhm), label="cauchy")
plt.legend()
plt.show()


# def plx_to_peak_dist(mean_plx, e_plx):
#     """
#     Computes peak of distance distribution given the
#     parallax & the uncertainty in the parallax (assuming the parallax is Gaussian)

#     TODO: finish docstring
#     """

#     mean_dist = 1 / mean_plx
#     sigma_sq = e_plx * e_plx
#     return (np.sqrt(8 * sigma_sq * mean_dist * mean_dist + 1) - 1) / (
#         4 * sigma_sq * mean_dist
#     )


# def dist_PDF(dist, mean_dist, sigma_plx):
#     # plx = 1 / dist
#     term1 = 1 / (dist**2 * sigma_plx * np.sqrt(2*np.pi))
#     term2 = np.exp(-0.5 * (dist - mean_dist)**2 / (dist**2 * mean_dist**2 * sigma_plx**2))
#     return term1 * term2


# mean_plx = 0.5
# std_plx = 0.2

# plx = np.random.normal(loc=mean_plx, scale=std_plx, size=50000)
# dist = 1 / plx
# peak = plx_to_peak_dist(mean_plx, std_plx)
# plt.hist(dist[(dist<3/mean_plx) & (dist>0)], bins=50, density=True)
# x = np.linspace(0.3 / mean_plx, 3 / mean_plx, 101)
# y = dist_PDF(x, 1 / mean_plx, std_plx)
# plt.plot(x, y, "r-")
# plt.axvline(peak, color="k")
# plt.show()

# plx_orig = np.array([20.,21.,22.])
# plx = np.array([[1.,2.,3.],
#                 [4.,5.,6.],
#                 [7.,8.,9.],
#                 [10.,11.,12.],
#                 [13.,14.,15.]], float)

# _PLX_BOUND = 8.  # minimum parallax allowed
# print(f"# plx <= {_PLX_BOUND} before correction:", np.count_nonzero(plx<=_PLX_BOUND))
# # Find indices where plx <= _PLX_BOUND
# for idx1, idx2 in zip(np.where(plx<=_PLX_BOUND)[0], np.where(plx<=_PLX_BOUND)[1]):
#     # Replace parallax <= _PLX_BOUND with original (aka database) value
#     plx[idx1, idx2] = plx_orig[idx2]
# print(plx)
# print(f"# plx <= {_PLX_BOUND} after correction:", np.count_nonzero(plx<=_PLX_BOUND))
# print(np.count_nonzero(plx==_PLX_BOUND))
# print("min plx after correction:", np.min(plx))
