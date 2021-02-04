"""
Following emcee tutorial from https://emcee.readthedocs.io/en/stable/tutorials/line/
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import emcee    # uses emsenble sampler (not good for high dimensional problems)
import corner
from IPython.display import display, Math

np.random.seed(123)

# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534  # factor that increases the error without changing the error bar
                # (i.e. error bars are underestimated)

# Generate some synthetic data from the model.
N = 50
x = np.sort(10 * np.random.rand(N))
yerr = 0.1 + 0.5 * np.random.rand(N)
y = m_true * x + b_true
y += np.abs(f_true * y) * np.random.randn(N)
y += yerr * np.random.randn(N)

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
x0 = np.linspace(0, 10, 500)
plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(
    Path(__file__).parent / "emcee_try_data.jpg",
    format="jpg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# === NAIVE LEAST SQUARES ===
# Assumes errors are correct, Gaussian, and independent
A = np.vander(x, 2)
C = np.diag(yerr * yerr)
ATA = np.dot(A.T, A / (yerr ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / yerr ** 2))
print("Least-squares estimates:")
print("m = {0:.3f} ± {1:.3f}".format(w[0], np.sqrt(cov[0, 0])))
print("b = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))

# plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
# plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3, label="truth")
# plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="LS")
# plt.legend(fontsize=14)
# plt.xlim(0, 10)
# plt.xlabel("x")
# plt.ylabel("y")

# # LS with numpy
# p_coeffs = np.polyfit(x, y, 1, w=1/yerr)
# print(p_coeffs)
# plt.plot(x0, np.polyval(p_coeffs, x0), "--b", label="numpy LS")
# plt.legend(fontsize=14)
# plt.xlim(0, 10)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.savefig(
#     Path(__file__).parent / "emcee_try_numpy.jpg",
#     format="jpg",
#     dpi=300,
#     bbox_inches="tight",
# )
# plt.show()

# # LS with curve_fit
# def linear_model(x, m, b):
#     return m * x + b
# popt = curve_fit(linear_model, x, y, sigma=yerr, absolute_sigma=True)[0]
# print(popt)

# === MAXIMIZE LIKELIHOOD ===
# Create function that returns ln(likelihood)
def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([m_true, b_true, np.log(f_true)]) + 0.1 * np.random.randn(3)
# Maximize likelihood by minimizing negative ln(likelihood)
soln = minimize(nll, initial, args=(x, y, yerr))
m_ml, b_ml, log_f_ml = soln.x

print("Maximum likelihood estimates:")  # notice no uncertainty values!
print("m = {0:.3f}".format(m_ml))
print("b = {0:.3f}".format(b_ml))
print("f = {0:.3f}".format(np.exp(log_f_ml)))

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3, label="truth")
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="LS")
plt.plot(x0, np.dot(np.vander(x0, 2), [m_ml, b_ml]), ":k", label="ML")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(
    Path(__file__).parent / "emcee_try_LS_ML.jpg",  # least squares & maximum likelihood
    format="jpg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# === UNCERTAINTY ESTIMATION VIA MCMC ===
def log_prior(theta):
    m, b, log_f = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < log_f < 1.0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

# Start MCMC
pos = soln.x + 1e-4 * np.random.randn(32, 3)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
sampler.run_mcmc(pos, 5000, progress=True);

fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["m", "b", "log(f)"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3, linewidth=0.5)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
fig.savefig(
    Path(__file__).parent / "emcee_try_walkers.jpg",
    format="jpg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# Estimate integrated autocorrelation time
tau = sampler.get_autocorr_time()
print("integrated autocorelation time:", tau)

# Discard first 100 steps (no convergence yet), thin samples by ~0.5 autocorrelation time
# Then flatten chain to have flat list of samples
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

# === VIEW RESULTS ===
# Histogram of parameters
fig2 = corner.corner(
    flat_samples, labels=labels, truths=[m_true, b_true, np.log(f_true)]
)
fig2.savefig(
    Path(__file__).parent / "emcee_try_corner.jpg",
    format="jpg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# Final plot results
inds = np.random.randint(len(flat_samples), size=100)
for ind in inds:
    sample = flat_samples[ind]
    plt.plot(x0, np.dot(np.vander(x0, 2), sample[:2]), "C1", alpha=0.1)
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, m_true * x0 + b_true, "k", label="truth")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(
    Path(__file__).parent / "emcee_try_MCMC_final_result.jpg",
    format="jpg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# Quote numbers based on 16th, 50th, and 84th percentile in distributions
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))