import theano.tensor as tt  # can take fast derivatives
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import corner

# # Trying theano.tensor
# # pymc3 has fast convergence bc of gradient info from theano.tensor
# a = tt.dscalar("a")
# b = tt.dscalar("b")
# f = a * a + tt.sqrt(b)  # theano implementation of sqrt (knows derivative)
# f.eval({a: 5.0, b: 2.5})  # argument is dictionary
# # First time it runs, theano compiles into C code, will be slow. Next time will be very fast
# df_da = tt.grad(f, a)  # partial derivative of f wrt a
# df_da.eval({a: 5.0, b: 2.0})

# RECREATE EMCEE TUTORIAL USING PYMC3
# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534  # factor that increases the error without changing the error bar
                # (i.e. error bars are underestimated)
# Make N random data points
N = 50
x = np.sort(10 * np.random.rand(N))
yerr = 0.1 + 0.5 * np.random.rand(N)
y = m_true * x + b_true
y += np.abs(f_true * y) * np.random.randn(N)
y += yerr * np.random.randn(N)

# Using PyMC3
model = pm.Model()
with model:
# with pm.Model() as model:
    # with statement auto associates all pm objects with model. Necessary
    # Define priors
    m = pm.Uniform("m", lower=-5.0, upper=0.5)
    b = pm.Uniform("b", lower=0.0, upper=10.0)
    log_f = pm.Uniform("log_f", lower=-10.0, upper=1.0)

    # Predicted values (using x-data)
    y_pred = m * x + b

    # Likelihood function (assume Gaussian)
    sigma = tt.sqrt(yerr * yerr + tt.exp(log_f) * tt.exp(log_f) * y_pred * y_pred)
    # N.B. "observed" kwarg necessary for likelihood function below
    likelihood = pm.Normal("likelihood", mu=y_pred, sigma=sigma, observed=y)

# with model:
    trace = pm.sample(500, cores=2, chains=2)  # walker
    # Explanation:
    #   Tuning is the stuff we throw away. After tuning, we do 500 iterations as specified
    # other params:
    #   init='advi' is algorithm (variational inference) (default init is "auto"),
    #   tune=<NUM> is number of tuning steps

# See results
print(pm.summary(trace))
# Explanation of terms:
#   mcse = uncertainty due to algorithm itself,
#          usually much less than std so not too important.
#          Can increase iteration # to decrease.
#          Rule of thumb: MCSE < 5% of STD
#   ess = effective sample size (small means you need to run chain for longer)
#   N.B. want r_hat to be as close to 1 as possible (r_har >= 1 always)

iterations = len(trace)
print(iterations)
chains = len(trace.chains)
print(chains)
samples = []
# print(trace['m'].shape)
samples.append(trace['m'])
samples.append(trace['b'])
samples.append(trace['log_f'])
samples = np.array(samples)
# print(np.shape(samples))

samples2 = [s.reshape((chains, iterations)) for s in samples]
# print(samples2[0])
# print(samples2[0].shape)
fig, axes = plt.subplots(3)
for ax, sample in zip(axes, samples2):
    for chain in sample:
        ax.plot(chain, "k-", alpha=0.5, linewidth=0.5)
plt.show()

fig2 = corner.corner(samples.T)
plt.show()
    