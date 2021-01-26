import numpy as np
import matplotlib.pyplot as plt

def lorentzian(x, mean, sigma):
    residual = (x-mean)/sigma
    return np.log((1-np.exp(-0.5 * residual * residual))/ (residual * residual))

def cauchy(x, mean ,hwhm):
    return np.log(1/(np.pi * hwhm) * (hwhm * hwhm / ((x - mean)**2 + hwhm * hwhm)))

x = np.linspace(0,17,101)
mean = 8
sigma = 1
hwhm = np.sqrt(2 * np.log(2)) * sigma

plt.plot(x, lorentzian(x, mean, sigma), label="lorentzian-like")
plt.plot(x,cauchy(x,mean,hwhm), label="cauchy")
plt.legend()
plt.show()
