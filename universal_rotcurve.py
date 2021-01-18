import numpy as np

# Universal rotation curve parameters (Persic et al. 1996)
_A_TWO = 0.96  # (Reid et al. 2019)
_A_THREE = 1.62  # (Reid et al. 2019)

# Sun's distance from galactic centre
_RSUN = 8.15  # kpc (Reid et al. 2019)


def urc(R, a2=_A_TWO, a3=_A_THREE, R0=_RSUN):
    """
    Universal rotation curve from Persic et al. 1996
    
    Inputs:
      R : Array of scalars (kpc)
        Galactocentric radius of object
        (i.e. perpendicular distance from z-axis in cylindrical coordinates)
      a2 : Scalar (unitless)
        Defined as R_opt/R_0 (ratio of optical radius to Galactocentric radius of the Sun)
      a3 : Scalar (unitless)
        Defined as 1.5*(L/L*)^0.2
      R0 : Scalar (kpc)
        Galactocentric radius of the Sun perp. to z-axis (i.e. in cylindrical coordinates)
    
    Returns:
      Theta : Array of scalars (km/s)
        Circular rotation speed of objects at radius R
        (i.e. tangential velocity in cylindrical coordinates)
    """

    lam = (a3 / 1.5) * (a3 / 1.5) * (a3 / 1.5) * (a3 / 1.5) * (a3 / 1.5)
    Ropt = a2 * R0
    rho = R / Ropt

    v1 = (200 * lam ** 0.41) / np.sqrt(
        0.8
        + 0.49 * np.log10(lam)
        + (0.75 * np.exp(-0.4 * lam) / (0.47 + 2.25 * lam ** 0.4))
    )

    v2 = (
        (0.72 + 0.44 * np.log10(lam))
        * (1.97 * (rho) ** 1.22)
        / (rho * rho + 0.61) ** 1.43
    )

    v3 = 1.6 * np.exp(-0.4 * lam) * rho * rho / (rho * rho + 2.25 * lam ** 0.4)

    return v1 * np.sqrt(v2 + v3)  # km/s; circular rotation speed at radius R


def urc_odr(a_vals, R):
    """
    Universal rotation curve from Persic et al. 1996.
    This version has R after the parameter arguments for scipy.odr.
    
    Inputs:
      R : Array of scalars (kpc)
        Galactocentric radius of object
        (i.e. perpendicular distance from z-axis in cylindrical coordinates)
      a2 : Scalar (unitless)
        Defined as R_opt/R_0 (ratio of optical radius to Galactocentric radius of the Sun)
      a3 : Scalar (unitless)
        Defined as 1.5*(L/L*)^0.2
      R0 : Scalar (kpc)
        Galactocentric radius of the Sun perp. to z-axis (i.e. in cylindrical coordinates)
    
    Returns:
      Theta : Array of scalars (km/s)
        Circular rotation speed of objects at radius R
        (i.e. tangential velocity in cylindrical coordinates)
    """
    a2, a3 = a_vals
    
    lam = (a3 / 1.5) * (a3 / 1.5) * (a3 / 1.5) * (a3 / 1.5) * (a3 / 1.5)
    Ropt = a2 * _RSUN
    rho = R / Ropt

    v1 = (200 * lam ** 0.41) / np.sqrt(
        0.8
        + 0.49 * np.log10(lam)
        + (0.75 * np.exp(-0.4 * lam) / (0.47 + 2.25 * lam ** 0.4))
    )

    v2 = (
        (0.72 + 0.44 * np.log10(lam))
        * (1.97 * (rho) ** 1.22)
        / (rho * rho + 0.61) ** 1.43
    )

    v3 = 1.6 * np.exp(-0.4 * lam) * rho * rho / (rho * rho + 2.25 * lam ** 0.4)

    return v1 * np.sqrt(v2 + v3)  # km/s; circular rotation speed at radius R