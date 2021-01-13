"""
mytransforms.py

Utilities for transforming positions and velocities between frames:
- J2000 equatorial
- local standard of rest
- (barycentric) galactic
- barycentric Cartesian
- galactocentric Cartesian
- galactocentric cylindrical

Isaac Cheng - January 2021
"""

import numpy as np

# Useful constants
_DEG_TO_RAD = 0.017453292519943295  # pi/180
_RAD_TO_DEG = 57.29577951308232  # 180/pi (Don't forget to % 360 after)
_AU_PER_YR_TO_KM_PER_S = 4.740470463533348  # from astropy (uses tropical year)
_KPC_TO_KM = 3.085677581e16

# Roll angle between galactic midplane and galactocentric frame
_ROLL = 0  # deg (Anderson et al. 2019)

# Sun's position in the Milky Way
_RSUN = 8.15  # kpc; Sun's distance from galactic centre (Reid et al. 2019)
_ZSUN = 5.5  # pc; Sun's height above galactic midplane (Reid et al. 2019)

# Sun's motion relative to the LSR (km/s)
_USUN = 10.6  # km/s; i.e. vel_x (Reid et al. 2019)
_VSUN = 10.7  # km/s; i.e. vel_y (Reid et al. 2019)
_WSUN = 7.6  # km/s; i.e. vel_z (Reid et al. 2019)

# Circular rotation speed at Sun's position in Milky Way
_THETA_0 = 236  # km/s (Reid et al. 2019)

# IAU definition of the local standard of rest (km/s)
_USTD = 10.27  # km/s
_VSTD = 15.32  # km/s
_WSTD = 7.74  # km/s

# J2000 Coordinate conversion constants from astropy
# Right ascension of North Galactic Pole (deg)
_RA_NGP = 192.8594812065348
# Declination of North Galactic Pole (deg)
_DEC_NGP = 27.12825118085622
_COS_DEC_NGP = np.cos(_DEC_NGP * _DEG_TO_RAD)
_SIN_DEC_NGP = np.sin(_DEC_NGP * _DEG_TO_RAD)
# Galactic longitude of North Celestial Pole (deg)
_L_NCP = 122.9319185680026


def eq_to_gal(ra, dec, mux, muy):
    """
    Convert J2000 equatorial positions and proper motions to the
    Galactic frame.
    
    Function taken (almost) verbatim from Dr. Trey Wenger ((c) February 2020)

    Inputs:
      ra :: scalar or array of scalars (deg)
        Right ascension
      dec :: scalar or array of scalars (deg)
        Declination
      mux :: scalar or array of scalars (mas/yr)
        RA proper motion with cos(Declination) correction
      muy :: scalar or array of scalars (mas/yr)
        Declination proper motion

    Returns: glon, glat, mul, mub
      glon :: scalar or array of scalars (deg)
        Galactic longitude
      glat :: scalar or array of scalars (deg)
        Galactic latitude
      mul :: scalar or array of scalars (mas/yr)
        Galactic longitude proper motion with cos(Latitude) correction
      mub :: scalar or array of scalars (mas/yr)
        Galactic latitude proper motion
    """
    #
    # Useful constants
    #
    cos_dec = np.cos(dec * _DEG_TO_RAD)
    sin_dec = np.sin(dec * _DEG_TO_RAD)
    cos_ra_off = np.cos((ra - _RA_NGP) * _DEG_TO_RAD)
    sin_ra_off = np.sin((ra - _RA_NGP) * _DEG_TO_RAD)
    #
    # Binney & Merrifield (1998)
    #
    sin_glat = cos_dec * _COS_DEC_NGP * cos_ra_off + sin_dec * _SIN_DEC_NGP
    glat = np.arcsin(sin_glat) * _RAD_TO_DEG
    tan_glon_num = cos_dec * sin_ra_off
    tan_glon_den = sin_dec * _COS_DEC_NGP - cos_dec * _SIN_DEC_NGP * cos_ra_off
    glon = _L_NCP - np.arctan2(tan_glon_num, tan_glon_den) * _RAD_TO_DEG
    # get range 0 to 360 degrees
    glon = glon % 360.0
    #
    # Rotation matrix from Poleski (2018)
    #
    matc1 = _SIN_DEC_NGP * cos_dec - _COS_DEC_NGP * sin_dec * cos_ra_off
    matc2 = _COS_DEC_NGP * sin_ra_off
    cos_b = np.sqrt(matc1 ** 2.0 + matc2 ** 2.0)
    mul = (matc1 * mux + matc2 * muy) / cos_b
    mub = (-matc2 * mux + matc1 * muy) / cos_b
    return glon, glat, mul, mub


def gal_to_bar(glon, glat, dist):
    """
    Convert Galactic coordinates to a barycentric (heliocentric)
    Cartesian frame

    Inputs:
      glon : Array of scalars (deg)
        Galactic longitude
      glat : Array of scalars (deg)
        Galactic latitude
      dist : Array of scalars (kpc)
        Distance

    Returns: Xb, Yb, Zb
      Xb, Yb, Zb : Arrays of scalars (kpc)
        Barycentric Cartesian coordinates
    """

    Xb = dist * np.cos(glat * _DEG_TO_RAD) * np.cos(glon * _DEG_TO_RAD)
    Yb = dist * np.cos(glat * _DEG_TO_RAD) * np.sin(glon * _DEG_TO_RAD)
    Zb = dist * np.sin(glat * _DEG_TO_RAD)

    return Xb, Yb, Zb


def bar_to_gcen(Xb, Yb, Zb, R0=_RSUN, Zsun=_ZSUN, roll=_ROLL):
    """
    Convert barycentric Cartesian coordinates to the Galactocentric
    Cartesian frame

    Inputs:
      Xb, Yb, Zb : Arrays of scalars (kpc)
        Barycentric Cartesian coordinates
      R0 : scalar (kpc)
        Galactocentric radius of the Sun
      Zsun : scalar (pc)
        Height of the Sun above the galactic midplane
      roll : scalar (deg)
        Angle between galactic plane and b=0

    Returns: Xg, Yg, Zg
      Zg, Yg, Zg : Arrays of scalars (kpc)
        Galactocentric Cartesian coordinates
    """

    # Tilt of b=0 relative to galactic plane
    tilt = np.arcsin(0.001 * Zsun / R0)
    #
    # Roll CCW about the barycentric X-axis so that the Y-Z plane
    # is aligned with the Y-Z plane of the galactocentric frame
    #
    roll_rad = roll * _DEG_TO_RAD
    Xb1 = np.copy(Xb)  # OR: Xb1 = Xb
    Yb1 = np.cos(roll_rad) * Yb - np.sin(roll_rad) * Zb
    Zb1 = np.sin(roll_rad) * Yb + np.cos(roll_rad) * Zb
    #
    # Translate to the galactic center
    #
    Xb1 -= R0  # must use np.copy() above
    # OR: Xb1 = Xb1 - R0
    #
    # Tilt to correct for Sun's height above midplane
    #
    Xg = np.cos(tilt) * Xb1 + np.sin(tilt) * Zb1
    Yg = Yb1
    Zg = -np.sin(tilt) * Xb1 + np.cos(tilt) * Zb1

    return Xg, Yg, Zg


def gal_to_bar_vel(glon, glat, dist, vbary, gmul, gmub):
    """
    Convert Galactic velocities to a barycentric (heliocentric)
    Cartesian frame

    Inputs:
      glon : Array of scalars (deg)
        Galactic longitude
      glat : Array of scalars (deg)
        Galactic latitude
      dist : Array of scalars (kpc)
        Distance
      vbary : Array of scalars (km/s)
        Radial velocity relative to barycentre (NOT vlsr)
      gmul : Array of scalars (mas/yr)
        Galactic longitudinal velocity
      gmub : Array of scalars (mas/yr)
        Galactic latitudinal velocity

    Returns: Vxb, Vyb, Vzb
      Ub, Vb, Wb : Array of scalars (km/s)
        Barycentric Cartesian velocities (i.e. vel_x, vel_y, vel_z)
    """

    # NOTE: (Ub, Vb, Wb) "omits" Sun's LSR velocity (implicitly included)
    # (This is included in galactocentric transform frunction)

    # Adopt method used by Jo Bovy (2019). Inverse of eqns (62) & (64)
    # https://github.com/jobovy/stellarkinematics/blob/master/stellarkinematics.pdf
    l = glon * _DEG_TO_RAD
    b = glat * _DEG_TO_RAD
    vl = dist * gmul * _AU_PER_YR_TO_KM_PER_S  # recall gmul has cos(b) correction
    vb = dist * gmub * _AU_PER_YR_TO_KM_PER_S

    Ub = vbary * np.cos(l) * np.cos(b) - vl * np.sin(l) - vb * np.sin(b) * np.cos(l)
    Vb = vbary * np.sin(l) * np.cos(b) + vl * np.cos(l) - vb * np.sin(b) * np.sin(l)
    Wb = vbary * np.sin(b) + vb * np.cos(b)

    return Ub, Vb, Wb


def bar_to_gcen_vel(Ub, Vb, Wb, R0=_RSUN, Zsun=_ZSUN, roll=_ROLL):
    """
    Convert barycentric Cartesian velocities to the Galactocentric
    Cartesian frame

    Inputs:
      Ub, Vb, Wb : Arrays of scalars (km/s)
        Barycentric Cartesian velocities (i.e. vel_x, vel_y, vel_z)
      R0 : scalar (kpc)
        Galactocentric radius of the Sun
      Zsun : scalar (pc)
        Height of the Sun above the galactic midplane
      roll : scalar (deg)
        Angle between galactic plane and b=0

    Returns: vel_xg, vel_yg, vel_zg
      vel_xg, vel_yg, vel_zg : Arrays of scalars (kpc)
        Galactocentric Cartesian velocities
    """

    # Tilt of b=0 relative to galactic plane
    tilt = np.arcsin(0.001 * Zsun / R0)
    #
    # Roll CCW about the barycentric X-axis so that the Y-Z plane
    # is aligned with the Y-Z plane of the galactocentric frame
    #
    roll_rad = roll * _DEG_TO_RAD
    Ub1 = np.copy(Ub)
    Vb1 = np.cos(roll_rad) * Vb - np.sin(roll_rad) * Wb
    Wb1 = np.sin(roll_rad) * Vb + np.cos(roll_rad) * Wb
    #
    # Tilt to correct for Sun's height above midplane
    #
    vel_xg = np.cos(tilt) * Ub1 + np.sin(tilt) * Wb1 + _USUN
    vel_yg = Vb1 + _VSUN + _THETA_0
    vel_zg = -np.sin(tilt) * Ub1 + np.cos(tilt) * Wb1 + _WSUN

    return vel_xg, vel_yg, vel_zg


def gcen_cart_to_gcen_cyl(x_kpc, y_kpc, z_kpc, vx, vy, vz):
    """
    Convert galactocentric Cartesian velocities to the
    galactocentric cylindrical velocities
    
                +z +y
                 | /
                 |/
    Sun -x ------+------ +x
                /|
               / |

    Inputs:
      x_kpc, y_kpc, z_kpc : Arrays of scalars (kpc)
        Galactocentric Cartesian positions
      vx, vy, vz : Arrays of scalars (km/s)
        Galactocentric Cartesian velocities

    Returns: perp_distance, azimuth, height, v_radial, v_theta, v_vertical
      perp_distance : Array of scalars (kpc)
        Radial distance perpendicular to z-axis
      azimuth : Array of scalars (deg)
        Azimuthal angle; positive CW from +y-axis like a clock (left-hand convention!)
      height : Array of scalars (kpc)
        Height above xy-plane (i.e. z_kpc)
      v_radial : Array of scalars (km/s)
        Radial velocity; positive away from z-axis
      v_tangent : Array of scalars (km/s)
        Tangential velocity; positive CW (left-hand convention!)
      v_vertical : Array of scalars (km/s)
        Velocity perp. to xy-plane; positive if pointing above xy-plane (i.e. vz)
    """

    y = np.copy(y_kpc) * _KPC_TO_KM  # km
    x = np.copy(x_kpc) * _KPC_TO_KM  # km

    perp_distance = np.sqrt(x_kpc * x_kpc + y_kpc * y_kpc)  # kpc
    perp_distance_km = np.sqrt(x * x + y * y)  # km
    # azimuth = (np.arctan2(-y, x) * _RAD_TO_DEG + 90) % 360  # deg in [0,360). Method 1
    azimuth = (90 - np.arctan2(y, x) * _RAD_TO_DEG) % 360  # deg in [0,360). Method 2
    #
    # **Check if any object is on z-axis (i.e. object's x_kpc & y_kpc both zero)**
    #
    arr = np.array([x_kpc, y_kpc])  # array with x_kpc in 0th row, y_kpc in 1st row
    if np.any(np.all(arr == 0, axis=0)):  # at least 1 object is on z_axis
        # Ensure vx & vy are arrays
        vx = np.atleast_1d(vx)
        vy = np.atleast_1d(vy)
        # Initialize arrays to store values
        v_radial = np.zeros(len(vx))
        v_tangent = np.zeros(len(vx))
        for i in range(len(vx)):
            if x[i] == 0 and y[i] == 0:  # this object is on z-axis
                # **all velocity in xy-plane is radial velocity**
                v_radial[i] = np.sqrt(vx[i] * vx[i] + vy[i] * vy[i])  # km/s
            else:  # this object is not on z-axis
                v_radial[i] = (x[i] * vx[i] + y[i] * vy[i]) / perp_distance_km[i]  # km/s
                v_tangent[i] = (x[i] * vy[i] - y[i] * vx[i]) / perp_distance_km[i]  # km/s
    else:  # no object is on z-axis (no division by zero)
        v_radial = (x * vx + y * vy) / perp_distance_km  # km/s
        v_tangent = (x * vy - y * vx) / perp_distance_km  # km/s

    return perp_distance, azimuth, z_kpc, v_radial, v_tangent, vz


def vlsr_to_vbary(vlsr, glon, glat):
    """
    Converts LSR (radial) velocity to radial velocity in barycentric frame

    Inputs:
      vlsr : Array of scalars (km/s)
        Radial velocity relative to local standard of rest
      glon, glat : Array of scalars (deg)
        Galactic longitude and latitude

    Returns: vbary
      vbary : Array of scalars (km/s)
        Radial velocity relative to barycentre of Solar System (NOT vlsr)
    """

    vbary = (
        vlsr
        - _USTD * np.cos(glon * _DEG_TO_RAD) * np.cos(glat * _DEG_TO_RAD)
        - _VSTD * np.sin(glon * _DEG_TO_RAD) * np.cos(glat * _DEG_TO_RAD)
        - _WSTD * np.sin(glat * _DEG_TO_RAD)
    )
    return vbary


def vbary_to_vlsr(vbary, glon, glat):
    """
    Converts LSR (radial) velocity to radial velocity in barycentric frame

    Inputs:
      vbary : Array of scalars (km/s)
        Radial velocity relative to barycentre of Solar System (NOT vlsr)
      glon, glat : Array of scalars (deg)
        Galactic longitude and latitude

    Returns: vbary
      vlsr : Array of scalars (km/s)
        Radial velocity relative to local standard of rest
    """

    vlsr = (
        vbary
        + _USTD * np.cos(glon * _DEG_TO_RAD) * np.cos(glat * _DEG_TO_RAD)
        + _VSTD * np.sin(glon * _DEG_TO_RAD) * np.cos(glat * _DEG_TO_RAD)
        + _WSTD * np.sin(glat * _DEG_TO_RAD)
    )

    return vlsr
