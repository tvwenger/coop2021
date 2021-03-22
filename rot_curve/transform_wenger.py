"""
transform.py

Utilities for transforming positions and kinematics between frames:
- J2000 Equatorial
- Galactic
- Barycentric Cartesian
- Galactocentric Cartesian
- Galactocentric cylindrical

Trey Wenger - February 2020
"""

import numpy as np

# IAU-defined solar motion parameters (km/s)
_USTD = 10.27
_VSTD = 15.32
_WSTD = 7.74

# Reid+2019 Solar Galactocentric radius and Solar height above plane
_R0 = 8.15 # kpc
_ZSUN = 5.5 # pc

# J2000 Coordinate conversion constants from Astropy
# RA of North Galactic Pole (deg)
_RA_NGP = 192.8594812065348
# Declination of North Galactic Pole (deg)
_DEC_NGP = 27.12825118085622
_COS_DEC_NGP = np.cos(np.deg2rad(_DEC_NGP))
_SIN_DEC_NGP = np.sin(np.deg2rad(_DEC_NGP))
# Galactic longitude of North Celestial Pole (deg)
_L_NCP = 122.9319185680026

# useful constants
_KPCTOKM = 3.0856775814671916e+16 # km/kpc
_MASYRTORADS = 1.5362818500441604e-16 # rad/s / (mas/yr)
_KMKPCSTOMASYR = 0.21094952657135144 # (mas/yr) / (km/kpc/s)

def eq_to_gal(ra, dec, mux, muy):
    """
    Convert J2000 equatorial positions and proper motions to the
    Galactic frame.

    Inputs:
      ra :: scalar or array of scalars (deg)
        Right ascension
      dec :: scalar or array of scalars (deg)
        Declination
      mux :: scalar or array of scalars (mas/yr)
        RA proper motion with cos(Declination) correction
      muy :: scalar or array of scalars (mas/yr)
        Declination proper motion

    Returns: glong, glat, mul, mub
      glong :: scalar or array of scalars (deg)
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
    cos_dec = np.cos(np.deg2rad(dec))
    sin_dec = np.sin(np.deg2rad(dec))
    cos_ra_off = np.cos(np.deg2rad(ra - _RA_NGP))
    sin_ra_off = np.sin(np.deg2rad(ra - _RA_NGP))
    #
    # Binney & Merrifield (1998)
    #
    sin_glat = cos_dec*_COS_DEC_NGP*cos_ra_off + sin_dec*_SIN_DEC_NGP
    glat = np.rad2deg(np.arcsin(sin_glat))
    tan_glong_num = cos_dec*sin_ra_off
    tan_glong_den = sin_dec*_COS_DEC_NGP - cos_dec*_SIN_DEC_NGP*cos_ra_off
    glong = _L_NCP - np.rad2deg(np.arctan2(tan_glong_num, tan_glong_den))
    # get range 0 to 360 degrees
    glong = glong % 360.
    #
    # Rotation matrix from Poleski (2018)
    #
    matc1 = _SIN_DEC_NGP*cos_dec - _COS_DEC_NGP*sin_dec*cos_ra_off
    matc2 = _COS_DEC_NGP*sin_ra_off
    cos_b = np.sqrt(matc1**2. + matc2**2.)
    mul = (matc1*mux + matc2*muy)/cos_b
    mub = (-matc2*mux + matc1*muy)/cos_b
    return glong, glat, mul, mub

def gal_to_eq(glong, glat, mul, mub):
    """
    Convert Galactic proper motions to the J2000
    equatorial frame.

    Inputs:
      glong :: scalar or array of scalars (deg)
        Galactic longitude
      glat :: scalar or array of scalars (deg)
        Galactic latitude
      mul :: scalar or array of scalars (mas/yr)
        Galactic longitude proper motion with cos(Latitude) correction
      mub :: scalar or array of scalars (mas/yr)
        Galactic latitude proper motion

    Returns: mux, muy
      mux :: scalar or array of scalars (mas/yr)
        RA proper motion with cos(Declination) correction
      muy :: scalar or array of scalars (mas/yr)
        Declination proper motion
    """
    #
    # useful constants
    #
    cos_glat = np.cos(np.deg2rad(glat))
    sin_glat = np.sin(np.deg2rad(glat))
    cos_glong_off = np.cos(np.deg2rad(glong - _L_NCP))
    sin_glong_off = np.sin(np.deg2rad(glong - _L_NCP))
    #
    # Rotation matrix from Poleski (2018)
    #
    matc1 = _SIN_DEC_NGP*cos_glat - _COS_DEC_NGP*sin_glat*cos_glong_off
    matc2 = _COS_DEC_NGP*sin_glong_off
    cos_dec = np.sqrt(matc1**2. + matc2**2.)
    mux = (matc1*mul + matc2*mub)/cos_dec
    muy = (-matc2*mul + matc1*mub)/cos_dec
    return mux, muy

def gal_to_barycar(glong, glat, dist):
    """
    Convert Galactic positions to the barycentric Cartesian
    frame.

    Inputs:
      glong :: scalar or array of scalars (deg)
        Galactic longitude
      glat :: scalar or array of scalars (deg)
        Galactic latitude
      dist :: scalar or array of scalars (kpc)
        Distance

    Returns: X, Y, Z
      X, Y, Z :: scalars or arrays of scalars (kpc)
        Cartesian positions

    We use the convention that the Galactic Center is in the +X
    direction, the Sun orbits in the +Y direction, and the North
    Galactic Pole is in the +Z direction.
    """
    #
    # useful constants
    #
    cos_glong = np.cos(np.deg2rad(glong))
    sin_glong = np.sin(np.deg2rad(glong))
    cos_glat = np.cos(np.deg2rad(glat))
    sin_glat = np.sin(np.deg2rad(glat))
    #
    # Heliocentric Cartesian
    #
    X = dist*cos_glat*cos_glong # kpc
    Y = dist*cos_glat*sin_glong # kpc
    Z = dist*sin_glat # kpc
    return X, Y, Z

def barycar_to_gal(dist, glat, X, Y, Z, vX, vY, vZ):
    """
    Convert barycentric Cartesian velocities to the Galactic frame
    proper motions and radial velocity.

    Inputs:
      dist :: scalar or array of scalars (kpc)
        Distance
      glat :: scalar or array of scalars (deg)
        Galactic latitude
      X, Y, Z :: scalar or array of scalars (kpc)
        Cartesian positions
      vX, vY, vZ :: scalars or arrays of scalars (km/s)
        Cartesian velocities

    Returns: mul, mub, vbary
      mul :: scalar or array of scalars (mas/yr)
        Galactic longitude proper motion with cos(Latitude) correction
      mub :: scalar or array of scalars (mas/yr)
        Galactic latitude proper motion
      vbary :: scalar or array of scalars (km/s)
        Barycentric velocity

    We use the convention that the Galactic Center is in the +X
    direction, the Sun orbits in the +Y direction, and the North
    Galactic Pole is in the +Z direction.
    """
    #
    # Barycentric Cartesian to Spherical (Galactic)
    #
    vbary = (X*vX + Y*vY + Z*vZ)/dist # km/s
    mub = (dist*vZ - Z*vbary)/np.sqrt(dist**4. - Z**2.*dist**2.)
    mub = mub*_KMKPCSTOMASYR
    mul = (X*vY - Y*vX)/(X**2. + Y**2.)*_KMKPCSTOMASYR
    mul = mul*np.cos(np.deg2rad(glat))
    return mul, mub, vbary

def barycar_to_gcencar(Xh, Yh, Zh, roll=0., R0=_R0, Zsun=_ZSUN):
    """
    Convert barycentric Cartesian positions to
    Galactocentric Cartesian positions and velocities.

    Inputs:
      Xh, Yh, Zh :: scalars or arrays of scalars (kpc)
        Barycentric Cartesian positions
      roll :: scalar or array of scalars (deg)
        The roll angle of the Galactic plane relative to b=0
      R0 :: scalar or array of scalars (kpc)
        Solar Galactocentric radius
      Zsun :: scalar or array of scalars (pc)
        Sun's height above the Galactic plane

    Returns: Xg, Yg, Zg
      Xg, Yg, Zg :: scalars or arrays of scalars (kpc)
        Galactocentric Cartesian positions

    We use the convention that the Sun is at (X, Y, Z) = (-R0, 0, Zsun)
    in the Galactocentric Cartesian frame. The Sun orbits in the +Y
    direction, and the North Galactic Pole is in the +Z direction.
    Azimuth is defined as 0 in the direction of the Sun and increasing
    in the +Y direction.
    """
    #
    # Useful constants
    #
    sin_tilt = Zsun/1000./R0
    cos_tilt = np.cos(np.arcsin(sin_tilt))
    sin_roll = np.sin(np.deg2rad(roll))
    cos_roll = np.cos(np.deg2rad(roll))
    #
    # Roll CCW about the X axis so that the X-Z plane is parallel
    # with the Galactocentric frame
    #
    Xh1 = Xh
    Yh1 = Yh*cos_roll - Zh*sin_roll
    Zh1 = Yh*sin_roll + Zh*cos_roll
    #
    # Translate along new X axis to Galactic center
    #
    Xh1 = Xh1 - R0
    #
    # Tilt CCW about the Y axis to account for Sun's height
    #
    Xg = Xh1*cos_tilt + Zh1*sin_tilt
    Yg = Yh1
    Zg = -Xh1*sin_tilt + Zh1*cos_tilt
    return Xg, Yg, Zg

def gcencar_to_barycar(vXg, vYg, vZg, roll=0., R0=_R0, Usun=_USTD,
                       Vsun=_VSTD, Wsun=_WSTD, Zsun=_ZSUN):
    """
    Convert velocities to barycentric velocities.

    Inputs:
      vXg, vYg, vZg :: scalars or arrays of scalars (km/s)
        Galactocentric Cartesian velocities
      roll :: scalar or array of scalars (deg)
        The roll angle of the Galactic plane relative to b=0
      R0 :: scalar or array of scalars (kpc)
        Solar Galactocentric radius
      Usun, Vsun, Wsun :: scalars or arrays of scalars (km/s)
        The motion of the barycenter relative to the Galactocentric
        frame. U increases toward the Galactic Center, V increases in
        direction of Solar orbit, and W increases towards North
        Galactic Pole.
      Zsun :: scalar or array of scalars (pc)
        Sun's height above the Galactic plane

    Returns: vXh, vYh, vZh
      vXh, vYh, vZh :: scalars or arrays of scalars (km/s)
        Barycentric Cartesian velocities

    We use the convention that the Sun is at (X, Y, Z) = (-R0, 0, Zsun)
    in the Galactocentric Cartesian frame. The Sun orbits in the +Y
    direction, and the North Galactic Pole is in the +Z direction.
    Azimuth is defined as 0 in the direction of the Sun and increasing
    in the +Y direction.
    """
    #
    # Useful constants
    #
    sin_tilt = Zsun/1000./R0
    cos_tilt = np.cos(np.arcsin(sin_tilt))
    sin_roll = np.sin(np.deg2rad(roll))
    cos_roll = np.cos(np.deg2rad(roll))
    #
    # Add solar motion
    #
    vXg = vXg - Usun
    vYg = vYg - Vsun
    vZg = vZg - Wsun
    #
    # Tilt CW about the Y axis to account for Sun's height
    #
    vXg1 = vXg*cos_tilt - vZg*sin_tilt
    vYg1 = vYg
    vZg1 = vXg*sin_tilt + vZg*cos_tilt
    #
    # Roll CW about the X axis so that the X-Z plane is parallel
    # with the Galactocentric frame
    #
    vXh = vXg1
    vYh = vYg1*cos_roll + vZg1*sin_roll
    vZh = -vYg1*sin_roll + vZg1*cos_roll
    return vXh, vYh, vZh

def gcencar_to_gcencyl(X, Y, Z):
    """
    Convert Galactocentric Cartesian positions to
    Galactocentric cylindrical positions and velocities.

    Inputs:
      X, Y, Z :: scalars or arrays of scalars (kpc)
        Galactocentric Cartesian positions

    Returns: R, Az, Z
      R :: scalar or array of scalars (kpc)
        Galactocentric radius
      Az :: scalar or array of scalars (deg)
        Galactocentric azimuth
      Z :: scalar or array of scalars (kpc)
        Height above the Galactic plane

    We use the convention that Galactocentric azimuth is 0 in
    the direction of the Sun and increases in the direction of
    Galactic rotation (clockwise viewed from North Galactic Pole).
    """
    #
    # Convert to cylindrical
    #
    R = np.sqrt(X**2. + Y**2.) # kpc
    Az = np.rad2deg(np.arctan2(Y, -X))
    # get range 0 to 360 degrees
    Az = Az % 360.
    return R, Az, Z

def gcencyl_to_gcencar(Az, vR, vAz, vZ):
    """
    Convert Galactocentric cylindrical velocities to
    Galactocentric Cartesian velocities.

    Inputs:
      Az :: scalar or array of scalars (deg)
        Galactocentric azimuth
      vR :: scalar or array of scalars (km/s)
        Radial velocity
      vAz :: scalar or array of scalars (km/s)
        Azimuthal angular velocity * Galactocentric radius
      vZ :: scalar or array of scalars (km/s)
        Velocity in the +Z direction

    Returns: vX, vY, vZ
      vX, vY vZ :: scalars or arrays of scalars (km/s)
        Galactocentric Cartesian velocities

    We use the convention that Galactocentric azimuth is 0 in
    the direction of the Sun and increases in the direction of
    Galactic rotation (clockwise viewed from North Galactic Pole).
    """
    #
    # Useful constants
    #
    cos_Az = np.cos(np.deg2rad(Az))
    sin_Az = np.sin(np.deg2rad(Az))
    #
    # Convert cylindrical to Cartesian
    #
    vX = -vR*cos_Az + sin_Az*vAz # km/s
    vY = vR*sin_Az + cos_Az*vAz # km/s
    return vX, vY, vZ
