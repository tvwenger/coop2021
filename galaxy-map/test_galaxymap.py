import unittest

import numpy as np
from astropy.coordinates import Galactic, Galactocentric
import astropy.units as u

# Module to test (galaxymap.py)
import galaxymap as gm

# Allowed differences for passing tests
_POSITION = 0.001  # 1 pc


class TestGalaxyMap(unittest.TestCase):
    def test_gal_to_bar_single(self):
        """
        Test single galactic to barycentric Cartesian coordinate conversion.
        """

        glon = 34.0  # deg
        glat = -15.0  # deg
        dist = 1.0  # kpc

        X, Y, Z = gm.galactic_to_barycentric(glon, glat, dist)
        gal = Galactic(l=glon * u.deg, b=glat * u.deg, distance=dist * u.kpc)

        self.assertAlmostEqual(X, gal.cartesian.x.to("kpc").value, delta=_POSITION)
        self.assertAlmostEqual(Y, gal.cartesian.y.to("kpc").value, delta=_POSITION)
        self.assertAlmostEqual(Z, gal.cartesian.z.to("kpc").value, delta=_POSITION)

    def test_gal_to_bar_multi(self):
        """
        Test multiple galactic to barycentric Cartesian coordinate conversions.
        """

        glons = np.array([34.2, 0, 0.0, 51, 265, 5.883, 360, 360.0])
        glats = np.array([3.4, 0, 0.0, -12, -60.52, 5.0, 90.0, -90])
        dists = 1 / np.array([0.21, 1.0, 0.1, 3, 25, 541, 0.2, 0.234])

        Xs, Ys, Zs = gm.galactic_to_barycentric(glons, glats, dists)
        gals = Galactic(l=glons * u.deg, b=glats * u.deg, distance=dists * u.kpc)

        for X, Y, Z, gal in zip(Xs, Ys, Zs, gals):
            self.assertAlmostEqual(X, gal.cartesian.x.to("kpc").value, delta=_POSITION)
            self.assertAlmostEqual(Y, gal.cartesian.y.to("kpc").value, delta=_POSITION)
            self.assertAlmostEqual(Z, gal.cartesian.z.to("kpc").value, delta=_POSITION)

    def test_bar_to_gcen_single(self):
        """
        Test single barycentric Cartesian to galactocentric
        Cartesian coordinate conversion.
        """

        Xb = 4.0  # kpc
        Yb = -10.5  # kpc
        Zb = 1.0  # kpc

        Xg, Yg, Zg = gm.barycentric_to_galactocentric(Xb, Yb, Zb)
        gcen = Galactic(
            Xb * u.kpc, Yb * u.kpc, Zb * u.kpc, representation_type="cartesian"
        ).transform_to(
            Galactocentric(
                galcen_distance=8.15 * u.kpc, z_sun=5.5 * u.pc, roll=0 * u.deg
            )
        )

        self.assertAlmostEqual(Xg, gcen.cartesian.x.to("kpc").value, delta=_POSITION)
        self.assertAlmostEqual(Yg, gcen.cartesian.y.to("kpc").value, delta=_POSITION)
        self.assertAlmostEqual(Zg, gcen.cartesian.z.to("kpc").value, delta=_POSITION)

    def test_bar_to_gcen_multi(self):
        """
        Test multiple barycentric Cartesian to
        galactocentric Cartesian coordinate conversions.
        """

        Xbs = np.array([12, 14.0, -2.54, -0, 0.0, -18, 3, -1])
        Ybs = np.array([-9.0, 4, 10.4, 0.0, 5.84, -8.201, 2, -2])
        Zbs = np.array([9.0, 11, -0.3, 0, -2.414, -2.01, 1, -3])

        Xgs, Ygs, Zgs = gm.barycentric_to_galactocentric(Xbs, Ybs, Zbs)
        gcens = Galactic(
            Xbs * u.kpc, Ybs * u.kpc, Zbs * u.kpc, representation_type="cartesian"
        ).transform_to(
            Galactocentric(
                galcen_distance=8.15 * u.kpc, z_sun=5.5 * u.pc, roll=0 * u.deg
            )
        )

        for Xg, Yg, Zg, gcen in zip(Xgs, Ygs, Zgs, gcens):
            self.assertAlmostEqual(
                Xg, gcen.cartesian.x.to("kpc").value, delta=_POSITION
            )
            self.assertAlmostEqual(
                Yg, gcen.cartesian.y.to("kpc").value, delta=_POSITION
            )
            self.assertAlmostEqual(
                Zg, gcen.cartesian.z.to("kpc").value, delta=_POSITION
            )


if __name__ == "__main__":
    unittest.main()
