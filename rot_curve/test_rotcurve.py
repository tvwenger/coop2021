import unittest

import numpy as np
import astropy.coordinates as acoord
from astropy.coordinates import GalacticLSR, Galactocentric, Galactic
import astropy.units as u
from astropy.coordinates import CartesianDifferential as cd

# Module to test (rot.py)
import rotcurve as rc

# Allowed differences for passing tests
_VELOCITY = 0.001  # 1 m/s

# Constants
_LSR_VEL = cd(10.6, 10.7, 7.6, unit="km/s")  # Reid et al. 2019
_GAL_V_SUN = cd(10.6, 246.7, 7.6, unit="km/s")  # Reid et al. 2019


class TestGalaxyMap(unittest.TestCase):
    def test_gal_to_bar_vel_single(self):
        """
        Test single galactic to barycentric Cartesian velocity conversion.
        """

        glon = 34.0  # deg
        glat = -15.0  # deg
        dist = 1.0  # kpc
        gmul = 0.235  # mas/yr
        gmub = -0.11  # mas/yr
        vlsr = 9.854  # km/s

        U, V, W = rc.gal_to_bar_vel(glon, glat, dist, vlsr, gmul, gmub)

        # No difference in results between Galactic and GalacticLSR frames
        # gal_vel = GalacticLSR(
        #     l=glon * u.deg,
        #     b=glat * u.deg,
        #     distance=dist * u.kpc,
        #     pm_l_cosb=gmul * (u.mas / u.yr),
        #     pm_b=gmub * (u.mas / u.yr),
        #     radial_velocity=vlsr * (u.km / u.s),
        #     v_bary=_LSR_VEL,
        # )
        gal_vel_no_lsr = Galactic(
            l=glon * u.deg,
            b=glat * u.deg,
            distance=dist * u.kpc,
            pm_l_cosb=gmul * (u.mas / u.yr),
            pm_b=gmub * (u.mas / u.yr),
            radial_velocity=vlsr * (u.km / u.s),
        )
        vx = gal_vel_no_lsr.velocity.d_x.value
        vy = gal_vel_no_lsr.velocity.d_y.value
        vz = gal_vel_no_lsr.velocity.d_z.value

        # print("My method:", U, V, W)
        # print("astropy GalacticLSR:", vx, vy, vz)
        # print("astropy no LSR:", gal_vel_no_lsr.velocity)

        # # Total velocities (below) are the same
        # print("My total vel:", np.sqrt(U * U + V * V + W * W))
        # print("astropy total vel:", np.sqrt(vx * vx + vy * vy + vz * vz))

        self.assertAlmostEqual(U, vx, delta=_VELOCITY)
        self.assertAlmostEqual(V, vy, delta=_VELOCITY)
        self.assertAlmostEqual(W, vz, delta=_VELOCITY)

    def test_gal_to_bar_vel_multi(self):
        """
        Test multiple galactic to barycentric Cartesian velocity conversions.
        """

        glons = np.array([34.2, 0, 0.0, 51, 265, 5.883, 360, 360.0])
        glats = np.array([3.4, 0, 0.0, -12, -60.52, 5.0, 90.0, -90])
        dists = 1 / np.array([0.21, 1.0, 0.1, 3, 25, 541, 0.2, 0.234])
        gmuls = np.array([0.525, -0.25, 0, 1.0, 0.35, 0.22, 0.86, 0.45])
        gmubs = np.array([0.12, -0.1, 0, 0.58, 5.2, -0.35, 0.51, 0.82])
        vlsrs = np.array([12.2, 0.5, -2, 0.0, 25, 6.25, -17.5, 8.2])

        Us, Vs, Ws = rc.gal_to_bar_vel(glons, glats, dists, vlsrs, gmuls, gmubs)

        # No difference in results between Galactic and GalacticLSR frames
        gal_vels = Galactic(
            l=glons * u.deg,
            b=glats * u.deg,
            distance=dists * u.kpc,
            pm_l_cosb=gmuls * (u.mas / u.yr),
            pm_b=gmubs * (u.mas / u.yr),
            radial_velocity=vlsrs * (u.km / u.s),
            # v_bary=_LSR_VEL,
        )
        vxs = gal_vels.velocity.d_x.value
        vys = gal_vels.velocity.d_y.value
        vzs = gal_vels.velocity.d_z.value

        for U, V, W, vx, vy, vz in zip(Us, Vs, Ws, vxs, vys, vzs):
            self.assertAlmostEqual(U, vx, delta=_VELOCITY)
            self.assertAlmostEqual(V, vy, delta=_VELOCITY)
            self.assertAlmostEqual(W, vz, delta=_VELOCITY)

    def test_bar_to_gcen_vel_single(self):
        """
        Test single barycentric Cartesian to galactocentric
        Cartesian velocity conversion.
        """

        Xb = 4.0  # kpc
        Yb = -10.5  # kpc
        Zb = 1.0  # kpc
        Ub = 78.2  # km/s
        Vb = -1.25  # km/s
        Wb = -12.42  # km/s

        g_vx, g_vy, g_vz = rc.bar_to_gcen_vel(Ub, Vb, Wb)

        # Here, Galactic and GalacticLSR _does_ make a difference
        # gcen_vel = GalacticLSR(
        #     x=Xb * u.kpc,
        #     y=Yb * u.kpc,
        #     z=Zb * u.kpc,
        #     v_x=Ub * (u.km / u.s),
        #     v_y=Vb * (u.km / u.s),
        #     v_z=Wb * (u.km / u.s),
        #     v_bary=_LSR_VEL,
        #     representation_type="cartesian",
        #     differential_type="cartesian",
        # ).transform_to(
        #     Galactocentric(
        #         galcen_distance=8.15 * u.kpc,
        #         z_sun=5.5 * u.pc,
        #         roll=0 * u.deg,
        #         galcen_v_sun=_GAL_V_SUN,
        #     )
        # )
        gcen_vel_no_lsr = Galactic(
            u=Xb * u.kpc,
            v=Yb * u.kpc,
            w=Zb * u.kpc,
            U=Ub * (u.km / u.s),
            V=Vb * (u.km / u.s),
            W=Wb * (u.km / u.s),
            representation_type="cartesian",
            differential_type="cartesian",
        ).transform_to(
            Galactocentric(
                galcen_distance=8.15 * u.kpc,
                z_sun=5.5 * u.pc,
                roll=0 * u.deg,
                galcen_v_sun=_GAL_V_SUN,
            )
        )
        gcen_vx = gcen_vel_no_lsr.v_x.value
        gcen_vy = gcen_vel_no_lsr.v_y.value
        gcen_vz = gcen_vel_no_lsr.v_z.value

        # print("My method:", g_vx, g_vy, g_vz)
        # print("astropy LSR:", gcen_vx, gcen_vy, gcen_vz)
        # print("astropy LSR:", gcen_vel.velocity)
        # print("astropy no LSR:", gcen_vel_no_lsr.velocity)

        self.assertAlmostEqual(g_vx, gcen_vx, delta=_VELOCITY)
        self.assertAlmostEqual(g_vy, gcen_vy, delta=_VELOCITY)
        self.assertAlmostEqual(g_vz, gcen_vz, delta=_VELOCITY)

    def test_bar_to_gcen_vel_multi(self):
        """
        Test multiple barycentric Cartesian to
        galactocentric Cartesian velocity conversions.
        """

        Xbs = np.array([12, 14.0, -2.54, -0, 0.0, -18, 3, -1])
        Ybs = np.array([-9.0, 4, 10.4, 0.0, 5.84, -8.201, 2, -2])
        Zbs = np.array([9.0, 11, -0.3, 0, -2.414, -2.01, 1, -3])
        Ubs = np.array([78.2, -51, 0.25, 544.5, -0.001, 0, 0.58, -8.8])
        Vbs = np.array([-1.25, 0.211, 0, 12, 84.5, -0.214, 18.3, -2])
        Wbs = np.array([-12.42, 11, 2.421, 0, 0.875, -121, 254, 0.961])

        g_vxs, g_vys, g_vzs = rc.bar_to_gcen_vel(Ubs, Vbs, Wbs)

        # Here, Galactic and GalacticLSR _does_ make a difference
        # gcen_vels = GalacticLSR(
        #     x=Xbs * u.kpc,
        #     y=Ybs * u.kpc,
        #     z=Zbs * u.kpc,
        #     v_x=Ubs * (u.km / u.s),
        #     v_y=Vbs * (u.km / u.s),
        #     v_z=Wbs * (u.km / u.s),
        #     v_bary=_LSR_VEL,
        #     representation_type="cartesian",
        #     differential_type="cartesian",
        # ).transform_to(
        #     Galactocentric(
        #         galcen_distance=8.15 * u.kpc,
        #         z_sun=5.5 * u.pc,
        #         roll=0 * u.deg,
        #         galcen_v_sun=_GAL_V_SUN,
        #     )
        # )
        gcen_vels_no_lsr = Galactic(
            u=Xbs * u.kpc,
            v=Ybs * u.kpc,
            w=Zbs * u.kpc,
            U=Ubs * (u.km / u.s),
            V=Vbs * (u.km / u.s),
            W=Wbs * (u.km / u.s),
            representation_type="cartesian",
            differential_type="cartesian",
        ).transform_to(
            Galactocentric(
                galcen_distance=8.15 * u.kpc,
                z_sun=5.5 * u.pc,
                roll=0 * u.deg,
                galcen_v_sun=_GAL_V_SUN,
            )
        )
        gcen_vxs = gcen_vels_no_lsr.v_x.value
        gcen_vys = gcen_vels_no_lsr.v_y.value
        gcen_vzs = gcen_vels_no_lsr.v_z.value

        for g_vx, g_vy, g_vz, gcen_vx, gcen_vy, gcen_vz in zip(
            g_vxs, g_vys, g_vzs, gcen_vxs, gcen_vys, gcen_vzs
        ):
            self.assertAlmostEqual(g_vx, gcen_vx, delta=_VELOCITY)
            self.assertAlmostEqual(g_vy, gcen_vy, delta=_VELOCITY)
            self.assertAlmostEqual(g_vz, gcen_vz, delta=_VELOCITY)


if __name__ == "__main__":
    unittest.main()
