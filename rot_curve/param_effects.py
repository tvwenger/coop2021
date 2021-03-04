import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Want to add galaxymap.py as package:
# Make a $PATH to coop2021 (twice parent folder of this file)
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add coop2021 to $PATH
sys.path.append(_SCRIPT_DIR)
import mytransforms as trans

# User-defined constants
_RSUN = 8.15  # kpc (Reid et al. 2019)

# Universal rotation curve parameters (Persic et al. 1996)
_A_TWO = 0.96  # (Reid et al. 2019)
_A_THREE = 1.62  # (Reid et al. 2019)


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


def main():
    # Create and plot different rotation curves for a2
    my_a2_vals = np.arange(-0.2,0.6,0.1) + _A_TWO
    fig, ax = plt.subplots()
    Rvals = np.linspace(0, 17, 101)

    # for a2_val in my_a2_vals:
    #     Vvals = urc(Rvals, a2=a2_val, a3=_A_THREE)
    #     ax.plot(Rvals, Vvals, "-.", linewidth=0.5, label=f"{a2_val}")

    # # Set title and labels. Then save figure
    # ax.set_title("Galactic Rotation Curve with Different a2 Values")
    # ax.set_xlabel("R (kpc)")
    # ax.set_ylabel(r"$\Theta$ (km $\mathrm{s}^{-1})$")
    # ax.set_xlim(0, 17)
    # ax.set_ylim(0, 300)
    # ax.legend(title="a2 value", loc="lower right")
    # fig.savefig(
    #     Path(__file__).parent / "param_effects_a2_rot_curve.jpg",
    #     format="jpg",
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # plt.show()

    # # Create and plot different rotation curves for a3
    my_a3_vals = np.arange(-0.2,0.6,0.1) + _A_THREE
    # fig2, ax2 = plt.subplots()

    # for a3_val in my_a3_vals:
    #     Vvals2 = urc(Rvals, a2=_A_TWO, a3=a3_val)
    #     ax2.plot(Rvals, Vvals2, "-.", linewidth=0.5, label=f"{a3_val:.2f}")

    # # Set title and labels. Then save figure
    # ax2.set_title("Galactic Rotation Curve with Different a3 Values")
    # ax2.set_xlabel("R (kpc)")
    # ax2.set_ylabel(r"$\Theta$ (km $\mathrm{s}^{-1})$")
    # ax2.set_xlim(0, 17)
    # ax2.set_ylim(0, 300)
    # ax2.legend(title="a3 value", loc="lower right")
    # fig2.savefig(
    #     Path(__file__).parent / "param_effects_a3_rot_curve.jpg",
    #     format="jpg",
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # plt.show()

    # Create and plot different rotation curves for a2 & a3 on one plot
    fig3, ax3 = plt.subplots(1, 2, figsize=plt.figaspect(0.25))

    for a2_val in my_a2_vals:
        Vvals = urc(Rvals, a2=a2_val, a3=_A_THREE)
        ax3[0].plot(Rvals, Vvals, "-.", linewidth=0.5, label=f"{a2_val}")

    for a3_val in my_a3_vals:
        Vvals2 = urc(Rvals, a2=_A_TWO, a3=a3_val)
        ax3[1].plot(Rvals, Vvals2, "-.", linewidth=0.5, label=f"{a3_val:.2f}")

    # Set title and labels. Then save figure
    ax3[0].set_title("Galactic Rotation Curve with Different a2 Values")
    ax3[0].set_xlabel("R (kpc)")
    ax3[0].set_ylabel(r"$\Theta$ (km $\mathrm{s}^{-1})$")
    ax3[0].set_xlim(0, 17)
    ax3[0].set_ylim(0, 300)
    ax3[0].legend(title="a2 value", loc="lower right")
    ax3[1].set_title("Galactic Rotation Curve with Different a3 Values")
    ax3[1].set_xlabel("R (kpc)")
    # ax3[1].set_ylabel(r"$\Theta$ (km $\mathrm{s}^{-1})$")
    ax3[1].set_xlim(0, 17)
    ax3[1].set_ylim(0, 300)
    ax3[1].legend(title="a3 value", loc="lower right")
    fig3.savefig(
        Path(__file__).parent / "param_effects_a2a3_rot_curve.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
