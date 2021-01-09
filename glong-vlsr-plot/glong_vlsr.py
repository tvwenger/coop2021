from pathlib import Path
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd


def create_connection(db_file):
    """
    Creates SQLite database connection specified by db_file
    """

    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        print(e)

    return conn


def get_data(conn):
    """
    Retrieves glong and vlsr from database conn
    Returns DataFrame with glong and vlsr
    """

    cur = conn.cursor()
    cur.execute("SELECT glong, vlsr FROM Parallax")
    coords = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    return coords


def main():
    # Specifying database file name
    filename = "data/hii_v2_20201203.db"

    # Database in parent directory of this script (call dirname twice)
    db = Path(__file__).parent.parent / filename

    # Create database connection to db
    conn = create_connection(db)

    # Get data + put into DataFrame
    coords = get_data(conn)
    glong = coords["glong"]
    vlsr = coords["vlsr"]
    # print(coords.to_markdown())

    ###############################################################################
    # # Transform data
    # glong *= -1  # reflect data about y-axis
    # glong[glong < -180] += 360  # split & shift data to right of 0
    # # glong[glong < 180] *= -1

    # # Plot data
    # fig, ax = plt.subplots(figsize=(plt.figaspect(0.4)))
    # plt.scatter(glong, vlsr, s=7, color="C4")

    # # Set the glong ticks so that positive ticks represent [360,180] degrees
    # ticks = ax.get_xticks()
    # ticks_left = -1 * ticks[ticks < 0]
    # ticks_right = 360 - ticks[ticks > 0]
    # ticks_new = np.concatenate([ticks_left, np.array([0]), ticks_right])
    # ax.set_xticklabels([int(tick) for tick in ticks_new])
    ###############################################################################

    # Transform data
    glong[glong > 180] -= 360  # map points [180,360] --> [-180,0]

    # Plot data
    fig, ax = plt.subplots(figsize=(plt.figaspect(0.5)))
    ax.scatter(glong, vlsr, s=7, color="C4")

    # Re-labelling ticks
    # # Method 1
    # ax.xlim(180, -180)
    # plt.xticks(
    #     [180, 150, 120, 90, 60, 30, 0, -30, -60, -90, -120, -150, -180],
    #     labels=[180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180],
    # )
    # # Method 2
    ax.set_xlim([180, -180])
    ax.set_xticks([180, 150, 120, 90, 60, 30, 0, -30, -60, -90, -120, -150, -180])
    ax.set_xticklabels([180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180])

    # plt.ylim(-200, 200)

    # Set background colour
    ax.set_facecolor("k")

    # Set title and labels. Then save figure
    ax.set_title("Longitude–LSR Velocity Diagram")
    ax.set_xlabel("Galactic Longitude ($^{\circ}$)")
    ax.set_ylabel("$v_{LSR}$")
    fig.savefig(
        Path(__file__).parent /  "glong_vlsr.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
