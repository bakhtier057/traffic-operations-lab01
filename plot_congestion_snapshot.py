import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_congestion_snapshot(time_min_vec):
    """
    Plot congestion snapshots of the road network at specified times.

    Parameters
    ----------
    time_min_vec : list or array-like
        Requested snapshot times in minutes.

    Example
    -------
    plot_congestion_snapshot([60, 90, 120])

    Required files
    --------------
    links.csv
    nodes.csv
    occupancy.csv

    The CSV files should be located in the same folder as this
    Python file.

    Occupancy = 0%   -> white
    Occupancy = 100% -> black
    """

    # ----------------------------------------------------------
    # 1. Locate and read data
    # ----------------------------------------------------------

    # Folder containing this Python file
    data_dir = Path(__file__).resolve().parent

    # links.csv and nodes.csv contain numerical data without
    # a header row
    links = pd.read_csv(
        data_dir / "links.csv",
        header=None
    ).to_numpy()

    nodes = pd.read_csv(
        data_dir / "nodes.csv",
        header=None
    ).to_numpy()

    # occupancy.csv:
    # first row    = link IDs
    # first column = time [sec]
    occupancy_raw = pd.read_csv(
        data_dir / "occupancy.csv",
        header=None
    ).to_numpy()

    # ----------------------------------------------------------
    # 2. Network information
    # ----------------------------------------------------------

    # links.csv columns:
    # 0 = Link ID
    # 1 = Length
    # 2 = Number of lanes
    # 3 = Starting node ID
    # 4 = Ending node ID
    # 5 = Region

    link_id = links[:, 0].astype(int)
    start_node = links[:, 3].astype(int)
    end_node = links[:, 4].astype(int)

    # nodes.csv columns:
    # 0 = Node ID
    # 1 = x-coordinate
    # 2 = y-coordinate

    node_id = nodes[:, 0].astype(int)
    node_x = nodes[:, 1].astype(float)
    node_y = nodes[:, 2].astype(float)

    # ----------------------------------------------------------
    # 3. Occupancy data
    # ----------------------------------------------------------

    occupancy_link_id = occupancy_raw[0, 1:].astype(int)

    time_sec = occupancy_raw[1:, 0].astype(float)

    occupancy = occupancy_raw[1:, 1:].astype(float)

    # ----------------------------------------------------------
    # 4. Match occupancy columns with links.csv
    # ----------------------------------------------------------

    occupancy_id_to_col = {
        link: j
        for j, link in enumerate(occupancy_link_id)
    }

    occupancy_columns = []

    for link in link_id:

        if link not in occupancy_id_to_col:
            raise ValueError(
                f"Link {link} from links.csv was not found "
                f"in occupancy.csv."
            )

        occupancy_columns.append(
            occupancy_id_to_col[link]
        )

    # Reorder occupancy columns so they follow links.csv
    occupancy = occupancy[:, occupancy_columns]

    # ----------------------------------------------------------
    # 5. Map node IDs to coordinates
    # ----------------------------------------------------------

    node_coordinates = {
        node_id[i]: (node_x[i], node_y[i])
        for i in range(len(node_id))
    }

    segments = []

    for i in range(len(link_id)):

        s = start_node[i]
        e = end_node[i]

        if s not in node_coordinates:
            raise ValueError(
                f"Start node {s} was not found in nodes.csv."
            )

        if e not in node_coordinates:
            raise ValueError(
                f"End node {e} was not found in nodes.csv."
            )

        x1, y1 = node_coordinates[s]
        x2, y2 = node_coordinates[e]

        segments.append([
            [x1, y1],
            [x2, y2]
        ])

    # ----------------------------------------------------------
    # 6. Process requested times
    # ----------------------------------------------------------

    time_min_vec = np.atleast_1d(
        np.asarray(time_min_vec, dtype=float)
    )

    n_plots = len(time_min_vec)

    if n_plots == 0:
        raise ValueError(
            "time_min_vec must contain at least one time."
        )

    # ----------------------------------------------------------
    # 7. Determine subplot layout automatically
    # ----------------------------------------------------------

    if n_plots <= 3:

        n_rows = 1
        n_cols = n_plots

    else:

        n_rows = int(np.floor(np.sqrt(n_plots)))
        n_cols = int(np.ceil(n_plots / n_rows))

    # ----------------------------------------------------------
    # 8. Create figure
    # ----------------------------------------------------------

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        squeeze=False
    )

    axes = axes.flatten()

    # Common occupancy-to-color mapping
    norm = Normalize(
        vmin=0,
        vmax=100
    )

    # Greys:
    # 0   -> white
    # 100 -> black
    cmap = plt.get_cmap("Greys")

    # ----------------------------------------------------------
    # 9. Plot requested snapshots
    # ----------------------------------------------------------

    for j, time_min in enumerate(time_min_vec):

        ax = axes[j]

        target_time_sec = time_min * 60

        # Check whether requested time is within simulation
        if (
            target_time_sec < np.min(time_sec)
            or
            target_time_sec > np.max(time_sec)
        ):

            ax.text(
                0.5,
                0.5,
                f"t = {time_min:g} min\n"
                f"is outside the simulation period",
                ha="center",
                va="center",
                transform=ax.transAxes
            )

            ax.axis("off")
            continue

        # Find nearest available measurement
        row = np.argmin(
            np.abs(time_sec - target_time_sec)
        )

        actual_time_sec = time_sec[row]
        actual_time_min = actual_time_sec / 60

        occupancy_vec = occupancy[row, :]

        # Restrict occupancy to [0, 100]
        occupancy_vec = np.clip(
            occupancy_vec,
            0,
            100
        )

        # ------------------------------------------------------
        # Plot road links
        # ------------------------------------------------------

        collection = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=1.5
        )

        # Occupancy directly controls link color
        collection.set_array(
            occupancy_vec
        )

        ax.add_collection(
            collection
        )

        ax.autoscale()
        ax.set_aspect("equal")
        ax.axis("off")

        # If requested time exactly matches measurement time
        if np.isclose(
            actual_time_sec,
            target_time_sec
        ):

            title_text = (
                f"t = {time_min:g} min"
            )

        # Otherwise show actual measurement time used
        else:

            title_text = (
                f"t = {actual_time_min:g} min "
                f"(nearest to {time_min:g} min)"
            )

        ax.set_title(
            title_text,
            fontweight="bold"
        )

    # ----------------------------------------------------------
    # 10. Hide unused subplot positions
    # ----------------------------------------------------------

    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    # ----------------------------------------------------------
    # 11. Shared colorbar
    # ----------------------------------------------------------

    sm = ScalarMappable(
        norm=norm,
        cmap=cmap
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes.tolist(),
        fraction=0.025,
        pad=0.02
    )

    cbar.set_label(
        "Occupancy (%)"
    )

    cbar.set_ticks(
        [0, 20, 40, 60, 80, 100]
    )

    # ----------------------------------------------------------
    # 12. Figure title
    # ----------------------------------------------------------

    fig.suptitle(
        "Network Congestion",
        fontsize=14,
        fontweight="bold"
    )

    plt.show()


# --------------------------------------------------------------
# Example
# --------------------------------------------------------------

if __name__ == "__main__":
    plot_congestion_snapshot(
        [30, 60, 90, 120]
    )