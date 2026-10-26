import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.collections import LineCollection


def plot_partition(cluster_id):
    """
    Plot a road-network partition using different colors.

    Parameters
    ----------
    cluster_id : array-like
        Cluster assignment of each link. The order must correspond
        to the rows of links.csv.

    Required files
    --------------
    links.csv
    nodes.csv

    The CSV files should be located in the same folder as this
    Python file.

    Example
    -------
    idx = KMeans(n_clusters=4, n_init=10, random_state=1).fit_predict(X)
    plot_partition(idx)
    """

    # ----------------------------------------------------------
    # 1. Locate and read data
    # ----------------------------------------------------------

    # Folder containing this Python file
    data_dir = Path(__file__).resolve().parent

    links = pd.read_csv(
        data_dir / "links.csv",
        header=None
    ).to_numpy()

    nodes = pd.read_csv(
        data_dir / "nodes.csv",
        header=None
    ).to_numpy()

    # ----------------------------------------------------------
    # 2. Network information
    # ----------------------------------------------------------

    # links.csv:
    # column 0 = Link ID
    # column 3 = Starting node ID
    # column 4 = Ending node ID

    start_node = links[:, 3].astype(int)
    end_node = links[:, 4].astype(int)

    # nodes.csv:
    # column 0 = Node ID
    # column 1 = x-coordinate
    # column 2 = y-coordinate

    node_id = nodes[:, 0].astype(int)
    node_x = nodes[:, 1].astype(float)
    node_y = nodes[:, 2].astype(float)

    # ----------------------------------------------------------
    # 3. Check cluster_id
    # ----------------------------------------------------------

    cluster_id = np.asarray(cluster_id).flatten()

    n_links = len(start_node)

    if len(cluster_id) != n_links:
        raise ValueError(
            f"cluster_id must contain one cluster assignment for "
            f"each link in links.csv. Expected {n_links} values, "
            f"but got {len(cluster_id)}."
        )

    if not np.all(np.isfinite(cluster_id)):
        raise ValueError(
            "cluster_id contains invalid values."
        )

    cluster_labels = np.unique(cluster_id)
    K = len(cluster_labels)

    # ----------------------------------------------------------
    # 4. Map node IDs to coordinates
    # ----------------------------------------------------------

    node_coordinates = {
        node_id[i]: (node_x[i], node_y[i])
        for i in range(len(node_id))
    }

    segments = []

    for i in range(n_links):

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

    segments = np.asarray(segments)

    # ----------------------------------------------------------
    # 5. Plot network partition
    # ----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    # Use a categorical colormap
    cmap = plt.get_cmap("tab10")

    for k, label in enumerate(cluster_labels):

        link_mask = (cluster_id == label)

        cluster_segments = segments[link_mask]

        collection = LineCollection(
            cluster_segments,
            colors=[cmap(k % 10)],
            linewidths=1.5,
            label=f"Cluster {label:g}"
        )

        ax.add_collection(collection)

    # ----------------------------------------------------------
    # 6. Figure formatting
    # ----------------------------------------------------------

    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        "Network Partition",
        fontweight="bold"
    )

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)
    )

    plt.tight_layout()
    plt.show()