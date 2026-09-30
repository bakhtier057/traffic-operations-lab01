import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def load_traffic_data():
    """Load the network and time-series tables and align them by link ID."""
    data_dir = Path(__file__).resolve().parent

    links = pd.read_csv(
        data_dir / "links.csv",
        header=None,
        names=["link_id", "length_km", "lanes", "start_node", "end_node", "region"],
    )

    flow_raw = pd.read_csv(data_dir / "flow.csv", header=None)
    occupancy_raw = pd.read_csv(data_dir / "occupancy.csv", header=None)

    flow_ids = flow_raw.iloc[0, 1:].astype(int).to_numpy()
    occupancy_ids = occupancy_raw.iloc[0, 1:].astype(int).to_numpy()

    flow_times = flow_raw.iloc[1:, 0].to_numpy(dtype=float)
    occupancy_times = occupancy_raw.iloc[1:, 0].to_numpy(dtype=float)

    flow_matrix = flow_raw.iloc[1:, 1:].to_numpy(dtype=float)
    occupancy_matrix = occupancy_raw.iloc[1:, 1:].to_numpy(dtype=float)

    links_order = links["link_id"].to_numpy(dtype=int)

    flow_lookup = {link_id: idx for idx, link_id in enumerate(flow_ids)}
    occupancy_lookup = {link_id: idx for idx, link_id in enumerate(occupancy_ids)}

    flow_aligned = np.zeros((len(flow_times), len(links_order)))
    occupancy_aligned = np.zeros((len(occupancy_times), len(links_order)))

    for j, link_id in enumerate(links_order):
        if link_id not in flow_lookup:
            raise ValueError(f"Link {link_id} is missing from flow.csv")
        if link_id not in occupancy_lookup:
            raise ValueError(f"Link {link_id} is missing from occupancy.csv")

        flow_aligned[:, j] = flow_matrix[:, flow_lookup[link_id]]
        occupancy_aligned[:, j] = occupancy_matrix[:, occupancy_lookup[link_id]]

    return links, flow_times, occupancy_times, flow_aligned, occupancy_aligned


def compute_link_metrics(links, flow_matrix, occupancy_matrix):
    """Compute accumulation and production per link across time."""
    lengths = links["length_km"].to_numpy(dtype=float)

    if np.nanmax(occupancy_matrix) <= 100:
        density = occupancy_matrix / 100.0
    else:
        density = occupancy_matrix

    accumulation = density * lengths[np.newaxis, :]
    production = flow_matrix * lengths[np.newaxis, :]

    return accumulation, production


def plot_production_accumulation():
    data_dir = Path(__file__).resolve().parent

    links, _, _, flow_matrix, occupancy_matrix = load_traffic_data()
    accumulation, production = compute_link_metrics(links, flow_matrix, occupancy_matrix)

    rng = np.random.default_rng(42)
    all_link_ids = links["link_id"].to_numpy(dtype=int)

    random_link_id = int(rng.choice(all_link_ids))
    random_link_index = int(np.where(all_link_ids == random_link_id)[0][0])

    eligible_links = np.delete(all_link_ids, np.where(all_link_ids == random_link_id)[0][0])
    second_link_id = int(rng.choice(eligible_links))
    second_link_index = int(np.where(all_link_ids == second_link_id)[0][0])

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)

    regions = sorted(links["region"].unique())
    n_regions = len(regions)
    ncols = min(3, n_regions)
    nrows = int(np.ceil(n_regions / ncols)) + 1
    gs = fig.add_gridspec(nrows, 3)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(
        accumulation[:, random_link_index],
        production[:, random_link_index],
        s=18,
        alpha=0.7,
        color="tab:blue",
    )
    ax.set_xlabel("Accumulation (veh)")
    ax.set_ylabel("Production (veh·km/h)")
    ax.set_title(f"Single link {random_link_id}")
    ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(gs[0, 1])
    acc_sum = accumulation[:, random_link_index] + accumulation[:, second_link_index]
    prod_sum = production[:, random_link_index] + production[:, second_link_index]
    ax.scatter(acc_sum, prod_sum, s=18, alpha=0.7, color="tab:orange")
    ax.set_xlabel("Accumulation (veh)")
    ax.set_ylabel("Production (veh·km/h)")
    ax.set_title(f"Sum of links {random_link_id} and {second_link_id}")
    ax.grid(True, alpha=0.2)

    for i, region in enumerate(regions):
        row = i // ncols
        col = i % ncols
        ax_region = fig.add_subplot(gs[1 + row, col])

        region_link_ids = links.loc[links["region"] == region, "link_id"].to_numpy(dtype=int)
        region_indices = [int(np.where(all_link_ids == link_id)[0][0]) for link_id in region_link_ids]

        region_accumulation = accumulation[:, region_indices].sum(axis=1)
        region_production = production[:, region_indices].sum(axis=1)

        ax_region.scatter(region_accumulation, region_production, s=16, alpha=0.7, color="tab:green")
        ax_region.set_title(f"Region {region}")
        ax_region.set_xlabel("Accumulation (veh)")
        ax_region.set_ylabel("Production (veh·km/h)")
        ax_region.grid(True, alpha=0.2)

    fig.suptitle("Production vs. accumulation", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.savefig(data_dir / "production_vs_accumulation.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    plot_production_accumulation()
