from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
DETECTOR_LENGTH_M = 2.0
AVERAGE_VEHICLE_LENGTH_M = 5.0


def estimate_region_traffic(data_dir=DATA_DIR, excel_path=None):
    links = pd.read_csv(
        data_dir / "links.csv",
        header=None,
        names=["link_id", "length_m", "lanes", "start_node", "end_node", "region"],
    )

    def read_measurements(filename):
        raw = pd.read_csv(data_dir / filename, header=None)
        link_ids = raw.iloc[0, 1:].astype(int).to_numpy()
        if len(np.unique(link_ids)) != len(link_ids):
            raise ValueError(f"Duplicate link IDs found in {filename}.")
        times_s = raw.iloc[1:, 0].astype(float).to_numpy()
        values = raw.iloc[1:, 1:].astype(float).to_numpy()
        return times_s, link_ids, values

    occupancy_times, occupancy_ids, occupancy = read_measurements("occupancy.csv")
    flow_times, flow_ids, flow = read_measurements("flow.csv")

    if not np.array_equal(occupancy_times, flow_times):
        raise ValueError("flow.csv and occupancy.csv have different time steps.")
    intervals_s = np.diff(flow_times)
    if len(intervals_s) == 0 or np.any(intervals_s <= 0):
        raise ValueError("Measurement times must contain positive intervals.")
    if not np.allclose(intervals_s, intervals_s[0]):
        raise ValueError("flow.csv must use a constant measurement interval.")

    link_ids = links["link_id"].astype(int).to_numpy()

    def align_columns(values, measurement_ids, filename):
        id_to_column = {link_id: index for index, link_id in enumerate(measurement_ids)}
        missing = [link_id for link_id in link_ids if link_id not in id_to_column]
        if missing:
            raise ValueError(f"Links missing from {filename}: {missing[:10]}")
        return values[:, [id_to_column[link_id] for link_id in link_ids]]

    occupancy = align_columns(occupancy, occupancy_ids, "occupancy.csv")
    flow = align_columns(flow, flow_ids, "flow.csv")
    flow_rate = flow * (3600.0 / intervals_s[0])

    lanes = links["lanes"].to_numpy(dtype=float)
    density = (
        (occupancy / 100.0)
        * lanes[None, :]
        / (AVERAGE_VEHICLE_LENGTH_M + DETECTOR_LENGTH_M)
        * 1000.0
    )

    regions = links["region"].to_numpy()
    link_lengths = links["length_m"].to_numpy(dtype=float)
    time_min = occupancy_times / 60.0

    if excel_path is not None:
        n_times, n_links = density.shape
        link_density_table = pd.DataFrame(
            {
                "time_s": np.repeat(occupancy_times, n_links),
                "time_min": np.repeat(time_min, n_links),
                "link_id": np.tile(link_ids, n_times),
                "region": np.tile(regions, n_times),
                "link_length_m": np.tile(link_lengths, n_times),
                "lanes": np.tile(lanes, n_times),
                "occupancy_percent": occupancy.reshape(-1),
                "average_vehicle_length_m": AVERAGE_VEHICLE_LENGTH_M,
                "detector_length_m": DETECTOR_LENGTH_M,
                "density_veh_per_km": density.reshape(-1),
            }
        )
        link_density_table.to_excel(
            excel_path,
            sheet_name="Link densities",
            index=False,
        )

    records = []

    for region in np.sort(pd.unique(regions)):
        in_region = regions == region
        lengths = link_lengths[in_region]
        density_length = density[:, in_region] * lengths[None, :]
        denominator = density_length.sum(axis=1)
        mean_density = denominator / lengths.sum()
        mean_speed = np.divide(
            (flow_rate[:, in_region] * lengths[None, :]).sum(axis=1),
            denominator,
            out=np.full(len(time_min), np.nan),
            where=denominator > 0,
        )

        records.extend(
            {
                "time_min": time,
                "region": region,
                "mean_density_veh_per_km": average_density,
                "mean_speed_km_per_h": average_speed,
            }
            for time, average_density, average_speed in zip(
                time_min, mean_density, mean_speed
            )
        )

    return pd.DataFrame(records)


def plot_region_traffic(results):
    regions = np.sort(results["region"].unique())
    n_regions = len(regions)
    n_cols = int(np.ceil(np.sqrt(n_regions)))
    n_rows = int(np.ceil(n_regions / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    axes = axes.ravel()
    time_min = results["time_min"]
    scatter = None

    for ax, region in zip(axes, regions):
        region_data = results[results["region"] == region]
        scatter = ax.scatter(
            region_data["mean_density_veh_per_km"],
            region_data["mean_speed_km_per_h"],
            c=region_data["time_min"],
            cmap="viridis",
            vmin=time_min.min(),
            vmax=time_min.max(),
            s=22,
            alpha=0.8,
        )
        ax.set_title(f"Region {region}")
        ax.set_xlabel("Mean density [veh/km]")
        ax.set_ylabel("Mean speed [km/h]")
        ax.grid(True, alpha=0.25)

    for ax in axes[n_regions:]:
        ax.set_visible(False)

    fig.suptitle("Regional mean speed vs. mean density")
    fig.subplots_adjust(top=0.88, right=0.87, wspace=0.3, hspace=0.35)
    colorbar_ax = fig.add_axes([0.90, 0.2, 0.025, 0.65])
    fig.colorbar(scatter, cax=colorbar_ax, label="Time [min]")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for region in regions:
        region_data = results[results["region"] == region]
        ax.plot(
            region_data["time_min"],
            region_data["mean_speed_km_per_h"],
            label=f"Region {region}",
        )

    ax.set_title("Regional mean speed over time")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Mean speed [km/h]")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Region", ncol=2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    excel_path = DATA_DIR / "link_density_estimates.xlsx"
    regional_results = estimate_region_traffic(excel_path=excel_path)
    valid_speeds = regional_results["mean_speed_km_per_h"].dropna()
    print(f"Link density table saved to: {excel_path}")
    print(
        "Regional mean speed range: "
        f"{valid_speeds.min():.2f} to {valid_speeds.max():.2f} km/h"
    )
    plot_region_traffic(regional_results)