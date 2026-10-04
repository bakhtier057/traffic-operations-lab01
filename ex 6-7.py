from re import sub

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

data_dir = Path(__file__).resolve().parent
flow = pd.read_csv(
        data_dir / "flow.csv",
        header=None
    ).to_numpy()
occupancy = pd.read_csv(
        data_dir / "occupancy.csv",
        header=None
    ).to_numpy()
links = pd.read_csv(
        data_dir / "links.csv",
        header=None
    ).to_numpy()
nodes = pd.read_csv(
        data_dir / "nodes.csv",
        header=None
    ).to_numpy()
density = pd.read_csv(
        data_dir / "density.csv",
        header=None
    ).to_numpy()

np.mean(occupancy[1:,1:], axis=0)
np.std(occupancy[1:,1:], axis=0)
meandensity=np.mean(density[:,:], axis=0)

print(meandensity)
standardized_mean_density = np.zeros_like(meandensity)
print(np.shape(meandensity))
standardized_mean_density[:] = (meandensity - np.mean(meandensity[:], axis=0)) / np.std(meandensity, axis=0)

matrice = standardized_mean_density[:].astype(float)
matrice = np.nan_to_num(matrice, nan=0.0, posinf=0.0, neginf=0.0)
links2 = pd.read_csv(
        data_dir / "links.csv",
        header=None
    )
nodes2 = pd.read_csv(
        data_dir / "nodes.csv",
        header=None
    )
df_left = pd.merge(links2, nodes2, left_on=3, right_on=0, how='left')
df_final = pd.merge(df_left, nodes2, left_on=4, right_on=0, how='left')
df_final['meandensity'] = meandensity
numberk=[2]
for k in numberk:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=1).fit_predict(matrice.reshape(-1, 1))
    df_final[f'cluster_{k}'] = kmeans



#plt.scatter(nodes2[1], nodes2[2], s=1, zorder=2, color="black")
#plt.title(f"KMeans Clustering of Links ({k} clusters)")
#plt.xlabel("Longitude")
#plt.ylabel("Latitude")
#plt.show()




cluster_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

for k in numberk:
    labels = df_final[f"cluster_{k}"]

    #plt.figure(figsize=(8, 6))
    for _, sub in df_final.iterrows():
        xs = [sub.iloc[7], sub.iloc[10]]
        ys = [sub.iloc[8], sub.iloc[11]]
        cluster = int(sub[f"cluster_{k}"])
        #plt.plot(
            #xs,
            #ys,
            #color=cluster_colors[cluster % len(cluster_colors)],
            #linewidth=3,
            #alpha=1,
            #zorder=1,
        #)

    #plt.scatter(nodes2[1], nodes2[2], s=1, zorder=2, color="black")
    #plt.title(f"KMeans Clustering of Links ({k} clusters)")
    #plt.xlabel("Longitude")
    #plt.ylabel("Latitude")
    #plt.show()





    rows = int(np.ceil(k / 2))
    #fig, axes = plt.subplots(rows, 2, figsize=(10, 5 * rows), squeeze=False)
    for i in range(k):
        #ax = axes[i // 2, i % 2]
        #ax.set_title(f"Cluster {i}")
        #ax.set_xlabel("Longitude")
        #ax.set_ylabel("Latitude")
        for _, sub in df_final.iterrows():
            if sub[f"cluster_{k}"] == i:
                xs = [sub.iloc[7], sub.iloc[10]]
                ys = [sub.iloc[8], sub.iloc[11]]
                #ax.plot(
                    #xs,
                    #ys,
                    #color=cluster_colors[i % len(cluster_colors)],
                    #linewidth=3,
                    #alpha=1,
                    #zorder=1,
                #)
        #ax.scatter(nodes2[1], nodes2[2], s=1, zorder=2, color="black")

    #for i in range(k, rows * 2):
        #axes[i // 2, i % 2].set_visible(False)
    #plt.tight_layout()
    #plt.show()


##ex 7
kpoint=[1.82,2.63,1.76,2.91,2.97]
print(df_final.head())
df_final['xmean']=(df_final.iloc[:, 10] + df_final.iloc[:, 7])/2
df_final['ymean']=(df_final.iloc[:, 11] + df_final.iloc[:, 8])/2
df_final['xmeannormalized']=(df_final['xmean']-np.mean(df_final['xmean']))/np.std(df_final['xmean'])
df_final['ymeannormalized']=(df_final['ymean']-np.mean(df_final['ymean']))/np.std(df_final['ymean'])
df_final["standardized_meandensity"] = (df_final["meandensity"] - np.mean(df_final["meandensity"])) / np.std(df_final["meandensity"])

numberk=[2,3,4,5,6]
for k in numberk:
    matrice = df_final[['xmeannormalized', 'ymeannormalized', 'standardized_meandensity']].to_numpy()
    print("avant",matrice)
    matrice[:, 0]=kpoint[k-2]*matrice[:, 0]
    matrice[:, 1]=kpoint[k-2]*matrice[:, 1]
    print("apres",matrice)
    print(np.shape(matrice))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=1).fit_predict(matrice)
    print(f"Cluster labels for k={k}: {kmeans}")
    print(np.shape(kmeans))
    df_final[f'clusterxy_{k}'] = kmeans


segments = []
for i in range(len(df_final)):
    x1 = df_final.iloc[i, 5]
    y1 = df_final.iloc[i, 6]
    x2 = df_final.iloc[i, 8]
    y2 = df_final.iloc[i, 9]
    segments.append([(x1, y1), (x2, y2)])


#plt.figure(figsize=(8, 6))
for i, sub in df_final.iterrows():
    xs = [sub.iloc[7], sub.iloc[10]]
    ys = [sub.iloc[8], sub.iloc[11]]
    cluster = int(sub[f"clusterxy_{k}"])
    plt.plot(
        xs,
        ys,
        color=plt.cm.viridis(
            plt.Normalize(np.nanmin(meandensity), np.nanmax(meandensity))(meandensity[i])
        ),
        linewidth=3,
        alpha=1,
        zorder=1,
    )

#plt.scatter(nodes2[1], nodes2[2], s=1, zorder=2, color="black")
#plt.title(f"KMeans Clustering of Links ({k} clusters with xy coordinates)")
#plt.xlabel("Longitude")
#plt.ylabel("Latitude")
#plt.show()




cluster_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

for k in numberk:
    labels = df_final[f"clusterxy_{k}"]

    plt.figure(figsize=(8, 6))
    for _, sub in df_final.iterrows():
        xs = [sub.iloc[7], sub.iloc[10]]
        ys = [sub.iloc[8], sub.iloc[11]]
        cluster = int(sub[f"clusterxy_{k}"])
        plt.plot(
            xs,
            ys,
            color=cluster_colors[cluster % len(cluster_colors)],
            linewidth=3,
            alpha=1,
            zorder=1,
        )

    plt.scatter(nodes2[1], nodes2[2], s=1, zorder=2, color="black")
    plt.title(f"KMeans Clustering of Links ({k} clusters with xy coordinates)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()





    rows = int(np.ceil(k / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(10, 5 * rows), squeeze=False)
    for i in range(k):
        ax = axes[i // 2, i % 2]
        ax.set_title(f"Cluster {i}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        for _, sub in df_final.iterrows():
            if sub[f"clusterxy_{k}"] == i:
                xs = [sub.iloc[7], sub.iloc[10]]
                ys = [sub.iloc[8], sub.iloc[11]]
                ax.plot(
                    xs,
                    ys,
                    color=cluster_colors[i % len(cluster_colors)],
                    linewidth=3,
                    alpha=1,
                    zorder=1,
                )
        ax.scatter(nodes2[1], nodes2[2], s=1, zorder=2, color="black")

    for i in range(k, rows * 2):
        axes[i // 2, i % 2].set_visible(False)
    plt.tight_layout()
    plt.show()

for i in numberk:
    mean_density_per_cluster = df_final.groupby(f"clusterxy_{i}")["meandensity"].mean()
    print(f"Mean density for k={i}:")
    print(mean_density_per_cluster)