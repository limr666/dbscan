
import numpy as np

class PointInfo:
    '''
    Class to store information of a point
    for noise points:
        core_index = 0
    for core points and border points:
        core_index = 1-based index of its core point
    for core points:
        cluster_index = 1-based index of its cluster
    '''
    def __init__(self):
        self.n_neighbours = 0   # Number of neighbours, including itself
        self.core_index = 0     # 0 for noise points, 1-based index of its core point
        self.cluster_index = 0  # 0 for noise points and border points, 1-based index of its cluster

def dist2(data, i, j):
    return (data[i,0] - data[j,0])**2 + (data[i,1] - data[j,1])**2

def dbscan(data, eps, min_pts, dist_func=dist2):
    n = len(data)
    point_info = [PointInfo() for i in range(n)]
    neighbours = np.zeros((n, n), np.int32)  # To store indices of neighbours (can also use list of lists)
    core_pt_indices = []    # To store indices of core points, 0-based index
    eps2 = eps**2

    # Find core points & neighbours
    for i, pt in enumerate(point_info):
        neighbours[i, pt.n_neighbours] = i  # Add itself as neighbour
        pt.n_neighbours += 1
        for j in range(i + 1, n):
            if dist_func(data, i, j) < eps2:
                neighbours[i, pt.n_neighbours] = j  # Add neighbour j
                pt.n_neighbours += 1
                pt2 = point_info[j]                 # Add itself as neighbour to j
                neighbours[j, pt2.n_neighbours] = i
                pt2.n_neighbours += 1
        # Check if core point
        if pt.n_neighbours >= min_pts:
            core_pt_indices.append(i)

    clusters = [None for i in range(len(core_pt_indices))]  # To store core points as clusters
    len_clusters = 0        # number of slots used for clusters, may have None slots (empty clusters)

    for i in core_pt_indices:
        # get cluster index or assign a new one
        pt = point_info[i]
        if pt.cluster_index > 0:    # If it is already assigned to a cluster
            cluster_index = pt.cluster_index - 1    # Get cluster index
        else:                       # If it is not assigned to any cluster
            # Assign a new cluster for this core point
            cluster_index = len_clusters
            len_clusters += 1
            pt.cluster_index = cluster_index + 1    # 1-based index
            clusters[cluster_index] = [i]           # Create a new cluster

        for j in range(pt.n_neighbours):
            pt2_index = neighbours[i, j]
            pt2 = point_info[pt2_index]
            if pt2.core_index == 0:             # If it is not a core point
                pt2.core_index = i + 1          # 1-based index
            else:
                # Check if it is a better core point
                if dist_func(data, pt2_index, pt2.core_index - 1) > dist_func(data, pt2_index, i):
                    pt2.core_index = i + 1      # Update core point

            if pt2.n_neighbours >= min_pts:     # If it is a core point
                if pt2.cluster_index == 0:      # If it is not assigned to any cluster
                    pt2.cluster_index = cluster_index + 1       # add to current cluster
                    clusters[cluster_index].append(pt2_index)
                else:                           # If it is already assigned to a cluster
                    cluster_index2 = pt2.cluster_index - 1
                    if cluster_index2 != cluster_index:         # If it is assigned to a different cluster
                        # Merge clusters
                        cluster = clusters[cluster_index2]      # Get cluster
                        for k in cluster:
                            point_info[k].cluster_index = cluster_index + 1     # Reassign cluster index
                        clusters[cluster_index].extend(cluster) # Merge clusters
                        clusters[cluster_index2] = None         # Remove merged cluster

    # cluster id mapping
    cluster_ids = [0 for i in range(len_clusters)]  # To store cluster ids (0-based index)
    num_cluters = 0
    for i in range(len_clusters):
        if clusters[i] is not None:
            cluster_ids[i] = num_cluters
            num_cluters += 1

    clusters = [[] for i in range(num_cluters + 1)]  # first one for noise points
    for i, pt in enumerate(point_info):
        if pt.core_index > 0:
            core_pt = point_info[pt.core_index - 1]
            cluster_id = cluster_ids[core_pt.cluster_index - 1]
            clusters[cluster_id + 1].append(i)  # Core points and border points
        else:
            clusters[0].append(i)  # Noise points

    return clusters

# ------------------------------------------------------------------------------------------------------
# Test with a sample dataset
# ------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    import pandas as pd
    from sklearn import datasets
    import matplotlib.pyplot as plt

    s = 50
    alpha = 0.6

    X,_ = datasets.make_moons(500, noise=0.15, random_state=1)
    df = pd.DataFrame(X, columns=['feature1', 'feature2'])
    df.plot.scatter('feature1', 'feature2', s=s, alpha=alpha, title='dataset by make_moon')
    plt.show()

    clusters = dbscan(df.values, 0.2, 20, dist2)
    print(len(clusters), 'clusters found')
    for i, cluster in enumerate(clusters):
        print(f'Cluster {i}: {len(cluster)} points')

    # plot clusters
    colors = ['k', 'r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']
    for i, cluster in enumerate(clusters):
        for idx in cluster:
            plt.scatter(df.iloc[idx, 0], df.iloc[idx, 1], c=colors[i%len(colors)], s=s, alpha=alpha)
    plt.title('DBSCAN clustering')
    plt.show()

