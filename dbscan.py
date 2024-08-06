
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
        first_in_cluster = 1-based index of the first point in the cluster
        next_in_cluster = 1-based index of the next point in the cluster
    '''
    def __init__(self):
        self.n_neighbours = 0       # Number of neighbours, including itself
        self.core_index = 0         # 0 for noise points, 1-based index of its core point
        self.cluster_index = 0      # 0 for noise points and border points, 1-based index of its cluster
        self.first_in_cluster = 0   # first point in cluster, 1-based index
        self.next_in_cluster = 0;   # next point in cluster, 1-based index

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

    for i in core_pt_indices:
        # for each core point
        pt = point_info[i]
        if pt.first_in_cluster > 0:         # If it is already assigned to a cluster
            first_in_cluster = pt.first_in_cluster - 1
        else:
            first_in_cluster = i
            pt.first_in_cluster = i + 1     # 1-based index (itself)
            pt.next_in_cluster = 0          # 0 for the last point in the cluster

        for j in range(pt.n_neighbours):
            # for each neighbour
            pt2_index = neighbours[i, j]
            pt2 = point_info[pt2_index]

            if pt2.n_neighbours >= min_pts:     # If it is a core point
                pt2.core_index = i + 1          # Set core point
                if pt2.first_in_cluster > 0:    # If it is already assigned to a cluster
                    first_in_cluster2 = pt2.first_in_cluster - 1
                    if first_in_cluster2 != first_in_cluster:   # If it is assigned to a different cluster
                        # Merge clusters : insert cluster after the first point of cluster2
                        first_pt = point_info[first_in_cluster]
                        p = first_pt
                        while True:
                            p.first_in_cluster = first_in_cluster2 + 1
                            if p.next_in_cluster == 0:
                                break
                            p = point_info[p.next_in_cluster - 1]
                        first_pt2 = point_info[first_in_cluster2]
                        p.next_in_cluster = first_pt2.next_in_cluster
                        first_pt2.next_in_cluster = first_in_cluster + 1
                        first_in_cluster = first_in_cluster2
                else:
                    # Add to the cluster: insert after the first point in the cluster
                    first_pt = point_info[first_in_cluster]
                    pt2.first_in_cluster = first_in_cluster + 1
                    pt2.next_in_cluster = first_pt.next_in_cluster
                    first_pt.next_in_cluster = pt2_index + 1

            else:      # it is a border point
                if pt2.core_index == 0:
                    pt2.core_index = i + 1
                else:
                    if dist_func(data, pt2_index, pt2.core_index - 1) > dist_func(data, pt2_index, i):
                        pt2.core_index = i + 1

    num_clusters = 0
    for i in core_pt_indices:
        pt = point_info[i]
        if pt.first_in_cluster == i + 1:    # If it is the first point in the cluster
            num_clusters += 1
            pt.cluster_index = num_clusters

    clusters = [[] for i in range(num_clusters + 1)]  # first one for noise points
    for i, pt in enumerate(point_info):
        if pt.core_index > 0:
            core_pt = point_info[pt.core_index - 1]
            assert core_pt.core_index > 0
            first_pt = point_info[core_pt.first_in_cluster - 1]
            cluster_id = first_pt.cluster_index
            assert cluster_id > 0
            clusters[cluster_id].append(i)
        else:
            clusters[0].append(i)   # noise points

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

    X,_ = datasets.make_moons(500, noise=0.2, random_state=1)
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

