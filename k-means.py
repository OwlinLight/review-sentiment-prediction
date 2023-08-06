import numpy as np

def main():
    X = np.random.randn(4, 2)
    X = np.floor(X)
    k = 2
    k_means(X, k)


def k_means(X, k, centers=None, num_iter=100):
    if centers is None:
        rnd_centers_idx = np.random.choice(np.arange(X.shape[0]), k, replace=False)
        centers = X[rnd_centers_idx]
    for _ in range(num_iter):
        distances = np.sum(np.sqrt((X - centers[:, np.newaxis]) ** 2), axis=-1)
        distances_ = np.linalg.norm(X - centers[:, np.newaxis], axis=-1)
        cluster_assignments = np.argmin(distances, axis=0)
        for i in range(k):
            msk = (cluster_assignments == i)
            centers[i] = np.mean(X[msk], axis=0) if np.any(msk) else centers[i]
    return cluster_assignments, centers

if __name__ == '__main__':
    main()