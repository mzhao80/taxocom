import warnings
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed

from sklearn.cluster import KMeans
from sklearn.cluster._kmeans import (
    _labels_inertia_threadpool_limit,
    _validate_center_shape,
    _tolerance,
    _init_centroids,
)
from sklearn.preprocessing import normalize
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import _check_sample_weight, _num_samples

def _spherical_kmeans_single_lloyd(
    X,
    n_clusters,
    sample_weight=None,
    max_iter=300,
    init="k-means++",
    verbose=False,
    x_squared_norms=None,
    random_state=None,
    tol=1e-4,
    precompute_distances=True,
):
    """
    Modified from sklearn.cluster.k_means_.k_means_single_lloyd.
    """
    random_state = check_random_state(random_state)

    sample_weight = _check_sample_weight(sample_weight, X)

    best_labels, best_inertia, best_centers = None, None, None

    # init
    centers = _init_centroids(
        X, n_clusters, init, random_state=random_state, x_squared_norms=x_squared_norms
    )
    if verbose:
        print("Initialization complete")

    # Allocate memory to store the distances for each sample to its closer center
    distances = np.zeros(X.shape[0], dtype=X.dtype)

    for i in range(max_iter):
        centers_old = centers.copy()

        # Labels assignment using ThreadPoolExecutor
        labels, inertia = _labels_inertia_threadpool_limit(
            X, sample_weight, centers, distances=distances
        )

        # Normalize the centers
        centers = np.zeros_like(centers)
        for center_idx in range(n_clusters):
            mask = labels == center_idx
            if np.sum(mask) > 0:
                center_vals = np.sum(X[mask] * sample_weight[mask, np.newaxis], axis=0)
                centers[center_idx] = center_vals
        
        # Normalize centers to unit length
        center_norms = np.linalg.norm(centers, axis=1)
        centers[center_norms > 0] /= center_norms[center_norms > 0, np.newaxis]
        
        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift = np.sum(np.abs(centers_old - centers))
        if center_shift <= tol:
            if verbose:
                print(f"Converged at iteration {i + 1}: center shift {center_shift} within tolerance {tol}")
            break

    if center_shift > tol:
        warnings.warn(
            "Algorithm did not converge: center shift "
            f"{center_shift} > tolerance {tol}.",
            ConvergenceWarning,
            stacklevel=2,
        )

    return best_labels, best_inertia, best_centers, i + 1


def spherical_k_means(
    X,
    n_clusters,
    sample_weight=None,
    init="k-means++",
    n_init=10,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    random_state=None,
    copy_x=True,
    n_jobs=1,
    algorithm="auto",
    return_n_iter=False,
):
    """Modified from sklearn.cluster.k_means_.k_means."""
    if n_init <= 0:
        raise ValueError(f"n_init={n_init} must be > 0.")
    
    X = check_array(
        X, accept_sparse="csr", dtype=[np.float64, np.float32], copy=copy_x
    )
    tol = _tolerance(X, tol)
    
    # Validate init array
    init = _init_centroids(X, n_clusters, init, random_state=random_state)
    
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)
    
    # parallelization of k-means runs
    seed = random_state.randint(np.iinfo(np.int32).max + 1) if random_state else None
    
    if n_jobs == 1:
        # Special case: single thread, don't override threading backend
        results = [
            _spherical_kmeans_single_lloyd(
                X,
                n_clusters,
                sample_weight=sample_weight,
                init=init,
                max_iter=max_iter,
                verbose=verbose,
                tol=tol,
                random_state=seed,
            )
            for _ in range(n_init)
        ]
    else:
        # Avoid using too many threads for each run
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_spherical_kmeans_single_lloyd)(
                X,
                n_clusters,
                sample_weight=sample_weight,
                init=init,
                max_iter=max_iter,
                verbose=verbose,
                tol=tol,
                random_state=seed,
            )
            for _ in range(n_init)
        )
    
    # Get results with the lowest inertia
    labels, inertia, centers, n_iters = zip(*results)
    best = np.argmin(inertia)
    best_labels = labels[best]
    best_inertia = inertia[best]
    best_centers = centers[best]
    best_n_iter = n_iters[best]
    
    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter
    else:
        return best_centers, best_labels, best_inertia


class SphericalKMeans:
    """Spherical K-Means clustering

    Modification of sklearn.cluster.KMeans where cluster centers are normalized
    (projected onto the sphere) in each iteration.

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence.

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.

    verbose : int, default=0
        Verbosity mode.

    copy_x : boolean, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True, then the original data is not
        modified. If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    normalize : boolean, default=True
        Normalize the input to have unit norm.

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.

    labels_ : array, [n_samples]
        Labels of each point.

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    n_iter_ : int
        Number of iterations run.
    """

    def __init__(
        self,
        n_clusters=8,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        n_jobs=1,
        verbose=0,
        random_state=None,
        copy_x=True,
        normalize=True,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.normalize = normalize

    def fit(self, X, y=None, sample_weight=None):
        """Compute spherical k-means clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)

        if self.normalize:
            X = normalize(X)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = spherical_k_means(
            X,
            n_clusters=self.n_clusters,
            sample_weight=sample_weight,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            verbose=self.verbose,
            tol=self.tol,
            random_state=random_state,
            copy_x=self.copy_x,
            n_jobs=self.n_jobs,
            return_n_iter=True,
        )

        return self
