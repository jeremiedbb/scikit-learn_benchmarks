import numpy as np
import scipy.sparse as sp

from sklearn.cluster import KMeans


class KmeansBenchmarks:
    """
    Benchmarks for KMeans.
    """
    # size = (n_samples, n_clusters, n_features)
    param_names = ['params']
    params = ([('dense', 10000, 100, 3, 'full'),
               ('sparse', 10000, 100, 100, 'full'),
               ('dense', 10000, 100, 3, 'elkan')],)

    def setup(self, params):
        self.issparse = False if params[0] is 'dense' else True
        self.n_samples = params[1]
        self.n_clusters = params[2]
        self.n_features = params[3]
        self.algo = params[4]

        if self.issparse:
            self.X = sp.random(self.n_samples, self.n_features, density=0.05,
                               format='csr')
        else:
            self.X = np.random.random_sample((self.n_samples, self.n_features))

        self.kmeans = KMeans(n_clusters=self.n_clusters, algorithm=self.algo,
                             init='random', max_iter=20, n_init=1, n_jobs=1)

    def time_iterations(self, params):
        self.kmeans.fit(self.X)
