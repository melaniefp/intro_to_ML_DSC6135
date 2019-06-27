## HW0: Intro to Data Analysis with NumPy

This homework is not graded, and you do not need to submit your results.

However, these functions will be used in HW1, so make sure you implement them correctly.


[Jupyter notebook](hw/hw0.ipynb).

We recommend you use your personal Python installation (for exampe, by downloading [Anaconda](https://www.anaconda.com/distribution/#download-section)), or without installing any software, using [Google Colab](colab.research.google.com/).


## <a name="problem-1">Problem 1: Finding Nearest Neighbors via Euclidean Distance</a>

Suppose we have a dataset of measurements collected about related instances. The instances could be these patients in the same hospital, or birds of the same species, or emails sent to the same person. Given any new "query" instance, we'd like to find the *K* "nearest neighbor" instances in the dataset. Here, *K* is an integer that could be 1 or 3 or 500.

We'll define nearest neighbors for a query feature vector $\tilde{x}_q$ as the *K* vectors in the dataset set (denoted $x_1, \ldots x_N$) that have the smallest *Euclidean* distance to $\tilde{x}_q$.

Given a specific dataset vector $x_n$ (of size $F$) and the query vector $\tilde{x}_q$ (also of size *F*), we define the Euclidean distance as:

$$
\text{dist}_{\text{Euclidean}}(\tilde{x}_q, x_n) = \sqrt{ \sum_{f=1}^F \Big( \tilde{x}_{qf} - x_{nf} \Big) ^2 }
$$


**Edge case:** When finding the k neighbors with smallest distance, sometimes two data vectors will have exact ties (both will be the same distance from the query). When this happens, you should always return *exactly* $K$ neighbor vectors from among the tied candidates. You can resolve the tie arbitrarily (e.g. pick at random, always pick the first data vector in index order).

We'd like you to solve *multiple* k-nearest queries simultaneously. That is, given a dataset of $N$ examples (stored as an N x F 2D array) and a set of $Q$ query examples (stored as a Q x F array), you should write effective NumPy code to return a 3D array (size Q x K x F) where each row (indexed by q) contains the $K$ nearest neighbor vectors of the q-th query vector.

We've drafted a function definition and defined a detailed specification in its docstring. You need to fill in the missing code.

    #!python
    def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
        ''' Compute and return k-nearest neighbors under Euclidean distance

        Any ties in distance may be broken arbitrarily.

        Args
        ----
        data_NF : 2D array, shape = (n_examples, n_features) aka (N, F)
            Each row is a feature vector for one example in dataset
        query_QF : 2D array, shape = (n_queries, n_features) aka (Q, F)
            Each row is a feature vector whose neighbors we want to find
        K : int, positive (must be >= 1)
            Number of neighbors to find per query vector

        Returns
        -------
        neighbors_QKF : 3D array, (n_queries, n_neighbors, n_feats) (Q, K, F)
            Entry q,k is feature vector of the k-th neighbor of the q-th query
        '''
        # TODO: Write Solution
        return None

Use *exclusively* NumPy functions. No calls to any functions in sklearn or other libraries.


Remember, use *exclusively* NumPy functions. No calls to any functions in sklearn or other libraries.
