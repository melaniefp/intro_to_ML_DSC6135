title: HW0: Intro to Data Analysis with NumPy
url: hw0.html
save_as: hw0.html
show_last_modified_date: True

**Due date**: Wed. Jan 23 at 11:59PM EST.

**Turn-in Link**: <https://www.gradescope.com/courses/33142/assignments/140409/submissions>

**Files to Turn In:**

ZIP file of source code, containing:

* hw0.py : Python file with your solution code
* COLLABORATORS.txt : a plain-text file [[example](https://github.com/tufts-ml-courses/comp135-19s-assignments/tree/master/hw0/COLLABORATORS.txt)]

* * Your full name
* * Number of hours you spent on the problem
* * Names of any people you talked to for help (TAs, fellow students, etc.). If none, write "No external help".
* * Brief description of what content you sought help about (1-3 sentences)

**Evaluation Rubric:**

* Collaboration statement in txt file: 5 points possible
* Problem 1 code: 10 points possible (number of autograder tests passed)
* Problem 2 code: 10 points possible (number of autograder tests passed)

** Prerequisites: **

Make sure you've followed the COMP 135-specific [Python Setup Instructions](python_setup.html), and activated your `comp135_env` environment.

** Starter code:**

See the hw0 folder of the public assignments repo for this class:

<https://github.com/tufts-ml-courses/comp135-19s-assignments/tree/master/hw0>


**Jump to**: [Problem 1](#problem-1) &nbsp; [Problem 2](#problem-2)


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


## <a name="problem-2">Problem 2: Splitting Datasets into Training and Testing</a>

A common task in ML is to divide a dataset of independent instances into a "training" set and a "test" set. These two sets can then be used to measure how well an ML method *generalizes* to data it hasn't seen before: we fit the method to the training set, then evaluate the trained method on the test set.

In this problem, you'll demonstrate basic understanding of NumPy array indexing and random number generation by writing a procedure to divide an input dataset of $L$ instances into a training set (of size $M$) and a test set (of size $N$). Each row of the original dataset should be exclusively assigned to either train or test. 

How do we set the values of M and N? Your function will take a keyword argument **frac_test** that specifies the number of test examples (N) as a *fraction* of the overall dataset size. To compute $N$, we always want to round up to the nearest whole number: $N = \text{ceil}(\textit{frac_test} * L)$

We want the test set to be a *uniform at random* subset. You should look at the [NumPy API for Random Sampling](https://docs.scipy.org/doc/numpy-1.15.1/reference/routines.random.html). Functions like `shuffle` or `permutation` might be helpful. 

We also want the test set to be *reproducible* by specifying particular random seed. That is, if I run the code now to extract a train/test set, if I need to rerun the code later I'd like to be able to recover the *exact same* train/test assignments if needed. With NumPy, the common way to do this is by specifying a `random_state` keyword argument that can either take an integer seed (0, 42, 1337, etc.) or an instance of the `RandomState` class. Specifying the same random_state should deliver the *same* pseudo-randomness across multiple calls to a function. 

We've drafted a function definition and defined a detailed specification in its docstring. You need to fill in the missing code. 

    #!python
    def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
        ''' Divide provided array into train and test set along first dimension

        User can provide a random number generator object to ensure reproducibility.

        Args
        ----
        x_all_LF : 2D array, shape = (n_total_examples, n_features) (L, F)
            Each row is a feature vector
        frac_test : float, fraction between 0 and 1
            Indicates fraction of all L examples to allocate to the "test" set
        random_state : np.random.RandomState instance or integer or None
            If int, code will create RandomState instance with provided value as seed
            If None, defaults to the current numpy random number generator np.random

        Returns
        -------
        x_train_MF : 2D array, shape = (n_train_examples, n_features) (M, F)
            Each row is a feature vector

        x_test_NF : 2D array, shape = (n_test_examples, n_features) (N, F)
            Each row is a feature vector

        Post Condition
        --------------
        This function should be side-effect free. The provided input array x_all_LF
        should not change at all (not be shuffled, etc.)

        Examples
        --------
        >>> x_LF = np.eye(10)
        >>> xcopy_LF = x_LF.copy() # preserve what input was before the call
        >>> train_MF, test_NF = split_into_train_and_test(
        ...     x_LF, frac_test=0.3, random_state=np.random.RandomState(0))
        >>> train_MF.shape
        (7, 10)
        >>> test_NF.shape
        (3, 10)
        >>> print(train_MF)
        [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
         [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
        >>> print(test_NF)
        [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
         [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]

        ## Verify that input array did not change due to function call
        >>> np.allclose(x_LF, xcopy_LF)
        True

        References
        ----------
        For more about RandomState, see:
        https://stackoverflow.com/questions/28064634/random-state-pseudo-random-numberin-scikit-learn
        '''
        if random_state is None:
            random_state = np.random
        ## TODO fixme
        return None, None

Remember, use *exclusively* NumPy functions. No calls to any functions in sklearn or other libraries.
