title: HW1: Practical Introduction to Regression
url: hw1.html
save_as: hw1.html
show_last_modified_date: True

**Due date**: Wed. Jan 30 at 11:59PM EST.

**Turn-in links**:

* PDF report turned in to: <https://www.gradescope.com/courses/33142/assignments/147357/>
* ZIP file of source code turned in to: <https://www.gradescope.com/courses/33142/assignments/143294/>

**Files to Turn In:**

ZIP file of source code should contain:

* hw1.ipynb : Jupyter Notebook file containing your plotting code and markup
* LeastSquaresLinearRegression.py : Python file
* evaluate_perf_metrics.py : Python file
* COLLABORATORS.txt : a plain text file [[example](https://github.com/tufts-ml-courses/comp135-19s-assignments/tree/master/hw1/COLLABORATORS.txt)], containing
* * Your full name
* * Estimate the hours you spent on each of Problem 1, Problem 2, and Problem 3
* * Names of any people you talked to for help (TAs, students, etc.). If none, write "No external help".
* * Brief description of what content you sought help about (1-3 sentences)

PDF report:

* Please export your completed hw1.ipynb notebook as a PDF (easiest way is likely in your browser, just do 'Print as PDF' or similar)
* This document will be manually graded

**Evaluation Rubric:**

See the PDF submission portal on Gradescope for the point values of each problem. Generally, tasks with more coding/effort will earn more potential points.

** Starter code:**

See the hw1 folder of the public assignments repo for this class:

<https://github.com/tufts-ml-courses/comp135-19s-assignments/tree/master/hw1>

**Jump to**: [Problem 1](#problem-1) &nbsp; [Problem 2](#problem-2)  &nbsp; [Problem 3](#problem-3)

## Best practices

Across all the problems here, be sure that all plots include readable axes labels and legends if needed.

## <a name="problem-1">Problem 1: Predicting the Age of Abalone Sea Creatures given Physiological Measurements </a>

Your ecologist colleagues from Australia have given you a dataset of physiological measurements related to abalone, an abundant shellfish [[Wikipedia article on abalone](https://simple.wikipedia.org/wiki/Abalone)]. Your colleagues are interested in monitoring abalone population health by tracking various measurements (length, weight) of these creatures, as well as their age. While the physical measurements are somewhat easy to obtain in the field, directly measuring age is a boring and time-consuming task (cut open the shell, stain it, count the number of rings on the sheel visible through a microscope). The age is known to be equal to 1.5 plus the number of rings.

You have been asked to build an *ring count* predictor for abalone, which is naturally a **regression** problem. You'll have the following input measurements for each abalone:

| column name      | type    | unit | description |
| ---------------- | ------- | ---- | ----------- |
| is_male          | binary  |      | 1 = 'male', 0 = 'female'
| length_mm        | numeric | mm   | longest shell measurement
| diam_mm          | numeric | mm   | diameter of shell, perpendicular to length
| height_mm        | numeric | mm   | height of shell (with meat inside)
| whole_weight_g   | numeric | gram | entire creature weight (shell + guts + meat)
| shucked_weight_g | numeric | gram | weight of the meat
| viscera_weight_g | numeric | gram | weight of the guts (after bleeding)
| shell_weight_g   | numeric | gram | weight of shell alone (after drying)

<br />

If you like, you can browse the web to see [visually what meat, guts, and shells look like](
https://www.thespruceeats.com/how-to-clean-abalone-2216416).

In the starter code, we have provided an existing train/validation/test split of this dataset, stored on-disk in comma-separated-value (CSV) files: x_train.csv, y_train.csv, x_valid.csv, y_valid.csv, x_test.csv, and y_test.csv.

Get the data here: <https://github.com/tufts-ml-courses/comp135-19s-assignments/tree/master/hw1/data_abalone>

**Loading $y$**: You'll want to load in the $y$ data into separate NumPy arrays for training, validation, and test. You can read from CSV files into arrays via the following:

````
    y_tr = np.loadtxt('path/to/y_train.csv', delimiter=',', skiprows=1)
    y_va = np.loadtxt('path/to/y_valid.csv', delimiter=',', skiprows=1)
    y_te = np.loadtxt('path/to/y_test.csv', delimiter=',', skiprows=1)
````

**Loading $x$**: You'll want to load in two versions of the $x$ data:

* all 8 features  : all columns in `x_train.csv`, `x_valid.csv`, etc.
* only 2 features : only the columns `diam_mm` and `shucked_weight_g`

Probably it's easiest to load the full dataset, then make a copy that has only the two relevant columns.

#### <a name="problem-1-a"> 1a: Response Variable Exploration </a>


**1a(i):** Produce one figure with three subplots, showing histograms of the $y$ values on training, validation, and test sets. Be sure to set the bin width of the histogram to 1.0 to show the essential features of the distribution.

**1a(ii):** Describe the **training set** distribution you see in a few sentences. Is it unimodal or multimodal? What kind of shape does it it have? Are there noticeable outliers?

**1a(iii):** Quantify the training set's descriptive statistics. What is the mean? The median? Minimum value? Maximum value?


#### <a name="problem-1-b"> 1b: Data Exploration for Prediction </a>

Using the training set $x$ data, consider only the **two** features 'diam_mm' and 'shucked_weight_g'.

**1b(i):** Create one figure with two subplots. First subplot: scatter plot of `diam_mm` vs `rings`. Second subplot: a scatter plot of `shucked_weight_g` vs `rings`.

**1b(ii):** Describe the trends you between diameter and rings (1-2 sentences). Could you predict rings from diameter?

**1b(iii):** Describe the trends you see between shucked weight and rings (1-2 sentences). Could you predict rings from shucked weight?


#### <a name="problem-1-c"> 1c: Baseline Predictions </a>

Given a training set of values $\{y_n \}_{n=1}^N$, we can **always** consider two simple baselines for prediction that return the same constant values regardless of the input $x_i$ feature vector:

* predict-mean-of-y : $\hat{y}(x_i) = \text{mean}( y_1, y_2, \ldots y_N)$
* predict-median-of-y : $\hat{y}(x_i) = \text{median}(y_1, y_2, \ldots y_N)$

We have provided fully-function Python classes to do this prediction, `MeanPredictor` and `MedianPredictor`. They follow the template for scikit-learn regression objects discussed in class, meaning they offer `fit` and `predict` methods.

For each of MeanPredictor and MedianPredictor, you should construct an regression object instance, fit it to the training set by calling `fit`, and then obtain predictions on training, validation, and test by calling `predict`.

To evaluate predictions, you'll need to implement the most common regression performance metric, **mean squared error** (via the `calc_perf_metric__squared_error` function). You should complete this template functions within the starter code file [evaluate_perf_metrics.py](https://github.com/tufts-ml-courses/comp135-19s-assignments/blob/master/hw1/evaluate_perf_metrics.py)

**1c(i):** Make a table of the **mean-squared-error** for each of the MeanPredictor and MedianPredictor predictors when evaluated on all 3 dataset splits (training, validation, and test).


#### <a name="problem-1-d"> 1d: Linear Regression Prediction </a>

Now, we'll try *linear regression* to predict the number of rings given physical measurements.

Look at the starter code file [LeastSquaresLinearRegression.py](https://github.com/tufts-ml-courses/comp135-19s-assignments/blob/master/hw1/LeastSquaresLinearRegression.py). This file defines a `LeastSquaresLinearRegressor` class with the two key methods of the usual sklearn regression API: `fit` and `predict`.

You will edit this file to complete the `fit` and the `predict` methods.

**Coding Step 1/2: The `fit` method** should take in a labeled dataset $\{x_n, y_n\}_{n=1}^N$ and instantiate two instance attributes:

* `w_F` : 1D array of weights, shape (n_features = F,)
* `b` : scalar bias

Nothing should be returned. You're updating the internal state of the object.

These attributes should be set using the formulas discussed in class for solving the "least squares" optimization problem (finding $w$ and $b$ values that minimize squared error on the training set).

Hint: Within a Python class, you can set an attribute like `self.b = 1.0`. 

**Coding Step 2/2: The `predict` method** should take in a set of feature vectors $\{x_n\}_{n=1}^N$ and produce (return) the predicted responses $\{ \hat{y}(x_n) \}_{n=1}^N$

Recall that for linear regression, we've defined the prediction function as $\hat{y}(x_n) = w^T x_n + b$.

**Back to the Notebook**: The starter code notebook has already imported your `LeastSquaresLinearRegressor` class (you might need to restart the notebook / reimport after any edits). Construct two instances of your `LeastSquaresLinearRegressor` class, one for each of the versions of the $x$ data ("2 features" and "all 8 features").

**1d(i):** Apply your linear regression code to the "2 features" $x$ data, and add a column to our results table showing the **mean-squared-error** when evaluated on all 3 dataset splits (training, validation, and test).

**1d(ii):** Apply your linear regression code to the "8 features" $x$ data, and add a column to our results table showing the **mean-squared-error** when evaluated on all 3 dataset splits (training, validation, and test).

**1d(iii):** Does using more features seem worthwhile? Do you think the improvement on the test data is significant? Why or why not?


#### <a name="problem-1-e"> 1e: K-Nearest-Neighbor Regression </a>

Now, we'll apply some more flexible machine learning to this problem: *K nearest neighbors* regression to the full "8 feature" version of the abalone dataset. You should use the [`KNeighborsRegressor` class](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) provided by `sklearn`.

You will explore various values for the number of neighbors $K$: 1, 3, 5, 7, 9, 11, 21, 41, 61, 81, 101, 201, 401, 801. Also include $N$, the total size of the training data.

For each value, train a `KNeighborsRegressor` object and evaluate it on the training set and the validation set. 

**1e(i):** Make a line plot showing the trend between mean-squared-error (MSE) and $K$ on the validation set (use line style 'rs-', a solid red line with square markers). Include a line plot representing the K-NN training set MSE vs $K$ (use line style 'r:', a thin dashed red line). Include two flat lines representing linear regression (solid blue) and guess-mean baseline (solid black).

**1e(ii):** Based on this plot, which value of $K$ would you recommend and why?

**1e(iii):** Add a column to our results table showing the **mean-squared-error** of the selected *best* K-NN regressor from 1e(ii) when evaluated on all 3 dataset splits (training, validation, and test).

#### <a name="problem-1-f"> 1f: Analyzing Residuals (not required; bonus points possible) </a>

We'd like to understand what kinds of mistakes we're making. Using the validation set, we can scatter plot the ground truth $y$ values (x-axis) versus the predicted $\hat{y}$ values (y-axis). This can tell us if our model might be underperforming on a specific subset of data.

**1f(i):** Create a figure with two subplots, one showing the $y$-vs-$\hat{y}$ scatter plot for linear regression (using all features), the other showing the best $K$-NN regressor. 

**1f(ii):** Describe your conclusions from these plots. What kinds of systematic errors does each method make?



## <a name="problem-2">Problem 2: Predicting the Number of Doctor Visits for Elderly People </a>

Your public health colleagues are keen to estimate how often elderly populations might visit their primary-care physician. They'd like to have an individual-subject-level predictor that can take in basic information about a subject and predict how many total doctor visits they'll have in a year. Given such a predictor, you could improve estimates that help decide the allocation of resources (budgeting dollars, number of doctors, etc) for a local clinic. 

*Important note*: For this analysis, your stakeholders care about your choice of evaluation metric. They say that this is mostly a budgeting problem, and thus if they make a mistake in resource allocation, costs accrue linearly (e.g. if they schedule 2 extra visits than necessary for the year, the cost is 2x the cost of 1 extra doctor).

This *doctor visits per year* predictor is naturally a **regression** problem. You'll have the following input measurements for each subject:

| column name           | type    | unit   | description |
| --------------------- | ------- | ------ | ----------- |
| age_in_decades        | numeric | decade | 
| health_excellent      | binary  |        | 1 if health is rated 'excellent', 0 otherwise
| health_poor           | binary  |        | 1 if health is rated 'poor', 0 otherwise
| n_chronic_cond        | numeric |        | count of subject's chronic health conditions (such as asthma, heart disease, obesity, etc.)
| n_years_schooling     | numeric | year   | total years of schooling
| income_in_10k_dollars | numeric | dollars | annual income for household/family (measured in $10,000s of dollars)
| limited_daily_activity| binary  |        | 1 if health limits daily activities like walking, 0 otherwise
| employed              | binary  |        | 1 if has employed, 0 o.w.
| has_private_insurance | binary  |        | 1 if on private health insurance policy, 0 o.w.
| has_medicaid          | binary  |        | 1 if on Medicaid (US govt insurance for low-income), 0 o.w.


<br />

In the starter code, we have provided an existing train/validation/test split of this dataset, stored on-disk in comma-separated-value (CSV) files: x_train.csv, y_train.csv, x_valid.csv, y_valid.csv, etc.

Get the DoctorVisits data here: https://github.com/tufts-ml-courses/comp135-19s-assignments/tree/master/hw1/data_doctorvisits

#### <a name="problem-2-a"> 2a: Baseline Predictions for DoctorVisits </a>

To evaluate predictions, you'll need to implement another common regression performance metric, **mean absolute error** (via the `calc_perf_metric__absolute_error` function). You should complete the template function within the starter code file [evaluate_perf_metrics.py](https://github.com/tufts-ml-courses/comp135-19s-assignments/blob/master/hw1/evaluate_perf_metrics.py)

**2a(i):** Given your stakeholder's preferences (defined above), which error metric is most appropriate for this problem, and why?

**2a(ii):** Make a table of the **mean absolute error** for each of MeanPredictor and MedianPredictor when evaluated on all 3 dataset splits (training, validation, and test).

#### <a name="problem-2-b"> 2b: Linear Regression for DoctorVisits </a>

Now, we'll try your *linear regression* implementation on this dataset. Construct an instance of your `LeastSquaresLinearRegressor` class, and fit it to two versions of the dataset:

* "2 features", which uses only age and number of chronic conditions
* "all features", which uses all 10 features

**2b(i):** Apply your linear regression code to the "2 features" $x$ data, and add a column to our results table showing the **mean absolute error** when evaluated on all 3 dataset splits (training, validation, and test).

**2b(ii):** Apply your linear regression code to the "all features" $x$ data, and add a column to our results table showing the **mean absolute error** when evaluated on all 3 dataset splits (training, validation, and test).

**2b(iii):** Does using more features seem worthwhile? Do you think the improvement on the test data is significant? Why or why not?


#### <a name="problem-2-c"> 2c: Decision Tree Regression for DoctorVisits</a>

Now, we'll apply some more flexible machine learning to this problem: decision tree regression. You should use the `DecisionTreeRegressor` class provided by sklearn. 

You should use the "all features" version of the dataset throughout **2c**. You will explore various values for the parameter `min_samples_leaf`, which as discussed in class controls the complexity of the learned decision tree.

You should try the following values for `min_samples_leaf`: 1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000, $N$, where $N$ is the size of the training set.

For each value of `min_samples_leaf`, train a `DecisionTreeRegressor` object as follows:

    #!python
    tree_regr = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, random_state=42)
    tree_regr.fit(x_tr_NF, y_tr_N)

FYI: providing an explicit random state is a good practice for any algorithm that might break ties at random. Then, you can always reproduce any results from earlier experiments.

Given a "trained" tree regression object, please evaluate its mean absolute error performance on the training, validation, and test sets.

**2c(i):** Make a line plot showing the trend between the validation set mean-absolute-error (MAE) on the y-axis and min_samples_leaf on the x-axis (use line style 'rs-', a solid red line with square markers). Include a line plot for training set MAE vs min_samples_leaf (use line style 'r:', a thin dashed red line). Include two flat lines representing linear regression (solid blue) and the guess-median baseline (solid black).

**2c(ii):** Based on the plot from **2c(i)**, which value of min_samples_leaf would you recommend and why?

**2c(iii):** Add a column to our results table showing the selected *best* decision tree regressor from **2c(ii)** when evaluated on all 3 dataset splits (training, validation, and test).


#### <a name="problem-2-d"> 2d: Decision Tree Regression with MAE criterion </a>

Repeat steps (i) - (iii) of 2c, but use the following constructor instead

    #!python
    tree_regr = DecisionTreeRegressor(criterion='mae', min_samples_leaf=min_samples_leaf, random_state=42)
    tree_regr.fit(x_tr_NF, y_tr_N)

**2d(i):** Repeat 2c(i) but with setting `criterion='mae'` in constructor

**2d(ii):** Repeat 2c(ii) but with setting `criterion='mae'` in constructor

**2d(iii):** Repeat 2c(iii) but with setting `criterion='mae'` in constructor

**2d(iv):** Read the sklearn documentation for [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor). Why is there a noticeable change in performance between 2d and 2c? What was the default criterion used in 2c? What makes the approach in 2d better for this task?



## <a name="problem-3"> 3: Regression Concept Questions </a>

#### 3a: Limits of $K$-NN

**Question**: When $K$ equals the total training set size $N$, the $K$-nearest-neighbor regression algorithm approaches the behavior of which other regression method discussed here? Why?

#### 3b: Modifications of $K$-NN

**Question**: Suppose in problem 2, when trying to minimize *mean absolute error* on heldout data, that instead of a DecisionTreeRegressor, we had used a $K$-NN regressor with Euclidean distance (as in <a href="#problem-1-e">Problem **1e**</a>).  

Would we expect $K$-NN with large $K$ to always beat the strongest constant-prediction baseline (e.g. guess-median or guess-mean)? To get better MAE performance using a nearest-neighbor like approach, should we change the distance function used to compute neighbors? Or would we need to change some other step of the $K$-NN prediction process?

#### <a name='problem-3c'> 3c: Linear Regression with Categorical Features </a>

**Question:** Your colleague trains a linear regression model on a subset of the DoctorVisits data using only the `has_medicaid` and `has_private_insurance` features. Thus, all features in the vector have a binary categorical type and can be represented via a redundant one-hot encoding. 

To your dismay, you discover that your colleague failed to include a bias term (aka intercept term) when training the model. You recall from class that including a bias term can be important.

To be concrete, you wish that predictions were made via $\hat{y}(x_i) = w^T x_i +b$, where each feature vector $x_i$ was represented as a length-two vector:
$$
x_i = [
    \texttt{has_medicaid}
    \quad \texttt{has_private_insurance}
] \quad \quad \quad ~
$$

However, your colleague made predictions $\hat{y}(\tilde{x}_i) = \tilde{x}_i^T \tilde{w}$ and used the following feature vector representation:
$$
\tilde{x}_i = [
    \texttt{has_medicaid}
    \quad \texttt{not(has_medicaid)}
    \quad \texttt{has_private_insurance}
    \quad \texttt{not(has_private_insurance)} 
]
$$

Your colleague has delivered to you a length-4 feature vector $\tilde{w}$ for the 4 features above, but then left for vacation without giving you access to the training data.

Can you manipulate the $\tilde{w}$ vector to estimate an appropriate $w$ and $b$ such that for all possible inputs $x_i$:

$$
    w^T x_i + b = \tilde{w}^T \tilde{x}_i
$$
