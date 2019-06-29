  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        inlineMath: [['$','$']]
      }
    });
  </script>
  <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# HW1: Practical Introduction to Regression
###### (credit: this homework has been adapted from Mike Hughes COM135 course at Tufts University)

**Due date**: Tuesday July 02 at 11:59PM.

**What to submit**:
IMPORTANT: please name the file as firstname_lastname.pdf and firstname_lastname.zip
* PDF report to upload here: <https://www.aeoncase.com/inbox/AA2YIsWADA3QpixzIReGc>
* ZIP file of source code turned in to: <https://www.aeoncase.com/inbox/AA2YIsV_5QvbnsJea54h0>

ZIP file of source code should contain:

* hw1.ipynb : Jupyter Notebook file containing your plotting code and markup
* LeastSquaresLinearRegression.py : Python file
* evaluate_perf_metrics.py : Python file
* COLLABORATORS.txt : your name, a plain text file saying who you collaborated with, and any feedback/comments you might want to add

PDF report:

* Please export your completed hw1.ipynb notebook as a PDF (easiest way is likely in your browser, just do 'Print as PDF' or similar)

**Starter code:**

Download material here:
<https://melaniefp.github.io/intro_to_ML_DSC6135/hw/HW1/hw1.zip>

Note: be sure that all plots include readable axes, labels and legends if needed.

## <a name="problem-1">Task: Predicting the Age of Abalone Sea Creatures given Physiological Measurements </a>

Your ecologist colleagues from Australia have given you a dataset of physiological measurements related to white abalone, an endangered specie [[Wikipedia article on abalone](https://simple.wikipedia.org/wiki/Abalone)]. Your colleagues are interested in monitoring abalone population health by tracking various measurements (length, weight) of these creatures, as well as their age. While the physical measurements are somewhat easy to obtain in the field, directly measuring age is a time-consuming task (cut open the shell, stain it, count the number of rings on the sheel visible through a microscope). It also requires killing the animal, which defeats the purpose of protecting the specie.

The age is known to be equal to 1.5 plus the number of rings inside the animal. You have been asked to build a *ring count* predictor for abalone, which is naturally a **regression** problem. You'll have the following input measurements for each abalone:

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

Get the data here: <https://melaniefp.github.io/intro_to_ML_DSC6135/hw/HW1/data_abalone.zip>

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

#### <a name="problem-1"> 1: Response Variable Exploration </a>

**1a:** Produce one figure with three subplots, showing histograms of the $y$ values on training, validation, and test sets. Be sure to set the bin width of the histogram to 1.0 to show the essential features of the distribution.

**1b:** Describe the **training set** distribution you see in a few sentences. Is it unimodal or multimodal? What kind of shape does it it have? Are there noticeable outliers?

**1c:** Quantify the training set's descriptive statistics. What is the mean? The median? Minimum value? Maximum value?


#### <a name="problem-2"> 2: Data Exploration for Prediction </a>

Using the training set $x$ data, consider only the **two** features 'diam_mm' and 'shucked_weight_g'.

**2a:** Create one figure with two subplots. First subplot: scatter plot of `diam_mm` vs `rings`. Second subplot: a scatter plot of `shucked_weight_g` vs `rings`.

**2b:** Describe the trends you see between diameter and rings (1-2 sentences). Could you predict rings from diameter?

**2c:** Describe the trends you see between shucked weight and rings (1-2 sentences). Could you predict rings from shucked weight?


#### <a name="problem-3"> 3: Baseline Predictions </a>

Given a training set with label values {% raw %} $$\{y_i \}_{i=1}^N$$ {% endraw %} , we can **always** consider two simple baselines for prediction that return the same constant values regardless of the input $x_i$ feature vector:

* predict-mean-of-y : $\hat{y}(x_i) = \text{mean}( y_1, y_2, \ldots y_N)$
* predict-median-of-y : $\hat{y}(x_i) = \text{median}(y_1, y_2, \ldots y_N)$

We have provided fully-function Python classes to do this prediction, `MeanPredictor` and `MedianPredictor`. They follow the template for scikit-learn regression objects discussed in class, meaning they offer `fit` and `predict` methods.

For each of MeanPredictor and MedianPredictor, you should construct a regression object instance, fit it to the training set by calling `fit`, and then obtain predictions on training, validation, and test by calling `predict`.

To evaluate predictions, you'll need to implement the most common regression performance metric, **mean squared error** (via the `calc_perf_metric__squared_error` function). You should complete this template functions within the starter code file [evaluate_perf_metrics.py]

**3a:** Make a table of the **mean-squared-error** for each of the MeanPredictor and MedianPredictor predictors when evaluated on all 3 dataset splits (training, validation, and test).


#### <a name="problem-4"> 4: Linear Regression Prediction </a>

Now, we'll try *linear regression* to predict the number of rings given physical measurements.

Look at the starter code file [LeastSquaresLinearRegression.py]. This file defines a `LeastSquaresLinearRegressor` class with the two key methods of the usual sklearn regression API: `fit` and `predict`.

You will edit this file to complete the `fit` and the `predict` methods.

**Coding Step 1/2: The `fit` method** should take in a labeled dataset {% raw %} $$\{x_n, y_n\}_{n=1}^N$$  {% endraw %} and instantiate two instance attributes:

* `w_F` : 1D array of weights, shape (n_features = F,)
* `b` : scalar bias

Nothing should be returned. You're updating the internal state of the object.

These attributes should be set using the formulas discussed in class for solving the "least squares" optimization problem (finding $w$ and $b$ values that minimize squared error on the training set).

Hint: Within a Python class, you can set an attribute like `self.b = 1.0`.

**Coding Step 2/2: The `predict` method** should take in a set of feature vectors {% raw %} $$\{x_n\}_{n=1}^N$$ {% endraw %} and produce (return) the predicted responses  {% raw %} $$\{ \hat{y}(x_n) \}_{n=1}^N$$ {% endraw %}

Recall that for linear regression, we've defined the prediction function as $\hat{y}(x_n) = w^T x_n + b$.

**Back to the Notebook**: The starter code notebook has already imported your `LeastSquaresLinearRegressor` class (you might need to restart the notebook / reimport after any edits). Construct two instances of your `LeastSquaresLinearRegressor` class, one for each of the versions of the $x$ data ("2 features" and "all 8 features").

**4a:** Apply your linear regression code to the "2 features" $x$ data, and add a column to our results table showing the **mean-squared-error** when evaluated on all 3 dataset splits (training, validation, and test).

**4b:** Apply your linear regression code to the "8 features" $x$ data, and add a column to our results table showing the **mean-squared-error** when evaluated on all 3 dataset splits (training, validation, and test).

**4c:** Does using more features seem worthwhile? Do you think the improvement on the test data is significant? Why or why not?


#### <a name="problem-5"> 5: K-Nearest-Neighbor Regression </a>

Now, we'll apply some more flexible machine learning to this problem: *K nearest neighbors* regression to the full "8 feature" version of the abalone dataset. You should use the [`KNeighborsRegressor` class](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) provided by `sklearn`.

You will explore various values for the number of neighbors $K$: 1, 3, 5, 7, 9, 11, 21, 41, 61, 81, 101, 201, 401, 801. Also include $N$, the total size of the training data.

For each value, train a `KNeighborsRegressor` object and evaluate it on the training set and the validation set.

**5a:** Make a line plot showing the trend between mean-squared-error (MSE) and $K$ on the validation set (use line style 'rs-', a solid red line with square markers). Include a line plot representing the K-NN training set MSE vs $K$ (use line style 'r:', a thin dashed red line). Include two flat lines representing linear regression (solid blue) and guess-mean baseline (solid black).

**5b:** Based on this plot, which value of $K$ would you recommend and why?

**5c:** Add a column to our results table showing the **mean-squared-error** of the selected *best* K-NN regressor from 5(ii) when evaluated on all 3 dataset splits (training, validation, and test).

#### <a name="problem-6"> 6: Analyzing Residuals (not required; bonus points possible) </a>

We'd like to understand what kinds of mistakes we're making. Using the validation set, we can scatter plot the ground truth $y$ values (x-axis) versus the predicted $\hat{y}$ values (y-axis). This can tell us if our model might be underperforming on a specific subset of data.

**6a:** Create a figure with two subplots, one showing the true labels $y$ in x-axis, and the predicted labels $\hat{y}$ in y-axis as a scatter plot for linear regression (using all features), the other showing the best $K$-NN regressor.

**6b:** Describe your conclusions from these plots. What kinds of systematic errors does each method make?
