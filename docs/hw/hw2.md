  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        inlineMath: [['$','$']]
      }
    });
  </script>
  <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# HW2: Cross-Validation and Regularization
###### (credit: this homework has been adapted from Mike Hughes COM135 course at Tufts University)

**Due date**: Fr. July 5 at 11:59PM.

**Turn-in links**:

IMPORTANT: please name the file as firstname_lastname.pdf and firstname_lastname.zip
* PDF report turned in to: <https://www.aeoncase.com/inbox/AA2YIsfVJA7lxyXJznA4c>
* ZIP file of source code turned in to: <https://www.aeoncase.com/inbox/AA2YIsfUsAQeq79ZvOmy8>

**Files to Turn In:**

ZIP file of source code should contain:

* hw2.ipynb : Jupyter Notebook file containing your code and markup
* COLLABORATORS.txt : a plain text file with your full name and collaborators

PDF report:

* Please export your completed hw2.ipynb notebook as a PDF (easiest way is likely in your browser, just do 'Print as PDF' or similar)
* This document will be manually graded

** Starter code and Data:**

Download material here:
<https://melaniefp.github.io/intro_to_ML_DSC6135/hw/HW2/HW2.zip>

Note: be sure that all plots include readable axes, labels and legends if needed, when multiple lines are shown.

## <a name="problem-1">Polynomial Basis Model Selection: From Validation Sets to Cross-Validation </a>

Car pollution contributes significantly to global warming. You have been given a data set containing gas mileage, horsepower, and other information for 395 makes and models of vehicles.  For each vehicle, we have the following information:

| column name      | type    | unit | description |
| ---------------- | ------- | ---- | ----------- |
| horsepower       | numeric | hp   | engine horsepower
| weight           | numeric | lb.  | vehicle weight
| cylinders        | numeric | #    | number of engine cylinders, from 4 to 8
| displacement     | numeric | cu. inches | engine displacement
| mpg              | numeric | mi. / gal | vehicle miles per gallon

You have been asked to build a predictor for vehicle mileage (mpg) as a function of other vehicle characteristics.

In the starter code, we have provided an existing train/validation/test split of this dataset, stored on-disk in comma-separated-value (CSV) files: x_train.csv, y_train.csv, x_valid.csv, y_valid.csv, x_test.csv, and y_test.csv.

We will train linear regression models that minimize mean-squared error.

For this task, using higher-order polynomial transformations of the input features result in improved predictive performance.

#### <a name="problem-1-a"> 1: Polynomial Degree Selection on Fixed Validation Set </a>

Using `sklearn`, we can perform a polynomial transform via the `PolynomialFeatures` class.

For example, to make a degree 2 transformation of an input feature array `x_tr_MF` with $M$ rows and $F$ columns, we have:

```
    >>> poly_transformer = sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)
    >>> x_tr_MG = poly_transformer.fit_transform(x_tr_MF)
```

This creates a new set of $G$ features, expanding the original $F$ features to include all possible combinations of degree 2 or less. `sklearn` provides an easy way to see the names of the new features:

```
    >>> poly_transformer.get_feature_names(['horsepower', 'weight', 'cylinders', 'displacement'])

    ['horsepower',
     'weight',
     'cylinders',
     'displacement',
     'horsepower^2',
     'horsepower weight',
     'horsepower cylinders',
     'horsepower displacement',
     'weight^2',
     'weight cylinders',
     'weight displacement',
     'cylinders^2',
     'cylinders displacement',
     'displacement^2']
```

**1a:** For this dataset, where the feature size is $F=4$, make a plot of the total number of polynomial features $G$ when the degree is in [1, 2, 3, 4, 5, 6, 7, 8]. What kind of trend to you observe?

**1b:** Fit a linear regression model to a polynomial feature transformation of the provided training set of $x$, $y$ values at each of these possible degrees: [1, 2, 3, 4, 5, 6]. Make a line plot of mean-squared error vs. polynomial degree on the training set (use style 'b:', a dotted blue line) and the validation set (use style 'rs-', a solid red line with square markers). Set the y-axis limits between [0,70]. Your code should chain together the `PolynomialFeatures` and `LinearRegression` class provided by `sklearn`.

**1c:** Based on this plot, which single degree value do you recommend? Why?

**1d:** Report the numerical values of the 5th percentile and 95th percentile of the coefficients (parameters) of your linear regression model for degrees 3, 4, 5, and 6. What seems to be happening?

**1e:** Comment on the training error observed at degree 6. Based on your plots from **Q1** and your knowledge of linear regression, what **should** the training error be at degree 6? What do you think is happening instead?


#### <a name="problem-1-b"> 2: Rescaling Features </a>

Although all our original feature are *positive* values, they all have different numerical scales and ranges. A typical weight value is in the 1000s. A typical cylinder value is 4-8. Especially when we take high-order polynomial combinations of features, we are likely to see extreme values that perhaps lead to problems.

To counteract this, we will *rescale* our numerical features in $x$ to be between 0.0 and 1.0. We can use `sklearn`'s convenient `MinMaxScaler` preprocessing tool to do this.

For best results, we want to apply rescaling *twice*, once to our original features, and once to the transformed polynomial features. This will ensure that the features fed *into* the polynomial featurizer have consistent scale, and the result of the polynomial featurizer also has consistent scale.

Tracking all these steps can be cumbersome, so we suggest a [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). Inside the pipeline, we specify a series of steps to apply in order to an original dataset. After creating a pipeline, we can treat it like an encapsulated regression object, which has a `fit` method and a `predict` method. We can call `fit` to train the pipeline, and `predict` to apply it to new data.

```
    pipeline = sklearn.pipeline.Pipeline(
        steps=[
         ('rescaler', sklearn.preprocessing.MinMaxScaler()),
         ('poly_transformer', sklearn.preprocessing.PolynomialFeatures(degree=degree, include_bias=False)),
         ('poly_rescaler', sklearn.preprocessing.MinMaxScaler()),
         ('linear_regr', sklearn.linear_model.LinearRegression()),
        ])
```    

**2a:** Fit a linear regression model to a *rescaled* polynomial feature transformation of the provided training set of $x$, $y$ values at each of these possible degrees: [1, 2, 3, 4, 5, 6]. Make a line plot of mean-squared error vs. polynomial degree on the training set (use style 'b:', a dotted blue line) and the validation set (use style 'rs-', a solid red line with square markers). Set the y-axis limits between [0,70].

**2b:** Using this new analysis, which degree do you recommend? 

**2c:** Report the numerical values of the 5th percentile and 95th percentile of the coefficients observed in this most recent linear regression model for degrees 3, 4, 5, and 6. What seems to be happening? What's different than in **Q1**?

**2d:** Comment on the training error observed at degree 6. Is this what we would expect? Why is this different than Question **Q1**?


#### <a name="problem-1-c"> 3: Tuning with cross validation </a>

Using the same rescaling pipeline from Q2, we will now look at *cross validation* as a possible way to use our scarce training dataset more effectively. 

First, you should stack all training and validation examples together:

```
x_trva_LF = np.vstack([x_tr_MF, x_va_NF])
y_trva_L = np.hstack([y_tr_M, y_va_N])
```

**Coding Step 1/1:** Complete the starter code function `calc_mean_squared_error_across_k_folds`, defined in in the starter notebook.

**3a** Using your `calc_mean_squared_error_across_k_folds` function with 10 folds, make a line plot of the *average* mean-squared-error at degrees 1, 2, 3, 4, 5, 6.

**3b** Based on this plot, what is your recommended degree? How do your recommendations differ from *Q2*?

**3c** Fix the degree at 3. We might hope that using *many* folds lets us maximize our chances of fitting a complex model well (by using as much training data as possible in each fold). We'll look at the distribution of single-fold estimates as a function of the number of folds. For each number of folds $K$ in the grid given below, make a scatter plot of the $K$ fold-specific estimates of MSE ($K$ is x-axis, MSE on y-axis). Also draw a line connecting the average MSE across $K$.

```
K_grid = [2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
```

**3d:** What happens to the distribution of the estimated heldout MSE as you use more and more folds? Is the trend observed in Plot **3c** what we should expect?

<br /> <br />

#### <a name="problem-2"> 4: L2 Regularization for Regression </a>

We'll use the same dataset, and now look at L2-penalized least-squares linear regression. In statistics, this is sometimes called "ridge" regression, so the `sklearn` implementation uses a regression class called [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), with the usual `fit` an `predict` methods.

**4a:** Train `Ridge` regression at a fine grid of 31 possible L2-penalty strengths $\alpha$: `alpha_grid = np.logspace(-9, 6, 31)`. Using *degree 2* polynomial features, plot the MSE vs. regularization strength on both validation (use style 'rs-') and training (use style 'b:'). Because $\alpha$ is log-scaled, show the base-10 log of $alpha$ on the x axis.

**4b:** Repeat the plot from 2a(i) with polynomial degree 6.

**4c:** Describe how the recommended value of $\alpha$ changes from degree 2 to 6.
