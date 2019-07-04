  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        inlineMath: [['$','$']]
      }
    });
  </script>
  <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# HW3: Practical Introduction to Binary Classifiers and Evaluation
###### (credit: this homework has been adapted from Mike Hughes COM135 course at Tufts University)

**Due date**: Mon. July 08 at 11:59PM EST.

**Turn-in links**:

IMPORTANT: please name the file as firstname_lastname.pdf and firstname_lastname.zip
* PDF report turned in to: <https://www.aeoncase.com/inbox/AA2YIsgkMwpQ2seV6ezio>
* ZIP file of source code turned in to: <https://www.aeoncase.com/inbox/AA2YIsgkBQ0WoGUHJ0lJc>

**Files to Turn In:**

ZIP file of source code should contain:

* hw3.ipynb : Jupyter Notebook file containing your code and markup
* COLLABORATORS.txt : a plain text file with your full name and collaborators

PDF report:

* Please export your completed hw3.ipynb notebook as a PDF (easiest way is likely in your browser, just do 'Print as PDF')
* This document will be manually graded

** Starter code and Data:**

Download material here:
<https://melaniefp.github.io/intro_to_ML_DSC6135/hw/HW3/hw3.zip>

Note: be sure that all plots include readable axes, labels and legends if needed, when multiple lines are shown.

## <a name="problem-1">Problem 1: Binary Classifier for Cancer-Risk Screening </a>

You have been given a dataset containing some medical history information for 750 patients that might be at risk of cancer. Dataset credit: A. Vickers, Memorial Sloan Kettering Cancer Center [[original link]](https://www.mskcc.org/sites/default/files/node/4509/documents/dca-tutorial-2015-2-26.pdf).

Each patient in our dataset has been biopsied (fyi: in this case a [biopsy](https://www.cancer.net/navigating-cancer-care/diagnosing-cancer/tests-and-procedures/biopsy) is a short surgical procedure that is painful but with virtually no lasting harmful effects) to obtain a direct "ground truth" label so we know each patient's actual cancer status (binary variable, 1 means "has cancer", 0 means does not, column name is `cancer` in the $ {% raw %} $$y$$  {% endraw %} data files). We want to build classifiers to predict whether a patient likely has cancer from easier-to-get information, so we could avoid painful biopsies unless they are necessary. Of course, if we skip the biopsy, a patient with cancer would be left undiagnosed and therefore untreated. We're told by the doctors this outcome would be life-threatening.

*Easiest* features: It is known that older patients with a family history of cancer have a higher probability of harboring cancer. So we can use `age` and `famhistory` variables  in the {% raw %} $$x$$  {% endraw %} dataset files as inputs to a simple predictor.

*Possible new feature*: A clinical chemist has recently discovered a real-valued marker (called `marker` in the  {% raw %} $$x$$  {% endraw %} dataset files) that she believes can distinguish between patients with and without cancer. We wish to assess whether or not the new marker does indeed identify patients with and without cancer well.

To summarize, there are two versions of the features {% raw %} $$x$$  {% endraw %} we'd like you to examine:

* 2 variable: 'age' and 'famhistory'
* 3 variable: 'age' and 'famhistory' and 'marker'

*Bottom-line*: We are building classifiers so that many patients might not need to undergo a painful biopsy if our classifier is reliable enough to be trusted to filter out low-risk patients.

In the starter code, we have provided an existing train/validation/test split of this dataset, stored on-disk in comma-separated-value (CSV) files: x_train.csv, y_train.csv, x_valid.csv, y_valid.csv, x_test.csv, and y_test.csv.

We will train binary classifiers that minimize log loss (aka binary cross entropy error).


#### <a name="problem-1-a"> 1: Data Exploration </a>


**1a:** What fraction of the provided patients have cancer in the training set, the validation set, and the test set?

**1b:** Looking at the features data contained in the training set $x$ array, what feature preprocessing (if any) would you recommend to improve a **decision tree**'s performance?

**1c:** Looking at the features data contained in the training set $x$ array, what feature preprocessing (if any) would you recommend to improve **logistic regression**'s performance?



#### <a name="problem-1-b"> 2: Baseline Predictions </a>

Given a training set of values $\{y_n \}_{n=1}^N$, we can **always** consider a simple baseline for prediction that returns the same constant predicted label regardless of the input $x_i$ feature vector:

* predict-0-always : $\hat{y}(x_i) = 0$


**2a:** Compute the accuracy of the predict-0-always classifier on validation and test set. Print the results neatly.

**2b:** Print a confusion matrix for predict-0-always on the validation set. Use the provided `calc_confusion_matrix_for_threshold`.

**2c:** This classifier gets pretty good accuracy! Why wouldn't we want to use it?

**2d:** For the intended application (screening patients before biopsy), describe the possible mistakes the classifier can make in task-specific terms. What costs does each mistake entail (lost time? lost money? life-threatening harm?). How do you recommend evaluating the classifier to be mindful of these costs?


#### <a name="problem-1-c"> 3: Logistic Regression </a>

Fit a logistic regression model using sklearn's `LogisticRegression` implementation `sklearn.linear_model.LogisticRegression` [docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). To avoid overfitting, be sure to call `fit` with an L2 penalty with inverse penalty strength `C`.  You should explore a range of `C` values, using a regularly-spaced grid: `C_grid = np.logspace(-9, 6, 31)`.

**3a:** Apply your logistic regression code to the "2 feature" $x$ data, and make a plot of the log loss [[`sklearn.metrics.log_loss`]](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html) (y-axis) vs. base-10 logarithm of C (x-axis) on the training set and validation set. Which value of $C$ should be selected?

**3b:** Make a performance plot that shows how good your probabilistic predictions from the best 3a) classifier are on the validation set.

Using your trained model from 3a) at the best $C$ value, compute the **probability** that each example in the validation set should be assigned a positive label. Use the sklearn function `predict_proba`, which is like `predict` but for probabilities rather than hard decisions. Use the provided function `make_plot_perf_vs_threshold` in the starter notebook to make a plot with 3 rows:

* top row: histogram of predicted probabilities for negative class examples (shaded red)
* middle row: histogram of predicted probabilities for positive class examples (shaded blue)
* bottom row: line plots of performance metrics that require hard decisions (ACC, TPR, TNR, etc.)

**3c:** Apply your logistic regression code to the "3 feature" $x$ data (which includes the new `marker` feature), and make a plot of the log loss (y-axis) vs. base-10 logarithm of C (x-axis) on the training set and validation set. Which value of $C$ should be selected?

**3d:** Make a performance plot that shows how good your probabilistic predictions from the best 1c(iii) classifier are on the validation set. Again, use the provided `make_plot_perf_vs_threshold` function.

#### <a name="problem-1-d"> 4: Decision Tree Predictions </a>

Now try fitting decision tree classifiers using sklearn's `DecisionTreeClassifier` implementation [sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier). Be sure to use the produced probabilities, not hard binary predictions. Try a range of `min_samples_leaf` values from [1, 2, 5, 10, 20, 50, 100, 200, `n_training_examples`].

**4a:**  Make a plot of the log loss (`sklearn.metrics.log_loss`, see [[sklearn docs for log loss]](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html) (y-axis) vs. `min_samples_leaf` (x-axis) on the training set and validation set. Which value of `min_samples_leaf` should be selected?

**4b:** Make a performance plot that shows how good your probabilistic predictions from the best 4a) classifier are on the validation set. Use the provided `make_plot_perf_vs_threshold` function.


#### <a name="problem-1-e"> 5: ROC Analysis </a>

**5a:** Plot a ROC curve for the best LR model with 2 features, best LR model with 3 features, and best decision tree, using the **validation** set. Use sklearn's existing ROC curve tools (`sklearn.metrics.roc_curve`).

**5b:** Plot a ROC curve for the best LR model with 2 features, best LR model with 3 features, and best decision tree, using the **test** set.

**5c:** Short Answer: Compare the 3-feature LR to 2-feature LR models: does one dominate the other in terms of ROC performance? Or are there some thresholds where one model wins, and other thresholds where the other model wins?

**5d:** Short Answer: Compare the 3-feature Tree to 2-feature LR models: does one dominate the other in terms of ROC performance? Or are there some thresholds where one model wins, and other thresholds where the other model wins?


#### <a name="problem-1-f"> 6: Selecting the best single threshold </a>

Throughout 6), use the best 3-feature logistic regression (LR) classifier from earlier in 3). Use the provided `calc_confusion_matrix_for_threshold` function to print confusion matrices nicely.

**6a:** Use the "default" probability threshold (0.5) to produce hard binary predictions given probabilities from your classifier. Print this threshold's confusion matrix on the **test** set.


**6b:** For the same classifier as above, compute performance metrics across a range of possible thresholds on validation, and pick the threshold that maximizes TPR while satisfying PPV >= 0.98 on the validation set. Print this threshold's confusion matrix on the **test** set.

**6c:** For the same classifier as above, compute performance metrics across a range of possible thresholds on validation, and pick the threshold that maximizes PPV while satisfying TPR >= 0.98 on the validation set. Print this threshold's confusion matrix on the **test** set.

**6d:** (Short Answer) Compare the confusion matrices between 6a) - 6c). Which thresholding strategy best meets our preferences: avoid life-threatening mistakes at all costs, while also eliminating unnecessary biopsies?

**6e:** (Short Answer) How many subjects in the test set are saved from unnecessary biopsies using your selected thresholding strategy? What fraction of current biopsies would be avoided if this classifier was adopted by the hospital?
