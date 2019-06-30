  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        inlineMath: [['$','$']]
      }
    });
  </script>
  <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


## HW0: Intro to Data Analysis with NumPy

This homework is not graded, and you do not need to submit your results.

[Jupyter notebook](HW0/hw0.ipynb).

We recommend you use your personal Python installation (for exampe, by downloading [Anaconda](https://www.anaconda.com/distribution/#download-section)), or without installing any software, using [Google Colab](https://colab.research.google.com/).

## <a name="problem-1">Task: Implement the k-Nearest Neighbors Prediction Step</a>

Given some dataset of input-output observations {% raw %} $$\{x_n,y_n \}^N_{n=1}$$ {% endraw %}, we would like to be able to make a label prediction {% raw %} $$y^{\star}$$ {% endraw %} for a new input {% raw %} $$x^{\star}$$ {% endraw %}. Let us assume that the input is 1-dimensional for simplicity. We will use the k-nearest neighbors algorithm to make new predictions.

*Instructions*: Implement your own function to make predictions in python.
Find below a function signature to help you in the process

    #!python
    def predict_knn_regression(x_test, x_train, y_train, k=1):
        '''
        Function to predict output y_test for input x_test given past data
        (x_train, y_train) in 1-dimension
        Parameters:
            x_train: (N_train,) numpy array, inputs observed in the past
            y_train: (N_train,) numpy array, outputs observed in the past
            x_test: (N_test,) numpy array, new inputs
            k: scalar, number of neighbors to consider
            Return: y: (N,) numpy array, outputs of interest
        '''
        # COMPLETE
        return y
