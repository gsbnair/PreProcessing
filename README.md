# PreProcessing
Techniques and Solutions used during Pre-processing.

Data preparation takes 60 to 80 percent of the whole analytical pipeline in a typical machine learning project.

A key task when you want to build an analytic model using machine learning, is the integration and preparation of data sets from various sources like files, databases, BigData, sensors or social networks. This step can take up to 80 percent of the whole analytics project.

Datasets are different and they posses unique characteristics and taming and reshaping them for use in ML are always challenging.

Data preprocessing is an important step of solving machine learning problem. Most of the datasets (inputs) arriving at ML area requires to be cleansed and transformed to get qualified to be used in Machine Learning algorithms.

**Data PreProcessing = Data Cleansing + Feature Engineering**

A small writeup on Data cleansing can be found [here](DataCleansing.md)

## Feature Engineering 

Feature Engineering selects the right attributes to analyze. Domain knowledge is much required to select or create attributes that make machine learning algorithms work better. Feature Engineering process includes:

- Brainstorming or testing of features
- Feature selection
- Validation of how the features work with your model
- Improvement of features if needed
- Return to brainstorming / creation of more features until the work is done

Let's explore **`Feature Selection`**.

A common problem in applied machine learning is determining whether input features are relevant to the outcome to be predicted.

This is the problem of feature selection.

In a classification problem where input variables are also categorical, we can use statistical tests to determine whether the output variable is `dependent` or `independent` of the input variables. If independent, then the input variable may be irrelevant to the problem and can be removed from the dataset.

The Karl Pearson’s `chi-squared` (Greek capital letter Chi (X) pronounced “ki” as in kite) statistical hypothesis is an example of a test for independence between categorical variables. In simple words, the Chi-Square statistic will test whether there is a significant difference in the `observed` vs the `expected` frequencies of both variables. 

Chi-Square is explained in details and with examples [here](ChiSquareExplanation.md)

Feature Selection explained using data from Titanic Survival Prediction can be seen [here](02.Chi-Square-Titanic.ipynb)

