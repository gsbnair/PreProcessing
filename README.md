# PreProcessing
Techniques and Solutions used during Pre-processing.

Data preparation takes 60 to 80 percent of the whole analytical pipeline in a typical machine learning project.

A key task when you want to build an analytic model using machine learning, is the integration and preparation of data sets from various sources like files, databases, BigData, sensors or social networks. This step can take up to 80 percent of the whole analytics project.

Datasets are different and they posses unique characteristics and taming and reshaping them for use in ML are always challenging.

Data preprocessing is an important step of solving machine learning problem. Most of the datasets (inputs) arriving at ML area requires to be cleansed and transformed to get qualified to be used in Machine Learning algorithms.

**Data PreProcessing = Data Cleansing + Feature Engineering**

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

Let us see how to calculate and interpret the chi-squared test for categorical variables in Python.

Look at a summary of a categorical variable as it pertains to another categorical variable. 

For example, sex and interest, where interest may have the labels ‘science‘, ‘math‘, or ‘art‘. We can collect observations from people with regard to these two categorical variables; for example:

`Sex`	`Interest`
Male,	Art
Female,	Math
Male, 	Science
Male,	Math
...

We can summarize the collected observations in a table with one variable corresponding to columns and another variable corresponding to rows. Each cell in the table corresponds to the count or frequency of observations that correspond to the row and column categories.

A table summarization of two categorical variables in this form is called a contingency table.

For example, the `Sex` = `rows` and `Interest` = `columns` table with contrived counts might look as follows:


| Gender | Science  |  Math |  Art  | Total |
|--------|----------|-------|-------|-------|
| Male   |      200 |   150 |    50 |   400 |
| Female |      250 |   300 |    50 |   600 |
| Total  |      450 |   450 |   100 |  1000 |


| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |   

