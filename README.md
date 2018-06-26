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

Let us see how to calculate and interpret the chi-squared test for categorical variables in Python.

Look at a summary of a categorical variable as it pertains to another categorical variable. 

For example, sex and interest, where interest may have the labels ‘science‘, ‘math‘, or ‘art‘. We can collect observations from people with regard to these two categorical variables; for example:

|Gender| Interest   |
|:-----|:-----------|
|Male  |	Art     |
|Female|	Math    |
|Male  | 	Science |
|Male  |	Math    |
|Female|	Art     |
|Male  | 	Science |
|...   |    ...     |

We can summarize the collected observations in a table with one variable corresponding to `columns` and another variable corresponding to `rows`. Each cell in the table corresponds to the count or frequency of observations that correspond to the row and column categories.

A table summarization of two categorical variables in this form is called a **contingency table**.

For example, the `Gender` = `rows` and `Interest` = `columns` table with contrived counts might look as follows:

| Gender | Science  |  Math |  Art  | Total |
|:-------|---------:|------:|------:|------:|
| Male   |      200 |   150 |    50 |   400 |
| Female |      250 |   300 |    50 |   600 |
| Total  |      450 |   450 |   100 |  1000 |

Question: Does an interest in math, science or art depend on gender, or are they independent?

This is challenging to determine from the table alone; instead, we can use a statistical method called the Pearson’s Chi-Squared test.

Wait... there are constraints for using Chi-square !

**When to Use Chi-Square Test for Independence**

The test can be conducted when the following conditions are met:

- The sampling method is `simple random sampling`.
- The variables under study are each `categorical`.
- If sample data are displayed in a contingency table, the expected `frequency count` for each cell of the table is `at least 5`.

The chi-square approach consists of **four steps**: 
- (1) State the hypotheses
- (2) Formulate an analysis plan
- (3) Analyze sample data
- (4) Interpret results

1. **State the Hypotheses** 

>***Hypotheses `Ho`***: Variable A and Variable B are independent.

>***Alternative Hypothesis `Ha`***: Variable A and Variable B are not independent.
The `alternative hypothesis` is that knowing the level of Variable A helps you predict the level of Variable B (`one causes the other`).

2. **Formulate an Analysis Plan**

The analysis plan describes how to use sample data to accept or reject the null hypothesis. The plan should specify the following elements.

***Significance level***. Usually, researchers choose significance levels equal to 0.01, 0.05, or 0.10; but any value between 0 and 1 can be used. 
***Test method***. Use the chi-square test for independence to determine whether there is a significant relationship between two categorical variables.

3. **Analyze Sample Data**
Using sample data, find the `degrees of freedom`, `expected frequencies`, `test statistic`, and the `P-value` associated with the test statistic.

***Degrees of freedom***. The degrees of freedom (DF) is equal to:
> degrees of freedom: (rows - 1) * (cols - 1)

>				DF = (r - 1) * (c - 1)

where r is the number of levels for one catagorical variable, and c is the number of levels for the other categorical variable.
In our case, it is `DF = (r - 1) * (c - 1) = (2 - 1) * (3 - 1) = 2`

***Expected frequencies***. The expected frequency counts are computed separately for each level of one categorical variable at each level of the other categorical variable. Compute r * c expected frequencies, according to the following formula: `E(r,c) = (nr * nc) / n`

where E(r,c) is the expected frequency count for level r of Variable A and level c of Variable B, nr is the total number of sample observations at level r of Variable A, nc is the total number of sample observations at level c of Variable B, and n is the total sample size.

Which means: E(r,c) = (nr * nc) / n
E1,1 = (400 * 450) / 1000 = 180000/1000 = 180
E1,2 = (400 * 450) / 1000 = 180000/1000 = 180
E1,3 = (400 * 100) / 1000 = 40000/1000 = 40
E2,1 = (600 * 450) / 1000 = 270000/1000 = 270
E2,2 = (600 * 450) / 1000 = 270000/1000 = 270
E2,3 = (600 * 100) / 1000 = 60000/1000 = 60

***Test statistic***. The test statistic is a chi-square random variable (Χ2) defined by the following equation.
Χ^2 = Σ [ (O(r,c) - E(r,c))^2 / E(r,c) ]

where O(r,c) is the observed frequency count at level r of Variable A and level c of Variable B, and E(r,c) is the expected frequency count at level r of Variable A and level c of Variable B.

Which means: Χ^2 = Σ [ (O(r,c) - E(r,c))^2 / E(r,c) ] 
Χ^2 = (200 - 180)^2/180 + (150 - 180)^2/180 + (50 - 40)^2/40
    + (250 - 270)^2/270 + (300 - 270)^2/270 + (50 - 60)^2/60
Χ^2 = 400/180 + 900/180 + 100/40 + 400/270 + 900/270 + 100/60
Χ^2 = 2.22 + 5.00 + 2.50 + 1.48 + 3.33 + 1.67 = 16.2

The result of the test is a test statistic that has a chi-squared distribution and can be interpreted to reject or fail to reject the assumption or null hypothesis that the observed and expected frequencies are the same.

There are two ways to proceed:
- 1. We can interpret the test statistic in the context of the chi-squared distribution with the requisite number of degress of freedom as follows:

>If Statistic >= Critical Value: significant result, reject null hypothesis (H0), dependent.
>If Statistic < Critical Value: not significant result, fail to reject null hypothesis (H0), independent.

- 2. In terms of a p-value and a chosen significance level (alpha), the test can be interpreted as follows:

>If p-value <= alpha: significant result, reject null hypothesis (H0), dependent.
>If p-value > alpha: not significant result, fail to reject null hypothesis (H0), independent.

The Pearson’s chi-squared test for independence can be calculated in Python using the `chi2_contingency()` function from `SciPy` library.

The function takes an array as input representing the contingency table for the two categorical variables. It returns the calculated statistic and p-value for interpretation as well as the calculated degrees of freedom and table of expected frequencies.

>statistic, p, dof, expected = chi2_contingency(table)

In our [example](01.ChiSquare.ipynb) we got the results:
prob=0.95
statistic = 16.2
p = 0.0003
critical=5.991

***Option-1***: A probability of 95% can be used, suggesting that the finding of the test is quite likely given the assumption of the test that the variable is independent. If the statistic is less than or equal to the critical value, it can be rejected.

Verdict:
--------
statistic >= critical 
-> Dependent (reject H0)

***Option-2***: We can also interpret the p-value by comparing it to a chosen significance level, which would be 5%, calculated by inverting the 95% probability used in the critical value interpretation.
significance [alpha =(1.0 - Prob)] = 0.050, 
p = 0.00030298

Verdict:
--------
p <= alpha 
-> Dependent (reject H0)

Both ways we can conclude that the `interest` is __dependent__ on `gender`

