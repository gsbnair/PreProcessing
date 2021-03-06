# PreProcessing
Techniques and Solutions used during Pre-processing.

**Data PreProcessing = Data Cleansing + Feature Engineering**

## Data Cleansing

Few common cleansing activities are:
- missing value imputation
- encoding categorical variables
- scaling
- etc.

**Too many nulls** - When most (over 60% to 70%) of the values in a column are null, it’s better to drop the column. This percentage/threshold can be decided based on problem and experience.

**Same values/skew** - Sometimes, a majority of values in a column might be same values with very few different values. We need to check if the occurrence of such values is due to a skew in dataset or is it natural for that dataset. If it’s skewed, dataset should be resampled (sub-sample or over-sample, as appropriate). If it’s not a skew and the values occur naturally in that way, it’s better to drop the column.

**Data types** - Check the datatypes of the columns, particularly date columns and type cast appropriately.

**Missing value imputation** - Usually median is used with numeric columns and mode with non-numeric columns.

**When column doesn’t have missing values** - It’s possible that a column doesn’t have any null values in the train dataset, but it’s very possible that it might have null values in test dataset. Hence, it’s important to review the columns/data and perform missing value imputation of all columns that can possibly have missing values, even if the train dataset doesn’t have any missing values.

**Categorical Attributes** - When the number unique values in a categorical column are too high, check the value counts of each of those values. Replace rarely occurring values together into a single value like ‘Other’ before encoding.

**Many unique values** - When number of unique values is huge and even the values are equally distributed, try to find some related values and see if the multiple categorical values can be clubbed into single (grouping), thereby reducing the count of categorical values.

**Related Attributes** - If there multiple attributes with same information with different granularity, like city and state, it’s better to keep columns like state and delete city column. Additionally, keeping both columns and assessing feature importance might help in eliminating one column.


