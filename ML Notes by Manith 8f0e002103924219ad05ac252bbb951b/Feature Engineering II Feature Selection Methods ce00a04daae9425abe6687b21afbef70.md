# Feature Engineering II : Feature Selection Methods

# **Introduction to Feature Selection Methods**

**Learn about the pros and cons of different feature selection methods.**

## **Introduction and motivation**

Imagine you want to build a machine learning model for predicting the presence of a disease, and you have a dataset of patient records containing hundreds of different metrics, from demographic characteristics to medical test results to past treatment history. While having access to so many data points is always a good starting point, you most likely would not want to use all of the available features to construct your model, as many of them can be redundant or irrelevant, and would only serve to contribute unnecessary noise. Instead, a more effective approach is to select only a subset of relevant, predictive features to use in your model — this process is known as *feature selection*.

Feature selection is an important step in the machine learning pipeline, and when done right, can optimize the performance and predictive power of your model. The goal is to improve the model’s accuracy and efficiency by eliminating extraneous variables that do not contribute any useful information. In addition, simplifying the model through feature selection not only makes the results more easily interpretable, but also reduces the time and resources required to run the model.

There are three broad categories of feature selection that we will discuss in this article: *filter methods*, *wrapper methods*, and *embedded methods*.

## **Feature Selection Methods**

### **Filter methods**

*Filter methods* are the simplest type of feature selection method. They work by filtering features prior to model building based on some criteria.

### **Advantages**

- They are computationally inexpensive, since they do not involve testing the subsetted features using a model.
- They can work for any type of machine learning model.

### **Disadvantages**

- It is more difficult to take multivariate relationships into account because we are not evaluating model performance. For example, a variable might not have much predictive power on its own, but can be informative when combined with other variables.
- They are not tailored toward specific types of models.

### **Examples**

- Variance thresholds
- Correlation
- Mutual information

### **Wrapper methods**

*Wrapper methods* involve fitting a model and evaluating its performance for a particular subset of features. They work by using a search algorithm to find which combination of features can optimize the performance of a given model.

### **Advantages**

- They can determine the optimal set of features that produce the best results for a specific machine learning problem.
- They can better account for multivariate relationships because model performance is evaluated.

### **Disadvantages**

- They are computationally expensive because the model needs to be re-fitted for each feature set being tested.

### **Examples**

- Forward/backward/bidirectional sequential feature selection
- Recursive feature elimination

### **Embedded methods**

*Embedded methods* also involve building and evaluating models for different feature subsets, but their feature selection process happens at the same time as their model fitting step.

### **Advantages**

- Like wrapper methods, they can optimize the feature set for a particular model and account for multivariate relationships.
- They are also generally less computationally expensive because feature selection happens during model training.

### **Examples**

- Regularization (e.g., lasso/ridge regression)
- Tree-based feature importance

## **Conclusion**

When it comes to the number of features to keep for your model, more is not always better. It is worth being selective about which features to retain in order to maximize performance and reduce noise. Therefore, finding and implementing the right feature selection method is a key part of developing an effective and reliable model.

# **Filter Methods**

**Learn about the different filter methods for evaluating and selecting features.**

## **Introduction**

**Filter methods** are a type of feature selection method that works by selecting features based on some criteria prior to building the model. Because they don’t involve actually testing the subsetted features using a model, they are computationally inexpensive and flexible to use for any type of machine learning algorithm. This makes filter methods an efficient initial step for narrowing down the pool of features to only the most relevant, predictive ones.

There are many different filter methods that can be used for evaluating and selecting features. In this article, we will use *variance thresholds*, *correlation*, and *mutual information* to rank and select the top features. To demonstrate how these methods work in Python, we will use the **`[feature_selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection)`** module from [scikit-learn](https://www.codecademy.com/article/scikit-learn) as well as the **`[pandas](https://www.codecademy.com/resources/docs/pandas)`** library.

## **Example dataset**

Let’s suppose we have the following dataset containing information on a class of middle school students:

```python
import pandas as pd

df = pd.DataFrame(data={
    'edu_goal': ['bachelors', 'bachelors', 'bachelors', 'masters', 'masters', 'masters', 'masters', 'phd', 'phd', 'phd'],
    'hours_study': [1, 2, 3, 3, 3, 4, 3, 4, 5, 5],
    'hours_TV': [4, 3, 4, 3, 2, 3, 2, 2, 1, 1],
    'hours_sleep': [10, 10, 8, 8, 6, 6, 8, 8, 10, 10],
    'height_cm': [155, 151, 160, 160, 156, 150, 164, 151, 158, 152],
    'grade_level': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    'exam_score': [71, 72, 78, 79, 85, 86, 92, 93, 99, 100]
})

print(df)

```

| edu_goal | hours_study | hours_TV | hours_sleep | height_cm | grade_level | exam_score |
| --- | --- | --- | --- | --- | --- | --- |
| bachelors | 1 | 4 | 10 | 155 | 8 | 71 |
| bachelors | 2 | 3 | 10 | 151 | 8 | 72 |
| bachelors | 3 | 4 | 8 | 160 | 8 | 78 |
| masters | 3 | 3 | 8 | 160 | 8 | 79 |
| masters | 3 | 2 | 6 | 156 | 8 | 85 |
| masters | 4 | 3 | 6 | 150 | 8 | 86 |
| masters | 3 | 2 | 8 | 164 | 8 | 92 |
| phd | 4 | 2 | 8 | 151 | 8 | 93 |
| phd | 5 | 1 | 10 | 158 | 8 | 99 |
| phd | 5 | 1 | 10 | 152 | 8 | 100 |

Our goal is to use the data to predict how well each student will perform on the exam. Thus, our target variable is **`exam_score`** and the remaining 6 variables are our features. We’ll prepare the data by separating the features matrix (**`X`**) and the target vector (**`y`**):

```python
# 10 x 6 features matrix
X = df.drop(columns=['exam_score'])

print(X)

```

| edu_goal | hours_study | hours_TV | hours_sleep | height_cm | grade_level |
| --- | --- | --- | --- | --- | --- |
| bachelors | 1 | 4 | 10 | 155 | 8 |
| bachelors | 2 | 3 | 10 | 151 | 8 |
| bachelors | 3 | 4 | 8 | 160 | 8 |
| masters | 3 | 3 | 8 | 160 | 8 |
| masters | 3 | 2 | 6 | 156 | 8 |
| masters | 4 | 3 | 6 | 150 | 8 |
| masters | 3 | 2 | 8 | 164 | 8 |
| phd | 4 | 2 | 8 | 151 | 8 |
| phd | 5 | 1 | 10 | 158 | 8 |
| phd | 5 | 1 | 10 | 152 | 8 |

```python
# 10 x 1 target vector
y = df['exam_score']

print(y)

```

exam_score

---

71

---

72

---

78

---

79

---

85

---

86

---

92

---

93

---

99

---

100

---

## **Variance threshold**

One of the most basic filter methods is to use a *variance threshold* to remove any features that have little to no variation in their values. This is because features with low variance do not contribute much information to a model. Since variance can only be calculated on numeric values, this method only works on quantitative features. That said, we may also want to remove categorical features for which all or a majority of the values are the same. To do that, we would need to dummy code the categorical variables first, but we won’t demonstrate that here.

In our example dataset, **`edu_goal`** is the only feature that is not numeric. We can use the **`[.drop()](https://www.codecademy.com/resources/docs/pandas/dataframe/drop)`** method to remove it from our features DataFrame and store the remaining numeric features in **`X_num`**:

```python
X_num = X.drop(columns=['edu_goal'])

print(X_num)

```

| hours_study | hours_TV | hours_sleep | height_cm | grade_level |
| --- | --- | --- | --- | --- |
| 1 | 4 | 10 | 155 | 8 |
| 2 | 3 | 10 | 151 | 8 |
| 3 | 4 | 8 | 160 | 8 |
| 3 | 3 | 8 | 160 | 8 |
| 3 | 2 | 6 | 156 | 8 |
| 4 | 3 | 6 | 150 | 8 |
| 3 | 2 | 8 | 164 | 8 |
| 4 | 2 | 8 | 151 | 8 |
| 5 | 1 | 10 | 158 | 8 |
| 5 | 1 | 10 | 152 | 8 |

Now, we’ll be able to use the **`[VarianceThreshold](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html)`** class from **`scikit-learn`** to help remove the low-variance features from **`X_num`**. By default, it drops all features with zero variance, but we can adjust the threshold during class instantiation using the **`threshold`** parameter if we want to allow some variation. The **`.fit_transform()`** method returns the filtered features as a numpy array:

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0)  # 0 is default

print(selector.fit_transform(X_num))

```

The output will look like this:

```
[[  1   4  10 155]
 [  2   3  10 151]
 [  3   4   8 160]
 [  3   3   8 160]
 [  3   2   6 156]
 [  4   3   6 150]
 [  3   2   8 164]
 [  4   2   8 151]
 [  5   1  10 158]
 [  5   1  10 152]]
```

As we can see, **`grade_level`** was removed because there is no variation in its values — all students are 8th graders. Since this data is the same across the board, a student’s grade level will not be able to provide any useful predictive information about their exam score, so it makes sense to drop **`grade_level`** as a feature.

Something to note is that [loading datasets with **`scikit-learn`**](https://scikit-learn.org/stable/faq.html#how-can-i-load-my-own-datasets-into-a-format-usable-by-scikit-learn) generally works with numpy arrays internally, hence the output type of **`.fit_transform()`**. However, the methods can also accept other data types that can be converted to numpy arrays, such as Python lists or pandas DataFrames, like the **`X_num`** we used. From a human perspective, one downside of working with numpy arrays as compared to pandas DataFrame is that we lose information like column headings, making the data harder to visually inspect.

Luckily, **`VarianceThreshold`** offers another method called **`.get_support()`** that can return the indices of the selected features, which we can use to manually subset our numeric features DataFrame:

```python
# Specify `indices=True` to get indices of selected features
print(selector.get_support(indices=True))

```

```
[0 1 2 3]
```

```python
# Use indices to get the corresponding column names of selected features
num_cols = list(X_num.columns[selector.get_support(indices=True)])

print(num_cols)
```

```
['hours_study', 'hours_TV', 'hours_sleep', 'height_cm']
```

```python
# Subset `X_num` to retain only selected features
X_num = X_num[num_cols]

print(X_num)
```

| hours_study | hours_TV | hours_sleep | height_cm |
| --- | --- | --- | --- |
| 1 | 4 | 10 | 155 |
| 2 | 3 | 10 | 151 |
| 3 | 4 | 8 | 160 |
| 3 | 3 | 8 | 160 |
| 3 | 2 | 6 | 156 |
| 4 | 3 | 6 | 150 |
| 3 | 2 | 8 | 164 |
| 4 | 2 | 8 | 151 |
| 5 | 1 | 10 | 158 |
| 5 | 1 | 10 | 152 |

Finally, to obtain our entire features DataFrame, including the categorical column **`edu_goal`**, we could do:

```python
X = X[['edu_goal'] + num_cols]

print(X)
```

| edu_goal | hours_study | hours_TV | hours_sleep | height_cm |
| --- | --- | --- | --- | --- |
| bachelors | 1 | 4 | 10 | 155 |
| bachelors | 2 | 3 | 10 | 151 |
| bachelors | 3 | 4 | 8 | 160 |
| masters | 3 | 3 | 8 | 160 |
| masters | 3 | 2 | 6 | 156 |
| masters | 4 | 3 | 6 | 150 |
| masters | 3 | 2 | 8 | 164 |
| phd | 4 | 2 | 8 | 151 |
| phd | 5 | 1 | 10 | 158 |
| phd | 5 | 1 | 10 | 152 |

## **Pearson’s correlation**

Another type of filter method involves finding the *correlation* between variables. In particular, the [Pearson’s correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) is useful for measuring the linear relationship between two numeric, continuous variables — a coefficient close to **`1`** represents a positive correlation, **`-1`** represents a negative correlation, and **`0`** represents no correlation. Like variance, Pearson’s correlation coefficient cannot be calculated for categorical variables. Although, there is a related [point biserial correlation coefficient](https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient) that can be computed when one variable is dichotomous, but we won’t focus on that here.

There are 2 main ways of using correlation for feature selection — to detect correlation between features and to detect correlation between a feature and the target variable.

### **Correlation between features**

When two features are highly correlated with one another, then keeping just one to be used in the model will be enough because otherwise they provide duplicate information. The second variable would only be redundant and serve to contribute unnecessary noise.

To determine which variables are correlated with one another, we can use the **`[.corr()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html)`** method from **`pandas`** to find the correlation coefficient between each pair of numeric features in a DataFrame. By default, **`.corr()`** computes the Pearson’s correlation coefficient, but alternative methods can be specified using the **`method`** parameter. We can visualize the resulting correlation matrix using a heatmap:

```python
import matplotlib.pyplot as plt
import seaborn as sns

corr_matrix = X_num.corr(method='pearson')  # 'pearson' is default

sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r')
plt.show()
```

![https://static-assets.codecademy.com/skillpaths/feature-engineering/filter-methods/heatmap_corr_features.svg](https://static-assets.codecademy.com/skillpaths/feature-engineering/filter-methods/heatmap_corr_features.svg)

Let’s define high correlation as having a coefficient of greater than **`0.7`** or less than **`-0.7`**. We can [loop through the correlation matrix to identify the highly correlated variables](https://heartbeat.fritz.ai/hands-on-with-feature-selection-techniques-filter-methods-f248e0436ce5):

```python
# Loop over bottom diagonal of correlation matrix
for i in range(len(corr_matrix.columns)):
    for j in range(i):

        # Print variables with high correlation
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            print(corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
```

The output to our code is:

```
hours_TV hours_study -0.780763315142435
```

As seen, **`hours_TV`** appears to be highly negatively correlated with **`hours_study`** — a student who watches a lot of TV tends to spend fewer hours studying, and vice versa. Because they provide redundant information, we can choose to remove one of those variables. To decide which one, we can look at their correlation with the target variable, then remove the one that is less associated with the target. This is explored in the next section.

### **Correlation between feature and target**

As mentioned, the second way correlation can be used is to determine if there is a relationship between a feature and the target variable. In the case of Pearson’s correlation, this is especially useful if we intend to fit a linear model, which assumes a linear relationship between the target and predictor variables. If a feature is not very correlated with the target variable, such as having a coefficient of between **`-0.3`** and **`0.3`**, then it may not be very predictive and can potentially be filtered out.

We can use the same **`.corr()`** method seen previously to obtain the correlation between the target variable and the rest of the features. First, we’ll need to create a new DataFrame containing the numeric features with the **`exam_score`** column:

```python
X_y = X_num.copy()
X_y['exam_score'] = y

print(X_y)
```

| hours_study | hours_TV | hours_sleep | height_cm | exam_score |
| --- | --- | --- | --- | --- |
| 1 | 4 | 10 | 155 | 71 |
| 2 | 3 | 10 | 151 | 72 |
| 3 | 4 | 8 | 160 | 78 |
| 3 | 3 | 8 | 160 | 79 |
| 3 | 2 | 6 | 156 | 85 |
| 4 | 3 | 6 | 150 | 86 |
| 3 | 2 | 8 | 164 | 92 |
| 4 | 2 | 8 | 151 | 93 |
| 5 | 1 | 10 | 158 | 99 |
| 5 | 1 | 10 | 152 | 100 |

Then, we can generate the correlation matrix and isolate the column corresponding to the target variable to see how strongly each feature is correlated with it:

```python
corr_matrix = X_y.corr()

# Isolate the column corresponding to `exam_score`
corr_target = corr_matrix[['exam_score']].drop(labels=['exam_score'])

sns.heatmap(corr_target, annot=True, fmt='.3', cmap='RdBu_r')
plt.show()
```

![https://static-assets.codecademy.com/skillpaths/feature-engineering/filter-methods/heatmap_corr_features_target.svg](https://static-assets.codecademy.com/skillpaths/feature-engineering/filter-methods/heatmap_corr_features_target.svg)

As seen, **`hours_study`** is positively correlated with **`exam_score`** and **`hours_TV`** is negatively correlated with it. It makes sense that **`hours_study`** and **`hours_TV`** would be negatively correlated with each other as we saw earlier, and just one of those features would suffice for predicting **`exam_score`**. Since **`hours_study`** has a stronger correlation with the target variable, let’s remove **`hours_TV`** as the redundant feature:

```python
X = X.drop(columns=['hours_TV'])

print(X)
```

| edu_goal | hours_study | hours_sleep | height_cm |
| --- | --- | --- | --- |
| bachelors | 1 | 10 | 155 |
| bachelors | 2 | 10 | 151 |
| bachelors | 3 | 8 | 160 |
| masters | 3 | 8 | 160 |
| masters | 3 | 6 | 156 |
| masters | 4 | 6 | 150 |
| masters | 3 | 8 | 164 |
| phd | 4 | 8 | 151 |
| phd | 5 | 10 | 158 |
| phd | 5 | 10 | 152 |

The other two features, **`hours_sleep`** and **`height_cm`**, both do not seem to be correlated with **`exam_score`**, suggesting they would not be very good predictors. We could potentially remove either or both of them as being uninformative. But before we do, it is a good idea to use other methods to double check that the features truly are not predictive. We will do that in the next section by using mutual information to see if there are any non-linear associations between the features and target variable.

To conclude this section, we’ll briefly note an alternative approach for assessing the correlation between variables. Instead of generating the full correlation matrix, we could use the **`[f_regression()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html)`** function from **`scikit-learn`** to find the F-statistic for a model with each predictor on its own. The F-statistic will be larger (and p-value will be smaller) for predictors that are more highly correlated with the target variable, thus it will perform the same filtering:

```python
from sklearn.feature_selection import f_regression

print(f_regression(X_num, y))
```

```
(array([3.61362007e+01, 3.44537037e+01, 0.00000000e+00, 1.70259066e-03]),
 array([3.19334945e-04, 3.74322763e-04, 1.00000000e+00, 9.68097878e-01]))
```

The function returns the F-statistic in the first array and the p-value in the second. As seen, the result is consistent with what we had observed in the correlation matrix — the stronger the correlation (either positive or negative) between the feature and target, the higher the corresponding F-statistic and lower the p-value. For example, amongst all the features, **`hours_study`** has the largest correlation coefficient (**`0.905`**), highest F-statistic (**`3.61e+01`**), and lowest p-value (**`3.19e-04`**).

## **Mutual information**

The final filter method we’ll look at is using *mutual information* to rank and select the top features. [Mutual information](https://en.wikipedia.org/wiki/Mutual_information) is a measure of dependence between two variables and can be used to gauge how much a feature contributes to the prediction of the target variable. It is similar to Pearson’s correlation, but is not limited to detecting linear associations. This makes mutual information useful for more flexible models where a linear functional form is not assumed. Another advantage of mutual information is that it also works on discrete features or target, unlike correlation. Although, categorical variables need to be numerically encoded first.

In our example, we can encode the **`edu_goal`** column using the **`[LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)`** class from **`scikit-learn`**‘s **`preprocessing`** module:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Create copy of `X` for encoded version
X_enc = X.copy()
X_enc['edu_goal'] = le.fit_transform(X['edu_goal'])

print(X_enc)
```

| edu_goal | hours_study | hours_sleep | height_cm |
| --- | --- | --- | --- |
| 0 | 1 | 10 | 155 |
| 0 | 2 | 10 | 151 |
| 0 | 3 | 8 | 160 |
| 1 | 3 | 8 | 160 |
| 1 | 3 | 6 | 156 |
| 1 | 4 | 6 | 150 |
| 1 | 3 | 8 | 164 |
| 2 | 4 | 8 | 151 |
| 2 | 5 | 10 | 158 |
| 2 | 5 | 10 | 152 |

Now, we can compute the mutual information between each feature and **`exam_score`** using **`[mutual_info_regression()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html)`**. This function is used because our target variable is continuous, but if we had a discrete target variable, we would use **`[mutual_info_classif()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)`**. We specify the **`random_state`** in the function in order obtain reproducible results:

```python
from sklearn.feature_selection import mutual_info_regression

print(mutual_info_regression(X_enc, y, random_state=68))
```

```
[0.50396825 0.40896825 0.06896825 0.        ]
```

The estimated mutual information between each feature and the target is returned in a numpy array, where each value is a non-negative number — the higher the value, the more predictive power is assumed.

However, we are missing one more important piece here. Earlier, even though we encoded **`edu_goal`** to be numeric, that does not mean it should be treated as a continuous variable. In other words, the values of **`edu_goal`** are still discrete and should be interpreted as such. If we plot **`edu_goal`** against **`exam_score`** on a graph, we can clearly see the steps between the values of **`edu_goal`**:

![https://static-assets.codecademy.com/skillpaths/feature-engineering/filter-methods/scatterplot_edu_goal.svg](https://static-assets.codecademy.com/skillpaths/feature-engineering/filter-methods/scatterplot_edu_goal.svg)

In order to properly calculate the mutual information, we need to tell **`mutual_info_regression()`** which features are discrete by providing their index positions using the **`discrete_features`** parameter:

```python
print(mutual_info_regression(X_enc, y, discrete_features=[0], random_state=68))
```

```
[0.75563492 0.38896825 0.18563492 0.        ]
```

Compared to the earlier results, we now get greater mutual information between **`edu_goal`** and the target variable once it is correctly interpreted as a discrete feature.

From the results, we can also see that there is **`0`** mutual information between **`height_cm`** and **`exam_score`**, suggesting that these variables are largely independent. This is consistent with what we saw earlier with Pearson’s correlation, where the correlation coefficient between them is very close to **`0`** as well.

What is interesting to note is that the mutual information between **`hours_sleep`** and **`exam_score`** is a positive value, even though their Pearson’s correlation coefficient is **`0`**. The answer becomes more clear when we plot the relationship between **`hours_sleep`** and **`exam_score`**:

![https://static-assets.codecademy.com/skillpaths/feature-engineering/filter-methods/scatterplot_hours_sleep.svg](https://static-assets.codecademy.com/skillpaths/feature-engineering/filter-methods/scatterplot_hours_sleep.svg)

As seen, there do seem to be some association between the variables, only it is not a linear one, which is why it was detected using mutual information but not Pearson’s correlation coefficient.

Finally, let’s look at using the **`[SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)`** class from **`scikit-learn`** to help pick out the top **`k`** features with the highest ranked scores. In our case, we are looking to select features that share the most mutual information with the target variable. When we instantiate **`SelectKBest`**, we’ll specify which scoring function to use and how many top features to select. Here, our scoring function is **`mutual_info_regression()`**, but because we want to specify additional arguments besides the **`X`** and **`y`** inputs, we’ll need the help of the **`[partial()](https://docs.python.org/3/library/functools.html#functools.partial)`** function from Python’s built-in **`functools`** module. Then, the **`.fit_transform()`** method will return the filtered features as a numpy array:

```python
from sklearn.feature_selection import SelectKBest
from functools import partial

score_func = partial(mutual_info_regression, discrete_features=[0], random_state=68)

# Select top 3 features with the most mutual information
selection = SelectKBest(score_func=score_func, k=3)

print(selection.fit_transform(X_enc, y))
```

```
[[ 0  1 10]
 [ 0  2 10]
 [ 0  3  8]
 [ 1  3  8]
 [ 1  3  6]
 [ 1  4  6]
 [ 1  3  8]
 [ 2  4  8]
 [ 2  5 10]
 [ 2  5 10]]
```

As seen above, we selected the top 3 features based on mutual information, thus dropping **`height_cm`**. Like **`VarianceThreshold`**, **`SelectKBest`** also offers the **`.get_support()`** method that returns the indices of the selected features, so we could subset our original features DataFrame:

```python
X = X[X.columns[selection.get_support(indices=True)]]

print(X)
```

| edu_goal | hours_study | hours_sleep |
| --- | --- | --- |
| bachelors | 1 | 10 |
| bachelors | 2 | 10 |
| bachelors | 3 | 8 |
| masters | 3 | 8 |
| masters | 3 | 6 |
| masters | 4 | 6 |
| masters | 3 | 8 |
| phd | 4 | 8 |
| phd | 5 | 10 |
| phd | 5 | 10 |

## **Conclusion**

In our example dataset, we started out with 6 features for predicting the **`exam_score`** of students. Using various filter methods, we narrowed down that set to just the top most relevant and informative ones. First, we eliminated **`grade_level`** because it has zero variance and would contribute nothing to the model. Then, we dropped **`hours_TV`** since it is highly correlated with **`hours_study`** and is therefore redundant. Lastly, we filtered out **`height_cm`** based on mutual information, which suggested that it does not have any meaningful association with the target variable, linear or otherwise, and would not have been very predictive.

Phew! That was a lot we were able to accomplish using filter methods. Being the most simple type of feature selection method, they sure do not lack power nor potential. It is certainly worth considering how you might want to incorporate filter methods into your next machine learning project.

[**WRAPPER METHODS**](Feature%20Engineering%20II%20Feature%20Selection%20Methods%20ce00a04daae9425abe6687b21afbef70/WRAPPER%20METHODS%20fc56aaf2f2ac4e9b82b9f9367d760523.md)