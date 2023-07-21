# WRAPPER METHODS

## **Introduction to Wrapper Methods**

Machine learning problems often involve datasets with many features. Some of those features might be very important for a specific machine learning model. Other features might be irrelevant. Given a feature set and a model, we would like to be able to distinguish between important and unimportant features (or even important combinations of features). Wrapper methods do exactly that.

A *wrapper method* for feature selection is an algorithm that selects features by evaluating the performance of a machine learning model on different subsets of features. These algorithms add or remove features one at a time based on how useful those features are to the model.

Wrapper methods have some advantages over filter methods. The main advantage is that wrapper methods evaluate features based on their performance with a specific model. Filter methods, on the other hand, can’t tell how important a feature is to a model.

Another upside of wrapper methods is that they can take into account relationships between features. Sometimes certain features aren’t very useful on their own but instead perform well only when combined with other features. Since wrapper methods test subsets of features, they can account for those situations.

This lesson will explain five different wrapper methods:

- Sequential forward selection
- Sequential backward selection
- Sequential forward floating selection
- Sequential backward floating selection
- Recursive feature elimination

You’ll learn how to implement these algorithms in Python and evaluate the results.

Before we get started, let’s take a look at a dataset that you’ll use throughout this lesson.

## **Setting Up a Logistic Regression Model**

Before we can use a wrapper method, we need to specify a machine learning model. We’ll train a logistic regression model on the `health` data and see how well it performs.

We’ll prepare the data by splitting it into a pandas DataFrame `X` and a pandas Series `y`. `X` will contain the observations of the independent variables, and `y` will contain the observations of the dependent variable.

Here’s an example of how to do this. The `fire` dataset below was taken from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset++) and cleaned for our analysis. Its features are `Temperature`, `RH` (relative humidity), `Ws` (wind speed), `Rain`, `DMC` (Duff Moisture Code), and `FWI` (Fire Weather Index). The final column, `Classes`, contains a `1` if there is a forest fire at a specific location on a given day and `0` if not.

```python
import pandas as pd

# Load the data
fire = pd.read_csv("fire.csv")
# Split independent and dependent variables
X = fire.iloc[:,:-1]
y = fire.iloc[:,-1]
```

We can create a logistic regression model and fit it to `X` and `y` with scikit-learn using the following code.

```python
from sklearn.linear_model import LogisticRegression

# Create and fit the logistic regression model
lr = LogisticRegression()
lr.fit(X, y)
```

Logistic regression models give a probability that an observation belongs to a category. In the `fire` dataset, probabilities greater than 0.5 are considered predictions that there is a fire, and probabilities less than 0.5 are considered predictions that there is no fire. In the `health` dataset, probabilities greater than 0.5 are considered predictions that a patient has breast cancer.

The *accuracy* of a logistic regression model is the percentage of correct predictions that it makes on a testing set. In scikit-learn, you can check the accuracy of a model with the `.score()` method.

```python
print(lr.score(X,y))
```

This outputs:

```python
0.9836065573770492
```

For our testing set, our logistic regression model correctly predicts whether a fire occurred 98.4% of the time.

## **Sequential Forward Selection**

Now that we have a specific machine learning model, we can use a wrapper method to choose a smaller feature subset.

*Sequential forward selection* is a wrapper method that builds a feature set by starting with no features and then adding one feature at a time until a desired number of features is reached. In the first step, the algorithm will train and test a model using only one feature at a time. The algorithm keeps the feature that performs best.

In each subsequent step, the algorithm will test the model on each possible new feature addition. Whichever feature improves model performance the most is then added to the feature subset. This process stops once we have the desired number of features.

Let’s say we want to use three features, and we have five features to choose from: `age`, `height`, `weight`, `blood_pressure`, and `resting_heart_rate`. Sequential forward selection will train your machine learning model on five different feature subsets: one for each feature.

If the model performs best on the subset {`age`}, the algorithm will then train and test the model on the following four subsets:

- {`age`, `height`}
- {`age`, `weight`}
- {`age`, `blood_pressure`}
- {`age`, `resting_heart_rate`}

If the model performs best on {`age`, `resting_heart_rate`}, the algorithm will test the model on the following three subsets:

- {`age`, `height`, `resting_heart_rate`}
- {`age`, `weight`, `resting_heart_rate`}
- {`age`, `blood_pressure`, `resting_heart_rate`}

If the model performs best on {`age`, `weight`, `resting_heart_rate`}, it will stop the algorithm and use that feature set.

Sequential forward selection is a *greedy* algorithm: instead of checking every possible feature set by brute force, it adds whichever feature gives the best immediate performance gain.

## **Sequential Forward Selection with mlxtend**

Recall from a previous exercise that the logistic regression model was about 80.2% accurate at predicting if a patient had breast cancer. But there were NINE different features. Are all of those features necessary, or is it possible that the model could make accurate predictions with fewer features? That would make the model easier to understand, and it could simplify diagnosis.

We will use the `SFS` class from Python’s mlxtend library to implement sequential forward selection and choose a subset of just THREE features for the logistic regression model that we used earlier.

```python
# Set up SFS parameters
sfs = SFS(lr,
           k_features=3, # number of features to select
           forward=True,
           floating=False,
           scoring='accuracy',
           cv=0)
# Fit SFS to our features X and outcome y
sfs.fit(X, y)
```

- The first parameter is the name of the model that you’re using. In the previous exercise, we called the logistic regression model `lr`.
- The parameter `k_features` determines how many features the algorithm will select.
- `forward=True` and `floating=False` ensure that we are using sequential forward selection.
- `scoring` determines how the algorithm will evaluate each feature subset. It’s often okay to use the default value `None` because mlxtend will automatically use a metric that is suitable for whatever scikit-learn model you are using. For this lesson, we’ll set it to `'accuracy'`.
- `cv` allows you to do k-fold cross-validation. We’ll leave it at `0` for this lesson and only evaluate performance on the training set.

We’ll see which features were selected in the next exercise. For now, we just want to fit the `SFS` model.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Load the data
health = pd.read_csv("dataR2.csv")
X = health.iloc[:,:-1]
y = health.iloc[:,-1]

# Logistic regression model
lr = LogisticRegression(max_iter=1000)

# Sequential forward selection
sfs = SFS(lr,k_features=3,forward=True,floating=False,scoring='accuracy',cv=0)
# Fit the equential forward selection model
sfs.fit(X,y)
```

## **Evaluating the Result of Sequential Forward Selection**

The `sfs` object that you fit in the previous exercise contains information about the sequential forward selection that was applied to your feature set. The `.subsets_` attribute allows you to see much of that information, including which feature was chosen at each step and the model’s accuracy after each feature addition.

`sfs.subsets_` is a dictionary that looks something like this:

```python
{1: {'feature_idx': (7,),
  'cv_scores': array([0.93852459]),
  'avg_score': 0.9385245901639344,
  'feature_names': ('FWI',)},
 2: {'feature_idx': (4, 7),
  'cv_scores': array([0.97540984]),
  'avg_score': 0.9754098360655737,
  'feature_names': ('DMC', 'FWI')},
 3: {'feature_idx': (1, 4, 7),
  'cv_scores': array([0.9795082]),
  'avg_score': 0.9795081967213115,
  'feature_names': (' RH', 'DMC', 'FWI')}}
```

The keys in this dictionary are the numbers of features at each step in the sequential forward selection algorithm. The values in the dictionary are dictionaries with information about the feature set at each step. `'avg_score'` is the accuracy of the model with the specified number of features.

In this particular example, the model had an accuracy of about 93.9% after the feature `FWI` was added. The accuracy improved to about 97.5% after a second feature, `DMC`, was added. Once three features were added the accuracy improved to about 98.0%.

You can use this dictionary to easily get a tuple of chosen features or the accuracy of the model after any step.

```python
# Print a tuple of feature names after 5 features are added
print(sfs.subsets_[5]['feature_names'])
```

This outputs:

```python
(' RH', ' Ws', 'Rain ', 'DMC', 'FWI')
```

```python
# Print the accuracy of the model after 5 features are added
print(sfs.subsets_[5]['avg_score'])
```

This outputs:

```python
0.9836065573770492
```

The mlxtend library also makes it easy to visualize how the accuracy of a model changes as sequential forward selection adds features. You can use the code `plot_sfs(sfs.get_metric_dict())` to create a matplotlib figure that plots the model’s performance as a function of the number of features used.

```python
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt

# Plot the accuracy of the model as a function of the number of features
plot_sfs(sfs.get_metric_dict())
plt.show()
```

![https://static-assets.codecademy.com/skillpaths/feature-engineering/wrapper-methods/sfs_fire_plot.png](https://static-assets.codecademy.com/skillpaths/feature-engineering/wrapper-methods/sfs_fire_plot.png)

This plot shows you some of the same information that was in `sfs.subsets_`. The accuracy after one feature was added is about 93.9%, then 97.5% after a second feature is added, and so on.

```python
import pandas as pd
import codecademylib3
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt

# Load the data
health = pd.read_csv("dataR2.csv")
X = health.iloc[:,:-1]
y = health.iloc[:,-1]

# Logistic regression model
lr = LogisticRegression(max_iter=1000)

# Sequential forward selection
sfs = SFS(lr,
          k_features=3,
          forward=True,
          floating=False,
          scoring='accuracy',
          cv=0)
sfs.fit(X, y)

# Print the chosen feature names
print(sfs.subsets_[3]['feature_names'])
# Print the accuracy of the model after sequential forward selection
print(sfs.subsets_[3]['avg_score'])
# Plot the model accuracy
plot_sfs(sfs.get_metric_dict())
plt.show()
```

output

```python
('Age', 'Glucose', 'Insulin')
0.7672413793103449
```

![Untitled](WRAPPER%20METHODS%20fc56aaf2f2ac4e9b82b9f9367d760523/Untitled.png)

## **Sequential Backward Selection with mlxtend**

*Sequential backward selection* is another wrapper method for feature selection. It is very similar to sequential forward selection, but there is one key difference. Instead of starting with no features and adding one feature at a time, sequential backward selection starts with all of the available features and removes one feature at a time.

Let’s again say we want to use three of the following five features: `age`, `height`, `weight`, `blood_pressure`, and `resting_heart_rate`. Sequential backward selection will start by training whatever machine learning model you are using on five different feature subsets, one for each possible feature removal:

- {`height`, `weight`, `blood_pressure`, `resting_heart_rate`}
- {`age`, `weight`, `blood_pressure`, `resting_heart_rate`}
- {`age`, `height`, `blood_pressure`, `resting_heart_rate`}
- {`age`, `height`, `weight`, `resting_heart_rate`}
- {`age`, `height`, `weight`, `blood_pressure`}

Let’s say that out of the five subsets, the model performed best on the subset without `blood_pressure`. Then the algorithm will proceed with the feature set {`age`, `height`, `weight`, `resting_heart_rate`}. It then tries removing each of `age`, `height`, `weight`, and `resting_heart_rate`.

Let’s say that of those four subsets, the model performed best without `weight`. Then it will arrive at the subset {`age`, `height`, `resting_heart_rate`}. The algorithm will stop there since it arrived at the desired number of features.

To implement sequential backward selection in mlxtend you can use the same `SFS` class you used for sequential forward selection. The only difference is that you have to set the parameter `forward` to `False`.

## **Evaluating the Result of Sequential Backward Selection**

As you learned in a previous exercise, `model.subsets_` is a dictionary that contains information about feature subsets from each step of an `SFS` selection model. This works with sequential backward selection just like it did with sequential forward selection.

```python
print(sbs.subsets_)
```

```python
{6: {'feature_idx': (0, 1, 2, 3, 4, 5),
  'cv_scores': array([0.98360656]),
  'avg_score': 0.9836065573770492,
  'feature_names': ('Temperature', ' RH', ' Ws', 'Rain ', 'DMC', 'FWI')},
 5: {'feature_idx': (1, 2, 3, 4, 5),
  'cv_scores': array([0.98360656]),
  'avg_score': 0.9836065573770492,
  'feature_names': (' RH', ' Ws', 'Rain ', 'DMC', 'FWI')},
 4: {'feature_idx': (2, 3, 4, 5),
  'cv_scores': array([0.98360656]),
  'avg_score': 0.9836065573770492,
  'feature_names': (' Ws', 'Rain ', 'DMC', 'FWI')},
 3: {'feature_idx': (2, 4, 5),
  'cv_scores': array([0.9795082]),
  'avg_score': 0.9795081967213115,
  'feature_names': (' Ws', 'DMC', 'FWI')}}
```

You can also use `plot_sfs(sfs.get_metric_dict())` to visualize the results of sequential backward selection.

```python
# Plot the accuracy of the model as a function of the number of features
plot_sfs(sbs.get_metric_dict())
plt.show()
```

![https://static-assets.codecademy.com/skillpaths/feature-engineering/wrapper-methods/sbs_fire_plot.png](https://static-assets.codecademy.com/skillpaths/feature-engineering/wrapper-methods/sbs_fire_plot.png)

```python
import pandas as pd
import codecademylib3
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt

# Load the data
health = pd.read_csv("dataR2.csv")
X = health.iloc[:,:-1]
y = health.iloc[:,-1]

# Logistic regression model
lr = LogisticRegression(max_iter=1000)

# Sequential backward selection
sbs = SFS(lr,
          k_features=3,
          forward=False,
          floating=False,
          scoring='accuracy',
          cv=0)
sbs.fit(X, y)

# Print the chosen feature names
print(sbs.subsets_[3]['feature_names'])
# Print the accuracy of the model after sequential backward selection
print(sbs.subsets_[3]['avg_score'])
# Plot the model accuracy
plot_sfs(sbs.get_metric_dict())
plt.show()
```

output

```python
('Age', 'Glucose', 'Resistin')
0.7413793103448276
```

![Untitled](WRAPPER%20METHODS%20fc56aaf2f2ac4e9b82b9f9367d760523/Untitled%201.png)

Notice that the accuracy sometimes decreases when a feature is removed. Do you think that too much accuracy was lost, or is that a reasonable trade-off to have a simpler model?

## **Sequential Forward and Backward Floating Selection**

*Sequential forward floating selection* is a variation of sequential forward selection. It starts with zero features and adds one feature at a time, just like sequential forward selection, but after each addition, it checks to see if we can improve performance by removing a feature.

- If performance can’t be improved, the floating algorithm will continue to the next step and add another feature.
- If performance can be improved, the algorithm will make the removal that improves performance the most (unless removal of that feature would lead to an infinite loop of adding and removing the same feature over and over again).

For example, let’s say that the algorithm has just added `weight` to the feature set {`age`, `resting_heart_rate`}, resulting in the set {`age`, `weight`, `resting_heart_rate`}. The floating algorithm will test whether it can improve performance by removing `age` or `resting_heart_rate`. If the removal of `age` improves performance, then the algorithm will proceed with the set {`weight`, `resting_heart_rate`}.

*Sequential backward floating selection* works similarly. Starting with all available features, it removes one feature at a time. After each feature removal, it will check to see if any feature additions will improve performance (but it won’t add any features that would result in an infinite loop).

Floating selection algorithms are sometimes preferable to their non-floating counterparts because they test the model on more possible feature subsets. They can detect useful relationships between variables that plain sequential forward and backward selection might miss.

## **Sequential Forward and Backward Floating Selection with mlxtend**

We can implement sequential forward or backward floating selection in mlxtend by setting the parameter `floating` to `True`. The parameter `forward` determines whether mlxtend will use sequential forward floating selection or sequential backward floating selection. As usual, the dictionary `model.subsets_` will contain useful information about the chosen features.

Here’s an implementation of sequential backward floating selection.

```python
# Sequential backward floating selection
sbfs = SFS(lr,
          k_features=5,
          forward=False,
          floating=True,
          scoring='accuracy',
          cv=0)
sbfs.fit(X, y)
```

We can use the `.subsets_` attribute to look at feature names, just like we did with the non-floating sequential selection algorithms.

```python
print(sbfs.subsets_[5]['feature_names'])
```

This outputs:

```python
(' RH', ' Ws', 'Rain ', 'DMC', 'FWI')
```

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Load the data
health = pd.read_csv("dataR2.csv")
X = health.iloc[:,:-1]
y = health.iloc[:,-1]

# Logistic regression model
lr = LogisticRegression(max_iter=1000)

# Sequential forward floating selection
sffs = SFS(lr,
          k_features=3,
          forward=True,
          floating=True,
          scoring='accuracy',
          cv=0)
sffs.fit(X, y)

# Print a tuple with the names of the features chosen by sequential forward floating selection.
print(sffs.subsets_[3]['feature_names'])

# Sequential backward floating selection
sbfs = SFS(lr,
          k_features=3,
          forward=False,
          floating=True,
          scoring='accuracy',
          cv=0)
sbfs.fit(X, y)

# Print a tuple with the names of the features chosen by sequential backward floating selection.
print(sbfs.subsets_[3]['feature_names'])
```

output

```python
('Age', 'Glucose', 'Insulin')
('Age', 'Glucose', 'Resistin')
```

## **Recursive Feature Elimination**

*Recursive feature elimination* is another wrapper method for feature selection. It starts by training a model with all available features. It then ranks each feature according to an importance metric and removes the least important feature. The algorithm then trains the model on the smaller feature set, ranks those features, and removes the least important one. The process stops when the desired number of features is reached.

In regression problems, features are ranked by the size of the absolute value of their coefficients. For example, let’s say that we trained a regression model with five features and got the following regression coefficients.

| Feature | Regression coefficient |
| --- | --- |
| age | 2.5 |
| height | 7.0 |
| weight | -4.3 |
| blood_pressure | -5.7 |
| resting_heart_rate | -4.6 |

The regression coefficient for `age` has the smallest absolute value, so it is ranked least important by recursive feature elimination. It will be removed, and the remaining four features will be re-ranked after the model is trained without `age`.

It’s important to note that you might need to standardize data before doing recursive feature elimination. In regression problems in particular, it’s necessary to standardize data so that the scale of features doesn’t affect the size of the coefficients.

Note that recursive feature elimination is different from sequential backward selection. Sequential backward selection removes features by training a model on a collection of subsets (one for each possible feature removal) and greedily proceeding with whatever subset performs best. Recursive feature elimination, on the other hand, only trains a model on one feature subset before deciding which feature to remove next.

This is one advantage of recursive feature elimination. Since it only needs to train and test a model on one feature subset per feature removal, it can be much faster than the sequential selection methods that we’ve covered.

## **Recursive Feature Elimination with scikit-learn**

We can use scikit-learn to implement recursive feature elimination. Since we’re using a logistic regression model, it’s important to standardize data before we proceed.

We can standardize features using scikit-learn’s `StandardScaler()`.

```python
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
```

Once the data is standardized, you can train the model and do recursive feature elimination using `RFE()` from scikit-learn. As before with the sequential feature selection methods, you have to specify a scikit-learn model for the `estimator` parameter (in this case, `lr` for our logistic regression model). `n_features_to_select` is self-explanatory: set it to the number of features you want to select.

```python
from sklearn.feature_selection import RFE

# Recursive feature elimination
rfe = RFE(lr, n_features_to_select=2)
rfe.fit(X, y)
```

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

# Load the data
health = pd.read_csv("dataR2.csv")
X = np.array(health.iloc[:,:-1])
y = np.array(health.iloc[:,-1])

# Standardize the data
X = StandardScaler().fit_transform(X)

# Logistic regression model
lr = LogisticRegression(max_iter=1000)

# Recursive feature elimination
rfe = RFE(lr,n_features_to_select=3)
rfe.fit(X,y)
```

## **Evaluating the Result of Recursive Feature Elimination**

You can inspect the results of recursive feature elimination by looking at `rfe.ranking_` and `rfe.support_`.

`rfe.ranking_` is an array that contains the rank of each feature. Here are the features from the `fire` dataset:

```python
['Temperature', 'RH', 'Ws', 'Rain', 'DMC', 'FWI']
```

Here are the feature rankings after recursive feature elimination is done on the `fire` dataset.

```python
print(rfe.ranking_)
```

```python
[2 5 4 1 3 1]
```

A `1` at a certain index indicates that recursive feature elimination kept the feature at the same index. In this example, the model kept the features at indices 3 and 5: `Rain` and `FWI`. The other numbers indicate at which step a feature was removed. The `5` (the highest rank in the array) at index 1 means that `RH` (the feature at index 1) was removed first. The `4` at index 2 means that `Ws` (the feature at index 2) was removed in the next step, and so on.

`rfe.support_` is an array with `True` and `False` values that indicate which features were chosen. Here’s an example of what this looks like, again using the `fire` dataset.

```python
print(rfe.support_)
```

```python
[False False False  True False  True]
```

This array indicates that the features at indices 3 and 5 were chosen. The features at indices 0, 1, 2, and 4 were eliminated.

If you have a list of feature names, you can use a list comprehension and `rfe.support_` to get a list of chosen feature names.

```python
# features is a list of feature names
# Get a list of features chosen by rfe
rfe_features = [f for (f, support) in zip(features, rfe.support_) if support]

print(rfe_features)
```

```python
['Rain ', 'FWI']
```

You can use `rfe.score(X, y)` to check the accuracy of the model.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

# Load the data
health = pd.read_csv("dataR2.csv")
X = health.iloc[:,:-1]
y = health.iloc[:,-1]

# Create a list of feature names
feature_list = list(X.columns)

# Standardize the data
X = StandardScaler().fit_transform(X)

# Logistic regression
lr = LogisticRegression(max_iter=1000)

# Recursive feature elimination
rfe = RFE(estimator=lr, n_features_to_select=3)
rfe.fit(X, y)

# List of features chosen by recursive feature elimination
rfe_features = [f for (f, support) in zip(feature_list, rfe.support_) if support]

# Print the accuracy of the model with features chosen by recursive feature elimination
print(rfe.score(X, y))
```

output

```python
0.7327586206896551
```

# **Feature Importance**

**Learn about feature importance and how to calculate it.**

## **Introduction**

When we fit a supervised machine learning (ML) model, we often want to understand which features are most associated with our outcome of interest. Features that are highly associated with the outcome are considered more “important.” In this article, we’ll introduce you to the concept of **feature importance** through a discussion of:

- Tree-based feature importance
    - [Gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)
    - Implementation in [scikit-learn](https://scikit-learn.org/stable/)
- Other methods for estimating feature importance

### **Feature importance in an ML workflow**

There are many reasons why we might be interested in calculating feature importances as part of our machine learning workflow. For example:

- Feature importance is often used for dimensionality reduction.
    - We can use it as a filter method to remove irrelevant features from our model and only retain the ones that are most highly associated with our outcome of interest.
    - Wrapper methods such as recursive feature elimination use feature importance to more efficiently search the feature space for a model.
- Feature importance may also be used for model inspection and communication. For example, stakeholders may be interested in understanding which features are most important for prediction. Feature importance can help us answer this question.

### **Calculating feature importance**

There are many different ways to calculate feature importance for different kinds of machine learning models. In this section, we’ll investigate one tree-based method in a little more detail: **Gini impurity**.

### **Gini impurity**

Imagine, for a moment, that you’re interested in building a model to screen candidates for a particular job. In order to build this model, you’ve collected some data about candidates who you’ve hired and rejected in the past. For each of these candidates, suppose that you have data on years of experience and certification status. Consider the following two simple decision trees that use these features to predict whether the candidate was hired:

![https://static-assets.codecademy.com/skillpaths/feature-engineering/feature-importance/simple_tree_gini.png](https://static-assets.codecademy.com/skillpaths/feature-engineering/feature-importance/simple_tree_gini.png)

![https://static-assets.codecademy.com/skillpaths/feature-engineering/feature-importance/simple_tree_certified.png](https://static-assets.codecademy.com/skillpaths/feature-engineering/feature-importance/simple_tree_certified.png)

Which of these features seems to be more important for predicting whether a candidate will be hired? In the first example, we saw that *most* candidates who had >5 years of experience were hired and *most* candidates with <5 years were rejected; however, *all* candidates with certifications were hired and *all* candidates without them were rejected.

Gini impurity is related to the extent to which observations are well separated based on the outcome variable at each node of the decision tree. For example, in the two trees above, the Gini impurity is higher in the node with all candidates (where there are an equal number of rejected and hired candidates) and lower in the nodes after the split (where most or all of the candidates in each grouping have the same outcome — either hired or rejected).

To estimate feature importance, we can calculate the Gini gain: the amount of Gini impurity that was eliminated at each branch of the decision tree. In this example, certification status has a higher Gini gain and is therefore considered to be more important based on this metric.

### **Gini importance in scikit-learn**

To demonstrate how we can estimate feature importance using Gini impurity, we’ll use the breast cancer dataset from **`sklearn`**. This dataset contains features related to breast tumors. The outcome variable is the diagnosis: either malignant or benign. To start, we’ll load the dataset and split it into a training and test set:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets

dataset = datasets.load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
```

Next, we’ll fit a decision tree to predict the diagnosis using **`sklearn.tree.DecisionTreeClassifier()`**. Note that we’re setting **`criterion= 'gini'`**. This actually tells the function to build the decision tree by splitting each node based on the feature that has the highest Gini gain. By building the tree in this way, we’ll be able to access the Gini importances later.

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion='gini')

# Fit the decision tree classifier
clf = clf.fit(X_train, y_train)
```

Next, we can access the feature importances based on Gini impurity as follows:

```python
# Print the feature importances
feature_importances = clf.feature_importances_
```

Finally, we’ll visualize these values using a bar chart:

```python
import seaborn as sns

# Sort the feature importances from greatest to least using the sorted indices
sorted_indices = feature_importances.argsort()[::-1]
sorted_feature_names = data.feature_names[sorted_indices]
sorted_importances = feature_importances[sorted_indices]

# Create a bar plot of the feature importances
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(sorted_importances, sorted_feature_names)
```

![https://static-assets.codecademy.com/Paths/machine-learning-engineer-career-path/feature_imp.png](https://static-assets.codecademy.com/Paths/machine-learning-engineer-career-path/feature_imp.png)

Based on this output, we could conclude that the features **`mean concave points`**, **`worst area`** and **`worst texture`** are most predictive of a malignant tumor. There are also many features with importances close to zero which we may want to exclude from our model.

### **Pros and cons of using Gini importance**

Because Gini impurity is used to train the decision tree itself, it is computationally inexpensive to calculate. However, Gini impurity is somewhat biased toward selecting numerical features (rather than categorical features). It also does not take into account the correlation between features. For example, if two highly correlated features are both equally important for predicting the outcome variable, one of those features may have low Gini-based importance because all of it’s explanatory power was ascribed to the other feature. This issue can be mediated by removing redundant features before fitting the decision tree.

### **Other measures of feature importance**

There are many other methods for estimating feature importance beyond calculating Gini gain for a single decision tree. We’ll explore a few of these methods below.

### **Aggregate methods**

Random forests are an ensemble-based machine learning algorithm that utilize many decision trees (each with a subset of features) to predict the outcome variable. Just as we can calculate Gini importance for a single tree, we can calculate average Gini importance across an entire random forest to get a more robust estimate.

### **Permutation-based methods**

Another way to test the importance of particular features is to essentially remove them from the model (one at a time) and see how much predictive accuracy suffers. One way to “remove” a feature is to randomly permute the values for that feature, then refit the model. This can be implemented with any machine learning model, including non-tree-based- methods. However, one potential drawback is that it is computationally expensive because it requires us to refit the model many times.

### **Coefficients**

When we fit a general(ized) linear model (for example, a linear or logistic regression), we estimate coefficients for each predictor. If the original features were standardized, these coefficients can be used to estimate relative feature importance; larger absolute value coefficients are more important. This method is computationally inexpensive because coefficients are calculated when we fit the model. It is also useful for both classification and regression problems (i.e., categorical and continuous outcomes). However, similar to the other methods described above, these coefficients do not take highly correlated features into account.

## **Conclusion**

In this article, we’ve covered a few different examples of feature importance metrics, including how to interpret and calculate them. We learned about:

- Gini impurity
- How to calculate Gini-based feature importance for a decision tree in **`sklearn`**
- Other methods for calculating feature importance, including:
    - Aggregate methods
    - Permutation-based methods
    - Coefficients

Feature importance is an important part of the machine learning workflow and is useful for feature engineering and model explanation, alike!