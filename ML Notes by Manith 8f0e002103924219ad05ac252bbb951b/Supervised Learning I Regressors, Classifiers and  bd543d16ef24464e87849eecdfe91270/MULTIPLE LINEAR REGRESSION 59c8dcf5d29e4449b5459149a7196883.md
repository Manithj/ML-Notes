# MULTIPLE LINEAR REGRESSION

**Introduction to Multiple Linear Regression**

Linear regression is useful when we want to predict the values of a variable from its relationship with other variables. There are two different types of linear regression models ([simple linear regression](https://www.codecademy.com/paths/data-science/tracks/dspath-supervised/modules/dspath-linear-regression/lessons/linear-regression) and multiple linear regression).

In predicting the price of a home, one factor to consider is the size of the home. The relationship between those two variables, price and size, is important, but there are other variables that factor in to pricing a home: location, air quality, demographics, parking, and more. When making predictions for price, our *dependent variable*, we’ll want to use multiple *independent variables*. To do this, we’ll use Multiple Linear Regression.

**Multiple Linear Regression** uses two or more independent variables to predict the values of the dependent variable. It is based on the following equation that we’ll explore later on:

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled.png)

**StreetEasy Dataset**

You’ll learn multiple linear regression by performing it on this dataset. It contains information about apartments in New York.

```python
import codecademylib3_seaborn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['size_sqft','building_age_yrs']]
y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

ols = LinearRegression()

ols.fit(x_train, y_train)

# Plot the figure

fig = plt.figure(1, figsize=(6, 4))
plt.clf()

elev = 43.5
azim = -110

ax = Axes3D(fig, elev=elev, azim=azim)

ax.scatter(x_train[['size_sqft']], x_train[['building_age_yrs']], y_train, c='k', marker='+')

ax.plot_surface(np.array([[0, 0], [4500, 4500]]), np.array([[0, 140], [0, 140]]), ols.predict(np.array([[0, 0, 4500, 4500], [0, 140, 0, 140]]).T).reshape((2, 2)), alpha=.7)

ax.set_xlabel('Size (ft$^2$)')
ax.set_ylabel('Building Age (Years)')
ax.set_zlabel('Rent ($)')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

# Add the code below:
plt.show()
```

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%201.png)

## **StreetEasy Dataset**

![https://content.codecademy.com/programs/machine-learning/multiple-linear-regression/streeteasy.jpg](https://content.codecademy.com/programs/machine-learning/multiple-linear-regression/streeteasy.jpg)

**[StreetEasy](https://www.streeteasy.com/)** is New York City’s leading real estate marketplace — from studios to high-rises, Brooklyn Heights to Harlem.

In this lesson, you will be working with a dataset that contains a sample of 5,000 rentals listings in `Manhattan`, `Brooklyn`, and `Queens`, active on StreetEasy in June 2016.

It has the following columns:

- `rental_id`: rental ID
- `rent`: price of rent in dollars
- `bedrooms`: number of bedrooms
- `bathrooms`: number of bathrooms
- `size_sqft`: size in square feet
- `min_to_subway`: distance from subway station in minutes
- `floor`: floor number
- `building_age_yrs`: building’s age in years
- `no_fee`: does it have a broker fee? (0 for fee, 1 for no fee)
- `has_roofdeck`: does it have a roof deck? (0 for no, 1 for yes)
- `has_washer_dryer`: does it have washer/dryer in unit? (0/1)
- `has_doorman`: does it have a doorman? (0/1)
- `has_elevator`: does it have an elevator? (0/1)
- `has_dishwasher`: does it have a dishwasher (0/1)
- `has_patio`: does it have a patio? (0/1)
- `has_gym`: does the building have a gym? (0/1)
- `neighborhood`: (ex: Greenpoint)
- `borough`: (ex: Brooklyn)

More information about this dataset can be found in the [StreetEasy Dataset](https://www.codecademy.com/content-items/d19f2f770877c419fdbfa64ddcc16edc) article.

Let’s start by doing exploratory data analysis to understand the dataset better. We have broken the dataset for you into:

- **[manhattan.csv](https://raw.githubusercontent.com/Codecademy/datasets/master/streeteasy/manhattan.csv)**
- **[brooklyn.csv](https://raw.githubusercontent.com/Codecademy/datasets/master/streeteasy/brooklyn.csv)**
- **[queens.csv](https://raw.githubusercontent.com/Codecademy/datasets/master/streeteasy/queens.csv)**

```python
import codecademylib3_seaborn
import pandas as pd

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/queens.csv")

df = pd.DataFrame(streeteasy)

print(df.head())
```

## **Training Set vs. Test Set**

As with most machine learning algorithms, we have to split our dataset into:

- **Training set**: the data used to fit the model
- **Test set**: the data partitioned away at the very start of the experiment (to provide an unbiased evaluation of the model)

![https://content.codecademy.com/programs/machine-learning/multiple-linear-regression/split.svg](https://content.codecademy.com/programs/machine-learning/multiple-linear-regression/split.svg)

In general, putting 80% of your data in the training set and 20% of your data in the test set is a good place to start.

Suppose you have some values in `x` and some values in `y`:

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
```

Here are the parameters:

- `train_size`: the proportion of the dataset to include in the train split (between 0.0 and 1.0)
- `test_size`: the proportion of the dataset to include in the test split (between 0.0 and 1.0)
- `random_state`: the seed used by the random number generator [optional]

To learn more, here is a [Training Set vs Validation Set vs Test Set article](https://www.codecademy.com/articles/training-set-vs-validation-set-vs-test-set).

```python
import codecademylib3_seaborn
import pandas as pd

# import train_test_split
from sklearn.model_selection import train_test_split

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms','bathrooms','size_sqft','min_to_subway','floor','building_age_yrs','no_fee','has_roofdeck','has_washer_dryer','has_doorman','has_elevator','has_dishwasher','has_patio','has_gym']]

y = df['rent']

x_train , x_test, y_train,y_test = train_test_split(x,y,train_size = 0.8, test_size = 0.2 ,random_state = 6)

print(x_train.shape)
print(x_test.shape)
 
print(y_train.shape)
print(y_test.shape)
```

## **Multiple Linear Regression: Scikit-Learn**

Now we have the training set and the test set, let’s use scikit-learn to build the linear regression model!

The steps for multiple linear regression in scikit-learn are identical to the steps for simple linear regression. Just like simple linear regression, we need to import `LinearRegression` from the `linear_model` module:

```python
from sklearn.linear_model import LinearRegression
```

Then, create a `LinearRegression` model, and then fit it to your `x_train` and `y_train` data:

```python
mlr = LinearRegression()

mlr.fit(x_train, y_train)
# finds the coefficients and the intercept value
```

We can also use the `.predict()` function to pass in x-values. It returns the y-values that this plane would predict:

```python
y_predicted = mlr.predict(x_test)
# takes values calculated by `.fit()` and the `x` values, plugs them into the multiple linear regression equation, and calculates the predicted y values.
```

We will start by using two of these columns to teach you how to predict the values of the dependent variable, prices.

```python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

# Add the code here:
mlr = LinearRegression()
mlr.fit(x_train,y_train)

y_predict = mlr.predict(x_train)

# Sonny doesn't have an elevator so the 11th item in the list is a 0
sonny_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]
 
predict = mlr.predict(sonny_apartment)
 
print("Predicted rent: $%.2f" % predict)
```

## **Visualizing Results with Matplotlib**

You’ve performed Multiple Linear Regression, and you also have the predictions in `y_predict`. However, we don’t have insight into the data, yet. In this exercise, you’ll create a 2D scatterplot to see how the independent variables impact prices.

**How do you create 2D graphs?**

Graphs can be created using Matplotlib’s `pyplot` module. Here is the code with inline comments explaining how to plot using Matplotlib’s `.scatter()`:

```
# Create a scatter plot
plt.scatter(x, y, alpha=0.4)

# Create x-axis label and y-axis label
plt.xlabel("the x-axis label")
plt.ylabel("the y-axis label")

# Create a title
plt.title("title!")

# Show the plot
plt.show()

```

We want to create a scatterplot like this:

![https://content.codecademy.com/courses/matplotlib/visualization.png](https://content.codecademy.com/courses/matplotlib/visualization.png)

```python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

lm = LinearRegression()

model=lm.fit(x_train, y_train)

y_predict = lm.predict(x_test)

plt.scatter(y_test, y_predict, alpha=0.4)

plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual Rent vs Predicted Rent")

plt.show()
```

## **Multiple Linear Regression Equation**

Now that we have implemented Multiple Linear Regression, we will learn how to tune and evaluate the model. Before we do that, however, it’s essential to learn the equation behind it.

**Equation 6.1** The equation for multiple linear regression that uses two independent variables is this:

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%202.png)

**Equation 6.2** The equation for multiple linear regression that uses three independent variables is this:

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%203.png)

**Equation 6.3** As a result, since multiple linear regression can use any number of independent variables, its general equation becomes:

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%204.png)

Here, *m1*, *m2*, *m3*, … *mn* refer to the **coefficients**, and *b* refers to the **intercept** that you want to find. You can plug these values back into the equation to compute the predicted *y* values.

Remember, with `sklearn`‘s `LinearRegression()` method, we can get these values with ease.

The `.fit()` method gives the model two variables that are useful to us:

- `.coef_`, which contains the coefficients
- `.intercept_`, which contains the intercept

After performing multiple linear regression, you can print the coefficients using `.coef_`.

Coefficients are most helpful in determining which independent variable carries more weight. For example, a coefficient of -1.345 will impact the rent more than a coefficient of 0.238, with the former impacting prices negatively and latter positively.

## **Correlations**

In our Manhattan model, we used 14 variables, so there are 14 coefficients:

```
[ -302.73009383  1199.3859951  4.79976742
 -24.28993151  24.19824177 -7.58272473
-140.90664773  48.85017415 191.4257324
-151.11453388  89.408889  -57.89714551
-19.31948556  -38.92369828 ]
```

- `bedrooms` - number of bedrooms
- `bathrooms` - number of bathrooms
- `size_sqft` - size in square feet
- `min_to_subway` - distance from subway station in minutes
- `floor` - floor number
- `building_age_yrs` - building’s age in years
- `no_fee` - has no broker fee (0 for fee, 1 for no fee)
- `has_roofdeck` - has roof deck (0 for no, 1 for yes)
- `has_washer_dryer` - has in-unit washer/dryer (0/1)
- `has_doorman` - has doorman (0/1)
- `has_elevator` - has elevator (0/1)
- `has_dishwasher` - has dishwasher (0/1)
- `has_patio` - has patio (0/1)
- `has_gym` - has gym (0/1)

To see if there are any features that don’t affect price linearly, let’s graph the different features against `rent`.

**Interpreting graphs**

In regression, the independent variables will either have a positive linear relationship to the dependent variable, a negative linear relationship, or no relationship. A negative linear relationship means that as X values *increase*, Y values will *decrease*. Similarly, a positive linear relationship means that as X values *increase*, Y values will also *increase*.

Graphically, when you see a downward trend, it means a negative linear relationship exists. When you find an upward trend, it indicates a positive linear relationship. Here are two graphs indicating positive and negative linear relationships:

![https://content.codecademy.com/programs/machine-learning/multiple-linear-regression/correlations.png](https://content.codecademy.com/programs/machine-learning/multiple-linear-regression/correlations.png)

```python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

# Input code here:

plt.scatter(df[['size_sqft']], df[['rent']], alpha=0.4)
# plt.scatter(df[['floor']], df[['rent']])
# plt.scatter(df[['min_to_subway']], df[['rent']])

plt.show()
```

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%205.png)

## **Evaluating the Model's Accuracy**

When trying to evaluate the accuracy of our multiple linear regression model, one technique we can use is **Residual Analysis**.

The difference between the actual value *y*, and the predicted value *ŷ* is the **residual *e***. The equation is:

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%206.png)

In the StreetEasy dataset, *y* is the actual rent and the *ŷ* is the predicted rent. The real *y* values should be pretty close to these predicted *y* values.

`sklearn`‘s `linear_model.LinearRegression` comes with a `.score()` method that returns the coefficient of determination R² of the prediction.

The coefficient R² is defined as:

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%207.png)

where *u* is the residual sum of squares:

```
((y - y_predict) ** 2).sum()
```

and *v* is the total sum of squares (TSS):

```
((y - y.mean()) ** 2).sum()
```

The TSS tells you how much variation there is in the y variable.

R² is the percentage variation in y explained by all the x variables together.

For example, say we are trying to predict `rent` based on the `size_sqft` and the `bedrooms` in the apartment and the R² for our model is 0.72 — that means that all the x variables (square feet and number of bedrooms) together explain 72% variation in y (`rent`).

Now let’s say we add another x variable, building’s age, to our model. By adding this third relevant x variable, the R² is expected to go up. Let say the new R² is 0.95. This means that square feet, number of bedrooms and age of the building *together* explain 95% of the variation in the rent.

The best possible R² is 1.00 (and it can be negative because the model can be arbitrarily worse). Usually, a R² of 0.70 is considered good.

```python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

mlr = LinearRegression()

model=mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)

# Input code here:
print("Train score:")
print(mlr.score(x_train, y_train))

print("Test score:")
print(mlr.score(x_test, y_test))
```

**Review**

Great work! Let’s review the concepts before you move on:

- **Multiple Linear Regression** uses two or more variables to make predictions about another variable:

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%208.png)

- Multiple linear regression uses a set of independent variables and a dependent variable. It uses these variables to learn how to find optimal parameters. It takes a labeled dataset and learns from it. Once we confirm that it’s learned correctly, we can then use it to make predictions by plugging in new `x` values.
- We can use scikit-learn’s `LinearRegression()` to perform multiple linear regression.
- **Residual Analysis** is used to evaluate the regression model’s accuracy. In other words, it’s used to see if the model has learned the coefficients correctly.
- Scikit-learn’s `linear_model.LinearRegression` comes with a `.score()` method that returns the coefficient of determination R² of the prediction. The best score is 1.0.

# **Solving a Regression Problem: Ordinary Least Squares to Gradient Descent**

**Learn about how the linear regression problem is solved analytically (using Ordinary Least Squares) and algorithmically (using Gradient Descent) and how these two methods are connected!**

Linear regression finds a linear relationship between one or more predictor variables and an outcome variable. This article will explore two different ways of finding linear relationships: *Ordinary Least Squares* and *Gradient Descent*.

## **Ordinary Least Squares**

To understand the method of least squares, let’s take a look at how to set up the linear regression problem with linear equations. We’ll use the [Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) as an example. The outcome variable, Y, is a measure of disease progression. There are 10 predictor variables in this dataset, but to simplify things let’s take a look at just one of them: BP (average blood pressure). Here are the first five rows of data:

| BP | Y |
| --- | --- |
| 32.1 | 151 |
| 21.6 | 75 |
| 0.5 | 141 |
| 25.3 | 206 |
| 23 | 135 |

We can fit the data with the following simple linear regression model with slope *m* and intercept *b*:

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%209.png)

This equation is actually short-hand for a large number of equations — one for each patient in our dataset. The first five equations (corresponding to the first five rows of the dataset) are:

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%2010.png)

When we fit this linear regression model, we are trying to find the values of *m* and *b* such that the sum of the squared error terms above (e.g., *error_1^2 + error_2^2 + error_3^2 + error_4^2 + error_5^2 + ….*) is minimized.

We can create a column matrix of Y (the outcome variable), a column matrix of BP (the predictor variable), and a column matrix of the errors and rewrite the five equations above as one matrix equation:

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%2011.png)

Using the rules of matrix addition and multiplication, it is possible to simplify this to the following.

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%2012.png)

In total we have 4 matrices in this equation:

- A one-column matrix on the left hand side of the equation containing the outcome variable values that we will call *Y*
- A two-column matrix on the right hand side that contains a column of 1’s and a column of the predictor variable values (*BP* here) that we will call *X*.
- A one-column matrix containing the intercept *b* and the slope *m*, i.e, the solution matrix that we will denote by the Greek letter *beta*. The goal of the regression problem is to find this matrix.
- A one-column matrix of the residuals or errors, the error matrix. The regression problem can be solved by minimizing the sum of the squares of the elements of this matrix. The error matrix will be denoted by the Greek letter *epsilon*.

Using these shorthands, the matrix representation of the regression equation is thus:

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%2013.png)

Ordinary Least Squares gives us an explicit formula for *beta*. Here’s the formula:

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%2014.png)

A couple of reminders: X^T is the *transpose* of X. M^{-1} is the *inverse* of a matrix. We won’t review these terms here, but if you want to know more you can check the Wikipedia articles for the [transpose](https://en.wikipedia.org/wiki/Transpose) and [invertible matrices](https://en.wikipedia.org/wiki/Invertible_matrix).

This looks like a fairly simple formula. In theory, you should be able to plug in X and Y, do the computations, and get *beta*. But it’s not always so simple.

First of all, it’s possible that the matrix XX^T might not even have an inverse. This will be the case if there happens to be an exact linear relationship between some of the columns of X. If there is such a relationship between the columns of X, we say that X is *multicollinear*. For example, your data set might contain temperature readings in both Fahrenheit and Celsius. Those columns would be linearly related, and thus XX^T would not have an inverse.

In practice, you also have to watch out for data that is almost multicollinear. For example, a data set might have Fahrenheit and Celsius readings that are rounded to the nearest degree. Due to rounding error, those columns would not be perfectly correlated. In that case it would still be possible to compute the inverse of XX^T, but it would lead to other problems. Dealing with situations like that is beyond the scope of this article, but you should be aware that multicollinearity can be troublesome.

Another drawback of the OLS equation for *beta* is that it can take a long time to compute. Matrix multiplication and matrix inversion are both computationally intensive operations. Data sets with a large of number of predictor variables and rows can make these computations impractical.

## **Gradient Descent**

*Gradient descent* is a numerical technique that can determine regression parameters without resorting to OLS. It’s an iterative process that uses calculus to get closer and closer to the exact coefficients one step at a time. To introduce the concept, we’ll look at a simple example: linear regression with one predictor variable. For each row of data, we have the following equation:

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%2015.png)

The sum of the squared errors is

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%2016.png)

This is the loss function. It depends on two variables: m and b. [Here](https://content.codecademy.com/programs/data-science-path/line-fitter/line-fitter.html)‘s an interactive plot where you can tune the parameters m and b and see how it affects the loss.

Try changing m and b and observe how those changes affect the loss. You can also try to come up with an algorithm for how to adjust m and b in order to minimize the loss.

As you adjust the sliders and try to minimize the loss, you might notice that there is a sweet spot where changing either m or b too far in either direction will increase the loss. Get too far away from that sweet spot, and small changes in m or b will result in bigger changes to the loss.

If playing with the sliders made you think about rates of change, derivatives, and calculus, then you’re well on your way toward understanding gradient descent. The *gradient* of a function is a calculus concept that’s very similar to the derivative of a function. We won’t cover the technical definition here, but you can think of it as a vector that points uphill in the steepest direction. The steeper the slope, the larger the gradient. If you step in the direction opposite of the gradient, you will move downhill.

That’s where the *descent* part of the gradient descent comes in. Imagine you’re standing on some undulating terrain with peaks and valleys. Now take a step in the opposite direction of the gradient. If the gradient is large (in other words, if the slope you’re on is steep), take a big step. If the gradient is small, take a small step. If you repeat this process enough times, you’ll probably end up at the bottom of a valley. By going against the gradient, you’ve minimized your elevation!

Here’s an example of how this works in two dimensions.

![https://content.codecademy.com/programs/data-science-path/linear_regression/Linear_regression_gif_1.gif](https://content.codecademy.com/programs/data-science-path/linear_regression/Linear_regression_gif_1.gif)

Let’s take a closer look at how gradient descent works in the case of linear regression with one predictor variable. The process always starts by making starting guesses for m and b. The initial guesses aren’t important. You could make random guesses, or just start with m=0 and b=0. The initial guesses will be adjusted by using gradient formulas.

Here’s the gradient formula for b. This formula can be obtained by differentiating the average of the squared error terms with respect to b.

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%2017.png)

In this formula,

- N is the total number of observations in the data set,
- x_i and y_i are the observations,
- m is the current guess for the slope of the linear regression equation, and
- b is the current guess for the intercept of the linear regression equation.

Here’s the gradient formula for m. Again, this can be obtained by differentiating the average of the squared error terms with respect to m.

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%2018.png)

The next step of gradient descent is to adjust the current guesses for m and b by subtracting a number proportional the gradient.

Our new guess for b is

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%2019.png)

Our new guess for m is

![Untitled](MULTIPLE%20LINEAR%20REGRESSION%2059c8dcf5d29e4449b5459149a7196883/Untitled%2020.png)

For now, *eta* is just a constant. We’ll explain it in the next section. Once we get new guesses for m and b, we recompute the gradient and continue the process.

Multiple choice

Which of the following statements is FALSE?

In the context of linear regression, Ordinary Least Squares and gradient descent are two different techniques that minimize two different loss functions.

Ordinary Least Squares regression uses a matrix formula to find parameters that minimize loss.

Gradient descent can be preferable to Ordinary Least Squares regression when dealing with a very large number of variables.

Gradient descent is an iterative process.

## **Learning Rate and Convergence**

How big should your steps be when doing gradient descent? Imagine you’re trying to get to the bottom of a valley, one step at a time. If you step one inch at time, it could take a very long time to get to the bottom. You might want to take bigger steps. On the other extreme, imagine you could cover a whole mile in a single step. You’d cover a lot of ground, but you might step over the bottom of the valley and end up on a mountain!

The size of the steps that you take during gradient descent depend on the gradient (remember that we take big steps when the gradient is steep, and small steps when the gradient is small). In order to further tune the size of the steps, machine learning algorithms multiply the gradient by a factor called the *learning rate*. If this factor is big, gradient descent will take bigger steps and hopefully reach the bottom of the valley faster. In other words, it “learns” the regression parameters faster. But if the learning rate is too big, gradient descent can overshoot the bottom of the valley and fail to converge.

How do you know when to stop doing gradient descent? Imagine trying to find the bottom of a valley if you were blindfolded. How would you know when you reached the lowest point?

If you’re walking downhill (or doing gradient descent on a loss function), sooner or later you’ll reach a point where everything flattens out and moving against the gradient will only reduce your elevation by a negligible amount. When that happens, we say that gradient descent *converges*. You might have noticed this when you were adjusting m and b on the interactive graph: when m and b are both near their sweet spot, small adjustments to m and b didn’t affect the loss much.

To summarize what we’ve learned so far, here are the steps of gradient descent. We’ll denote the learning rate by *eta*.

1. Set initial guesses for m and b
2. Replace m with m + *eta* * (-gradient) and replace b with b + *eta* * (-gradient)
3. Repeat step 2 until convergence

If the algorithm fails to converge because the loss increases after some steps, the learning rate is probably too large. If the algorithm runs for a long time without converging, then the learning rate is probably too small.

Multiple choice

Gradient descent is being used to do linear regression. After each iteration of gradient descent, the loss gets larger. Which of the following should you try?

Make the learning rate smaller.

Make the learning rate larger.

Keep running the gradient descent algorithm until it converges.

## **Implementation in sci-kit learn**

Version 1.0.3 of the scikit-learn library has two different linear regression models: one that uses OLS and another that uses a variation of gradient descent.

The **`LinearRegression`** model uses OLS. For most applications this is a good approach. Even if a data set has hundreds of predictor variables or thousands of observations, your computer will have no problem computing the parameters using OLS. One advantage of OLS is that it is guaranteed to find the exact optimal parameters for linear regression. Another advantage is that you don’t have to worry about what the learning rate is or whether the gradient descent algorithm will converge.

Here’s some code that uses **`LinearRegression`**.

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

# Import the data set
X, y = load_diabetes(return_X_y=True)

# Create the OLS linear regression model
ols = LinearRegression()

# Fit the model to the data
ols.fit(X, y)

# Print the coefficients of the model
print(ols.coef_)

# Print R^2
print(ols.score(X, y))

```

```
[ -10.01219782 -239.81908937  519.83978679  324.39042769 -792.18416163
  476.74583782  101.04457032  177.06417623  751.27932109   67.62538639]
0.5177494254132934

```

Scikit-learn’s **`SGDRegressor`** model uses a variant of gradient descent called *stochastic gradient descent* (or *SGD* for short). SGD is very similar to gradient descent, but instead of using the actual gradient it uses an approximation of the gradient that is more efficient to compute. This model is also sophisticated enough to adjust the learning rate as the SGD algorithm iterates, so in many cases you won’t have to worry about setting the learning rate.

**`SGDRegressor`** also uses a technique called *regularization* that encourages the model to find smaller parameters. Regularization is beyond the scope of this article, but it’s important to note that the use of regularization can sometimes result in finding different coefficients than OLS would have.

If your data set is simply too large for your computer to handle OLS, you can use **`SGDRegressor`**. It will not find the exact optimal parameters, but it will get close enough for all practical purposes and it will do so without using too much computing power. Here’s an example.

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import SGDRegressor

# Import the data set
X, y = load_diabetes(return_X_y=True)

# Create the SGD linear regression model
# max_iter is the maximum number of iterations of SGD to try before halting
sgd = SGDRegressor(max_iter = 10000)

# Fit the model to the data
sgd.fit(X, y)

# Print the coefficients of the model
print(sgd.coef_)

# Print R^2
print(sgd.score(X, y))

```

```
[  12.19842555 -177.93853188  463.50601685  290.64175509  -33.34621692
  -94.62205923 -202.87056914  129.75873577  386.77536299  123.17079841]
0.5078357600233131

```

## **Gradient Descent in Other Machine Learning Algorithms**

Gradient descent can be used for much more than just linear regression. In fact, it can be used to train any machine learning algorithm as long as the ML algorithm has a loss function that is a differentiable function of the ML algorithm’s parameters. In more intuitive terms, gradient descent can be used whenever the loss function looks like smooth terrain with hills and valleys (even if those hills and valleys live in a space with more than 3 dimensions).

Gradient descent (or variations of it) can be used to find parameters in logistic regression models, support vector machines, neural networks, and other ML models. Gradient descent’s flexibility makes it an essential part of machine learning.