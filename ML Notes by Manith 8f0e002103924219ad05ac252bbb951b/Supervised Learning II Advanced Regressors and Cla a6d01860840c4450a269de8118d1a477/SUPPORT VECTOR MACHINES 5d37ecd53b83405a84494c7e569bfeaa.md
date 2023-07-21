# SUPPORT VECTOR MACHINES

## **Support Vector Machines**

A **Support Vector Machine** (SVM) is a powerful supervised machine learning model used for classification. An SVM makes classifications by defining a decision boundary and then seeing what side of the boundary an unclassified point falls on. In the next few exercises, we’ll learn how these decision boundaries get defined, but for now, know that they’re defined by using a training set of classified points. That’s why SVMs are *supervised* machine learning models.

Decision boundaries are easiest to wrap your head around when the data has two features. In this case, the decision boundary is a line. Take a look at the example below.

![https://content.codecademy.com/programs/machine-learning/svm/two_dimensions.png](https://content.codecademy.com/programs/machine-learning/svm/two_dimensions.png)

Note that if the labels on the figures in this lesson are too small to read, you can resize this pane to increase the size of the images.

This SVM is using data about fictional games of Quidditch from the Harry Potter universe! The classifier is trying to predict whether a team will make the playoffs or not. Every point in the training set represents a “historical” Quidditch team. Each point has two features — the average number of goals the team scores and the average number of minutes it takes the team to catch the Golden Snitch.

After finding a decision boundary using the training set, you could give the SVM an unlabeled data point, and it will predict whether or not that team will make the playoffs.

Decision boundaries exist even when your data has more than two features. If there are three features, the decision boundary is now a plane rather than a line.

![https://content.codecademy.com/programs/machine-learning/svm/three_dimensions.png](https://content.codecademy.com/programs/machine-learning/svm/three_dimensions.png)

As the number of dimensions grows past 3, it becomes very difficult to visualize these points in space. Nonetheless, SVMs can still find a decision boundary. However, rather than being a separating line, or a separating plane, the decision boundary is called a *separating hyperplane*.

```python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from graph import ax, x_1, y_1, x_2, y_2

#Top graph intercept and slope
intercept_one = 8
slope_one = -2

x_vals = np.array(ax.get_xlim())
y_vals = intercept_one + slope_one * x_vals
plt.plot(x_vals, y_vals, '-')

#Bottom Graph
ax = plt.subplot(2, 1, 2)
plt.title('Good Decision Boundary')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

plt.scatter(x_1, y_1, color = "b")
plt.scatter(x_2, y_2, color = "r")

#Change the intercept to separate the clusters
intercept_two = 15
slope_two = -2

x_vals = np.array(ax.get_xlim())
y_vals = intercept_two + slope_two * x_vals
plt.plot(x_vals, y_vals, '-')

plt.tight_layout()
plt.show()
```

![Untitled](SUPPORT%20VECTOR%20MACHINES%205d37ecd53b83405a84494c7e569bfeaa/Untitled.png)

## **Optimal Decision Boundaries**

One problem that SVMs need to solve is figuring out what decision boundary to use. After all, there could be an infinite number of decision boundaries that correctly separate the two classes. Take a look at the image below:

![https://content.codecademy.com/programs/machine-learning/svm/decision_boundaries.png](https://content.codecademy.com/programs/machine-learning/svm/decision_boundaries.png)

There are so many valid decision boundaries, but which one is best? In general, we want our decision boundary to be as far away from training points as possible.

Maximizing the distance between the decision boundary and points in each class will decrease the chance of false classification. Take graph C for example.

![https://content.codecademy.com/programs/machine-learning/svm/graph_c.png](https://content.codecademy.com/programs/machine-learning/svm/graph_c.png)

The decision boundary is close to the blue class, so it is possible that a new point close to the blue cluster would fall on the red side of the line.

Out of all the graphs shown here, graph F has the best decision boundary.

```python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from graph import ax, x_1, y_1, x_2, y_2

#Top graph intercept and slope
intercept_one = 98
slope_one = -20

x_vals = np.array(ax.get_xlim())
y_vals = intercept_one + slope_one * x_vals
plt.plot(x_vals, y_vals, '-')

#Bottom graph
ax = plt.subplot(2, 1, 2)
plt.title('Good Decision Boundary')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

plt.scatter(x_1, y_1, color = "b")
plt.scatter(x_2, y_2, color = "r")

#Bottom graph intercept and slope
intercept_two = 8
slope_two = -0.5

x_vals = np.array(ax.get_xlim())
y_vals = intercept_two + slope_two * x_vals
plt.plot(x_vals, y_vals, '-')

plt.tight_layout()
plt.show()
```

![Untitled](SUPPORT%20VECTOR%20MACHINES%205d37ecd53b83405a84494c7e569bfeaa/Untitled%201.png)

## **Support Vectors and Margins**

We now know that we want our decision boundary to be as far away from our training points as possible. Let’s introduce some new terms that can help explain this idea.

The *support vectors* are the points in the training set closest to the decision boundary. In fact, these vectors are what define the decision boundary. But why are they called vectors? Instead of thinking about the training data as points, we can think of them as vectors coming from the origin.

![https://content.codecademy.com/programs/machine-learning/svm/vectors.png](https://content.codecademy.com/programs/machine-learning/svm/vectors.png)

These vectors are crucial in defining the decision boundary — that’s where the “support” comes from. If you are using `n` features, there are at least `n+1` support vectors.

The distance between a support vector and the decision boundary is called the *margin*. We want to make the margin as large as possible. The support vectors are highlighted in the image below:

![https://content.codecademy.com/programs/machine-learning/svm/margin.png](https://content.codecademy.com/programs/machine-learning/svm/margin.png)

Because the support vectors are so critical in defining the decision boundary, many of the other training points can be ignored. This is one of the advantages of SVMs. Many supervised machine learning algorithms use every training point in order to make a prediction, even though many of those training points aren’t relevant. SVMs are fast because they only use the support vectors!

```python
red_support_vector = [1, 6]
blue_support_vector_one = [2.5, 2]
blue_support_vector_two = [0.5, 2]
margin_size = 2
```

## **scikit-learn**

Now that we know the concepts behind SVMs we need to write the code that will find the decision boundary that maximizes the margin. All of the code that we’ve written so far has been guessing and checking — we don’t actually know if we’ve found the best line. Unfortunately, calculating the parameters of the best decision boundary is a fairly complex optimization problem. Luckily, Python’s scikit-learn library has implemented an SVM that will do this for us.

Note that while it is not important to understand how the optimal parameters are found, you should have a strong conceptual understanding of what the model is optimizing.

To use scikit-learn’s SVM we first need to create an SVC object. It is called an SVC because scikit-learn is calling the model a Support Vector Classifier rather than a Support Vector Machine.

```python
classifier = SVC(kernel = 'linear')
```

We’ll soon go into what the `kernel` parameter is doing, but for now, let’s use a `'linear'` kernel.

Next, the model needs to be trained on a list of data points and a list of labels associated with those data points. The labels are analogous to the color of the point — you can think of a `1` as a red point and a `0` as a blue point. The training is done using the `.fit()` method:

```python
training_points = [[1, 2], [1, 5], [2, 2], [7, 5], [9, 4], [8, 2]]
labels = [1, 1, 1, 0, 0, 0]
classifier.fit(training_points, labels)
```

The graph of this dataset would look like this:

![https://content.codecademy.com/programs/machine-learning/svm/example_dataset.png](https://content.codecademy.com/programs/machine-learning/svm/example_dataset.png)

Calling `.fit()` creates the line between the points.

Finally, the classifier predicts the label of new points using the `.predict()` method. The `.predict()` method takes a list of points you want to classify. Even if you only want to classify one point, make sure it is in a list:

```python
print(classifier.predict([[3, 2]]))
```

In the image below, you can see the unclassified point `[3, 2]` as a black dot. It falls on the red side of the line, so the SVM would predict it is red.

![https://content.codecademy.com/programs/machine-learning/svm/predict.png](https://content.codecademy.com/programs/machine-learning/svm/predict.png)

In addition to using the SVM to make predictions, you can inspect some of its attributes. For example, if you can print `classifier.support_vectors_` to see which points from the training set are the support vectors.

In this case, the support vectors look like this:

```python
[[7, 5],
 [8, 2],
 [2, 2]]
```

```python
from sklearn.svm import SVC
from graph import points, labels

classifier = SVC(kernel = 'linear')
classifier.fit(points, labels)
print(classifier.predict([[3, 4], [6, 7]]))
```

Output

```
[0 1]
```

## **Outliers**

SVMs try to maximize the size of the margin while still correctly separating the points of each class. As a result, outliers can be a problem. Consider the image below.

![https://content.codecademy.com/programs/machine-learning/svm/outliers.png](https://content.codecademy.com/programs/machine-learning/svm/outliers.png)

The size of the margin decreases when a single outlier is present, and as a result, the decision boundary changes as well. However, if we allowed the decision boundary to have some error, we could still use the original line.

SVMs have a parameter `C` that determines how much error the SVM will allow for. If `C` is large, then the SVM has a hard margin — it won’t allow for many misclassifications, and as a result, the margin could be fairly small. If `C` is too large, the model runs the risk of overfitting. It relies too heavily on the training data, including the outliers.

On the other hand, if `C` is small, the SVM has a soft margin. Some points might fall on the wrong side of the line, but the margin will be large. This is resistant to outliers, but if `C` gets too small, you run the risk of underfitting. The SVM will allow for so much error that the training data won’t be represented.

When using scikit-learn’s SVM, you can set the value of `C` when you create the object:

```python
classifier = SVC(C = 0.01)
```

The optimal value of `C` will depend on your data. Don’t always maximize margin size at the expense of error. Don’t always minimize error at the expense of margin size. The best strategy is to validate your model by testing many different values for `C`.

```python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from graph import points, labels, draw_points, draw_margin

points.append([3, 3])
labels.append(0)

points.append([10, 8])
labels.append(1)

points.append([11, 7])
labels.append(1)

classifier = SVC(kernel='linear', C = 0.5)
classifier.fit(points, labels)

draw_points(points, labels)
draw_margin(classifier)

plt.show()
```

![Untitled](SUPPORT%20VECTOR%20MACHINES%205d37ecd53b83405a84494c7e569bfeaa/Untitled%202.png)