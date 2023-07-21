# K-MEANS CLUSTERING

## **Introduction to Clustering**

Often, the data you encounter in the real world won’t be sorted into categories and won’t have labeled answers to your question. Finding patterns in this type of data, unlabeled data, is a common theme in many machine learning applications. *Unsupervised Learning* is how we find patterns and structure in these data.

**Clustering** is the most well-known unsupervised learning technique. It finds structure in unlabeled data by identifying similar groups, or *clusters*. Examples of clustering applications are:

- **Recommendation engines:** group products to personalize the user experience
- **Search engines:** group news topics and search results
- **Market segmentation:** group customers based on geography, demography, and behaviors
- **Image segmentation:** medical imaging or road scene segmentation on self-driving cars
- **Text clustering:** group similar texts together based on word usage

The *Iris* data set is a famous example of unlabeled data. It consists of measurements of sepals and petals on 50 different iris flowers. Here you can see a visualization of this data set that shows how the flowers naturally form three distinct clusters. We’ll learn how to find those clusters in this lesson.

*The image for this exercise will be replaced with a visualization of the iris data set.*

![Untitled](K-MEANS%20CLUSTERING%2079f9a91ed96244d99dab9c940ab8ddce/Untitled.gif)

## **K-Means Clustering**

The goal of clustering is to separate data so that data similar to one another are in the same group, while data different from one another are in different groups. So two questions arise:

- How many groups do we choose?
- How do we define similarity?

*k-means* is the most popular and well-known clustering algorithm, and it tries to address these two questions.

- The “k” refers to the number of clusters (groups) we expect to find in a dataset.
- The “Means” refers to the average distance of data to each cluster center, also known as the *centroid*, which we are trying to minimize.

It is an iterative approach:

1. Place `k` random centroids for the initial clusters.
2. Assign data samples to the nearest centroid.
3. Calculate new centroids based on the above-assigned data samples.
4. Repeat Steps 2 and 3 until convergence.

*Convergence* occurs when points don’t move between clusters and centroids stabilize. This iterative process of updating clusters and centroids is called *training*.

Once we are happy with our clusters, we can take a new unlabeled datapoint and quickly assign it to the appropriate cluster. This is called *inference*.

In practice it can be tricky to know how many clusters to look for. In the example here, the algorithm is sorting the data into `k=2` clusters.

## **Iris Dataset**

Before we implement the k-means algorithm, let’s find a dataset. The `sklearn` package embeds some datasets and sample images. One of them is the [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set).

The Iris dataset consists of measurements of sepals and petals of 3 different plant species:

- *Iris setosa*
- *Iris versicolor*
- *Iris virginica*

![https://content.codecademy.com/programs/machine-learning/k-means/iris.svg](https://content.codecademy.com/programs/machine-learning/k-means/iris.svg)

The sepal is the part that encases and protects the flower when it is in the bud stage. A petal is a leaflike part that is often colorful.

From `sklearn` library, import the `datasets` module:

```python
from sklearn import datasets
```

To load the Iris dataset:

```python
iris = datasets.load_iris()
```

The Iris dataset looks like:

```python
[[ 5.1  3.5  1.4  0.2 ]
 [ 4.9  3.   1.4  0.2 ]
 [ 4.7  3.2  1.3  0.2 ]
 [ 4.6  3.1  1.5  0.2 ]
   . . .
 [ 5.9  3.   5.1  1.8 ]]
```

We call each row of data a *sample*. For example, each flower is one sample.

Each characteristic we are interested in is a *feature*. For example, petal length is a feature of this dataset.

The features of the dataset are:

- **Column 0:** Sepal length
- **Column 1:** Sepal width
- **Column 2:** Petal length
- **Column 3:** Petal width

The 3 species of Iris plants are what we are going to cluster later in this lesson.

```python
Instructions
1.
Import the datasets module and load the Iris data.

Checkpoint 2 Passed

Hint
From sklearn library, import the datasets module, and load the Iris dataset:

from sklearn import datasets
 
iris = datasets.load_iris()
2.
Every dataset from sklearn comes with a bunch of different information (not just the data) and is stored in a similar fashion.

First, let’s take a look at the most important thing, the sample data:

print(iris.data)
Each row is a plant!

Checkpoint 3 Passed

Stuck? Get a hint
3.
The iris dataset comes with target values. The target values indicate which cluster each flower belongs to. In real life clustering problems, you will work with unlabeled data sets that don’t come with targets. For the sake of practice, we can ignore the targets while we are clustering. After we have clustered the data the targets can be used to check our work.

Take a look at the target values:

print(iris.target)
The iris.target values give the ground truth for the Iris dataset. Ground truth, in this case, is the number corresponding to the flower that we are trying to learn.

Checkpoint 4 Passed

Hint
The ground truth is what’s measured for the target variable for the training and testing examples.

It should look like:

[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
  2 2 ]
4.
Let’s take a look at one single row of data and the corresponding target.

print(iris.data[0, :], iris.target[0])
Checkpoint 5 Passed

Stuck? Get a hint
5.
It is always a good idea to read the descriptions of the data:

print(iris.DESCR)
Expand the terminal (right panel):

When was the Iris dataset published?
What is the unit of measurement?
Checkpoint 6 Passed

Hint
DESCR needs to be capitalized.

This dataset was published in 1936, over eighty years ago:

Fisher, R.A. "The use of multiple measurements in taxonomic problems" Annual Eugenics, 7, Part II, 179-188 (1936)
The unit of measurement is cm (centimeter):

    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
```

```python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

print(iris.data)
print(iris.target)
print(iris.data[0, :], iris.target[0])
print(iris.DESCR)
```

## **Visualize Before K-Means**

To get a better sense of the data in the `iris.data` matrix, let’s visualize it!

With Matplotlib, we can create a 2D scatter plot of the Iris dataset using two of its features (sepal length vs. petal length). Of course there are four different features that we could plot, but it’s much easier to visualize only two dimensions.

The sepal length measurements are stored in column `0` of the matrix, and the petal length measurements are stored in column `2` of the matrix.

But how do we get these values?

Suppose we only want to retrieve the values that are in column `0` of a matrix, we can use the NumPy/pandas notation `[:,0]` like so:

```python
matrix[:,0]
```

`[:,0]` can be translated to `[all_rows , column_0]`

Once you have the measurements we need, we can make a scatter plot like this:

```python
plt.scatter(x, y)
```

To show the plot:

```python
plt.show()
```

```python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn import datasets
 
iris = datasets.load_iris()
 
samples = iris.data
 
x = samples[:,0]
y = samples[:,1]
 
plt.scatter(x, y, alpha=0.5)
 
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
 
plt.show()
```

![Untitled](K-MEANS%20CLUSTERING%2079f9a91ed96244d99dab9c940ab8ddce/Untitled.png)

## **Implementing K-Means: Step 1**

The K-Means algorithm:

1. **Place `k` random centroids for the initial clusters.**
2. Assign data samples to the nearest centroid.
3. Update centroids based on the above-assigned data samples.
4. Repeat Steps 2 and 3 until convergence.

---

After looking at the scatter plot and having a better understanding of the Iris data, let’s start implementing the k-means algorithm.

In this exercise, we will implement Step 1.

Because we expect there to be three clusters (for the three species of flowers), let’s implement k-means where the `k` is 3. In real-life situations you won’t always know how many clusters to look for. We’ll learn more about how to choose `k` later.

Using the NumPy library, we will create three *random* initial centroids and plot them along with our samples.

```python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

samples = iris.data

x = samples[:,0]
y = samples[:,1]

# Number of clusters
k = 3
# Create x coordinates of k random centroids
centroids_x = np.random.uniform(min(x), max(x), size=k)
# Create y coordinates of k random centroids
centroids_y = np.random.uniform(min(y), max(y), size=k)
# Create centroids array
centroids = np.array(list(zip(centroids_x, centroids_y)))
 
print(centroids)
# Make a scatter plot of x, y
plt.scatter(x, y, alpha=0.5)
# Make a scatter plot of the centroids
plt.scatter(centroids_x, centroids_y)
 
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
 
# Display plot
plt.show()
```

output

```python
[[5.22032337 3.37561517]
 [5.1873663  2.57609842]
 [4.57310118 3.54156383]]
```

![Untitled](K-MEANS%20CLUSTERING%2079f9a91ed96244d99dab9c940ab8ddce/Untitled%201.png)

## **Implementing K-Means: Step 2**

The k-means algorithm:

1. Place `k` random centroids for the initial clusters.
2. **Assign data samples to the nearest centroid.**
3. Update centroids based on the above-assigned data samples.
4. Repeat Steps 2 and 3 until convergence.

---

In this exercise, we will implement Step 2.

Now we have the three random centroids. Let’s assign data points to their nearest centroids.

To do this we’re going to use a distance formula to write a `distance()` function.

There are many different kinds of distance formulas. The one you’re probably most familiar with is called *Euclidean distance*. To find the Euclidean distance between two points on a 2-d plane, make a right triangle so that the hypotenuse connects the points. The distance between them is the length of the hypotenuse.

Another common distance formula is the *taxicab distance*. The taxicab distance between two points on a 2-d plane is the distance you would travel if you took the long way around the right triangle via the two shorter sides, just like a taxicab would have to do if it wanted to travel to the opposite corner of a city block.

Different distance formulas are useful in different situations. If you’re curious, you can learn more about various distance formulas [here](https://machinelearningmastery.com/distance-measures-for-machine-learning/). For this lesson, we’ll use Euclidean distance.

After we write the `distance()` function, we are going to iterate through our data samples and compute the distance from each data point to each of the 3 centroids.

Suppose we have a point and a list of three distances in `distances` and it looks like `[15, 20, 5]`, then we would want to assign the data point to the 3rd centroid. The `argmin(distances)` would return the index of the lowest corresponding distance, `2`, because the index `2` contains the minimum value.

```python
Instructions
1.
Write a distance() function.

It should be able to take in a and b and return the distance between the two points.

Checkpoint 2 Passed

Hint
For 2D:

def distance(a, b):
  one = (a[0] - b[0]) ** 2
  two = (a[1] - b[1]) ** 2
  distance = (one+two) ** 0.5
  return distance
2.
Create an array called labels that will hold the cluster labels for each data point. Its size should be the length of the data sample.

It should look something like:

[ 0.  0.  0.  0.  0.  0.  ...  0.]
Checkpoint 3 Passed

Hint
# Cluster labels for each point (either 0, 1, or 2)
labels = np.zeros(len(samples))
3.
Create a function called assign_to_centroid() that assigns the nearest centroid to a sample. You’ll need to compute the distance to each centroid to find the closest one.

def assign_to_centroid(sample, centroids):
  # Fill in the code here
  #
  #
  #
  #
  return closest_centroid
Then, assign the cluster to each index of the labels array.

Checkpoint 4 Passed

Hint
The code should look like this:

def assign_to_centroid(sample, centroids):
  k = len(centroids)
  distances = np.zeros(k)
  for i in range(k):
    distances[i] = distance(sample[i], centroids[i])
  closest_centroid = np.argmin(distances)
  return closest_centroid
4.
Write a loop that iterates through the whole data sample and assigns each sample’s closest centroid to the corresponding index of the labels array. Use the function that you created in the previous exercise.

Checkpoint 5 Passed

Hint
The code should look like this:

for i in range(len(samples)):
  labels[i] = assign_to_centroid(samples[i], centroids)
5.
Then, print labels (outside of the for loop).

Awesome! You have just finished Step 2 of the k-means algorithm.

Checkpoint 6 Passed

Hint
print(labels)
The result labels should look like:

[ 0.  0.  0.  1.  0.  2. 0.  1.  1.  ... ]
```

code

```python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

samples = iris.data

x = samples[:,0]
y = samples[:,1]

sepal_length_width = np.array(list(zip(x, y)))

# Step 1: Place K random centroids

k = 3

centroids_x = np.random.uniform(min(x), max(x), size=k)
centroids_y = np.random.uniform(min(y), max(y), size=k)

centroids = np.array(list(zip(centroids_x, centroids_y)))

# Step 2: Assign samples to nearest centroid

# Distance formula

def distance(a, b):
  one = (a[0] - b[0]) **2
  two = (a[1] - b[1]) **2
  distance = (one+two) ** 0.5
  return distance

# Cluster labels for each point (either 0, 1, or 2)

labels = np.zeros(len(samples))

# A function that assigns the nearest centroid to a sample

def assign_to_centroid(sample, centroids):
  k = len(centroids)
  distances = np.zeros(k)
  for i in range(k):
    distances[i] = distance(sample, centroids[i])
  closest_centroid = np.argmin(distances)
  return closest_centroid

# Assign the nearest centroid to each sample

for i in range(len(samples)):
  labels[i] = assign_to_centroid(samples[i], centroids)

# Print labels

print(labels)
```

output

```python
[1. 2. 2. 2. 1. 1. 2. 1. 2. 2. 1. 2. 2. 2. 1. 1. 1. 1. 1. 1. 1. 1. 2. 1.
 2. 1. 1. 1. 1. 2. 2. 1. 1. 1. 2. 1. 1. 1. 2. 1. 1. 2. 2. 1. 1. 2. 1. 2.
 1. 1. 0. 0. 0. 1. 0. 1. 1. 2. 0. 1. 2. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1.
 1. 1. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 2. 1. 1.
 1. 1. 1. 1. 1. 1. 0. 1. 0. 0. 2. 0. 0. 0. 0. 1. 0. 1. 1. 0. 0. 0. 0. 1.
 0. 1. 0. 1. 0. 0. 1. 1. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 1. 0. 0. 0. 1. 0.
 0. 0. 1. 0. 1. 1.]
```

## **Implementing K-Means: Step 3**

The k-means algorithm:

1. Place `k` random centroids for the initial clusters.
2. Assign data samples to the nearest centroid.
3. **Update centroids based on the above-assigned data samples.**
4. Repeat Steps 2 and 3 until convergence.

---

In this exercise, we will implement Step 3.

Find *new* cluster centers by taking the average of the assigned points. To find the average of the assigned points, we can use the `.mean()` function.

- Instructions
    
    **Instructions**
    
    **1.**
    
    Save the old `centroids` value before updating.
    
    We have already imported `deepcopy` for you:
    
    ```python
    from copy import deepcopy
    ```
    
    Store `centroids` into `centroids_old` using `deepcopy()`:
    
    ```python
    centroids_old = deepcopy(centroids)
    ```
    
    Checkpoint 2 Passed
    
    Stuck? Get a hint
    
    **2.**
    
    Then, create a `for` loop that iterates `k` times.
    
    Since `k = 3`, as we are iterating through the `for`loop each time, we can calculate the mean of the points that have the same cluster label.
    
    Inside the `for` loop, create an array named `points` where we get all the data points that have the cluster label `i`.
    
    There are two ways to do this, check the hints to see both!
    
    Checkpoint 3 Passed
    
    Hint
    
    One way to do this is:
    
    ```python
    for i in range(k):
      points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i]
    ```
    
    Another way is to use nested `for` loop:
    
    ```python
    for i in range(k):
      points = []
      for j in range(len(sepal_length_width)):
        if labels[j] == i:
          points.append(sepal_length_width[j])
    ```
    
    Here, we create an empty list named `points` first, and use `.append()` to add values into the list.
    
    **3.**
    
    Now that we have assigned each input to its closest centroid, we can update the position of that centroid to the true center. Inside the `for` loop, calculate the mean of those points using `.mean()` to get the new centroid.
    
    Store the new centroid in `centroids[i]`.
    
    The `.mean()` fucntion looks like:
    
    ```python
    np.mean(input, axis=0)
    ```
    
    Checkpoint 4 Passed
    
    Hint
    
    ```python
    for i in range(k):
      ...
      centroids[i] = np.mean(points, axis=0)
    ```
    
    If you don’t have `axis=0` parameter, the default is to compute the mean of the flattened array. We need the `axis=0` here to specify that we want to compute the means along the rows.
    
    **4.**
    
    Oustide of the `for` loop, print `centroids_old` and `centroids` to see how centroids changed.
    
    Checkpoint 5 Passed
    
    Hint
    `print(centroids_old)print("- - - - - - - - - - - - - -")print(centroids)`
    

```python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from copy import deepcopy

iris = datasets.load_iris()

samples = iris.data
samples = iris.data

x = samples[:,0]
y = samples[:,1]

sepal_length_width = np.array(list(zip(x, y)))

# Step 1: Place K random centroids

k = 3

centroids_x = np.random.uniform(min(x), max(x), size=k)
centroids_y = np.random.uniform(min(y), max(y), size=k)

centroids = np.array(list(zip(centroids_x, centroids_y)))

# Step 2: Assign samples to nearest centroid

def distance(a, b):
  one = (a[0] - b[0]) **2
  two = (a[1] - b[1]) **2
  distance = (one+two) ** 0.5
  return distance

# Cluster labels for each point (either 0, 1, or 2)

labels = np.zeros(len(samples))

# A function that assigns the nearest centroid to a sample

def assign_to_centroid(sample, centroids):
  k = len(centroids)
  distances = np.zeros(k)
  for i in range(k):
    distances[i] = distance(sample, centroids[i])
  closest_centroid = np.argmin(distances)
  return closest_centroid

# Assign the nearest centroid to each sample

for i in range(len(samples)):
  labels[i] = assign_to_centroid(samples[i], centroids)

# Step 3: Update centroids

centroids_old = deepcopy(centroids)

for i in range(k):
  points = []
  for j in range(len(sepal_length_width)):
    if labels[j] == i:
      points.append(sepal_length_width[j])
  
print(centroids_old)
print("- - - - - - - - - - - - - -")
print(centroids)
```

output

```python
[[7.16667945 4.36231251]
 [6.55314665 2.52319503]
 [5.84591595 3.58562006]]
- - - - - - - - - - - - - -
[[7.16667945 4.36231251]
 [6.55314665 2.52319503]
 [5.84591595 3.58562006]]
```

## **Implementing K-Means: Step 4**

The k-means algorithm:

1. Place `k` random centroids for the initial clusters.
2. Assign data samples to the nearest centroid.
3. Update centroids based on the above-assigned data samples.
4. **Repeat Steps 2 and 3 until convergence.**

---

In this exercise, we will implement Step 4.

This is the part of the algorithm where we repeatedly execute Step 2 and 3 until the centroids stabilize (convergence).

We can do this using a `while` loop. And everything from Step 2 and 3 goes inside the loop.

For the condition of the `while` loop, we need to create an array named `errors`. In each `error` index, we calculate the difference between the updated centroid (`centroids`) and the old centroid (`centroids_old`).

The loop ends when all three values in `errors` are `0`.

- Instructions
    
    **Instructions**
    
    **1.**
    
    On line 49 of **script.py**, initialize `error`:
    
    ```python
    error = np.zeros(3)
    ```
    
    Then, use the `distance()` function to calculate the distance between the updated centroids and the old centroids and put them in `error`. Here’s how to calculate the error for entry ‘0’. You can write a loop to compute each distance.
    
    ```python
    error[0] = distance(centroids[0], centroids_old[0])
    ```
    
    Checkpoint 2 Passed
    
    Hint
    
    `error = np.zeros(3) for i in range(k):  error[i] = distance(centroids[i], centroids_old[i])`
    
    **2.**
    
    After that, add a `while` loop:
    
    ```python
    while error.all() != 0:
    ```
    
    And move *everything* below (from Step 2 and 3) inside.
    
    And recalculate `error` again at the end of each iteration of the `while` loop. You can put this line inside the ‘for’ loop that computes the new centroids:
    
    ```python
    error[i] = distance(centroids[i], centroids_old[i])
    ```
    
    Checkpoint 3 Passed
    
    Hint
    
    `while error.all() != 0:   # Step 2: Assign samples to nearest centroid   for i in range(len(samples)):    labels[i] = assign_to_centroid(samples[i], centroids)   # Step 3: Update centroids   centroids_old = deepcopy(centroids)   for i in range(k):    points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i]    centroids[i] = np.mean(points, axis=0)    error[i] = distance(centroids[i], centroids_old[i])`
    
    **3.**
    
    Awesome, now you have everything, let’s visualize it.
    
    After the `while` loop finishes, let’s create an array of colors:
    
    ```python
    colors = ['r', 'g', 'b']
    ```
    
    Then, create a `for` loop that iterates `k` times.
    
    Inside the `for` loop (similar to what we did in the last exercise), create an array named `points` where we get all the data points that have the cluster label `i`.
    
    Then we are going to make a scatter plot of `points[:, 0]` vs `points[:, 1]` using the `scatter()` function:
    
    ```python
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.5)
    ```
    
    Checkpoint 4 Passed
    
    Hint
    
    `colors = ['r', 'g', 'b'] for i in range(k):  points = np.array([sepal_length_width[j] for j in range(len(samples)) if labels[j] == i])  plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.5)`
    
    **4.**
    
    Then, paste the following code at the very end. Here, we are visualizing all the points in each of the `labels` a different color.
    
    ```python
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=150)
    
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    
    plt.show()
    ```
    

```python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from copy import deepcopy

iris = datasets.load_iris()

samples = iris.data

x = samples[:,0]
y = samples[:,1]

sepal_length_width = np.array(list(zip(x, y)))

# Step 1: Place K random centroids

k = 3

centroids_x = np.random.uniform(min(x), max(x), size=k)
centroids_y = np.random.uniform(min(y), max(y), size=k)

centroids = np.array(list(zip(centroids_x, centroids_y)))

def distance(a, b):
  one = (a[0] - b[0]) ** 2
  two = (a[1] - b[1]) ** 2
  distance = (one + two) ** 0.5
  return distance

# A function that assigns the nearest centroid to a sample
def assign_to_centroid(sample, centroids):
  k = len(centroids)
  distances = np.zeros(k)
  for i in range(k):
    distances[i] = distance(sample, centroids[i])
  closest_centroid = np.argmin(distances)
  return closest_centroid

# To store the value of centroids when it updates
centroids_old = np.zeros(centroids.shape)

# Cluster labeles (either 0, 1, or 2)
labels = np.zeros(len(samples))

distances = np.zeros(3)

# Initialize error:
error = np.zeros(3)

for i in range(k):
  error[i] = distance(centroids[i], centroids_old[i])

# Repeat Steps 2 and 3 until convergence:

while error.all() != 0:

  # Step 2: Assign samples to nearest centroid

  for i in range(len(samples)):
    labels[i] = assign_to_centroid(samples[i], centroids)

  # Step 3: Update centroids

  centroids_old = deepcopy(centroids)

  for i in range(k):
    points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i]
    centroids[i] = np.mean(points, axis=0)
    error[i] = distance(centroids[i], centroids_old[i])

colors = ['r', 'g', 'b']

for i in range(k):
  points = np.array([sepal_length_width[j] for j in range(len(samples)) if labels[j] == i])
  plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.5)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=150)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

plt.show()
```

![Untitled](K-MEANS%20CLUSTERING%2079f9a91ed96244d99dab9c940ab8ddce/Untitled%202.png)

## **Implementing K-Means: Scikit-Learn**

Awesome, you have implemented k-means clustering from scratch!

Writing an algorithm whenever you need it can be very time-consuming and you might make mistakes and typos along the way. We will now show you how to implement k-means more efficiently – using the [scikit-learn](https://scikit-learn.org/stable/) library.

There are many advantages to using scikit-learn. It can run k-means on datasets with as many features as your computer can handle, so it will be easy for us to use all four features of the iris data set instead of the two features that we used in the previous exercises.

Another big advantage of scikit-learn is that it is a widely-used open-source library. It is very well-tested, so it is much less likely to contain mistakes. Since so many people use it, there are many online resources that can help you if you get stuck. If you have a specific question about scikit-learn, it’s very likely that other users have already asked and answered your question on public forums.

To import `KMeans` from `sklearn.cluster`:

```python
from sklearn.cluster import KMeans
```

For Step 1, use the `KMeans()` method to build a model that finds `k` clusters. To specify the number of clusters (`k`), use the `n_clusters` keyword argument:

```python
model = KMeans(n_clusters = k)
```

For Steps 2 and 3, use the `.fit()` method to compute k-means clustering:

```python
model.fit(X)
```

After k-means, we can now predict the closest cluster each sample in X belongs to. Use the `.predict()` method to compute cluster centers and predict cluster index for each sample:

```python
model.predict(X)
```

```python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn import datasets

# From sklearn.cluster, import Kmeans class
from sklearn.cluster import KMeans
iris = datasets.load_iris()

samples = iris.data

# Use KMeans() to create a model that finds 3 clusters
model = KMeans(n_clusters=3)
# Use .fit() to fit the model to samples
model.fit(samples)
# Use .predict() to determine the labels of samples 
labels = model.predict(samples)
# Print the labels
print(labels)
```

## **New Data?**

You used k-means and found three clusters of the `samples` data. But it gets cooler!

Since you have created a model that computed k-means clustering, you can now feed *new* data samples into it and obtain the cluster labels using the `.predict()` method.

So, suppose we went to the florist and bought 3 more Irises with the measurements:

```python
[[ 5.1  3.5  1.4  0.2 ]
 [ 3.4  3.1  1.6  0.3 ]
 [ 4.9  3.   1.4  0.2 ]]
```

We can feed this new data into the model and obtain the labels for them.

```python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()

samples = iris.data

model = KMeans(n_clusters=3)

model.fit(samples)

# Store the new Iris measurements
new_samples = np.array([[5.7, 4.4, 1.5, 0.4],
   [6.5, 3. , 5.5, 0.4],
   [5.8, 2.7, 5.1, 1.9]])

# Predict labels for the new_samples
new_labels = model.predict(new_samples)

print(new_labels)

new_names = [iris.target_names[label] for label in new_labels]

print(new_names)
```

Output

```python
[0 1 1]
['setosa', 'versicolor', 'versicolor']
```

## **Visualize After K-Means**

We have done the following using `sklearn` library:

- Load the embedded dataset
- Compute k-means on the dataset (where `k` is 3)
- Predict the labels of the data samples

And the labels resulted in either `0`, `1`, or `2`.

Let’s finish it by making a scatter plot of the data again!

This time, however, use the `labels` numbers as the colors.

To edit colors of the scatter plot, we can set `c = labels`:

```python
plt.scatter(x, y, c=labels, alpha=0.5)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
```

```python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()

samples = iris.data

model = KMeans(n_clusters=3)

model.fit(samples)

labels = model.predict(samples)

print(labels)

# Make a scatter plot of x and y and using labels to define the colors
x = samples[:,0]
y = samples[:,1]

plt.scatter(x, y, c=labels, alpha=0.5)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

plt.show()
```

![Untitled](K-MEANS%20CLUSTERING%2079f9a91ed96244d99dab9c940ab8ddce/Untitled%203.png)

## **Evaluation**

At this point, we have clustered the Iris data into 3 different groups (implemented using Python and using scikit-learn). But do the clusters correspond to the actual species? Let’s find out!

First, remember that the Iris dataset comes with target values:

```python
target = iris.target
```

It looks like:

```python
[ 0 0 0 0 0 ... 2 2 2]
```

According to the metadata:

- All the `0`‘s are *Iris-setosa*
- All the `1`‘s are *Iris-versicolor*
- All the `2`‘s are *Iris-virginica*

Let’s change these values into the corresponding species using the following code:

```python
species = [iris.target_names[t] for t in list(target)]
```

Then we are going to use the Pandas library to perform a *cross-tabulation*.

Cross-tabulations enable you to examine relationships within the data that might not be readily apparent when analyzing total survey responses.

The result should look something like:

```python
labels    setosa    versicolor    virginica
0             50             0            0
1              0             2           36
2              0            48           14
```

(You might need to expand this narrative panel in order to the read the table better.)

The first column has the cluster labels. The second to fourth columns have the Iris species that are clustered into each of the labels.

By looking at this, you can conclude that:

- *Iris-setosa* was clustered with 100% accuracy.
- *Iris-versicolor* was clustered with 96% accuracy.
- *Iris-virginica* didn’t do so well.

Follow the instructions below to learn how to do a cross-tabulation.

## **The Number of Clusters**

At this point, we have grouped the Iris plants into 3 clusters. But suppose we didn’t know there are three species of Iris in the dataset, what is the best number of clusters? And how do we determine that?

Before we answer that, we need to define what is a *good* cluster?

Good clustering results in tight clusters, meaning that the samples in each cluster are bunched together. How spread out the clusters are is measured by *inertia*. Inertia is the distance from each sample to the centroid of its cluster. The lower the inertia is, the better our model has done.

You can check the inertia of a model by:

```python
print(model.inertia_)
```

For the Iris dataset, if we graph all the `k`s (number of clusters) with their inertias:

![https://content.codecademy.com/programs/machine-learning/k-means/number-of-clusters.svg](https://content.codecademy.com/programs/machine-learning/k-means/number-of-clusters.svg)

Notice how the graph keeps decreasing.

Ultimately, this will always be a trade-off. If the inertia is too large, then the clusters probably aren’t clumped close together. On the other hand, if there are too many clusters, the individual clusters might not be different enough from each other. The goal is to have low inertia *and* a small number of clusters.

One of the ways to interpret this graph is to use the *elbow method*: choose an “elbow” in the inertia plot - when inertia begins to decrease more slowly.

In the graph above, 3 is the optimal number of clusters.

```python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()

samples = iris.data

# Code Start here:

num_clusters = list(range(1, 9))
inertias = []

for k in num_clusters:
  model = KMeans(n_clusters=k)
  model.fit(samples)
  inertias.append(model.inertia_)
  
plt.plot(num_clusters, inertias, '-o')

plt.xlabel('number of clusters (k)')
plt.ylabel('inertia')

plt.show()
```

![Untitled](K-MEANS%20CLUSTERING%2079f9a91ed96244d99dab9c940ab8ddce/Untitled%204.png)