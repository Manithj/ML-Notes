# K-NEAREST NEIGHBORS

## **K-Nearest Neighbors Classifier**

**K-Nearest Neighbors (KNN)** is a classification algorithm. The central idea is that data points with similar attributes tend to fall into similar categories.

Consider the image to the right. This image is complicated, but for now, let’s just focus on where the data points are being placed. Every data point — whether its color is red, green, or white — has an `x` value and a `y` value. As a result, it can be plotted on this two-dimensional graph.

Next, let’s consider the color of the data. The color represents the class that the K-Nearest Neighbor algorithm is trying to classify. In this image, data points can either have the class `green` or the class `red`. If a data point is white, this means that it doesn’t have a class yet. The purpose of the algorithm is to classify these unknown points.

Finally, consider the expanding circle around the white point. This circle is finding the `k` nearest neighbors to the white point. When `k = 3`, the circle is fairly small. Two of the three nearest neighbors are green, and one is red. So in this case, the algorithm would classify the white point as green. However, when we increase `k` to `5`, the circle expands, and the classification changes. Three of the nearest neighbors are red and two are green, so now the white point will be classified as red.

This is the central idea behind the K-Nearest Neighbor algorithm. If you have a dataset of points where the class of each point is known, you can take a new point with an unknown class, find it’s nearest neighbors, and classify it.

![nearest_neighbor.gif](K-NEAREST%20NEIGHBORS%20b9b13851cd904fecad5892778e65fbb1/nearest_neighbor.gif)

## **Introduction**

Before diving into the K-Nearest Neighbors algorithm, let’s first take a minute to think about an example.

Consider a dataset of movies. Let’s brainstorm some features of a movie data point. A feature is a piece of information associated with a data point. Here are some potential features of movie data points:

- the *length* of the movie in minutes.
- the *budget* of a movie in dollars.

If you think back to the previous exercise, you could imagine movies being places in that two-dimensional space based on those numeric features. There could also be some boolean features: features that are either true or false. For example, here are some potential boolean features:

- *Black and white*. This feature would be `True` for black and white movies and `False` otherwise.
- *Directed by Stanley Kubrick*. This feature would be `False` for almost every movie, but for the few movies that were directed by Kubrick, it would be `True`.

Finally, let’s think about how we might want to classify a movie. For the rest of this lesson, we’re going to be classifying movies as either good or bad. In our dataset, we’ve classified a movie as good if it had an IMDb rating of 7.0 or greater. Every “good” movie will have a class of `1`, while every bad movie will have a class of `0`.

To the right, we’ve created some movie data points where the first item in the list is the length, the second is the budget, and the third is whether the movie was directed by Stanley Kubrick.

```python
mean_girls = [97, 17000000, False]
the_shining = [146, 19000000, True]
gone_with_the_wind = [238, 3977000, False]
```

## **SciPy Distances**

Now that you’ve written these three distance formulas yourself, let’s look at how to use them using Python’s [SciPy](https://www.scipy.org/) library:

- Euclidean Distance `.euclidean()`
- Manhattan Distance `.cityblock()`
- Hamming Distance `.hamming()`

There are a few noteworthy details to talk about:

First, the `scipy` implementation of Manhattan distance is called `cityblock()`. Remember, computing Manhattan distance is like asking how many blocks away you are from a point.

Second, the `scipy` implementation of Hamming distance will always return a number between `0` an `1`. Rather than summing the number of differences in dimensions, this implementation sums those differences and then divides by the total number of dimensions. For example, in your implementation, the Hamming distance between `[1, 2, 3]` and `[7, 2, -10]` would be `2`. In `scipy`‘s version, it would be `2/3`.

```python
from scipy.spatial import distance

def euclidean_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += (pt1[i] - pt2[i]) ** 2
  return distance ** 0.5

def manhattan_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += abs(pt1[i] - pt2[i])
  return distance

def hamming_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    if pt1[i] != pt2[i]:
      distance += 1
  return distance

print(euclidean_distance([1, 2], [4, 0]))
print(manhattan_distance([1, 2], [4, 0]))
print(hamming_distance([5, 4, 9], [1, 7, 9]))

print(distance.euclidean([1, 2], [4, 0]))
print(distance.cityblock([1, 2], [4, 0]))
print(distance.hamming([5, 4, 9], [1, 7, 9]))
```

## **Distance Between Points - 3D**

Making a movie rating predictor based on just the length and release date of movies is pretty limited. There are so many more interesting pieces of data about movies that we could use! So let’s add another dimension.

Let’s say this third dimension is the movie’s budget. We now have to find the distance between these two points in three dimensions.

![https://content.codecademy.com/courses/learn-knn/threed.png](https://content.codecademy.com/courses/learn-knn/threed.png)

What if we’re not happy with just three dimensions? Unfortunately, it becomes pretty difficult to visualize points in dimensions higher than 3. But that doesn’t mean we can’t find the distance between them.

The generalized distance formula between points A and B is as follows:

![Untitled](K-NEAREST%20NEIGHBORS%20b9b13851cd904fecad5892778e65fbb1/Untitled.png)

Here, A1-B1 is the difference between the first feature of each point. An-Bn is the difference between the last feature of each point.

Using this formula, we can find the K-Nearest Neighbors of a point in N-dimensional space! We now can use as much information about our movies as we want.

We will eventually use these distances to find the nearest neighbors to an unlabeled point.

```python
star_wars = [125, 1977, 11000000]
raiders = [115, 1981, 18000000]
mean_girls = [97, 2004, 17000000]

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

print(distance(star_wars, raiders))
print(distance(star_wars, mean_girls))
```

## **Data with Different Scales: Normalization**

In the next three lessons, we’ll implement the three steps of the K-Nearest Neighbor Algorithm:

1. **Normalize the data**
2. Find the `k` nearest neighbors
3. Classify the new point based on those neighbors

---

When we added the dimension of budget, you might have realized there are some problems with the way our data currently looks.

Consider the two dimensions of release date and budget. The maximum difference between two movies’ release dates is about 125 years (The Lumière Brothers were making movies in the 1890s). However, the difference between two movies’ budget can be millions of dollars.

The problem is that the distance formula treats all dimensions equally, regardless of their scale. If two movies came out 70 years apart, that should be a pretty big deal. However, right now, that’s exactly equivalent to two movies that have a difference in budget of 70 dollars. The difference in one year is exactly equal to the difference in one dollar of budget. That’s absurd!

Another way of thinking about this is that the budget completely outweighs the importance of all other dimensions because it is on such a huge scale. The fact that two movies were 70 years apart is essentially meaningless compared to the difference in millions in the other dimension.

The solution to this problem is to [normalize the data](https://www.codecademy.com/articles/normalization) so every value is between 0 and 1. In this lesson, we’re going to be using min-max normalization.

```python
release_dates = [1897, 1998, 2000, 1948, 1962, 1950, 1975, 1960, 2017, 1937, 1968, 1996, 1944, 1891, 1995, 1948, 2011, 1965, 1891, 1978]

def min_max_normalize(lst):
  maximum = max(lst)
  minimum = min(lst)

  normalized = []

  for value in lst:
    norm = (value - minimum)/(maximum - minimum)
    normalized.append(norm)

  return normalized

print(min_max_normalize(release_dates))
```

## **Finding the Nearest Neighbors**

The K-Nearest Neighbor Algorithm:

1. Normalize the data
2. **Find the `k` nearest neighbors**
3. Classify the new point based on those neighbors

---

Now that our data has been normalized and we know how to find the distance between two points, we can begin classifying unknown data!

To do this, we want to find the `k` nearest neighbors of the unclassified point. In a few exercises, we’ll learn how to properly choose `k`, but for now, let’s choose a number that seems somewhat reasonable. Let’s choose 5.

In order to find the 5 nearest neighbors, we need to compare this new unclassified movie to every other movie in the dataset. This means we’re going to be using the distance formula again and again. We ultimately want to end up with a sorted list of distances and the movies associated with those distances.

It might look something like this:

```python
[
  [0.30, 'Superman II'],
  [0.31, 'Finding Nemo'],
  ...
  ...
  [0.38, 'Blazing Saddles']
]
```

In this example, the unknown movie has a distance of `0.30` to Superman II.

In the next exercise, we’ll use the labels associated with these movies to classify the unlabeled point.

```python
from movies import movie_dataset, movie_labels

print(movie_dataset['Bruce Almighty'])
print(movie_labels['Bruce Almighty'])

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def classify(unknown,dataset,k):
  distances = []
  for title in dataset:
   distance_to_point = distance(dataset[title], unknown)
   distances.append([distance_to_point, title])

   distances.sort()
   neighbors = distances[0:k]

  return neighbors

print(classify([.4, .2, .9], movie_dataset, 5))
```

## **Count Neighbors**

The K-Nearest Neighbor Algorithm:

1. Normalize the data
2. Find the `k` nearest neighbors
3. **Classify the new point based on those neighbors**

---

We’ve now found the `k` nearest neighbors, and have stored them in a list that looks like this:

```python
[
  [0.083, 'Lady Vengeance'],
  [0.236, 'Steamboy'],
  ...
  ...
  [0.331, 'Godzilla 2000']
]
```

Our goal now is to count the number of good movies and bad movies in the list of neighbors. If more of the neighbors were good, then the algorithm will classify the unknown movie as good. Otherwise, it will classify it as bad.

In order to find the class of each of the labels, we’ll need to look at our `movie_labels` dataset. For example, `movie_labels['Akira']` would give us `1` because Akira is classified as a good movie.

You may be wondering what happens if there’s a tie. What if `k = 8` and four neighbors were good and four neighbors were bad? There are different strategies, but one way to break the tie would be to choose the class of the closest point.

```python
from movies import movie_dataset, movie_labels

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def classify(unknown, dataset, labels, k):
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  num_good = 0
  num_bad = 0
  for neighbor in neighbors:
    title = neighbor[1]
    if labels[title] == 0:
      num_bad += 1
    elif labels[title] == 1:
      num_good += 1
  if num_good > num_bad:
    return 1
  else:
    return 0

print(classify([.4, .2, .9], movie_dataset, movie_labels, 5))
```

## **Classify Your Favorite Movie**

Nice work! Your classifier is now able to predict whether a movie will be good or bad. So far, we’ve only tested this on a completely random point `[.4, .2, .9]`. In this exercise we’re going to pick a real movie, normalize it, and run it through our classifier to see what it predicts!

In the instructions below, we are going to be testing our classifier using the 2017 movie *Call Me By Your Name*. Feel free to pick your favorite movie instead!

```python
from movies import movie_dataset, movie_labels, normalize_point

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def classify(unknown, dataset, labels, k):
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  num_good = 0
  num_bad = 0
  for neighbor in neighbors:
    title = neighbor[1]
    if labels[title] == 0:
      num_bad += 1
    elif labels[title] == 1:
      num_good += 1
  if num_good > num_bad:
    return 1
  else:
    return 0

print("Call Me By Your Name" in movie_dataset)
my_movie = [3500000, 132, 2017]
normalized_my_movie = normalize_point(my_movie)

print(classify(normalized_my_movie, movie_dataset, movie_labels, 5))
```

## **Training and Validation Sets**

You’ve now built your first K Nearest Neighbors algorithm capable of classification. You can feed your program a never-before-seen movie and it can predict whether its IMDb rating was above or below 7.0. However, we’re not done yet. We now need to report how effective our algorithm is. After all, it’s possible our predictions are totally wrong!

As with most machine learning algorithms, we have split our data into a training set and validation set.

Once these sets are created, we will want to use every point in the validation set as input to the K Nearest Neighbor algorithm. We will take a movie from the validation set, compare it to all the movies in the training set, find the K Nearest Neighbors, and make a prediction. After making that prediction, we can then peek at the real answer (found in the validation labels) to see if our classifier got the answer correct.

If we do this for every movie in the validation set, we can count the number of times the classifier got the answer right and the number of times it got it wrong. Using those two numbers, we can compute the validation accuracy.

Validation accuracy will change depending on what K we use. In the next exercise, we’ll use the validation accuracy to pick the best possible K for our classifier.

```python
from movies import training_set, training_labels, validation_set, validation_labels

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def classify(unknown, dataset, labels, k):
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  num_good = 0
  num_bad = 0
  for neighbor in neighbors:
    title = neighbor[1]
    if labels[title] == 0:
      num_bad += 1
    elif labels[title] == 1:
      num_good += 1
  if num_good > num_bad:
    return 1
  else:
    return 0
  
print(validation_set["Bee Movie"])
print(validation_labels["Bee Movie"])

guess=classify(validation_set["Bee Movie"], training_set, training_labels, 5)
print(guess)
if guess == validation_labels["Bee Movie"]:
  print("Correct!")
else:
  print("Wrong!")
```

## **Choosing K**

In the previous exercise, we found that our classifier got one point in the training set correct. Now we can test *every* point to calculate the validation accuracy.

The validation accuracy changes as `k` changes. The first situation that will be useful to consider is when `k` is very small. Let’s say `k = 1`. We would expect the validation accuracy to be fairly low due to *overfitting*. Overfitting is a concept that will appear almost any time you are writing a machine learning algorithm. Overfitting occurs when you rely too heavily on your training data; you assume that data in the real world will always behave exactly like your training data. In the case of K-Nearest Neighbors, overfitting happens when you don’t consider enough neighbors. A single outlier could drastically determine the label of an unknown point. Consider the image below.

![https://content.codecademy.com/courses/learn-knn/overfit.png](https://content.codecademy.com/courses/learn-knn/overfit.png)

The dark blue point in the top left corner of the graph looks like a fairly significant outlier. When `k = 1`, all points in that general area will be classified as dark blue when it should probably be classified as green. Our classifier has relied too heavily on the small quirks in the training data.

On the other hand, if `k` is very large, our classifier will suffer from *underfitting*. Underfitting occurs when your classifier doesn’t pay enough attention to the small quirks in the training set. Imagine you have `100` points in your training set and you set `k = 100`. Every single unknown point will be classified in the same exact way. The distances between the points don’t matter at all! This is an extreme example, however, it demonstrates how the classifier can lose understanding of the training data if `k` is too big.

## **Graph of K**

The graph to the right shows the validation accuracy of our movie classifier as `k` increases. When `k` is small, overfitting occurs and the accuracy is relatively low. On the other hand, when `k` gets too large, underfitting occurs and accuracy starts to drop.

![Untitled](K-NEAREST%20NEIGHBORS%20b9b13851cd904fecad5892778e65fbb1/Untitled%201.png)

## **Using sklearn**

You’ve now written your own K-Nearest Neighbor classifier from scratch! However, rather than writing your own classifier every time, you can use Python’s `sklearn` library. `sklearn` is a Python library specifically used for Machine Learning. It has an amazing number of features, but for now, we’re only going to investigate its K-Nearest Neighbor classifier.

There are a couple of steps we’ll need to go through in order to use the library. First, you need to create a `KNeighborsClassifier` object. This object takes one parameter - `k`. For example, the code below will create a classifier where `k = 3`

```python
classifier = KNeighborsClassifier(n_neighbors = 3)
```

Next, we’ll need to train our classifier. The `.fit()` method takes two parameters. The first is a list of points, and the second is the labels associated with those points. So for our movie example, we might have something like this

```python
training_points = [
  [0.5, 0.2, 0.1],
  [0.9, 0.7, 0.3],
  [0.4, 0.5, 0.7]
]

training_labels = [0, 1, 1]
classifier.fit(training_points, training_labels)
```

Finally, after training the model, we can classify new points. The `.predict()` method takes a list of points that you want to classify. It returns a list of its guesses for those points.

```python
unknown_points = [
  [0.2, 0.1, 0.7],
  [0.4, 0.7, 0.6],
  [0.5, 0.8, 0.1]
]

guesses = classifier.predict(unknown_points)
```

## **Review**

Congratulations! You just implemented your very own classifier from scratch and used Python’s `sklearn` library. In this lesson, you learned some techniques very specific to the K-Nearest Neighbor algorithm, but some general machine learning techniques as well. Some of the major takeaways from this lesson include:

- Data with `n` features can be conceptualized as points lying in n-dimensional space.
- Data points can be compared by using the distance formula. Data points that are similar will have a smaller distance between them.
- A point with an unknown class can be classified by finding the `k` nearest neighbors
- To verify the effectiveness of a classifier, data with known classes can be split into a training set and a validation set. Validation error can then be calculated.
- Classifiers have parameters that can be tuned to increase their effectiveness. In the case of K-Nearest Neighbors, `k` can be changed.
- A classifier can be trained improperly and suffer from overfitting or underfitting. In the case of K-Nearest Neighbors, a low `k` often leads to overfitting and a large `k` often leads to underfitting.
- Python’s sklearn library can be used for many classification and machine learning algorithms.

To the right is an interactive visualization of K-Nearest Neighbors. If you move your mouse over the canvas, the location of your mouse will be classified as either green or blue. The nearest neighbors to your mouse are highlighted in yellow. Use the slider to change `k` to see how the boundaries of the classification change.

# **K-NEAREST NEIGHBOR REGRESSOR**

## **Regression**

The K-Nearest Neighbors algorithm is a powerful supervised machine learning algorithm typically used for classification. However, it can also perform regression.

In this lesson, we will use the movie dataset that was used in the [K-Nearest Neighbors classifier lesson](https://www.codecademy.com/content-items/e6a14b06673aae14c8262dd5c3998401/exercises/knn). However, instead of classifying a new movie as either good or bad, we are now going to predict its IMDb rating as a real number.

This process is almost identical to classification, except for the final step. Once again, we are going to find the `k` nearest neighbors of the new movie by using the distance formula. However, instead of counting the number of good and bad neighbors, the regressor averages their IMDb ratings.

For example, if the three nearest neighbors to an unrated movie have ratings of `5.0`, `9.2`, and `6.8`, then we could predict that this new movie will have a rating of `7.0`.

```python
from movies import movie_dataset, movie_ratings

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def predict(unknown, dataset, movie_ratings, k):
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  sum = 0
  for neighbor in neighbors:
    title = neighbor[1]
    sum += movie_ratings[title]
  return sum/len(neighbors)

print(movie_dataset["Life of Pi"])
print(movie_ratings["Life of Pi"])
print(predict([0.016, 0.300, 1.022], movie_dataset, movie_ratings, 5))
```

## **Weighted Regression**

We’re off to a good start, but we can be even more clever in the way that we compute the average. We can compute a *weighted* average based on how close each neighbor is.

Let’s say we’re trying to predict the rating of movie X and we’ve found its three nearest neighbors. Consider the following table:

| Movie | Rating | Distance to movie X |
| --- | --- | --- |
| A | 5.0 | 3.2 |
| B | 6.8 | 11.5 |
| C | 9.0 | 1.1 |

If we find the mean, the predicted rating for X would be 6.93. However, movie X is most similar to movie C, so movie C’s rating should be more important when computing the average. Using a weighted average, we can find movie X’s rating:

![https://content.codecademy.com/courses/learn-knn/weightedAverage.png](https://content.codecademy.com/courses/learn-knn/weightedAverage.png)

The numerator is the sum of every rating divided by their respective distances. The denominator is the sum of one over every distance. Even though the ratings are the same as before, the *weighted* average has now gone up to 7.9.

```python
from movies import movie_dataset, movie_ratings

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def predict(unknown, dataset, movie_ratings, k):
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  numerator = 0
  denominator = 0
  for neighbor in neighbors:
    rating = movie_ratings[neighbor[1]]
    distance_to_neighbor = neighbor[0]
    numerator += rating / distance_to_neighbor
    denominator += 1 / distance_to_neighbor
  return numerator / denominator

print(predict([0.016, 0.300, 1.022], movie_dataset, movie_ratings, 5))
```

## **Scikit-learn**

Now that you’ve written your own K-Nearest Neighbor regression model, let’s take a look at scikit-learn’s implementation. The `KNeighborsRegressor` class is very similar to `KNeighborsClassifier`.

We first need to create the regressor. We can use the parameter `n_neighbors` to define our value for `k`.

We can also choose whether or not to use a weighted average using the parameter `weights`. If `weights` equals `"uniform"`, all neighbors will be considered equally in the average. If `weights` equals `"distance"`, then a weighted average is used.

```python
classifier = KNeighborsRegressor(n_neighbors = 3, weights = "distance")
```

Next, we need to fit the model to our training data using the `.fit()` method. `.fit()` takes two parameters. The first is a list of points, and the second is a list of values associated with those points.

```python
training_points = [
  [0.5, 0.2, 0.1],
  [0.9, 0.7, 0.3],
  [0.4, 0.5, 0.7]
]

training_labels = [5.0, 6.8, 9.0]
classifier.fit(training_points, training_labels)
```

Finally, we can make predictions on new data points using the `.predict()` method. `.predict()` takes a list of points and returns a list of predictions for those points.

```python
unknown_points = [
  [0.2, 0.1, 0.7],
  [0.4, 0.7, 0.6],
  [0.5, 0.8, 0.1]
]

guesses = classifier.predict(unknown_points)
```

```python
from movies import movie_dataset, movie_ratings
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors=5,weights='distance')

regressor.fit(movie_dataset,movie_ratings)

unknown = [
  [0.016, 0.300, 1.022],
[0.0004092981, 0.283, 1.0112],
[0.00687649, 0.235, 1.0112]
]

print(regressor.predict(unknown))
```

## **Review**

- The K-Nearest Neighbor algorithm can be used for regression. Rather than returning a classification, it returns a number.
- By using a weighted average, data points that are extremely similar to the input point will have more of a say in the final result.
- scikit-learn has an implementation of a K-Nearest Neighbor regressor named `KNeighborsRegressor`.

In the browser, you’ll find an example of a K-Nearest Neighbor regressor in action. Instead of the training data coming from IMDb ratings, you can create the training data yourself! Rate the movies that you have seen. Once you’ve rated more than `k` movies, a K-Nearest Neighbor regressor will train on those ratings. It will then make predictions for every movie that you haven’t seen.

As you add more and more ratings, the predictor should become more accurate. After all, the regressor needs information from the user in order to make personalized recommendations. As a result, the system is somewhat useless to brand new users — it takes some time for the system to “warm up” and get enough data about a user. This conundrum is an example of the *cold start problem*.