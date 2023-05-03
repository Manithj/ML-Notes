# Training, Validation and Test Datasets

**This article teaches the importance of splitting a dataset into training, validation and test sets.**

Supervised machine learning models have the potential to learn patterns from labeled datasets and make predictions. But how do we know if the predictions our models are making are even useful? After all, it is possible that every prediction our email spam classifier makes is actually wrong, or that every housing price prediction our regression model makes is orders of magnitude off!

Luckily, we can leverage the fact that supervised machine learning models require a dataset of pre-labeled observations to help us determine the performance of our model. To do so, we will have to split our labeled dataset into three chunks – a training, validation and test (or holdout) set.

### **Training-Validation-Test Split**

Before we even begin to think about writing the code we will use to fit our machine learning model to the data, we must perform the split of our dataset to create our training set, validation set, and test set.

The *training set* is the data that the model will learn how to make predictions from. Learning looks different depending on which algorithm we are using. With Linear Regression, the observations in the training set are used to fit the parameters of a line of best fit. For choose different model (classification), the observations in the training set are used to fit the parameters being fit.

The *validation set* is the data that will be used during the training phase to evaluate the interim performance of the model, guide the tuning of hyper-parameters, and assist in other model improvement capacities (for example, feature selection). Some common metrics used to calculate the performance of machine learning models are [accuracy, recall, precision, and F1-Score](https://www.codecademy.com/content-items/1dd2cb55a893072f4dce2911004eeba2/exercises/accuracy). The metric we choose to use will vary depending on our particular use case.

The *test set* is the data that will determine the performance of our final model so we can estimate how our model will fare in the real world. To avoid introducing any bias to the final measurements of performance, we do not want the test set anywhere near the model training or tuning processes. That is why the test set is often referred to as the holdout set.

### **Evaluating the Model**

During model fitting, both the features (X) and the true labels (y) of the training set (Xtrain, ytrain) are used to learn. When evaluating the performance of the model with the validation (Xval, yval) or test (Xtest, ytest) set, we are going to temporarily pretend like we do not know the true label of every observation. If we use the observation features in our validation (Xval) or test (Xtest) sets as inputs to the trained model, we can receive a prediction as output for each observation (ypred). We can now compare each of the true labels (yval or ytest) with each of the predicted labels (ypred) and get a quantitative evaluation on the performance of the model.

![https://static-assets.codecademy.com/skillpaths/ml-fundamentals/evaluation-metrics/train%20test%20figure.png](https://static-assets.codecademy.com/skillpaths/ml-fundamentals/evaluation-metrics/train%20test%20figure.png)

### **How to Split**

Figuring out how much of our data should be split into training, validation, and test sets is a tricky question and there is no simple answer. If our training set is too small, then the model might not have enough data to effectively learn. On the other hand, if our validation and test sets are too small, then the performance metrics could have a large variance. In general, putting 70% of the data into the training set and 15% of the data into each of the validation and test sets is a good place to start.

### **N-Fold Cross-Validation**

Sometimes our dataset is so small that splitting it into training, validation, and test sets that are appropriate sizes is unfeasible. A potential solution is to perform N-Fold Cross-Validation. While we still first split the dataset into a training and test set, we are going to further split the training set into N chunks. In each iteration (or fold), N-1 of the chunks are treated as the training set and 1 of the chunks is treated as the validation set over which the evaluation metrics are calculated.

![https://content.codecademy.com/programs/data-science-path/cross_validation.svg](https://content.codecademy.com/programs/data-science-path/cross_validation.svg)

This process is repeated N times cycling through each chunk acting as the validation set and the evaluation metrics from each fold are averaged. For example, in 10-fold cross-validation, we’ll make the validation set the first 10% of the training set and calculate our evaluation metrics. We’ll then make the validation set the second 10% of the data and calculate these statistics once again. We can do this process 10 times, and every time the validation set will be a different chunk of the data. If we then average all of the accuracies, we will have a better sense of how our model does on average.