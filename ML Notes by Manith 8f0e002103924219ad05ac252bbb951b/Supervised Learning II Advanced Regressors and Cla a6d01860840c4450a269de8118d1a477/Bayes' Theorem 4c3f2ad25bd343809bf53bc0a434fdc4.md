# Bayes' Theorem

## **Introduction to Bayes' Theorem**

In this lesson, we’ll learn about **Bayes’ Theorem**. Bayes’ Theorem is the basis of a branch of statistics called *Bayesian Statistics*, where we take prior knowledge into account before calculating new probabilities.

This allows us to find narrow solutions from a huge universe of possibilities. British mathematician Alan Turing used it to crack the German Enigma code during WWII. And now it is used in:

- Machine Learning
- Statistical Modeling
- A/B Testing
- Robotics

By the end of this lesson, you’ll be able to solve simple problems involving prior knowledge.

![Untitled](Bayes'%20Theorem%204c3f2ad25bd343809bf53bc0a434fdc4/Untitled.png)

## **Independent Events**

The ability to determine whether two events are *independent* is an important skill for statistics.

If two events are **independent**, then the occurrence of one event does not affect the probability of the other event. Here are some examples of independent events:

- I wear a blue shirt; my coworker wears a blue shirt
- I take the subway to work; I eat sushi for lunch
- The NY Giants win their football game; the NY Rangers win their hockey game

If two events are **dependent**, then when one event occurs, the probability of the other event occurring changes in a predictable way.

Here are some examples of dependent events:

- It rains on Tuesday; I carry an umbrella on Tuesday
- I eat spaghetti; I have a red stain on my shirt
- I wear sunglasses; I go to the beach

## **Conditional Probability**

*Conditional probability* is the probability that two events happen. It’s easiest to calculate conditional probability when the two events are independent.

**Note:** For the rest of this lesson, we’ll be using the statistical convention that the probability of an event is written as `P(event)`.

If the probability of event `A` is `P(A)` and the probability of event `B` is `P(B)` and the two events are independent, then the probability of both events occurring is the product of the probabilities:

![Untitled](Bayes'%20Theorem%204c3f2ad25bd343809bf53bc0a434fdc4/Untitled%201.png)

The symbol ∩ just means “and”, so `P(A ∩ B)` means the probability that both `A` and `B` happen.

For instance, suppose we are rolling a pair of dice, and want to know the probability of rolling two sixes.

![Untitled](Bayes'%20Theorem%204c3f2ad25bd343809bf53bc0a434fdc4/Untitled%202.png)

Each die has six sides, so the probability of rolling a six is 1/6. Each die is independent (i.e., rolling one six does not increase or decrease our chance of rolling a second six), so:

![Untitled](Bayes'%20Theorem%204c3f2ad25bd343809bf53bc0a434fdc4/Untitled%203.png)

## **Testing for a Rare Disease**

Suppose you are a doctor and you need to test if a patient has a certain rare disease. The test is very accurate: it’s correct 99% of the time. The disease is very rare: only 1 in 100,000 patients have it.

You administer the test and it comes back positive, so your patient must have the disease, right?

Not necessarily. If we just consider the test, there is only a 1% chance that it is wrong, but we actually have more information: we know how rare the disease is.

Given that the test came back positive, there are two possibilities:

1. The patient had the disease, and the test correctly diagnosed the disease.
2. The patient didn’t have the disease and the test incorrectly diagnosed that they had the disease.

## **Bayes' Theorem**

In the previous exercise, we determined two probabilities:

1. The patient had the disease, and the test correctly diagnosed the disease ≈ 0.00001
2. The patient didn’t have the disease and the test incorrectly diagnosed that they had the disease ≈ 0.01

Both events are rare, but we can see that it was about 1,000 times more likely that the test was incorrect than that the patient had this rare disease.

We’re able to come to this conclusion because we had more information than just the accuracy of the test; we also knew the prevalence of this disease.

In statistics, if we have two events (`A` and `B`), we write the probability that event `A` will happen, given that event `B` already happened as `P(A|B)`. In our example, we want to find `P(rare disease | positive result)`. In other words, we want to find the probability that the patient has the disease *given* the test came back positive.

We can calculate `P(A|B)` using **Bayes’ Theorem**, which states:

![Untitled](Bayes'%20Theorem%204c3f2ad25bd343809bf53bc0a434fdc4/Untitled%204.png)

So in this case, we’d say:

![Untitled](Bayes'%20Theorem%204c3f2ad25bd343809bf53bc0a434fdc4/Untitled%205.png)

It is important to note that on the right side of the equation, we have the term `P(B|A)`. This is the probability that event `B` will happen given that event `A` has already happened. This is very different from `P(A|B)`, which is the probability we are trying to solve for. The order matters!

## **Spam Filters**

Let’s explore a different example. Email spam filters use Bayes’ Theorem to determine if certain words indicate that an email is [spam](https://en.wikipedia.org/wiki/Email_spam).

Let’s take a word that often appears in spam: “enhancement”.

With just 3 facts, we can make some preliminary steps towards a good spam filter:

1. “enhancement” appears in just 0.1% of non-spam emails
2. “enhancement” appears in 5% of spam emails
3. Spam emails make up about 20% of total emails

Given that an email contains “enhancement”, what is the probability that the email is spam?

```python
import numpy as np

a = 'spam'
b = 'enhancement'

p_spam = 0.2
p_enhancement_given_spam = 0.05
p_enhancement = 0.05 * 0.2 + 0.001 * (1 - 0.2)
p_spam_enhancement = p_enhancement_given_spam * p_spam / p_enhancement

print(p_spam_enhancement) #0.9259259259259259
```