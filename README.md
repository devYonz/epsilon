# Stanford ML Project CS229'18: epsilon

# 1 Abstract
<br><br>We envision using cutting-edge machine learning techniques and novel features to deliver higher accuracy predictions for assignees. During the duration of this project, we will focus on reducing the prediction error and view this project as an Application Result. In the future, we envision capabilities to infer priority and generalize the model to work with EHR diagnosis prediction, ticket triaging for customer support, bug reports, and exception tracking.

# 2 Introduction

Web and SaaS companies handle high volumes of tickets in the form of exceptions, support requests, user-reported bugs, and crash reports. JIRA and Asana are the most widely used task and ticketing systems to tame this beast with dedicated teams that work on aggregating, triaging and assigning these tickets to the right individual or team. However, effective automation is essential to improve productivity and obviate the tedious work of manually triaging tickets.

Project Epsilon aims to eliminate this overhead by experimenting with supervised-learning classifiers to intelligently and automatically assign tickets to a developer. It builds off of work done by Linkedin’s autotriager, a simple classifier that designates support tickets to pre-defined teams using an SVM and a simple text body input feature. The aforementioned autotriager has a high failure rate and only predicts a set of classes (teams). We aim to deliver higher accuracy for predicting assignee for a new ticket based on past tickets using more robust methods.

The input to our algorithm is a collection of historic JIRA tickets in JSON format. These tickets are preprocessed and featurized and following which we predict the assignee for new or unassigned tickets using 3 different methods: SVM, Naive Bayesian and Deep Neural Network Classifiers. We then compare the performance for these 3 methods as pertaining to 2 primary input datasets: a public Expium generated dataset as well as a dataset with real Jira tickets from Linkedin.

In the future, we envision extending Epsilon’s capabilities to infer priority and generalize the model to work with EHR diagnosis prediction, ticket triaging for customer support, bug reports, and exception tracking.



# 3 Introduction
Classification on open bug reports with supervised learning is a fairly common area of work. Common strategies are Naive Bayes and SVM classification on a multinomial event model input featurized as bag of words. The primary research revolves around interpreting text to predict assignee based on observed history. Additional research on time slicing methods, applicability of reinforcement learning, principal component analysis and feature extraction using neural networks prove promising.

Project Epsilon aims to eliminate this overhead by experimenting with supervised-learning classifiers to intelligently and automatically assign tickets to a developer. With high failure rates in state of the art solutions we aim to deliver higher accuracy for predicting assignee for a new ticket based on past tickets using deep learning models.

The input to our algorithm is a collection of historic JIRA tickets in JSON format. These tickets are preprocessed and featurized and following which we predict the assignee for new or unassigned tickets using 3 different methods: SVM, Naive Bayesian and Deep Neural Network Classifiers. We then compare the performance for these 3 methods as pertaining to 2 primary input datasets: a public Expium generated dataset as well as a dataset with real Jira tickets from Linkedin.


After complteing a grid search with 5-Fold cross vlaidation across multiple parameters the winner with the best performance has been found to be: 3-32.relu.0.005  Mean Train Accuracy: 0.999379073598363  Mean Test Accuracy: 0.28183346116770486  -0.7175456124306581

The next step is to onboard to word2vec and pandas to look through the data available.

## We invite you to look over the report and code for more details.