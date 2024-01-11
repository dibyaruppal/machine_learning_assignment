# Machine Learning Assisgment II
## 1. Linear regression and Regularization:
(a) Predict the “Overall” (target attribute: “overall”) rating of the players using Linear
regression report the Mean Absolute Error(MAE), Mean Square Error(MSE), R2 score.

(b) Compare the performance of linear regression, Ridge regression, and Lasso regression
models. Perform the hyperparameters tuning and observe how they affect the model’s
bias-variance trade-off, investigate the impact of the Lasso regularisation parameter on
this feature selection process.
### Note: Please carry out the necessary data preprocessing and test-train split as 20 : 80%.
**The use of the scikit-learn library is allowed for this question**. For (b), include necessary
metrics like MSE, MAE, R2 Score for performance analysis and necessary plots (Ex:
Scatter plots/line plots) for hyperparameters tuning.

## 2. Logistic Regression: 
You are given a dataset named football.csv containing information
about football players. Your task is to build a machine learning model to classify whether
a player’s contribution type is more inclined towards being type 1 or 0, where 1 indicates
players with contributions in the attacking half of the football field and 0 indicates players
with contributions in the defending half of the field. The classification column is “contribution type”.

(a) Train a Logistic Regression model using the training data. Implement logistic regression
from scratch. **You’re NOT ALLOWED to use sklearn for this question**.

(b) Make predictions on the test data using the trained model.

(c) Calculate the F1 score, accuracy score, and confusion matrix to evaluate the model’s
performance.

## 3. K-Means - Clustering of Football Clubs: 
You are given a dataset containing football
player information. The objective is to cluster different football clubs based on various attributes.

(a) Your initial step should be extracting the club information from the player dataset.
It will involve computing the “average player” of each club. One way is to group the
dataset by “club name id” and calculate the mean values for all relevant features. You
are encouraged to explore different ideas.

(b) Now, use K-Means to cluster the football clubs. You can determine the criteria for
clustering by considering various features. For example - First try clustering using all
features, after that try again using only features representing player stats or financial
attributes.

(c) To find the optimal number of clusters, K, you can use the elbow method.

## 4. Random Forest: 
For this question also, you have to use the same football dataset. The
aim is to use the Random Forest model to do classification and regression both.

(a) Classification: Your target column is ‘contribution type’. Process the data as you want,
modify/drop any columns that you want, and play around with the hyperparameters.
Try to understand and observe the difference in results. Try different losses (or quality
criterion) – ‘gini’, ‘entropy’, ‘log loss’. After training the model, report test accuracy
and f1 score.

(b) Regression: Your target column is ‘overall’. Again, you are free to process the dataset
and encouraged to try different hyperparameters. Use MSE and MAE one by one to
train the models, and report test MSE and MAE for both models.
### Note: You can use sklearn library to get Random Forest implementations.



# Machine Learning Assisgment II
## 1 Book Review
Given the review text and other information about it, you have to predict the rating associated with that review.
Check the attached excel file which contains the group details, Kaggle competition links, and WhatsApp group links.
Although the dataset is same for everyone, the competitions will be in groups. Each TA will hosting
his own competition on Kaggle. You have to participate ONLY in the assigned competition.
This is a classification problem. Rating 0, 1, 2, 3, 4, and 5 are the classes. The metric to be used
for leader board is weighted F1 score.
Each of the Kaggle competitions contain the dataset to be used. Your goal should be to try different
models, parameters, etc and climb up the leader board of your group. So, your model performance
matters. Final position on the private leader board (See Kaggle competition page to see what is
this) will contribute to your marks. But don’t worry, it is not the only thing that will bring you
marks. Your efforts and understanding matter the most. So don’t get discouraged by the
leader board at all.
### Note:
• For all the rows in test.csv file, you have to predict the rating (0-5). You will submit a csv file on the competition page with review id as first column, and rating as second. Check the sample submission.csv on the competition’s page.

• You can make many submissions to the competition, the best one will be reflected on the public leader board. Although there is a daily limit of 15 submissions

• You can choose any 2 of your submissions to be considered for the private leader board.
Note that this is a Natural Language Processing (NLP) dataset, so you are expected to do extensive research on how to tackle NLP problems as this is a fairly new domain for you. To help you, here are some hints:

• Analyze the text data and perform appropriate preprocessing steps such as text cleaning, tokenization, or stemming/lemmatization.

• You can create features using basic techniques like bag of words or TF-IDF.

• The above two steps will affect your model performance a lot, so do play around with different preprocessing techniques and feature engineering hyperparameters.

• This is a multi-class classification problem. The class distribution might be skewed towards higher ratings (1 star ratings are always rare). How would you tackle class imbalance?

• Remember that you don’t have test data labels. Use validation techniques to test your model before submitting on Kaggle.

Lastly, this is a fairly large dataset. Don’t waste time by running time-consuming code again
and again. Save your preprocessed data and read about saving models with pickle.
On LMS, you will submit the .ipynb that gave you the best results (the one you selected for
private leader board) and the report.

## 2 Neural Networks
You have to build a neural network using Numpy. So you CANNOT use TensorFlow, PyTorch
or any other library with built-in neural networks.
The dataset is uploaded on LMS along with this assignment. It is a regression task. You have to
predict the ’Price’ of the house.
