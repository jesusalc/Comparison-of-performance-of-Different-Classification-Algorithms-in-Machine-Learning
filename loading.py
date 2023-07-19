#!/usr/bin/env python 

#  Python code snippet that showcases the performance of various machine learning classification algorithms using a given dataset, implementation of KNN from scratch, and the generation of a PDF presentation.

# simplified version of the task due to the limitations of the text-based interface. I'll also provide explanations about what each part does. In an actual implementation, you would need to install the necessary Python libraries and run the code in a Python environment.

### Step 1: Classification Algorithm Comparison

# Required Libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
data = pd.read_csv('DatingAppReviewsDataset.csv')

# We will convert the 'Review' column into numerical features using Count Vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Review'])

# 'Rating' will be our target variable
y = data['Rating']

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# List of models
models = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('Support Vector Classifier', SVC(probability=True)),
    ('Logistic Regression', LogisticRegression()),
    ('KNN', KNeighborsClassifier()),
]

# Loop through models, train, predict, and get performance
for name, model in models:
    model.fit(X_train, y_train)  # Train model
    y_pred_prob = model.predict_proba(X_test)[:,1]  # Predict probabilities
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)  # Get ROC curve
    roc_auc = roc_auc_score(y_test, y_pred_prob)  # Get ROC AUC
    plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')  # Plot ROC curve

# Plot ROC curve settings
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel



