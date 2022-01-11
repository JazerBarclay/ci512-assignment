# This is the final version of my decision tree test application.
# This program will take in the test data and perform 5 tests using
# a test split as well as shuffle split. It will then print the
# results to the stdout which I have recorded in the results folder.

# Import required modules
from sklearn.model_selection import ShuffleSplit, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

# Set randomness based on seed '95'
np.random.seed(95)

# Read the data from the comma separated file with the pandas library
data = pd.read_csv("../emails.csv", skiprows=1)

# X uses all rows (:) and the 1st to 3000th columns as feature data (1-3000)
X = data.values[:, 1:3000]
# Cast all feature data as integers
X = X.astype('int')

# Y uses all rows (:) and the final column as the label data (3001) 
Y = data.values[:,3001]
# Cast label feature as integer
Y = Y.astype('int')

# Feature scaling - reduce data to scaled value from 0 to 1
scaler = StandardScaler()

# Calculate the mean value using the input data 
scaler.fit(X)

# Scale the input data to use the new scaling values based on the previous statement
X = scaler.transform(X)

# For every value from 1 to 5 (run this 5 times for multiple random results)
for i in range(0,5):

    # Shuffle dataset
    X, Y = shuffle(X, Y)

    # Create teh decision tree classifier model
    clf = DecisionTreeClassifier()

    # Split the data into training and test sets with a set percentage split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

    # Add the test dataset to the k-nearest neighbors classifier
    clf = clf.fit(X_train, Y_train)

    # Stores the prediction using the test set
    Y_prediction = clf.predict(X_test)

    # Set shuffle_split params
    cv = ShuffleSplit(n_splits=10, test_size=0.2)

    # Calculate cross test score using shuffle split params
    scores = cross_val_score(clf, X, Y, cv=cv)

    # Print the accuracy of the prediction compared to the actual results
    print("Train/test accuracy:",accuracy_score(Y_test,Y_prediction))

    # Print the mean value for all tests
    print("Cross fold validation accuracy mean:",scores.mean())

    # Store all values from confusion matrix as individual values
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_prediction).ravel()

    # Store total to match input size
    total = tn + fp + fn + tp

    # Print confusion matrix and classification report
    print("False Positive: " + str( (fp/total) ))
    print("False Negative: " + str( (fn/total) ))

    print("Confusion Matrix")
    print("TP: " + str(tp) + " | TN: " + str(tn))
    print("FP: " + str(fp) + " | FN: " + str(fn))
    print("Precision: " + str(tp/(tp+fp)))
    print("Classification Report: ")
    print(classification_report(Y_test, Y_prediction))
    print("= = = = = = = = = = = =")

    