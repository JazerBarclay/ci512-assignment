# This is a test application for the random forest algorithm. It
# stores the number of errors is gets for each forest size value from
# 1 to 500 and graphs the results using the matplotlib library.
# These results can be found in the evidence folder under figure 3.

# Import required modules
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# Create a scaler for the values
scaler = StandardScaler()

# Find mean value for the test data
scaler.fit(X)

# Transform the feature data using the scaler
X = scaler.transform(X)

# Set shuffle_split params for test batches
cv = ShuffleSplit(n_splits=10, test_size=0.2)

# Array to store the errors received for each k value
error = []

# Calculating error rate for forest size values between 1 and 500
for i in range(1, 500):

    # Create the random forest classifier with a forest size of 'i'
    rf = RandomForestClassifier(n_estimators=i)

    # Split the input data into training and test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
    
    # Train the random forest model using the training data
    rf.fit(X_train, Y_train)
    
    # Store predictions based on the test data
    pred_i = rf.predict(X_test)
    
    # Add the total number of errors received for this iteration's test
    error.append(np.mean(pred_i != Y_test))

# Plot the error data vs their corresponding k values
fig = plt.figure(figsize=(12, 6))
fig.canvas.manager.set_window_title('Figure 3 : Error Rate vs Forest Size')
plt.plot(range(1, 500), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate vs Forest Size')
plt.xlabel('Estimators (Forest Size)')
plt.ylabel('Mean Error')

# Show the plot data
plt.show()