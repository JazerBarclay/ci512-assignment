# This is a test application for the knn algorithm. It
# stores the number of errors is gets for each k value from
# 1 to 1000 and graphs the results using the matplotlib library.
# These results can be found in the evidence folder under figure 1.

# Import required modules
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

# Set shuffle_split params for test batches
cv = ShuffleSplit(n_splits=10, test_size=0.2)

# Array to store the errors received for each k value
error = []

# Calculating error for K values between 1 and 1000
for i in range(1, 1000):
    # Set the kNN classifier with 'i' number of neighbours
    knn = KNeighborsClassifier(n_neighbors=i)

    # Create a scaler for the values
    scaler = StandardScaler()

    # Find mean value for the test data
    scaler.fit(X)

    # Transform the feature data using the scaler
    X_scale = scaler.transform(X)

    # Split the input data into training and test data
    X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size = 0.2, random_state=0)

    # Train the kNN model using the training data
    knn.fit(X_train, Y_train)

    # Store predictions based on the test data
    pred_i = knn.predict(X_test)

    # Add the total number of errors received for this iteration's test
    error.append(np.mean(pred_i != Y_test))

# Plot the error data vs their corresponding k values
fig = plt.figure(figsize=(12, 6))
fig.canvas.manager.set_window_title('Figure 1 : Error Rate vs K Value')
plt.plot(range(1, 1000), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

# Show the plot data
plt.show()