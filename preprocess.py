# This is the final version of my knn test application. This
# program will take in the test data and perform 5 tests using
# a test split as well as shuffle split. It will then print the
# results to the stdout which I have recorded in the results folder.

# Import required modules
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd


# Read the data from the comma separated file with the pandas library
data = pd.read_csv("emails.csv")

print(data)
print()

print("Total Missing Values", data.isnull().sum().sum())
print()

# X uses all rows (:) and the 1st to 3000th columns as feature data (1-3000)
X = data.values[:, 1:3000]
# Cast all feature data as integers
X = X.astype('int')

# Y uses all rows (:) and the final column as the label data (3001) 
Y = data.values[:,3001]
# Cast label feature as integer
Y = Y.astype('int')

X = StandardScaler().fit_transform(X)

columnCount = data.iloc[:,-1].values
totalRecords = len(data.index)
totalSpam = columnCount.sum()
totalNonSpam = totalRecords - totalSpam

print("Total Records", totalRecords)
print("Total Spam Records", totalSpam)
print("Total Non-Spam Records", totalNonSpam )
print()
print("Total Spam %",  "{:.2f}".format((totalSpam/totalRecords)*100) )
print("Total Non-Spam %",  "{:.2f}".format((totalNonSpam/totalRecords)*100) )

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
    , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, data.iloc[:,-1]], axis = 1)

print(finalDf)


fig = plt.figure(figsize = (8,8))
fig.canvas.manager.set_window_title('Figure 0 : Principal Component Analysis of Spam Dataset')
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Prediction'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 20)

ax.legend(targets)
ax.grid()

plt.show()