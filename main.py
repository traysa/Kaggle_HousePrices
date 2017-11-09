import pandas as pd
import matplotlib
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Read training data into a dataframe and show summary of it
train = pd.read_csv("./Data/train.csv")
train.info()

# Read test data into a dataframe and show summary of it
test = pd.read_csv("./Data/test.csv")
test.info()

# Generate statistics (count, mean, std, min, 25%, 50%, 75%, max) for each column
# of the training dataframe (excluding NaN values)
train.describe()

# Return the first 5 rows of the training dataframe
train.head()

# Check if the row 'Utility' by counting the occurances of its values
train.groupby(['Utilities']).count() # Utilities not useful since all accept one have the same value

# Create releveant columns set
# LotArea, MSZoning, YrSold, OverallQual, OverallCond, GrLivArea
relevantColumns = ['LotArea', 'YrSold', 'OverallQual', 'OverallCond', 'GrLivArea']

# Generate statistics (count, mean, std, min, 25%, 50%, 75%, max) for the relevant columns
# of the training dataframe (excluding NaN values)
train[relevantColumns].describe()

# Boxplot the values of column 'OverallCond_quartiles' of the training dataframe
train['OverallCond_quartiles'] = pd.qcut(train['OverallCond'], 4, duplicates='drop')
train.boxplot(column='OverallCond', by='OverallCond_quartiles')

# Fit regression model on first 1400 data entries (of total 1460)
X_train = train[relevantColumns].head(1400)
y_train = train['SalePrice'].head(1400)
regr_1 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X_train, y_train)

# Predict the validation data in the last 10 data entries (of 1460)
X_validation = train[relevantColumns].tail(10)# pd.read_csv("./Data/test.csv")[relevantColumns]
y_validation = regr_1.predict(X_validation)
y_validation

# Correct answers of the validation rows
np.array(train['SalePrice'].tail(10))

# Get error rates of validation data
# (prediction data - real data) / real data * 100
(y_validation - np.array(train['SalePrice'].tail(10))) / np.array(train['SalePrice'].tail(10))*100

# Predict the test data
X_test = test[relevantColumns]
y_test = regr_1.predict(X_test)

# output results in a csv
test_dataframe = pd.DataFrame(data={
    'Id': np.array(test['Id']),
    'SalePrice': y_test
})
test_dataframe.to_csv('Data/results.csv', index=False)


# Plot correlation of single columns to the result
col = 2
relevantColumns[col]
col_X_test = (X_test.ix[:,col]).values
z_test = np.sort(np.vstack([col_X_test,y_test]),axis=1)
col_X_train = (X_train.ix[:,col]).values
z_train = np.sort(np.vstack([col_X_train,y_train]),axis=1)

# Plot the results
plt.figure()
plt.scatter(z_test[0],z_test[1],s=20, edgecolor="black", c="darkorange", label=relevantColumns[col])
# plt.plot(z_train[0],z_train[1], color="cornflowerblue", label="max_depth=5", linewidth=2)
# plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel(relevantColumns[col])
plt.ylabel("SalePrice")
plt.title("Correlation of "+relevantColumns[col])
plt.legend()
plt.show()


