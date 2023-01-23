import numpy as numpy
import pandas as pandas
import sklearn.datasets
import seaborn as seaborn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# loading the data to a data frame
cancer_data_frame = pandas.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

print(cancer_data_frame)

# adding the 'target' column to the data frame
cancer_data_frame['label'] = breast_cancer_dataset.target

# print last 5 rows of the dataframe
cancer_data_frame.tail()

### Data Analysis

cancer_data_frame.shape
cancer_data_frame.info()

# statistical measures about the data
cancer_data_frame.describe()

# checking the distribution of Target Varibale
print(cancer_data_frame['label'].value_counts())

print(cancer_data_frame.groupby('label').mean())


### Separting Features and Targets

X = cancer_data_frame.drop(columns='label', axis=1)
Y = cancer_data_frame['label']

print(X, Y)

### Training and Testing sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

### Model training - Logistic regression

model = LogisticRegression()

model.fit(X_train, Y_train)

### Model Evaluation - Accuracy Score

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy on training data = ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy on test data = ', test_data_accuracy)

### A predictive system

input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

# change the input data to a numpy array
input_data_as_numpy_array = numpy.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Malignant')

else:
  print('The Breast Cancer is Benign')