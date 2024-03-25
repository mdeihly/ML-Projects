import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Change the current working directory to the script directory
os.chdir(script_dir)

# loading the data from csv file to pandas dataframe
insurance_datasets = pd.read_csv("insurance.csv")

# inspecting the first 5 rows of the dataframe
print(insurance_datasets.head())

# checking the nummber of rows and columns
print(insurance_datasets.shape)

# getting some info about the datasets
print(insurance_datasets.info())

# checking for missing values
print(insurance_datasets.isnull().sum())

# statistical meaures of the dataset
print(insurance_datasets.describe())

# distribution of age value
sns.set_style("whitegrid")
plt.figure(figsize=(6,6))
sns.displot(insurance_datasets['age'], kind='hist', kde=True, rug=True)
plt.title('Age Distribution')
plt.show()

# Gender column
#Define a color palette for the two categories
colors = ["#1f77b4", "#ff7f0e"] 
plt.figure(figsize=(6,6))
sns.countplot(x='sex', hue='sex', data=insurance_datasets, palette=colors, legend=False)
plt.title('Sex Distribution')
plt.show()

# bmi distribution
sns.set_style("whitegrid")
plt.figure(figsize=(6,6))
sns.displot(insurance_datasets['bmi'], kind='hist', kde=True, rug=True)
plt.title('BMI Distribution')
plt.show()

# children column
# Get unique values of 'children' column
unique_children = insurance_datasets['children'].unique()
# Define a color palette with unique colors for each category
colors = sns.color_palette("pastel", len(unique_children))
plt.figure(figsize=(6,6))
sns.countplot(x='children', hue='children', data=insurance_datasets, palette=colors, legend=False)
plt.title('Children')
plt.show()

# smoker column
colors = ["#1f77b4", "#ff7f0e"] 
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', hue='smoker', data=insurance_datasets, palette=colors, legend=False)
plt.title('smoker')
plt.show()

# region column
plt.figure(figsize=(6,6))
sns.countplot(x='region', hue='region', data=insurance_datasets, legend=False)
plt.title('region')
plt.show()

# distribution of charges value
sns.set_style("whitegrid")
plt.figure(figsize=(6,6))
sns.displot(insurance_datasets['charges'], kind='hist', kde=True, rug=True)
plt.title('Charges Distribution')
plt.show()

# Data Pre-processing. encoding categorical featuresd
insurance_datasets.replace({'sex': {'male':0, 'female':1}},inplace=True)
insurance_datasets.replace({'smoker': {'yes':0, 'no':1}},inplace=True)
insurance_datasets.replace({'region': {'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}},inplace=True)

print(insurance_datasets.head())

# splitting the Features and Target
X = insurance_datasets.drop(columns='charges',axis=1)
Y = insurance_datasets['charges']

# Splitting Traning and Testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model training, loading the Linear Regression model
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, Y_train)

# prediction on training data
training_data_pred = linear_reg_model.predict(X_train)

# R squared Error
r_sq = metrics.r2_score(Y_train, training_data_pred)
print(f"coefficient of determination: {r_sq}")

# prediction on testing data
y_test_pred = linear_reg_model.predict(X_test)
print("r-squared: ", metrics.r2_score(Y_test, y_test_pred))

# build a predictive system (from csv file)
# female = 1, no = 1, southeast = 0
input_data =  [31,1,25.74,0,1,0]
# changing input_data (tuple) to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# Provide feature names to the reshaped input data
prediction = linear_reg_model.predict(input_data_reshaped)
print(prediction)


