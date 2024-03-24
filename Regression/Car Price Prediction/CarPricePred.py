import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics
import statsmodels.api as sm
import math
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Change the current working directory to the script directory
os.chdir(script_dir)

# loading the data from csv file to pandas dataframe
car_datasets = pd.read_csv("car_data.csv")

# inspecting the first 5 rows
print(car_datasets.head())

# checking the nummber of rows and columns
print(car_datasets.shape)

# getting some info about the datasets
print(car_datasets.info())

# checking the distribution of categorical data
print(car_datasets.Fuel_Type.value_counts())
print(car_datasets.Seller_Type.value_counts())
print(car_datasets.Transmission.value_counts())

# Encoding Fuel_Type, Seller_Type and Transmission columns
# Initialize LabelEncoder
label_encoder = LabelEncoder()
# Fit LabelEncoder to the labels and transform them to integers
car_datasets['Fuel_Type']    = label_encoder.fit_transform(car_datasets['Fuel_Type'])
car_datasets['Seller_Type']  = label_encoder.fit_transform(car_datasets['Seller_Type'])
car_datasets['Transmission'] = label_encoder.fit_transform(car_datasets['Transmission'])

print(car_datasets.head())

# splitting the data into Training and Testing data
X = car_datasets.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_datasets['Selling_Price']

# Splitting Traning and Testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model training, loading the Linear Regression model
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, Y_train)

# The coefficients in a dataframe
cdf = pd.DataFrame(linear_reg_model.coef_,X.columns,columns=['Coef'])
print(cdf)

# prediction on training data
training_data_pred = linear_reg_model.predict(X_train)

# R squared Error
r_sq = metrics.r2_score(Y_train, training_data_pred)
print(f"coefficient of determination: {r_sq}")

# Visualize actual and predicted prices
sns.scatterplot(x=Y_train, y=training_data_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs. Predicted Prices')
plt.show()

# prediction on testing data
y_test_pred = linear_reg_model.predict(X_test)
print("r-squared: ", linear_reg_model.score(X_test, Y_test))

sns.scatterplot(x=Y_test, y=y_test_pred)
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Prices (y_test_pred)')
plt.title('Actual Prices vs. Predicted Prices')
plt.show()


# statsmodel
X = sm.add_constant(X_train)
model = sm.OLS(Y_train, X)
model_fit = model.fit()
# Example: Print coefficients
print("Coefficients:")
print(model_fit.params)
print(model_fit.summary())
# Assuming you have already fitted your model and stored it in model_fit
# Get the p-values of the coefficients
p_values = model_fit.pvalues
# Print the p-values
print("P-values:")
print(p_values)