import math
import pandas as pd
import numpy as np
import scipy.stats as stats

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

mpl.rcParams["font.family"] = "Times New Roman"
plt.style.use("default")

# Functions for removing collinear predictor variables
def GetVFISeries(predictors):
	"""
	This function returns a Series containing the VFI's of the predictor variables
		predictors (DataFrame) : A DataFrame containing all the predictor variables
	"""
	X1 = sm.tools.add_constant(predictors)

	# Create a list of the variance inflation factors of all the predictor variables
	vfi_list = [variance_inflation_factor(X1, i) for i in range(1,X1.shape[1])]
	# Create a Series of the variance inflation factors
	vfi_series = pd.Series(vfi_list, index = predictors.columns)
	return vfi_series

def RemoveHighestVFI(predictors, show = True):
	"""
	This function removes the predictor variable from a Series with the highest VFI
		predictors (DataFrame) : A DataFrame containing all the predictor variables
		show (boolean) : Whether you want to see the predictor variable which got removed
	"""
	vfi_series = GetVFISeries(predictors)
	# Find the predictor variable with the highest VFI
	highest_vfi_predictor = vfi_series.idxmax()
	print(highest_vfi_predictor)
	# Remove this predictor variable from the DataFrame
	predictors.drop(columns = highest_vfi_predictor, inplace = True)

def RemoveCollinear(predictors, threshold = 10):
	"""
	This function removes collinear predictor variables for an DataFrame
		predictors (DataFrame) : A DataFrame containing all the predictor variables
		threshold (float) : The maximum variance inflation factor for a predictor variable
		that we keep in the model
	"""
	print("Removing collinear predictor variables...")
	vfi_series = GetVFISeries(predictors)
	# Create a boolean Series which is true where the VFI is greater than the threshold
	vfi_mask = vfi_series > threshold

	# While there are any predictors with VFI's above the threshold
	while vfi_mask.any():
		# Remove the predictor variable with the highest VFI
		RemoveHighestVFI(predictors)
		# Get the VFI's of the remaining predictor variables
		vfi_series = GetVFISeries(predictors)
		# Create a boolean Series which is true where the VFI is greater than the threshold
		vfi_mask = vfi_series > threshold
	print(f"All predictor variables with a variance inflation factor above {threshold} have been removed.")

def CorrelationFinder(df, corr_variable = "Log Returns"):
	"""
	This function prints a list of the correlations between one variable and the others in the DataFrame
		Parameters:
			df (DataFrame) : The complete data we want to analyse
			corr_variable (string) : The corr_variable we want to measure the correlations against
	"""
	corr_df = df.corr()
	print(corr_df.loc[:,corr_variable].sort_values(ascending = False))


def StatisticsDF(df):
	"""
	This function returns a DataFrame with descriptive statistics on each the column of the input DataFramne
		Parameters:
			df (DataFrame) : The input DataFrame to describe
		Returns:
			describe_df (DataFrame) : The DataFrame with descriptive statistics
	"""
	stats_functions = [len, min, max, np.mean, np.median, np.std]
	describe_df = df.aggregate(stats_functions)
	describe_df.rename(
		{"len":"Size",
		"min":"Minimum",
		"max":"Maximum",
		"mean":"Mean",
		"median":"Median",
		"std":"Standard Deviation"
		}, inplace = True)

	skew_arr = np.array(stats.skew(df)).reshape(1,-1)
	skew_df = pd.DataFrame(skew_arr, index = ["Skewness"], columns = df.columns)

	kurt_arr = np.array(stats.kurtosis(df)).reshape(1,-1)
	kurt_df = pd.DataFrame(kurt_arr, index = ["Kurtosis"], columns = df.columns)

	skew_test_arr = np.array(stats.skewtest(df)[1]).reshape(1,-1)
	skew_test_df = pd.DataFrame(skew_test_arr.round(4), index = ["Skew Test p-value"], columns = df.columns)

	kurt_test_arr = np.array(stats.kurtosistest(df)[1]).reshape(1,-1)
	kurt_test_df = pd.DataFrame(kurt_test_arr.round(4), index = ["Kurtosis Test p-value"], columns = df.columns)

	describe_df = pd.concat([describe_df, skew_df, kurt_df, skew_test_df, kurt_test_df], axis = 0)
	return describe_df

# Functions for visualising the data
def Correlations(df):
	"""
	This function
	- prints a DataFrame showing the correlations between the columns of the DataFrmae
	- produces a heatmap plot showing the correlations between the columns of the DataFrmae
		Parameters:
			df (DataFrame) : A DataFrame containing all the predictor variables
	"""
	corr_df = df.corr()
	print(corr_df)
	sns.heatmap(corr_df, xticklabels = corr_df.columns, yticklabels = corr_df.columns, cmap = "RdBu")
	plt.show()

def ScatterMatrix(df, variable_list):
	"""
	This function produces a plot of each variable against the every other variable and plots a histogram of the variable
		Parameters:
			df (DataFrame) : The data
			variable_list (array_like) : A list of the variables to include in the plot
		Returns:
			None
	"""
	from pandas.plotting import scatter_matrix
	scatter_matrix(df.loc[:,variable_list])
	plt.show()

def ScatterPlot(df, x, y):
	"""
	This function creates a scatter plot of 2 columns in a DataFrame
		Parameters:
			df (DataFrame) : The DataFrame containing the data we want to plot
			x (string) : The name of the column to plot as the dependent variable
			y (string) : The name of the column to plot as the independent variable
	"""
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 1, 1)
	ax1.set_title(f"{y} against {x}")
	ax1.set_xlabel(x)
	ax1.set_ylabel(y)
	ax1.scatter(df.loc[:,x], df.loc[:,y])
	plt.show()

def ResidualPlot(residuals, forecast_train):
	"""
	This function produces a residual plot
		residuals (ndarray) : A NumPy array containing the residuals
		forecast_train (ndarray) : A NumPy array conta
	"""
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 1, 1)
	ax1.set_title("Residual Plot")
	ax1.set_xlabel("Predicted Response Value")
	ax1.set_ylabel("Residual")
	ax1.hlines(y = 0, xmin = forecast_train.min(), xmax = forecast_train.max(), linewidth = 2, color = 'black')
	ax1.scatter(forecast_train, residuals)
	plt.show()

# Read the on-chain data and get it as log returns
onchain_df = pd.read_csv("ONCHAIN.csv",index_col = 0, parse_dates = True)
onchain_df.sort_index(ascending=False,inplace = True)
onchain_df[onchain_df == 0] = None
onchain_df.dropna(inplace = True)

onchain_df = np.log(onchain_df/onchain_df.shift(1))
# Read the btc data
btc_df = pd.read_csv("BTCUSD.csv",index_col=0,parse_dates=True)
btc_close = btc_df.loc[:,"Close"]

log_returns = np.log(btc_close/btc_close.shift(1))
log_returns.name = "Log Returns"

tom_log_returns = log_returns.shift(-1)
tom_log_returns.name = "Tomorrow's Log Returns"

# Combine the DataFrames
df = pd.concat([tom_log_returns, log_returns, onchain_df], axis = 1, join = "inner")
df.dropna(inplace = True)

X = df.drop(columns = ["Tomorrow's Log Returns","Log Returns"])
y = df.loc[:,"Tomorrow's Log Returns"]

# We want to remove predictor variables that are correlated to other predictor variables
RemoveCollinear(X, 5)

# Split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Fit the multiple linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
# Compute the predicted response value
forecast_train = lin_reg.predict(X_train)
forecast_test = lin_reg.predict(X_test)

# Training R2
train_r2 = r2_score(y_train, forecast_train)
print(f"Training R squared: {train_r2}")

# Test R2
test_r2 = r2_score(y_test, forecast_test)
print(f"Test R squared: {test_r2}")


