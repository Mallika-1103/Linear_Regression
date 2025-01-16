import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
warnings.simplefilter(action='ignore', category=FutureWarning)
df = pd.read_csv(r"C:\Users\VIGNESH BASKARAN\Downloads\archive\Ecommerce Customers.csv")
print('------------------First Five Rows-----------------------------------------------------')
print(df.head())
print('------------------Last Five Rows-----------------------------------------------------')
print(df.tail())
print('------------------Information About the DataFrame-----------------------------------------------------')
print(df.info)
print('------------------Summary Statistics-----------------------------------------------------')
print(df.describe())


# replacing infinite values with NaN 
df.replace([np.inf, -np.inf], np.nan, inplace=True)
# checking for NaN values, 0 = good
print(df[['Time on Website', 'Yearly Amount Spent']].isna().sum())

#EDA
#Compaere the Columns for Find the Best Correlations
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=df, alpha=0.5)
plt.gcf().canvas.manager.set_window_title('x="Time on Website", y="Yearly Amount Spent"')
sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=df, alpha=0.5)
plt.gcf().canvas.manager.set_window_title('x="Time on Website",y="Yearly Amount Spent"')

#Visualise All rows and Columns
sns.pairplot(df, kind='scatter', plot_kws={'alpha':0.4})
plt.gcf().canvas.manager.set_window_title('Analyse of All rows with columns')
 
#Linear Plot  
x_axis = 'Length of Membership'
y_axis = 'Yearly Amount Spent'
sns.lmplot(x=x_axis,y=y_axis,data = df,scatter_kws={'alpha':0.3})
plt.gcf().canvas.manager.set_window_title('My Linear  Plot')
plt.show()
 

#Train and Testing
X = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = df['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lm=LinearRegression()
lm.fit(X_train,y_train)

# review linear regression coefficients (Y = B0 + B1X)
coe = lm.coef_
cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coef'])
print('------------------Co-Efficients-----------------------------------------------------')
print(cdf)

#PREDICTIONS
# giving values for x, using the X-test [Predictions]
predictions = lm.predict(X_test)
print('------------------X-test Predictions-----------------------------------------------------')
print(predictions)
# calculating residuals: difference between actual values (y_test) and predicted values (predictions)
print('------------------Error Values-----------------------------------------------------')
residuals = y_test - predictions
print(residuals)

# visualizating predictions (x-axis) and actual y values (y-axis)
sns.scatterplot(x = predictions, y = y_test)
plt.xlabel("Predictions")
plt.title("Yearly Amount Spent vs Model Predictions")
plt.gcf().canvas.manager.set_window_title('Visualise the Predictions')
plt.show()

#Predicting the first 10 Rows
s = df.iloc[:10]
pre = s[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']].values
print('------------------Custom Input and Prediction Results-----------------------------------------------------')
print(lm.predict(pre))
 
print('------------------Accuracy and Evaluation-----------------------------------------------------')
# average distance of every point to actual line
print("Mean Absoulte Error", mean_absolute_error(y_test, predictions))

# squaring everytime there is a distance between the predicted value & the real value
print("Mean Squared Error", mean_squared_error(y_test, predictions))
print("RMSE", math.sqrt(mean_squared_error(y_test, predictions)))

