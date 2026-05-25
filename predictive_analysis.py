import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Walmart.csv")
print(data.head())
print(data.shape)
print(data.info())
print(data.isnull())
print(data.isnull().sum())
# converting the column
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
print(data.info())
#extract date Features
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Week'] = data['Date'].dt.isocalendar().week
#visualize sales trends
store1 = data[data['Store'] == 1]
plt.figure(figsize=(12,6))
plt.plot(store1['Date'], store1['Weekly_Sales'])
plt.title("Store 1 Weekly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.show()
# Features 
X = store1[['Holiday_Flag','Temperature','Fuel_Price','CPI','Unemployment','Year','Month','Week']]
y = store1['Weekly_Sales']
# importing libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
# evaluating accuracy
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)
#visualize predictions
plt.figure(figsize=(10,6))
plt.scatter(y_test, predictions)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
#Calculate Model Accuracy (R**2)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, predictions)
print("R2 Score:", r2)