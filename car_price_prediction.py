import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

data = pd.read_csv("CarPrice_Assignment.csv")

X = data[['horsepower', 'enginesize', 'curbweight']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

plt.figure(figsize=(15, 5))

# graph of the 'horsepower' property
plt.subplot(1, 3, 1)
sns.scatterplot(x='horsepower', y='price', data=data)
sns.regplot(x='horsepower', y='price', data=data, scatter=False, color='red')

# graph of the 'enginesize' property
plt.subplot(1, 3, 2)
sns.scatterplot(x='enginesize', y='price', data=data)
sns.regplot(x='enginesize', y='price', data=data, scatter=False, color='red')

# graph of the 'curbweight' property
plt.subplot(1, 3, 3)
sns.scatterplot(x='curbweight', y='price', data=data)
sns.regplot(x='curbweight', y='price', data=data, scatter=False, color='red')

plt.tight_layout()
plt.show()
