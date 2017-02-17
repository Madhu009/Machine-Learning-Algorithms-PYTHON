import pandas as pd
from sklearn.linear_model import LinearRegression

data=pd.read_csv("C:/Users/Madhu/Desktop/Advertising.csv",index_col=0)
data.head()

# create X and y
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data.Sales
z=data.TV

lr=LinearRegression()
lr.fit(X,y)

print(lr.coef_)
print(list(zip(feature_cols,lr.coef_)))
l=[50,25,25]
print(lr.predict(l))
