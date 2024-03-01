import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

df = pd.read_csv(r"C:\Users\moham\Desktop\MY AI\ML Projects\USA Housing\USA_Housing.csv")
df=pd.DataFrame(df)

# 1- some info about the data : 

# print(df.sample(5))
# print(df.shape)
# print(df.columns)
# print(df.info())
# print(df.describe())
# print(df.describe(exclude=[np.number]))


# 2 - EDA : 
# sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm")
# plt.show()

# df.boxplot()
# plt.show()


# 3 - data preprocessing :

df.drop('Address' ,  axis=1, inplace=True)
# print(df.corr())

from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()

df[['Avg. Area Income','Area Population']]= scaler.fit_transform(df[['Avg. Area Income','Area Population']])


# 4 - data Splitting : 

x= df.drop(['Price'],axis=1)
y= df['Price']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 , random_state=2) 


# 5 - fitting the model :

from sklearn.linear_model import LinearRegression
from sklearn import metrics
lm = LinearRegression()
lm.fit(x_train,y_train)
y_pred=lm.predict(x_train)
predict= lm.predict(x_test)

# 6 - model evaluation : 

# print("Mean absolute error (MAE):", metrics.mean_absolute_error(y_test,predict))
# print("Mean square error (MSE):", metrics.mean_squared_error(y_test,predict))
# print("Root mean square error (RMSE):", np.sqrt(metrics.mean_squared_error(y_test,predict)))
# print("R-squared value of predictions:",round(metrics.r2_score(y_test,predict),3))

# 7 - save the model : 

import pickle
pickle.dump(lm,open('USA_housing.pkl','wb'))
model = pickle.load(open( 'USA_housing.pkl', 'rb' ))

# 8 - now we can try the model : 

prediction_data = pd.DataFrame(data=np.array([79545.4585743167,5.68286132161558,7,4,23086.8005026864]).reshape(1,5))
prediction = model.predict(prediction_data)
# print(prediction)


