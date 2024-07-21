#income prediction model
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

salary_data=pd.read_csv('Salary_Data (1).csv')

#defining x & y

x=salary_data.drop(['Salary'],axis=1)
y=salary_data.drop(['YearsExperience'],axis=1)

X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=32,test_size=.3)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

#fitting the model

m=regressor.fit(X_train,y_train)  

#save the model

pickle.dump(regressor,open('model.pkl','wb'))
