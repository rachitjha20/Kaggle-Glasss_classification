import numpy as np
import pandas as pd 
import joblib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('glass.csv')

x= dataset.iloc[:,:-1]
y= dataset.iloc[:, 9 ]

x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.20, random_state=42)

s_x = StandardScaler()

x_train= s_x.fit_transform(x_train)
x_test = s_x.transform(x_test)


cls = RandomForestClassifier(criterion='entropy',n_estimators=300, random_state=42)
cls.fit(x_train, y_train)

print('Accuracy is' ,cls.score(x_test,y_test)*100,'%')

filename = 'finalize_model.sav'
joblib.dump(cls,filename)



