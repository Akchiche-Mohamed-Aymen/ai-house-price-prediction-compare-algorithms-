from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.metrics import r2_score , mean_squared_error , mean_absolute_error
import pandas as pd
from utils import target
import matplotlib.pyplot as plt
from math import sqrt
import json

"""-------------------------------------Import Tools ----------------------------------------------------------"""
cols = None
with open("cols.json" , 'r') as f:
    cols  = json.load(f)["cols"]
cols.remove(target)
df = pd.read_csv('clean_train.csv')
y = df[target]
X = df[cols]
num_cols = X.select_dtypes(include=["int32" , "int64" , "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

num_pipe =  Pipeline([
    ("scale" , StandardScaler())
])
cat_pipe =  Pipeline([
    ("impute" , SimpleImputer(strategy= 'most_frequent')),
    ("encoder" , OneHotEncoder(handle_unknown = 'ignore'))
])

""""""
prepocessor = ColumnTransformer([
    ("num" , num_pipe , num_cols),
    ("cat" , cat_pipe , cat_cols)
    ])

model_pipe = Pipeline([
    ('prepocessor' , prepocessor),
    ("regressor" , KNeighborsRegressor())
])
X_train , X_test ,y_train , y_test = train_test_split(X , y ,test_size = 0.3 , shuffle = True , random_state= 42)
model_pipe.fit(X_train , y_train)
y_pred = model_pipe.predict(X_train)
r2 , mse = r2_score(y_train , y_pred) , mean_squared_error(y_train , y_pred)
print(f'\n\n {"***"*5 } Model Train knn Evaluation {"***"*5}')
print("min" , min(y_train))
print("rmse = " , sqrt(mse) )
print("max" , max(y_train))
print("mae = " , mean_absolute_error(y_train , y_pred))
print("r2_score = " , r2)

print(f'\n\n {"***"*5 } Model Test knn Evaluation {"***"*5}')
y_pred = model_pipe.predict(X_test)
r2 , mse = r2_score(y_test , y_pred) , mean_squared_error(y_test , y_pred)
print("min" , min(y_test))
print("rmse = " , sqrt(mse) )
print("max" , max(y_test))
print("mae = " , mean_absolute_error(y_test , y_pred))
print("r2_score = " , r2 )

df = pd.read_csv("clean_test.csv")[cols]
solution = pd.DataFrame({ 'Id' :list(range(len(df))) , target : model_pipe.predict(df) })

""" save solution
solution.to_csv("solution_knn.csv")"""

"""cls ; py knn.py"""