from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.metrics import r2_score , mean_squared_error 
import pandas as pd
from utils import target , createMissedColumns
import matplotlib.pyplot as plt
from math import sqrt
import json
from numpy import log
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
    ("regressor" , LinearRegression())
])
X_train , X_test ,y_train , y_test = train_test_split(X , y ,test_size = 0.3 , shuffle = True , random_state= 42)
model_pipe.fit(X_train , y_train)
y_pred = model_pipe.predict(X_train)  
evaluate_performance  = {}
r2 , mse = r2_score(y_train , y_pred) , mean_squared_error(log(y_train) , log(y_pred))
evaluate_performance["min_train"] =[ min(y_train)]
evaluate_performance["max_train"] = [max(y_train)]
evaluate_performance["min_test"] = [min(y_test)]
evaluate_performance["max_test"] = [max(y_test)]
evaluate_performance["r2_score_train"] = [round(r2 , 3)]
evaluate_performance["log_rmse_train"] = [round(sqrt(mse) , 3)]
y_pred = model_pipe.predict(X_test)
r2 , mse = r2_score(y_test , y_pred) , mean_squared_error(y_test , y_pred)
evaluate_performance["r2_score_test"] = [round(r2 , 3)]
evaluate_performance["log_rmse_test"] = [round(sqrt(mse) , 3)]

pd.DataFrame(evaluate_performance).to_csv("evaluation_performance.csv")
print(pd.DataFrame(evaluate_performance))
""" save solution  git push origin main"""
try:
    df = pd.read_csv("clean_test.csv")
    df = df[cols]
    solution = pd.DataFrame({ 'Id' :list(range(len(df))) , target : model_pipe.predict(df) })
    solution.to_csv("solution_linear_regression.csv")
except Exception as e :
    print(e)
#cls ; py linear_regression.py