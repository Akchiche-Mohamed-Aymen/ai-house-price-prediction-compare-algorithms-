import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import json
from utils import cleanColumn , cleanByMode , showSummaryValuesColumn,createMissedColumns  , plotColumn , removeOutliers , target  , feature_selection
def confirm(df , col):
    print(df[col].isna().sum())
    showSummaryValuesColumn(df ,col)
#========================================================================
df = pd.read_csv("train.csv").drop(columns= ['Id']) 
rows = df.shape[0]
missing_percentage = createMissedColumns(df)
THRESHOLD_REQUIRED_TO_REMOVE_COLUMN = 50
droppedColumns =  missing_percentage[missing_percentage > THRESHOLD_REQUIRED_TO_REMOVE_COLUMN].index
df = df.drop(columns = droppedColumns).reset_index(drop = True)
missing_percentage = createMissedColumns(df)
#====================handle missed values===================
df = cleanColumn(df , "LotFrontage")
df = cleanColumn(df , 'MasVnrArea')
df = cleanByMode(df , 'BsmtQual')
df = cleanByMode(df , 'BsmtCond')
df = cleanByMode(df , 'BsmtExposure')
df = cleanByMode(df , 'BsmtFinType1')
df = cleanByMode(df , 'BsmtFinType2')
df = cleanByMode(df , 'Electrical')
df = cleanByMode(df , 'FireplaceQu')
df = cleanByMode(df , 'GarageType')
df = cleanByMode(df , 'GarageFinish')
df = cleanByMode(df , 'GarageQual')
df = cleanByMode(df , 'GarageCond')
df = cleanColumn(df , "GarageYrBlt") #error here 
#==============================================================
"""df = removeOutliers(df,target)"""
df["CentralAir"] = df["CentralAir"] == 'Y'
df["CentralAir"] = df["CentralAir"].map({True : 1 , False : 0})
ohe = OneHotEncoder(handle_unknown = 'ignore' , sparse_output = False ).set_output(transform  = "pandas" )
strColumns = df.select_dtypes(include=["object"]).columns
for col in strColumns:   
    encode = ohe.fit_transform(df[[col]])
    df = pd.concat([df , encode  ] , axis = 1 )
columns = list(feature_selection(df, 0.5))
columns.remove("Condition2")
columns.remove("BsmtFinType2")
df = df[columns]
df.to_csv('clean_train.csv')

with open("cols.json" , "w") as f:
    json.dump({"cols" : columns} , f)
#cls ; py prepare_data.py

