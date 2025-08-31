import pandas as pd
import json
from utils import cleanColumn , cleanByMode , showSummaryValuesColumn,createMissedColumns  , plotColumn , removeOutliers , target  , feature_selection
def confirm(df , col):
    print(df[col].isna().sum())
    showSummaryValuesColumn(df ,col)
#========================================================================
cols = None
with open("cols.json" , 'r') as f:
    cols  = json.load(f)["cols"]
cols.remove(target)
  
df = pd.read_csv("test.csv")[cols]
df = cleanColumn(df , 'TotalBsmtSF')
df = cleanColumn(df , "MasVnrArea")
df = cleanColumn(df , "GarageArea")
df = cleanColumn(df , "LotFrontage")
df = cleanByMode(df , 'GarageCars')
df = cleanByMode(df , 'GarageFinish')
df = cleanByMode(df , 'Utilities')
df = cleanByMode(df , 'BsmtExposure')
df = cleanByMode(df , 'MSZoning')
df = cleanByMode(df , 'BsmtCond')
df = cleanByMode(df , 'Exterior1st')
df = cleanByMode(df , 'GarageQual')
df = cleanByMode(df , 'Functional')
df = cleanByMode(df , 'Exterior2nd')
df = cleanByMode(df , 'GarageType')
df = cleanByMode(df , 'KitchenQual')
df = cleanByMode(df , 'BsmtFullBath')
df = cleanByMode(df , 'BsmtFinType1')
df = cleanByMode(df , 'BsmtQual')
df = cleanByMode(df , 'BsmtHalfBath')
df = cleanByMode(df , 'FireplaceQu')
df = cleanByMode(df , "SaleType")
df = cleanByMode(df , "GarageCond")
df["CentralAir"] = df["CentralAir"] == 'Y'
df["CentralAir"] = df["CentralAir"].map({True : 1 , False : 0})
df.to_csv("clean_test.csv")

""" cls ; py clean_test.py"""