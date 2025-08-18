import pandas as pd 
from utils import cleanColumn , cleanByMode , showSummaryValuesColumn 
def createMissedColumns(df):
    missing_percentage = 100 * df.isna().sum() / rows 
    missing_percentage = missing_percentage[missing_percentage > 0]
    return missing_percentage
def confirm(df , col):
    print(df[col].isna().sum())
    showSummaryValuesColumn(df ,col)
#========================================================================
target = "SalePrice"
df = pd.read_csv("train.csv").drop(columns=["Id"])
print(df.shape)
rows = df.shape[0]
missing_percentage = createMissedColumns(df)
droppedColumns =  missing_percentage[missing_percentage > 50].index
df = df.drop(columns = droppedColumns)
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

df.to_csv('clean_train.csv')

#py prepare_data.py
