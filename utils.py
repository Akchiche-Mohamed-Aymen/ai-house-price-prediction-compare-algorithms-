import matplotlib.pyplot as plt

target = "SalePrice"
def createMissedColumns(df):
    rows = len(df)
    missing_percentage = 100 * df.isna().sum() / rows 
    missing_percentage = missing_percentage[missing_percentage > 0]
    return missing_percentage
def cleanColumn(df , column):
    skew = abs(df[column].skew())
    if skew> 1:
        df.fillna({column: df[column].median()}, inplace=True)
    else:
        df.fillna({column: df[column].mean()}, inplace=True)
    return df

def cleanByMode(df , column):
    df[column] = df[column].fillna(df[column].mode()[0])
    return df

def showSummaryValuesColumn(df , column):
    print(f'set of values of column {column} is : {set(df[df[column].notna()][column])}')
def plotColumn(df , column):
    plt.figure(figsize=(6,4))
    plt.boxplot(df[column])
    plt.title(f'plotting of {column}')
    plt.ylabel(column)
    plt.grid(1)
    plt.savefig("boxplot")
    plt.show()

def removeOutliers(df , column):
    q1 = df[column].quantile(.25)
    q3 = df[column].quantile(.75)
    IQR = q3 - q1
    lower = q1 - IQR * 1.5
    upper = q3 + IQR * 1.5 
    df = df[(df[column] >= lower) & (df[column] <= upper)  ].reset_index(drop = True)  
    return df 

def correlation(df , thr):
    col_corr = set()
    not_corr = set()
    matrix = df[df.select_dtypes(include=["int32" , "int64" , "float64"]).columns].corr()
    for i in range(len(matrix.columns)):
        for j in range(i):
            cl1 = split(matrix.columns[i] )              
            cl2 = split(matrix.columns[j])
            if abs(matrix.iloc[i , j]) > thr :              
                col_corr.add((cl1 , cl2))
            else :
                not_corr.add((cl1 , cl2))
    return col_corr , not_corr 
def correlationTarget(df , thr):
    best_col = set()
    matrix = df[df.select_dtypes(include=["int32" , "int64" , "float64"]).columns].corr().SalePrice
    for key in matrix.keys():
        val = matrix[key]
        if abs(val) > thr and key != target:
            best_col.add(split(key))
    return best_col 

def split(s = "" ):
    return s.split("_")[0]

def clean_set(summary_corr , mul_corr):
    saved = list()
    for e in summary_corr:
        for ele in mul_corr:
            if e in ele:
                saved.append(e)
                break
    for e in saved :
        if e in summary_corr:
            summary_corr.remove(e)
    return summary_corr
def feature_selection(df , thr):
    mul_corr , not_corr =  correlation(df.drop(columns = [target] ) , thr)
    summary_corr = set()
    for e in not_corr:
        summary_corr.add(e[0])
        summary_corr.add(e[1])
    summary_corr = clean_set(summary_corr , mul_corr)           
    help_pred = correlationTarget(df , thr)
    selected = help_pred.copy()
    for e in mul_corr:
        if e[0] not in help_pred and e[1] not in help_pred:
            selected.add(e[0])
    selected.add(target)
    return selected.union(summary_corr)
