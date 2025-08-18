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