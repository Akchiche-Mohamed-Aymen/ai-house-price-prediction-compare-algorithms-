from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , OneHotEncoder , MinMaxScaler
from sklearn.metrics import r2_score , root_mean_squared_error , mean_absolute_percentage_error  
import pandas as pd
from utils import target 
import json
from numpy import log , exp

def get_columns():
    try:
        with open("cols.json" , 'r') as f:
            cols = json.load(f)["cols"]
            return cols
    except :
        return list()

cols = get_columns()
cols.remove(target)
def read_data(file):
    df = pd.read_csv(file)
    y = df[target]
    X = df[cols]
    return X , y
def split_data(X , y):
    return train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42
    )


def create_pipelines(distance_based , scaling):
    Scaler = MinMaxScaler if distance_based else StandardScaler
    steps = list()
    if scaling:
        steps.append(("scale" , Scaler()))
    steps.append(("impute" , SimpleImputer(strategy= 'median')))
    num_pipe =  Pipeline(steps=steps)
    cat_pipe =  Pipeline([
        ("impute" , SimpleImputer(strategy= 'most_frequent')),
        ("encoder" , OneHotEncoder(handle_unknown = 'ignore'))
    ])
    return  num_pipe ,  cat_pipe
def create_model(algo, param_grid=None , distance_based = False , scaling = True):
    # Load data
    X , y = read_data("clean_train.csv")
    # Separate numeric & categorical columns
    num_cols = X.select_dtypes(include=["int32", "int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    # Create preprocessing pipelines
    num_pipe, cat_pipe = create_pipelines(distance_based , scaling)
    prepocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    
    # Full pipeline
    pipe = Pipeline([
        ("prepocessor", prepocessor),
        ("regressor", algo)
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = split_data(X , y)
    
    # If no param_grid → normal fit
    if param_grid is None:
        pipe.fit(X_train, log(y_train))
        cls = pipe.named_steps["regressor"].__class__.__name__
        save_solution(pipe, f"sol_with_{cls}.csv")
        return pipe, X_train, X_test, y_train, y_test
    
    # If param_grid is given → GridSearchCV
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=5,
        scoring= "r2"
    )
    grid.fit(X_train, log(y_train))
    
    cls = grid.best_estimator_.named_steps["regressor"].__class__.__name__
    save_solution(grid.best_estimator_, f"sol_with_{cls}_grid.csv")
    
    return grid, X_train, X_test, y_train, y_test


def save_solution(model_pipe  , fname):
    try:
        df = pd.read_csv("clean_test.csv")
        df = df[cols]
        solution = pd.DataFrame({ 'Id' :list(range(len(df))) , target : model_pipe.predict(df) })
        solution.to_csv(fname)
        with open(".gitignore" , "a")as f:
            f.write(fname)
            f.write('\n')
    except Exception as e :
        print(e)
        
def evaluate_model( y_pred_train ,y_train ,y_pred_test ,  y_test):
    evaluate_performance  = {}
    r2 , mse = r2_score(y_train , exp(y_pred_train)) , root_mean_squared_error(log(y_train) , y_pred_train)
    evaluate_performance["min_train"] =[ min(y_train)]
    evaluate_performance["max_train"] = [max(y_train)]
    evaluate_performance["min_test"] = [min(y_test)]
    evaluate_performance["max_test"] = [max(y_test)]
    evaluate_performance["r2_score_train"] = [round(r2 , 3)]
    evaluate_performance["log_rmse_train"] = [round(mse , 3)]
    evaluate_performance["MAPE_train"] = [round(mean_absolute_percentage_error(y_train , exp(y_pred_train)) , 3)]
    r2 , mse = r2_score(y_test , exp(y_pred_test)) , root_mean_squared_error(log(y_test) , y_pred_test )
    evaluate_performance["r2_score_test"] = [round(r2 , 3)]
    evaluate_performance["log_rmse_test"] = [round(mse , 3)]
    evaluate_performance["MAPE_test"] = [round(mean_absolute_percentage_error(y_test , exp(y_pred_test)) , 3)]
    return evaluate_performance