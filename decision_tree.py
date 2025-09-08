from sharedCodeForAlgos import  evaluate_model , create_model
from sklearn.tree import DecisionTreeRegressor
from pandas import DataFrame
model_pipe , X_train , X_test ,y_train , y_test  = create_model(
    algo = DecisionTreeRegressor(
    max_depth =  5 , random_state = 42 , min_samples_split = 2 , min_samples_leaf = 2)  , 
    scaling = True
    )
y_pred_train  ,y_pred_test =  model_pipe.predict(X_train) , model_pipe.predict(X_test)
evaluate_performance  =  evaluate_model(y_pred_train , y_train  , y_pred_test , y_test)   
DataFrame(evaluate_performance).to_csv("DecisionTreeRegressor_evaluation_performance.csv")
print(DataFrame(evaluate_performance))

"""cls ; py decision_tree.py"""