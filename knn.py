from sharedCodeForAlgos import create_model , evaluate_model
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

"""-------------------------------------Import Tools ----------------------------------------------------------"""

model_pipe , X_train , X_test ,y_train , y_test  = create_model(algo = KNeighborsRegressor() , distance_based =True)
y_pred_train  ,y_pred_test =  model_pipe.predict(X_train) , model_pipe.predict(X_test)
evaluate_performance  =  evaluate_model(y_pred_train , y_train  , y_pred_test , y_test)   
pd.DataFrame(evaluate_performance).to_csv("knn_evaluation_performance.csv")
print(pd.DataFrame(evaluate_performance))
""" save solution  git push origin main"""


"""cls ; py knn.py"""