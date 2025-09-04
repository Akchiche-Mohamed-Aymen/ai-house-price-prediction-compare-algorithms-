from sharedCodeForAlgos import create_model , evaluate_model
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

"""-------------------------------------Import Tools ----------------------------------------------------------"""

param_grid = {
    "regressor__n_neighbors": [3, 5],
    "regressor__weights": ["uniform", "distance"],
    "regressor__metric": ["euclidean", "manhattan"]
}

model_pipe, X_train, X_test, y_train, y_test = create_model(
    KNeighborsRegressor(),
    param_grid=param_grid
)

print("Best Params:", model_pipe.best_params_)
print("Best Score:", model_pipe.best_score_)
best_model = model_pipe.best_estimator_

y_pred_train  ,y_pred_test =  best_model.predict(X_train) , best_model.predict(X_test)
evaluate_performance  =  evaluate_model(y_pred_train , y_train  , y_pred_test , y_test)   
pd.DataFrame(evaluate_performance).to_csv("knn_evaluation_performance.csv")
print(pd.DataFrame(evaluate_performance))
""" save solution  git push origin main"""


"""cls ; py knn.py"""