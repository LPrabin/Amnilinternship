from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
import numpy as np
def main():
    diabetes = datasets.load_diabetes()
    linear_reg = LinearRegression()
    X = diabetes.data
    y = diabetes.target
    X_train , X_test, y_train ,y_test = train_test_split(X,y , test_size=0.2)
    linear_reg.fit(X_train,y_train)
    predictions = linear_reg.predict(X_test)
    print(predictions[:5].round(),y_test[:5]) 
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    experiment_description = {
        "name" : "Linear Regression",
        "description" : "Linear Regression Model"
    }
    experiment_tags = {
        "type" : "regression",
        "author" : "seven"
    }
    regression_experiment = mlflow.create_experiment(
        name = "Linear Regression",
        artifact_location = "./mlruns",
        tags = experiment_tags
    )


    #metric definationfor regression
    mae = mean_absolute_error(y_test,predictions)
    mse = mean_squared_error(y_test,predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test,predictions)

    metric = {
        "mae" : mae,
        "mse" : mse,
        "rmse" : rmse,
        "r2" : r2
    }
    with mlflow.start_run(experiment_id=regression_experiment) as run:
        mlflow.log_metrics(metric)
        mlflow.sklearn.log_model(linear_reg,artifact_path="linear-regression-model")

    print(linear_reg.score(X_test,y_test))
if __name__ == "__main__":
    main()


