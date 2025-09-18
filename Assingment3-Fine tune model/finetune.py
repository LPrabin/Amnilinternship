from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , accuracy_score


iris = load_iris()
X,y = iris.data , iris.target

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

randomforestc = RandomForestClassifier()

param_grid = {
    'n_estimators':[50,100,200],
    'max_depth':[None,10,20,30],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4]
}

grid_search_ = GridSearchCV(estimator=randomforestc,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2) 

grid_search_.fit(X_train,y_train)

print("Best Hyperparameters:",grid_search_.best_params_)

best_model = grid_search_.best_estimator_

y_pred = best_model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))