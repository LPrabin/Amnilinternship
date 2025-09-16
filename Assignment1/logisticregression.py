from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def main():
    breast_cancer = datasets.load_breast_cancer()
    logistic_reg = LogisticRegression()
    X = breast_cancer.data
    y = breast_cancer.target
    X_train , X_test , y_train , y_test = train_test_split(X, y , test_size = 0.2)
    logistic_reg.fit(X_train, y_train)
    logistic_reg.predict(X_test)
    print(logistic_reg.score(X_test, y_test))

if __name__ == "__main__":
    main()
