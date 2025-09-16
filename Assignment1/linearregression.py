from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def main():
    diabetes = datasets.load_diabetes()
    linear_reg = LinearRegression()
    X = diabetes.data
    y = diabetes.target
    X_train , X_test, y_train ,y_test = train_test_split(X,y , test_size=0.2)
    linear_reg.fit(X_train,y_train)
    predictions = linear_reg.predict(X_test)
    print(predictions[:5].round(),y_test[:5]) 

    print(linear_reg.score(X_test,y_test))
if __name__ == "__main__":
    main()