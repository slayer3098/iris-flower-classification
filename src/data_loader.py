from sklearn.datasets import load_iris
import pandas as pd

def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    return X, y
