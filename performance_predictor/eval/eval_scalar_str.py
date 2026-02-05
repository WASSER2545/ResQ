import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error
import pandas as pd

df = pd.read_csv('./dataset/eval_str.csv')

X = df[['1']]  # rows, rows * expr_cnt
y = df['0']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

str_model = LinearRegression(fit_intercept=True)
str_model.fit(X_train, y_train)

y_pred = str_model.predict(X_test)

print("coef:", str_model.coef_)
print("intercept:", str_model.intercept_)
print("Median AE:", median_absolute_error(y_test, y_pred))
def str_predict_from_scalar(str_eval, card):
    return str_model.predict(np.array([[card * str_eval]]))
