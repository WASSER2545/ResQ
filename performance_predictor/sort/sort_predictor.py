import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error
import pandas as pd

df = pd.read_csv('./dataset/sort.csv')

X = df[['1']]  # rows * log(rows)
y = df['0']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sort_model = LinearRegression(fit_intercept=True)
sort_model.fit(X_train, y_train)

y_pred = sort_model.predict(X_test)

print("coef:", sort_model.coef_)
print("intercept:", sort_model.intercept_)
print("Median AE:", median_absolute_error(y_test, y_pred))
def sort_predict(width, res, agg_metrics):
    if not res['sort_key_list']:
        return 0
    card_list = []
    for node in agg_metrics:
        if "Sort" in node["name"]:
            output_rows = node["output_rows"]
            card_list.append(output_rows)
    return sort_model.predict(np.array([[card_list[0] * np.log1p(card_list[0]) * width]])).sum()