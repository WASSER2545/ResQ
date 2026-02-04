import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('/Users/zsy/Documents/codespace/python/FlexBench_original/Demo1/models/performance_predictor/dataset/joins_simple.csv')

X = df[['2', '3', '4', '5', '6', '7']] # build rows, probe rows, selectivity, 
y = df['0']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

hash_model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

y_train_log = np.log1p(y_train)   # log(1 + y)
y_test_log  = np.log1p(y_test)

hash_model.fit(X_train, y_train_log)

def hash_join_predict(join_node_info):
    if not join_node_info:
        return 0
    join_input = np.array(join_node_info)
    y_pred_log = hash_model.predict(join_input)
    y_pred = np.expm1(y_pred_log)
    
    return sum(y_pred)