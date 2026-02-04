import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('./dataset/joins_simple.csv')

X = df[['2', '3', '4', '5', '6', '7']] # build rows, probe rows, selectivity, 
y = df['0']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

y_train_log = np.log1p(y_train)   # log(1 + y)
y_test_log  = np.log1p(y_test)

model.fit(X_train, y_train_log)


from sklearn.metrics import mean_absolute_error, mean_squared_error
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
# print(y_test)
from datetime import datetime

log_df = X_test.copy()
log_df["cpu_true"] = y_test
log_df["cpu_pred"] = y_pred
log_df["relative_error"] = np.abs(y_pred - y_test) / np.maximum(y_test, 1e-6)

# 时间戳，方便多次实验
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"/Users/zsy/Documents/codespace/python/FlexBench_original/Demo1/models/performance_predictor/test_results/cpu_pred_log_{ts}.csv"

log_df.to_csv(log_path, index=False)

mae = mean_absolute_error(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"MAE  : {mae:.4f}")

rel_error = np.abs(y_pred - y_test) / np.maximum(y_test, 1e-6)

print("Median relative error:", np.median(rel_error))
print("90% relative error    :", np.percentile(rel_error, 90))

from scipy.stats import spearmanr

rho, _ = spearmanr(y_test, y_pred)
print("Spearman rho:", rho)
