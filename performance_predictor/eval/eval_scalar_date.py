import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error
import pandas as pd

df = pd.read_csv('./dataset/eval_date.csv')

X = df[['1']]  # rows, rows * expr_cnt
y = df['0']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

date_model = LinearRegression(fit_intercept=True)
date_model.fit(X_train, y_train)

y_pred = date_model.predict(X_test)

print("coef:", date_model.coef_)
print("intercept:", date_model.intercept_)
print("Median AE:", median_absolute_error(y_test, y_pred))

# import numpy as np
# from scipy.stats import spearmanr, pearsonr
# import matplotlib.pyplot as plt
# def evaluate():
#     # ===== 基础验证 =====
#     feat_test = X_test.iloc[:, 0].values
#     cpu_test  = y_test.values
# 
#     # 1️⃣ Spearman rank correlation（最重要）
#     rho, p_rho = spearmanr(feat_test, cpu_test)
#     print("Spearman rho (feat vs CPU):", rho, "p-value:", p_rho)
# 
#     # 2️⃣ Pearson correlation（辅助）
#     r, p_r = pearsonr(feat_test, cpu_test)
#     print("Pearson r (feat vs CPU):", r, "p-value:", p_r)
# 
#     # ===== log–log 稳定性验证（强烈推荐）=====
#     log_feat = np.log1p(feat_test)
#     log_cpu  = np.log1p(cpu_test)
# 
#     rho_log, p_rho_log = spearmanr(log_feat, log_cpu)
#     print("Spearman rho (log-log):", rho_log, "p-value:", p_rho_log)
# 
#     r_log, p_r_log = pearsonr(log_feat, log_cpu)
#     print("Pearson r (log-log):", r_log, "p-value:", p_r_log)
# 
#     # ===== 可视化（不会影响模型）=====
#     plt.figure(figsize=(5,4))
#     plt.scatter(feat_test, cpu_test, alpha=0.4)
#     plt.xlabel("rows × expr_cnt")
#     plt.ylabel("CPUTime")
#     plt.title("EvalScalar CPU vs rows×expr")
#     plt.tight_layout()
#     plt.show()
# 
#     plt.figure(figsize=(5,4))
#     plt.scatter(log_feat, log_cpu, alpha=0.4)
#     plt.xlabel("log(rows × expr_cnt)")
#     plt.ylabel("log(CPUTime)")
#     plt.title("EvalScalar CPU (log-log)")
#     plt.tight_layout()
#     plt.show()
    
def date_predict_from_scalar(date_eval, card):
    return date_model.predict(np.array([[card * date_eval]]))

# if __name__ == "__main__":
    # evaluate()