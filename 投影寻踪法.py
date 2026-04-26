import numpy as np
import pandas as pd
# 1. 读取Excel
def load_data(file_path):
    return pd.read_excel(file_path, sheet_name=None)
# 2. 标准化
def normalize(X):
    return (X - X.min()) / (X.max() - X.min() + 1e-10)
# 3. 定义局部密度函数
def density_numpy(z, h=0.1):
    z = z.reshape(-1, 1)
    diff = z - z.T
    K = np.exp(-(diff ** 2) / (h ** 2))
    np.fill_diagonal(K, 0)
    n = len(z)
    return np.sum(K) / (n * (n - 1))
# 4. 投影寻踪目标函数
def projection_index(theta, phi, X):
    # 非负约束（第一象限）
    a1 = np.cos(theta) * np.cos(phi)
    a2 = np.cos(theta) * np.sin(phi)
    a3 = np.sin(theta)
    a = np.array([a1, a2, a3])
    z = X @ a
    S = np.std(z)
    D = density_numpy(z)
    return S * D
# 5. 自适应步长 + 收敛判据
def adaptive_search(X, init_step=0.2, tol=0.01, max_iter=20):

    # 全局粗搜索
    step = init_step
    best_Q = -np.inf
    best_theta, best_phi = 0, 0
    for theta in np.arange(0, np.pi / 2, step):
        for phi in np.arange(0, np.pi / 2, step):

            Q = projection_index(theta, phi, X)

            if Q > best_Q:
                best_Q = Q
                best_theta, best_phi = theta, phi
    # 用于收敛判断（记录上一轮局部最优）
    prev_Q = None
    # 变步长迭代
    for iteration in range(max_iter):
        step = step / 2
        candidate_Q = -np.inf
        candidate_theta, candidate_phi = 0, 0
        # 局部搜索
        for theta in np.arange(max(0, best_theta - step), min(np.pi / 2, best_theta + step), step):
             for phi in np.arange(max(0, best_phi - step), min(np.pi / 2, best_phi + step),step):
                Q = projection_index(theta, phi, X)
                if Q > candidate_Q:
                    candidate_Q = Q
                    candidate_theta, candidate_phi = theta, phi
        # 收敛判据（前后两次局部搜索的结果）
        if prev_Q is not None:
            change_rate = abs(candidate_Q - prev_Q) / (abs(prev_Q) + 1e-10)
            if change_rate < tol:
                print(f"达到精度要求，共迭代 {iteration}次")
                break
        # 更新全局最优
        if candidate_Q > best_Q:
            best_Q = candidate_Q
            best_theta, best_phi = candidate_theta, candidate_phi
        # 更新上一轮局部最优
        prev_Q = candidate_Q
    return best_theta, best_phi, best_Q

# 6. 主程序
def run_pp_model(file_path):
    data_dict = load_data(file_path)
    results = {}

    for year, df in data_dict.items():
        # 第一列是城市名
        X = df.iloc[:, 1:4]
        # 标准化
        X = normalize(X).values
        theta, phi, Q = adaptive_search(X)
        # ===== 最优权重 =====
        a1 = np.cos(theta) * np.cos(phi)
        a2 = np.cos(theta) * np.sin(phi)
        a3 = np.sin(theta)

        weights = np.array([a1, a2, a3])

        # ===== 综合指数 =====
        z = X @ weights

        results[year] = {
            "weights": weights,
            "Q": Q,
            "index": z,
            "name":df.iloc[:,0].values
        }

    return results


# 7. 运行
if __name__ == "__main__":

    file_path = "your_data.xlsx"
    results = run_pp_model(file_path)

    output_path = "result.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for year, res in results.items():
            df= pd.DataFrame(res["name"],columns=["UA"]).copy()
            df["index"]= res["index"]
            df["w1"] = res["weights"][0]
            df["w2"] = res["weights"][1]
            df["w3"] = res["weights"][2]
            df["Q"] = res["Q"]
            # 写入对应年份sheet
            df.to_excel(writer, sheet_name=str(year), index=False)
    print(f"结果已保存至：{output_path}")