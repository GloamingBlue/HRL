import numpy as np
from typing import List
from numpy.typing import NDArray

np.random.seed(42)
# 定义状态转移矩阵P, 状态从s1到s6
P = np.array(
    [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
)
# 定义奖励函数r(s)
rewards = [-1, -2, -2, 10, 1, 0]
# 定义折扣因子
gamma = 0.5

def compute_return(start_index: int, chain: List[int], gamma: float) -> float:
    """给定状态序列(起始状态)到序列最后(终止状态)得到的回报(G)"""
    G = 0
    for i in reversed(range(start_index, len(chain))):  # 使用迭代的方式计算,所以从终止状态的奖励开始计算
        G = gamma * G + rewards[chain[i] - 1]  # 减去1是因为状态从1开始标号而非0
    return G

def compute_v(P: NDArray[np.float64], rewards: List[float], gamma: float, states_num: int) -> NDArray:
    """使用贝尔曼方程的矩阵形式计算价值函数的解析解,states_num是MRP的状态数"""
    rewards = np.array(rewards).reshape((-1, 1))  # 转换为列向量
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value

if __name__ == '__main__':

    # 一个状态序列,s1-s2-s3-s6
    chain = [1, 2, 3, 6]
    start_index = 0
    G = compute_return(start_index, chain, gamma)
    print(f"根据序列{chain}计算得到回报为：{G}.")

    V = compute_v(P, rewards, gamma, 6)
    print(f"MRP中每个状态价值分别为{V}\n")
