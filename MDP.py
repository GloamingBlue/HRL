import numpy as np
from MRP import compute_v


S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合
# 状态转移函数
P = {
    "s1-保持s1-s1": 1.0,
    "s1-前往s2-s2": 1.0,
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
}
# 奖励函数
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
}
gamma = 0.5  # 折扣因子
MDP = (S, A, P, R, gamma)

# 策略1,随机策略
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}
# 策略2
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}


def join(str1: str, str2: str) -> str:
    """把输入的两个字符串通过“-”连接,便于使用上述定义的P、R变量"""
    return str1 + '-' + str2

def sample(
        MDP: tuple[list[int], list[str], dict[str, float], dict[str, float], float], 
        Pi: dict, 
        timestep_max: int, 
        number: int
) -> list[list[tuple[str, str, float, str]]]:
    """采样函数,策略Pi,限制最长时间步timestep_max,总采样序列数number"""
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]  # 选择一个除终止状态s5以外的状态作为起点
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            for a_opt in A:  # 根据策略函数随机选择当前状态的动作
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            for s_opt in S:  # 根据状态转移函数选择下一个状态
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))
            s = s_next
        episodes.append(episode)
    return episodes


def MC(
        episodes: list[list[tuple[str, str, float, str]]], 
        V: dict[str, float], 
        N: dict[str, int], 
        gamma: float
) -> None:
    """使用蒙特卡洛采样计算采样序列所有状态的价值"""
    for episode in episodes:
        G = 0
        for i in range(len(episode) - 1, -1, -1):  # 从最后一个状态(其实是倒数第二个)向前迭代计算
            s, a, r, s_next = episode[i]
            G = G * gamma + r
            N[s] += 1
            V[s] += (G - V[s]) / N[s]
            

def occupancy(
        episodes: list[list[tuple[str, str, float, str]]], 
        s: str, 
        a: str, 
        timestep_max: int, 
        gamma: float
) -> float:
    """计算状态动作对(s, a)出现的频率,以此来估计策略的占用度量"""
    rho = 0
    total_times = np.zeros(timestep_max)
    occur_times = np.zeros(timestep_max)
    for episode in episodes:
        for i in range(len(episode)):
            s_opt, a_opt, _, __= episode[i]
            total_times[i] += 1  # 记录t时刻的状态动作对数
            if s == s_opt and a == a_opt:
                occur_times[i] += 1  # 记录t时刻的目标状态动作对数
    for i in reversed(range(timestep_max)):
        if total_times[i]:
            rho += (gamma ** i) * (occur_times[i] / total_times[i])
    return (1 - gamma) * rho


if __name__ == '__main__':

    # 转化后的MRP的状态转移矩阵
    P_from_mdp_to_mrp = [
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.5],
        [0.0, 0.1, 0.2, 0.2, 0.5],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
    R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

    V1 = compute_v(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)
    print(f"MDP中每个状态价值分别为\n{V1}\n")

    # 采样numbers次,每个序列最长不超过20步
    numbers = 10000
    episodes = sample(MDP, Pi_1, 20, numbers)
    print(f'第一条序列{episodes[0]}\n最后一条序列{episodes[numbers-1]}\n')
    
    V2 = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
    N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
    MC(episodes, V2, N, gamma)
    print(f"使用蒙特卡洛采样法计算MDP的状态价值为{V2}\n")  # 发现numbers越大,采样结果和解析解越接近

    # 采样numbers次,每个序列最长不超过timestep_max步
    timestep_max = 1000
    episodes_1 = sample(MDP, Pi_1, timestep_max, numbers)
    episodes_2 = sample(MDP, Pi_2, timestep_max, numbers)
    rho_1 = occupancy(episodes_1, "s4", "概率前往", timestep_max, gamma)
    rho_2 = occupancy(episodes_2, "s4", "概率前往", timestep_max, gamma)
    print(f"策略Pi_1的占用度量为{rho_1},策略Pi_2的占用度量为{rho_2}")
